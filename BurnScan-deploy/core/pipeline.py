"""
core/pipeline.py
================
Self-contained BurnScan image-analysis pipeline.
Replicates the Colab notebook logic without any external repo dependency.

Public API (imported by backend/main.py)
-----------------------------------------
decode_image(raw_bytes)           → img_bgr  (np.ndarray H×W×3 uint8)
run_full_pipeline(img_bgr, k)     → (rgb, burn_r, depth_r, texture_r)
overlay_grid_figure(rgb, r, title, k) → matplotlib Figure
classify_burn(burn_r, depth_r, texture_r) → dict
fig_to_png_bytes(fig)             → bytes
"""

from __future__ import annotations

import io
import math
from typing import Tuple

import cv2
import matplotlib
matplotlib.use("Agg")            # non-interactive backend — safe on server
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from skimage.feature import graycomatrix, graycoprops


# ── Type aliases ─────────────────────────────────────────────────────────────
ImgBGR  = np.ndarray   # H×W×3 uint8, BGR colour order (OpenCV default)
ImgRGB  = np.ndarray   # H×W×3 uint8, RGB colour order
GridArr = np.ndarray   # H×W float, block-averaged feature map


# ════════════════════════════════════════════════════════════════════════════
# 1. Image I/O
# ════════════════════════════════════════════════════════════════════════════

# Maximum allowed dimension for input images (memory cap on Render free tier).
# Larger images are downscaled while preserving aspect ratio. 1024px is more
# than enough for visual burn-feature extraction.
MAX_INPUT_DIM = 1024


def decode_image(raw: bytes) -> ImgBGR:
    """Decode raw JPEG/PNG bytes → BGR ndarray.  Raises ValueError on failure."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file.")
    return img


def fig_to_png_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib Figure to PNG bytes and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════════════
# 2. Feature extraction helpers
# ════════════════════════════════════════════════════════════════════════════

def _block_grid(img_bgr: ImgBGR, k: int, fn) -> GridArr:
    """
    Divide img_bgr into k×k non-overlapping blocks and apply fn(block_bgr)
    to each, collecting scalar results into a 2-D float array.
    """
    H, W = img_bgr.shape[:2]
    rows = H // k
    cols = W // k
    grid = np.zeros((rows, cols), dtype=float)
    for r in range(rows):
        for c in range(cols):
            block = img_bgr[r * k:(r + 1) * k, c * k:(c + 1) * k]
            grid[r, c] = fn(block)
    return grid


# ── Burn-mask feature ─────────────────────────────────────────────────────

def _burn_score(block_bgr: np.ndarray) -> float:
    """
    Heuristic burn probability for a BGR block.
    Score ranges ~[0, 1].  Higher → more likely burned tissue.

    Strategy:
      1. Convert to HSV.
      2. Red/orange hue range → active burn / erythema.
      3. Dark pixels (charring) also contribute.
      4. Desaturation contributes (pale/white blistered tissue).
    """
    hsv   = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2HSV).astype(float)
    H_ch  = hsv[:, :, 0]          # 0-179 in OpenCV
    S_ch  = hsv[:, :, 1] / 255.0  # 0-1
    V_ch  = hsv[:, :, 2] / 255.0  # 0-1

    n = H_ch.size

    # Red/orange mask (hue 0-15 or 160-179 in OpenCV, i.e. 0-30° or 320-360°)
    red_mask = ((H_ch <= 15) | (H_ch >= 160)) & (S_ch > 0.3) & (V_ch > 0.2)
    red_frac = red_mask.sum() / n

    # Charring: very dark pixels
    char_mask = V_ch < 0.18
    char_frac = char_mask.sum() / n

    # Blistering / pale tissue: low saturation, medium-high value
    blister_mask = (S_ch < 0.25) & (V_ch > 0.5)
    blister_frac = blister_mask.sum() / n

    score = 0.5 * red_frac + 0.35 * char_frac + 0.15 * blister_frac
    return float(np.clip(score, 0.0, 1.0))


# ── Depth feature ─────────────────────────────────────────────────────────

def _depth_score(block_bgr: np.ndarray) -> float:
    """
    Proxy for burn depth based on colour and intensity.
    • Darker / charred → deeper (score → 1).
    • Red / erythematous → superficial (score → 0.3–0.5).
    • Pale / white → intermediate–deep (score → 0.6–0.8).
    """
    hsv  = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2HSV).astype(float)
    V    = hsv[:, :, 2].mean() / 255.0   # mean brightness
    S    = hsv[:, :, 1].mean() / 255.0   # mean saturation

    # Charring: very dark
    if V < 0.20:
        return float(np.clip(1.0 - V / 0.20 * 0.2, 0.8, 1.0))

    # Pale / white (blister / full-thickness)
    if S < 0.20 and V > 0.55:
        return 0.65

    # Red / erythema (superficial)
    return float(np.clip(0.5 * (1.0 - V) + 0.1, 0.1, 0.55))


# ── Texture feature ───────────────────────────────────────────────────────

def _texture_score(block_bgr: np.ndarray) -> float:
    """
    GLCM-based texture heterogeneity (contrast) normalised to [0, 1].
    Higher → more irregular surface (deeper burn / eschar).
    """
    gray  = cv2.cvtColor(block_bgr, cv2.COLOR_BGR2GRAY)
    # Quantise to 8 levels to speed up GLCM
    gray8 = (gray // 32).astype(np.uint8)
    if gray8.shape[0] < 2 or gray8.shape[1] < 2:
        return 0.0
    try:
        glcm    = graycomatrix(gray8, distances=[1], angles=[0], levels=8,
                               symmetric=True, normed=True)
        contrast = graycoprops(glcm, "contrast")[0, 0]
        # Empirically contrast sits in ~[0, 20]; clip and normalise
        return float(np.clip(contrast / 20.0, 0.0, 1.0))
    except Exception:
        # Fall back to normalised std-dev if GLCM fails
        return float(np.clip(gray.std() / 128.0, 0.0, 1.0))


# ════════════════════════════════════════════════════════════════════════════
# 3. Full pipeline
# ════════════════════════════════════════════════════════════════════════════

def run_full_pipeline(
    img_bgr: ImgBGR,
    k: int = 10,
) -> Tuple[ImgRGB, GridArr, GridArr, GridArr]:
    """
    Run the three-channel feature extraction pipeline.

    Parameters
    ----------
    img_bgr : np.ndarray   BGR image (H×W×3 uint8)
    k       : int          block size in pixels (5–30)

    Returns
    -------
    rgb      : np.ndarray  RGB version of the input (for display)
    burn_r   : np.ndarray  block grid of burn scores
    depth_r  : np.ndarray  block grid of depth scores
    texture_r: np.ndarray  block grid of texture scores
    """
    rgb       = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    burn_r    = _block_grid(img_bgr, k, _burn_score)
    depth_r   = _block_grid(img_bgr, k, _depth_score)
    texture_r = _block_grid(img_bgr, k, _texture_score)
    return rgb, burn_r, depth_r, texture_r


# ════════════════════════════════════════════════════════════════════════════
# 4. Visualisation — overlay_grid_figure
# ════════════════════════════════════════════════════════════════════════════

_CMAP_BURN    = "Reds"
_CMAP_DEPTH   = "inferno"
_CMAP_TEXTURE = "Blues"


def overlay_grid_figure(
    rgb: ImgRGB,
    grid: GridArr,
    title: str,
    k: int,
    cmap: str = "Reds",
) -> plt.Figure:
    """
    Render the original image with a semi-transparent block-average heat-map
    overlay, exactly as produced in the Colab notebook.

    Parameters
    ----------
    rgb   : H×W×3 uint8 RGB image
    grid  : rows×cols float array of block scores (0-1)
    title : figure title
    k     : block size (used to determine grid line positions)
    cmap  : matplotlib colourmap name

    Returns
    -------
    matplotlib Figure (caller is responsible for closing it)
    """
    H, W     = rgb.shape[:2]
    rows, cols = grid.shape

    fig, ax = plt.subplots(figsize=(7, 7 * H / W), dpi=100)
    ax.imshow(rgb)

    # Colour-map look-up
    cmap_obj   = plt.get_cmap(cmap)
    norm_grid  = (grid - grid.min()) / (grid.max() - grid.min() + 1e-8)

    for r in range(rows):
        for c in range(cols):
            val   = norm_grid[r, c]
            rgba  = cmap_obj(val)
            alpha = 0.30 + 0.45 * val     # more opaque where signal is high
            rect  = patches.Rectangle(
                (c * k, r * k), k, k,
                linewidth=0.4,
                edgecolor="white",
                facecolor=(*rgba[:3], alpha),
            )
            ax.add_patch(rect)

    ax.set_title(title, fontsize=11, pad=8)
    ax.axis("off")

    # Colour-bar
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=grid.min(), vmax=grid.max()))
    sm.set_array([])
    fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)

    fig.tight_layout()
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 5. Classification
# ════════════════════════════════════════════════════════════════════════════

def classify_burn(
    burn_r:    GridArr,
    depth_r:   GridArr,
    texture_r: GridArr,
) -> dict:
    """
    Heuristic burn classification from the three grid feature maps.

    Returns a dict with keys:
      degree, confidence, tbsa_pct, colour, explanation
    """
    burn_mean    = float(burn_r.mean())
    depth_mean   = float(depth_r.mean())
    texture_mean = float(texture_r.mean())

    # ── Composite score ──────────────────────────────────────────────────
    composite = (
        0.50 * burn_mean
        + 0.30 * depth_mean
        + 0.20 * texture_mean
    )

    # ── TBSA estimate (placeholder — blocks above threshold / total) ─────
    burn_threshold = 0.25
    burned_blocks  = (burn_r > burn_threshold).sum()
    total_blocks   = burn_r.size
    tbsa_raw       = burned_blocks / total_blocks * 100.0
    # Paediatric surface: scale to realistic range (cap at 60%)
    tbsa_pct       = round(min(tbsa_raw * 0.6, 60.0), 1)

    # ── Degree classification ────────────────────────────────────────────
    if composite < 0.15:
        degree     = "Normal / No burn detected"
        confidence = 0.90 - composite
        colour     = "#38a169"       # green
        explanation = (
            "Feature scores across all channels are low. "
            "The image does not show characteristic burn signatures. "
            "If clinical concern persists, review under adequate lighting."
        )

    elif composite < 0.30:
        degree     = "Superficial (1st degree)"
        confidence = 0.75 + 0.1 * (composite - 0.15) / 0.15
        colour     = "#d97706"       # amber
        explanation = (
            f"Burn score {burn_mean:.2f} indicates erythema with intact epidermis. "
            f"Depth score {depth_mean:.2f} is consistent with superficial injury. "
            "Typical management: cool water irrigation, non-adherent dressings. "
            "Expect healing in 5–7 days."
        )

    elif composite < 0.50:
        degree     = "Superficial Partial Thickness (2nd degree)"
        confidence = 0.70 + 0.10 * (composite - 0.30) / 0.20
        colour     = "#f97316"       # orange
        explanation = (
            f"Burn score {burn_mean:.2f} and texture score {texture_mean:.2f} "
            "suggest blistering / partial dermal involvement. "
            "Wound may be painful with moist appearance. "
            "Management: silvadene or hydrocolloid dressings; review at 48 h."
        )

    elif composite < 0.70:
        degree     = "Deep Partial Thickness (2nd–3rd degree)"
        confidence = 0.68 + 0.08 * (composite - 0.50) / 0.20
        colour     = "#ef4444"       # red
        explanation = (
            f"High depth score ({depth_mean:.2f}) indicates reticular dermis "
            f"involvement. Texture irregularity ({texture_mean:.2f}) may indicate "
            "eschar formation. Likely requires surgical debridement and skin grafting."
        )

    else:
        degree     = "Full Thickness (3rd / 4th degree)"
        confidence = 0.72 + 0.08 * min((composite - 0.70) / 0.30, 1.0)
        colour     = "#7c3aed"       # purple
        explanation = (
            f"All three channels elevated (burn {burn_mean:.2f}, "
            f"depth {depth_mean:.2f}, texture {texture_mean:.2f}). "
            "Suggests full-thickness destruction of epidermis and dermis. "
            "Urgent surgical review required. Consider early excision and grafting."
        )

    return {
        "degree":      degree,
        "confidence":  round(float(np.clip(confidence, 0.0, 1.0)), 3),
        "tbsa_pct":    tbsa_pct,
        "colour":      colour,
        "explanation": explanation,
    }
