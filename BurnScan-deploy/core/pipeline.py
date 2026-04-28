"""
core/pipeline.py
================
BurnScan — image-analysis pipeline aligned with Colab notebook.

Colab logic preserved exactly:
  • LAB-a + HSV-S threshold + morph-close  → burn_mask
  • L-channel masked by burn_mask           → depth
  • LBP (P=8, R=1, uniform) on masked grey → texture
  • block_average reshape trick             → numeric grid arrays
  • make_grid_figure with per-cell labels   → numeric values on overlay

Public API (imported by backend/main.py)
-----------------------------------------
decode_image(raw_bytes)                          → img_bgr
run_pipeline(img_bgr, k)                         → (rgb, burn_r, depth_r, texture_r)
make_grid_figure(rgb, values, title, k, cmap,
                 vmin, vmax)                     → plt.Figure
classify_burn(burn_r, depth_r, texture_r)        → dict
fig_to_bytes(fig)                                → bytes
"""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from skimage.feature import local_binary_pattern

# Type aliases
ImgBGR  = np.ndarray   # H×W×3 uint8, BGR
ImgRGB  = np.ndarray   # H×W×3 uint8, RGB
GridArr = np.ndarray   # rows×cols float


# ════════════════════════════════════════════════════════════════════════════
# 1. I/O
# ════════════════════════════════════════════════════════════════════════════

def decode_image(raw: bytes) -> ImgBGR:
    """Decode raw JPEG/PNG bytes → BGR ndarray. Raises ValueError on failure."""
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image — unsupported format or corrupt file.")
    return img


def fig_to_bytes(fig: plt.Figure) -> bytes:
    """Render a matplotlib Figure to PNG bytes and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ════════════════════════════════════════════════════════════════════════════
# 2. Core pipeline — exact Colab notebook logic
# ════════════════════════════════════════════════════════════════════════════

def block_average(img: np.ndarray, k: int) -> GridArr:
    """
    Divide a 2-D array into k×k non-overlapping blocks and return block means.
    Exact Colab block_average() — reshape trick, no loops.
    """
    h, w = img.shape
    h2   = h // k
    w2   = w // k
    return img[:h2 * k, :w2 * k].reshape(h2, k, w2, k).mean(axis=(1, 3))


def extract_features(img_bgr: ImgBGR):
    """
    Exact Colab extract_features():
      1. Resize to 256×256
      2. BGR → RGB → HSV → LAB
      3. Burn mask: LAB-a > (mean + 0.8·std)  AND  HSV-S > mean
         + morphological closing (7×7 kernel)
      4. Depth   : L-channel masked by burn_mask
      5. Texture : LBP(P=8, R=1, uniform) on grey masked by burn_mask,
                   normalised to 0-255

    Returns (rgb, burn_mask, depth, texture) — all 256×256 uint8.
    """
    img  = cv2.resize(img_bgr, (256, 256))
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    hsv  = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    lab  = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)

    _, S, _  = cv2.split(hsv)
    L, A, _  = cv2.split(lab)
    gray     = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    meanA = float(np.mean(A))
    stdA  = float(np.std(A))
    meanS = float(np.mean(S))

    # Burn mask — identical condition to Colab
    burn_mask = ((A > meanA + 0.8 * stdA) & (S > meanS)).astype(np.uint8) * 255
    burn_mask = cv2.morphologyEx(burn_mask, cv2.MORPH_CLOSE,
                                  np.ones((7, 7), np.uint8))

    depth     = cv2.bitwise_and(L, L, mask=burn_mask)
    burn_gray = cv2.bitwise_and(gray, gray, mask=burn_mask)
    lbp       = local_binary_pattern(burn_gray, 8, 1, "uniform")
    texture   = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return rgb, burn_mask, depth, texture


def run_pipeline(img_bgr: ImgBGR, k: int = 10) -> Tuple[ImgRGB, GridArr, GridArr, GridArr]:
    """
    Full pipeline. Matches Colab run_pipeline() exactly.

    Returns
    -------
    rgb           : 256×256×3 uint8 RGB (for display)
    burn_r_binary : binary grid (0 or 1) — block burned or not
    depth_r       : float grid  — mean L-channel per block
    texture_r     : float grid  — mean LBP value per block
    """
    rgb, burn_mask, depth, texture = extract_features(img_bgr)

    burn_r    = block_average(burn_mask.astype(float), k)
    depth_r   = block_average(depth.astype(float),    k)
    texture_r = block_average(texture.astype(float),  k)

    burn_r_binary = (burn_r > 0).astype(np.uint8)   # matches Colab
    return rgb, burn_r_binary, depth_r, texture_r


# ════════════════════════════════════════════════════════════════════════════
# 3. Visualisation — with numeric cell labels (matches Colab output)
# ════════════════════════════════════════════════════════════════════════════

def make_grid_figure(
    rgb: ImgRGB,
    values: GridArr,
    title: str,
    k: int,
    cmap: str = "Reds",
    vmin=None,
    vmax=None,
) -> plt.Figure:
    """
    Render rgb (256×256) with semi-transparent block heat-map overlay.
    Each block shows its integer value as a white label — exact Colab output.

    Skips zero-value blocks on the burn-mask grid (cmap == 'Reds'),
    matching the `if val == 0 and cmap == 'Reds': continue` Colab logic.
    """
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=150)
    fig.patch.set_facecolor("#0d0f14")
    ax.set_facecolor("#0d0f14")
    ax.imshow(rgb, interpolation="lanczos")

    h, w   = values.shape
    vmin_  = float(values.min()) if vmin is None else float(vmin)
    vmax_  = float(values.max()) if vmax is None else float(vmax)
    cm     = plt.get_cmap(cmap)

    for i in range(h):
        for j in range(w):
            val = float(values[i, j])

            # Skip empty blocks on burn-mask grid (Colab behaviour)
            if val == 0 and cmap == "Reds":
                continue

            norm   = (val - vmin_) / (vmax_ - vmin_ + 1e-6)
            colour = cm(norm)

            rect = mpatches.Rectangle(
                (j * k, i * k), k, k,
                linewidth=0.3,
                edgecolor=(1, 1, 1, 0.15),
                facecolor=(*colour[:3], 0.45),
            )
            ax.add_patch(rect)

            # Numeric label on every block
            ax.text(
                j * k + k / 2,
                i * k + k / 2,
                f"{int(val)}",
                color="white",
                fontsize=3.5,
                ha="center",
                va="center",
                fontweight="bold",
            )

    ax.set_xlim(0, 256)
    ax.set_ylim(256, 0)
    ax.set_title(title, fontsize=7, color="#9ca3af", pad=5)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(
        cmap=cmap,
        norm=plt.Normalize(vmin=vmin_, vmax=vmax_),
    )
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=5, colors="#6b7280")
    cb.outline.set_edgecolor("#2a2d38")

    plt.tight_layout(pad=0.5)
    return fig


# ════════════════════════════════════════════════════════════════════════════
# 4. Classification — exact Colab scoring thresholds
# ════════════════════════════════════════════════════════════════════════════

def classify_burn(
    burn_r: GridArr,
    depth_r: GridArr,
    texture_r: GridArr,
) -> dict:
    """
    Heuristic classifier. Exact Colab classify_burn() thresholds:
      mean_d < 70   → +2,  mean_d < 120  → +1
      mean_t > 160  → +2,  mean_t > 110  → +1
      score >= 4    → 3rd Degree
      score >= 2    → 2nd Degree
      else          → 1st Degree
      TBSA          → round(cov * 90, 1)

    Returns dict: degree, confidence, tbsa_pct, colour, explanation
    """
    active = burn_r > 0

    if active.sum() == 0:
        return {
            "degree":      "No burn detected",
            "confidence":  0.0,
            "tbsa_pct":    0.0,
            "colour":      "#6b7280",
            "explanation": (
                "No burned regions detected. "
                "The image does not contain characteristic burn signatures."
            ),
        }

    mean_d = float(depth_r[active].mean())
    mean_t = float(texture_r[active].mean())
    cov    = float(active.mean())

    # Exact Colab scoring
    score = 0
    if mean_d < 70:      score += 2
    elif mean_d < 120:   score += 1
    if mean_t > 160:     score += 2
    elif mean_t > 110:   score += 1

    tbsa_pct = round(cov * 90, 1)

    if score >= 4:
        degree     = "3rd Degree"
        confidence = min(0.55 + cov * 0.2, 0.85)
        colour     = "#e53e3e"
        explanation = (
            f"High texture score ({mean_t:.1f} > 160) and low L-channel depth "
            f"({mean_d:.1f} < 70) indicate full-thickness destruction. "
            "Urgent surgical review — early excision and grafting expected."
        )
    elif score >= 2:
        degree     = "2nd Degree"
        confidence = min(0.50 + cov * 0.2, 0.80)
        colour     = "#dd6b20"
        explanation = (
            f"Moderate texture ({mean_t:.1f}) and depth ({mean_d:.1f}) scores "
            "suggest partial-thickness burn with blistering / dermal involvement. "
            "Management: silvadene or hydrocolloid dressings; review at 48 h."
        )
    else:
        degree     = "1st Degree"
        confidence = min(0.60 + cov * 0.15, 0.80)
        colour     = "#38a169"
        explanation = (
            f"Low texture ({mean_t:.1f}) and depth ({mean_d:.1f}) scores "
            "indicate superficial erythema with intact epidermis. "
            "Management: cool water irrigation, non-adherent dressings."
        )

    return {
        "degree":      degree,
        "confidence":  round(float(np.clip(confidence, 0.0, 1.0)), 3),
        "tbsa_pct":    tbsa_pct,
        "colour":      colour,
        "explanation": explanation,
    }
