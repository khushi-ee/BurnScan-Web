"""
backend/main.py
===============
FastAPI backend for BurnScan.
Uses core/pipeline.py aligned with the Colab notebook.

Endpoints
---------
POST /api/analyse?view=burn|depth|texture|all
    • view=burn|depth|texture  → single grid PNG (frontend fires 3 parallel
                                  fetch() calls simultaneously — one per view,
                                  each resolves as it finishes → smoother UI)
    • view=all (default)       → all three grids in one response (fallback)

    All responses include:
      - classification result (degree, confidence, tbsa_pct, colour, explanation)
      - grid PNG(s) as base64, rendered with per-cell numeric labels
        (exact Colab make_grid_figure output)

GET  /api/health  — liveness probe
GET  /            — serves frontend/index.html
GET  /{path}      — static file fallback (MUST stay last)
"""

from __future__ import annotations

import base64
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

# ── Make core/ importable ─────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent
CORE_DIR = ROOT_DIR / "core"
sys.path.insert(0, str(CORE_DIR))

from pipeline import (                              # noqa: E402
    classify_burn,
    decode_image,
    fig_to_bytes,
    make_grid_figure,
    run_pipeline,
)

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BurnScan API",
    description="AIIMS Paediatric Burns Analysis API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your AIIMS domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Frontend static files ─────────────────────────────────────────────────
FRONTEND_DIR = ROOT_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/", include_in_schema=False)
def serve_index():
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"status": "BurnScan API running — no frontend found."})


# ── Health ────────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ── View config ───────────────────────────────────────────────────────────
# Cmaps match Colab exactly: Reds / inferno_r / viridis
# vmin/vmax for burn is fixed at 0,1 because burn_r is binary (0 or 1)
_VIEW_CONFIG = {
    "burn":    ("Burn Mask Grid", "Reds",      "burn_mask", 0,    1   ),
    "depth":   ("Depth Values",   "inferno_r", "depth",     None, None),
    "texture": ("Texture Values", "viridis",   "texture",   None, None),
}


# ── Analyse ───────────────────────────────────────────────────────────────
@app.post("/api/analyse")
async def analyse(
    file: UploadFile = File(...),
    k: int = Form(10),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    burn_cause: Optional[str] = Form(None),
    view: str = Query("all", regex="^(burn|depth|texture|all)$"),
):
    # ── Validate ─────────────────────────────────────────────────────────
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPG and PNG images are accepted.")
    if not (5 <= k <= 30):
        raise HTTPException(400, "Block size k must be between 5 and 30.")

    raw = await file.read()
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large. Maximum size is 20 MB.")

    # ── Decode ───────────────────────────────────────────────────────────
    try:
        img_bgr = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    # ── Pipeline (Colab-aligned) ──────────────────────────────────────────
    # run_pipeline:
    #   - resizes to 256×256
    #   - LAB-a + HSV-S → burn_mask (morph closed)
    #   - L-channel masked → depth
    #   - LBP (P=8,R=1,uniform) masked → texture
    #   - block_average (reshape trick, no loops) on each
    #   - returns burn_r as binary (0 or 1) grid — exact Colab
    rgb, burn_r, depth_r, texture_r = run_pipeline(img_bgr, k=k)
    result = classify_burn(burn_r, depth_r, texture_r)

    grids_data = {
        "burn":    burn_r,
        "depth":   depth_r,
        "texture": texture_r,
    }

    def render(view_key: str) -> str:
        """
        Render one grid → base64 PNG.
        make_grid_figure draws per-cell numeric int(val) labels on every
        block — exact Colab output. Skips zero blocks on burn grid.
        """
        title_suffix, cmap, _, vmin, vmax = _VIEW_CONFIG[view_key]
        fig = make_grid_figure(
            rgb,
            grids_data[view_key],
            f"{patient_id or 'case'} – {title_suffix}",
            k,
            cmap,
            vmin,
            vmax,
        )
        return base64.b64encode(fig_to_bytes(fig)).decode()

    # ── Common fields in every response ───────────────────────────────────
    common = {
        "status":       "ok",
        "timestamp":    datetime.utcnow().isoformat(),
        "patient_id":   patient_id,
        "patient_age":  patient_age,
        "burn_cause":   burn_cause,
        "block_size_k": k,
        "classification": {
            "degree":      result["degree"],
            "confidence":  result["confidence"],
            "tbsa_pct":    result["tbsa_pct"],
            "colour":      result["colour"],
            "explanation": result["explanation"],
        },
    }

    # ── Single-view response — parallel calls pattern ─────────────────────
    # The frontend fires these three fetch() calls simultaneously:
    #
    #   const [burn, depth, texture] = await Promise.all([
    #     fetch("/api/analyse?view=burn",    { method:"POST", body:fd }),
    #     fetch("/api/analyse?view=depth",   { method:"POST", body:fd }),
    #     fetch("/api/analyse?view=texture", { method:"POST", body:fd }),
    #   ]);
    #
    # Each resolves independently as soon as its grid is rendered,
    # so the UI can display each image the moment it arrives rather
    # than waiting for all three to finish together.
    if view in _VIEW_CONFIG:
        _, _, response_key, _, _ = _VIEW_CONFIG[view]
        return JSONResponse({
            **common,
            "view": view,
            "grid": {
                "name": response_key,
                "png":  render(view),
            },
        })

    # ── "all" response — all three grids in one call (fallback) ──────────
    return JSONResponse({
        **common,
        "grids": {
            "burn_mask": render("burn"),
            "depth":     render("depth"),
            "texture":   render("texture"),
        },
    })


# ── Catch-all frontend route — MUST stay at the very bottom ───────────────
@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    requested = FRONTEND_DIR / full_path
    if requested.exists() and requested.is_file():
        return FileResponse(str(requested))
    index = FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    raise HTTPException(404, "Not found")


# ── Entry point (local dev) ───────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
