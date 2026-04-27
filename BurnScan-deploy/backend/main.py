"""
backend/main.py
===============
FastAPI backend for BurnScan.

Endpoints
---------
POST /api/analyse?view=burn|depth|texture|all
    Upload image → JSON + base64 PNG overlay(s).
      • view=burn|depth|texture → returns a single overlay
        (frontend fires three parallel calls; each finishes in ~25 s,
         comfortably under Render's 100 s HTTP proxy timeout).
      • view=all (default)      → returns all three overlays in one response
        (kept for legacy / direct-API use).
GET  /api/health     – liveness probe.
GET  /               – serves frontend/index.html.
GET  /{path}         – serves any static file from frontend/, falls back to
                       index.html (catch-all, MUST stay LAST).
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
    fig_to_png_bytes,
    overlay_grid_figure,
    run_full_pipeline,
)

# ── App ───────────────────────────────────────────────────────────────────
app = FastAPI(
    title="BurnScan API",
    description="AIIMS Paediatric Burns Analysis API",
    version="1.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten to your AIIMS domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Serve frontend static files ───────────────────────────────────────────
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


# ══════════════════════════════════════════════════════════════════════════
# Analyse — one endpoint, three "views"
# ══════════════════════════════════════════════════════════════════════════

# view name → (title suffix, matplotlib colour-map, response key for "name")
_VIEW_CONFIG = {
    "burn":    ("Burn Mask Grid", "Reds",    "burn_mask"),
    "depth":   ("Depth Values",   "inferno", "depth"),
    "texture": ("Texture Values", "Blues",   "texture"),
}


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

    # ── Feature pipeline (always produces all three grids) ──────────────
    rgb, burn_r, depth_r, texture_r = run_full_pipeline(img_bgr, k=k)
    classification = classify_burn(burn_r, depth_r, texture_r)

    grids_by_view = {"burn": burn_r, "depth": depth_r, "texture": texture_r}

    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode()

    def render(view_key: str) -> str:
        title_suffix, cmap, _ = _VIEW_CONFIG[view_key]
        title = f"{patient_id or 'case'} – {title_suffix}"
        fig   = overlay_grid_figure(rgb, grids_by_view[view_key], title, k, cmap=cmap)
        return b64(fig_to_png_bytes(fig))

    # ── Common response fields ───────────────────────────────────────────
    common = {
        "status":       "ok",
        "timestamp":    datetime.utcnow().isoformat(),
        "patient_id":   patient_id,
        "patient_age":  patient_age,
        "burn_cause":   burn_cause,
        "block_size_k": k,
        "classification": {
            "degree":      classification["degree"],
            "confidence":  classification["confidence"],
            "tbsa_pct":    classification["tbsa_pct"],
            "colour":      classification["colour"],
            "explanation": classification["explanation"],
        },
    }

    # ── Single-view response (parallel-calls pattern) ────────────────────
    if view in _VIEW_CONFIG:
        return JSONResponse({
            **common,
            "view": view,
            "grid": {
                "name": _VIEW_CONFIG[view][2],
                "png":  render(view),
            },
        })

    # ── Legacy "all" response ────────────────────────────────────────────
    return JSONResponse({
        **common,
        "grids": {
            "burn_mask": render("burn"),
            "depth":     render("depth"),
            "texture":   render("texture"),
        },
    })


# ── Catch-all frontend route (MUST stay at bottom, after every /api/* route)
@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    """Serve any static file from frontend/, fall back to index.html."""
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
