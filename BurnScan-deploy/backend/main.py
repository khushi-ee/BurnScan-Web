"""
backend/main.py
===============
FastAPI backend for BurnScan.

Endpoints
---------
POST /api/analyse   – upload image → JSON + base64 PNG grid overlays
GET  /api/health    – liveness probe
GET  /              – serves frontend/index.html
GET  /{path}        – serves static frontend files
"""

from __future__ import annotations

import base64
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
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
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


# ── Analyse ───────────────────────────────────────────────────────────────

@app.post("/api/analyse")
async def analyse(
    file: UploadFile = File(...),
    k: int = Form(10),
    patient_id: Optional[str] = Form(None),
    patient_age: Optional[int] = Form(None),
    burn_cause: Optional[str] = Form(None),
):
    t0 = time.time()
    print(f"[1] /api/analyse called  file={file.filename}  k={k}", flush=True)

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(400, "Only JPG and PNG images are accepted.")
    if not (5 <= k <= 30):
        raise HTTPException(400, "Block size k must be between 5 and 30.")

    raw = await file.read()
    print(f"[2] file read  bytes={len(raw)}  ({time.time()-t0:.1f}s)", flush=True)
    if len(raw) > 20 * 1024 * 1024:
        raise HTTPException(413, "Image too large. Maximum size is 20 MB.")

    try:
        img_bgr = decode_image(raw)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    print(f"[3] decoded  shape={img_bgr.shape}  ({time.time()-t0:.1f}s)", flush=True)

    rgb, burn_r, depth_r, texture_r = run_full_pipeline(img_bgr, k=k)
    print(f"[4] pipeline done  grid_shape={burn_r.shape}  ({time.time()-t0:.1f}s)", flush=True)

    result = classify_burn(burn_r, depth_r, texture_r)
    print(f"[5] classified  degree={result['degree']}  ({time.time()-t0:.1f}s)", flush=True)

    name = patient_id or "case"

    fig_burn = overlay_grid_figure(rgb, burn_r, f"{name} – Burn Mask Grid", k, cmap="Reds")
    print(f"[6] burn figure built  ({time.time()-t0:.1f}s)", flush=True)
    png_burn = fig_to_png_bytes(fig_burn)
    print(f"[7] burn PNG rendered  bytes={len(png_burn)}  ({time.time()-t0:.1f}s)", flush=True)

    fig_depth = overlay_grid_figure(rgb, depth_r, f"{name} – Depth Values", k, cmap="inferno")
    print(f"[8] depth figure built  ({time.time()-t0:.1f}s)", flush=True)
    png_depth = fig_to_png_bytes(fig_depth)
    print(f"[9] depth PNG rendered  bytes={len(png_depth)}  ({time.time()-t0:.1f}s)", flush=True)

    fig_texture = overlay_grid_figure(rgb, texture_r, f"{name} – Texture Values", k, cmap="Blues")
    print(f"[10] texture figure built  ({time.time()-t0:.1f}s)", flush=True)
    png_texture = fig_to_png_bytes(fig_texture)
    print(f"[11] texture PNG rendered  bytes={len(png_texture)}  ({time.time()-t0:.1f}s)", flush=True)

    def b64(data: bytes) -> str:
        return base64.b64encode(data).decode()

    response = JSONResponse({
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
        "grids": {
            "burn_mask": b64(png_burn),
            "depth":     b64(png_depth),
            "texture":   b64(png_texture),
        },
    })
    print(f"[12] /api/analyse SUCCESS  total={time.time()-t0:.1f}s", flush=True)
    return response


# ── Catch-all frontend route (MUST stay at bottom, after all /api/* routes) ──

@app.get("/{full_path:path}", include_in_schema=False)
def serve_frontend(full_path: str):
    """Catch-all: serve any static file from frontend/, fall back to index.html."""
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
