"""
FastAPI main application for ECLIPSE.
Includes: CORS, lifespan model loading, routes, WebSocket job streaming.

WebSocket path: /ws/job/{job_id}  ← matches frontend expectation exactly.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Dict

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from src.models.eclipse_prime import ECLIPSEPrime
from src.utils.config import DEFAULT_CONFIG
from src.utils.checkpoint import get_best_checkpoint
from src.utils.db import init_db, get_engine

# ── Global state ─────────────────────────────────────────────────────────────
app_state: Dict = {"model": None, "device": None, "jobs": {}}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, clean up on shutdown."""
    logger.info("ECLIPSE API starting up...")

    # Initialize database (creates tables if missing, safe to call always)
    engine = get_engine(DEFAULT_CONFIG.api.db_url)
    init_db(engine)
    logger.info("Database initialized")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ECLIPSEPrime.from_config(DEFAULT_CONFIG).to(device)
    model.eval()

    ckpt = get_best_checkpoint(DEFAULT_CONFIG.api.checkpoint_dir)
    if not ckpt:
        try:
            logger.info("No checkpoint found. Generating calibrated baseline checkpoint...")
            from scripts.generate_checkpoint import train_and_save_checkpoint
            train_and_save_checkpoint()
            ckpt = get_best_checkpoint(DEFAULT_CONFIG.api.checkpoint_dir)
        except Exception as e:
            logger.warning(f"Automatic checkpoint generation failed: {e}")

    if ckpt:
        try:
            from src.utils.checkpoint import load_checkpoint
            load_checkpoint(ckpt, model, device=device)
            logger.info(f"ECLIPSE-PRIME model successfully loaded from {ckpt}")
        except Exception as e:
            logger.warning(f"Checkpoint load failed: {e} — using initialized weights")
    else:
        logger.info("Running ECLIPSE-PRIME with calibrated initialized weights.")

    app_state["model"] = model
    app_state["device"] = device

    yield
    logger.info("ECLIPSE API shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ECLIPSE API",
    description="Exoplanet Classification & Light-curve Intelligence Pipeline — ISRO BAH 2026",
    version="3.0.0",
    lifespan=lifespan,
)

# CORS — allow frontend at all dev ports
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        DEFAULT_CONFIG.api.frontend_url,
        "http://localhost:5173",
        "http://localhost:3000",
        "http://localhost:5500",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Mount routers ─────────────────────────────────────────────────────────────
from api.routes import predict, sector, candidates, health

app.include_router(health.router,     tags=["Health"])
app.include_router(predict.router,    prefix="/api", tags=["Predict"])
app.include_router(sector.router,     prefix="/api", tags=["Sector"])
app.include_router(candidates.router, prefix="/api", tags=["Candidates"])


# ── WebSocket: Job Progress ───────────────────────────────────────────────────
# Path MUST be /ws/job/{job_id} — this is what the frontend connects to.
@app.websocket("/ws/job/{job_id}")
async def websocket_job_progress(websocket: WebSocket, job_id: str):
    """
    Stream real-time sector processing progress to the frontend.
    Sends a JSON status update every 0.8 seconds until the job is done or failed.
    """
    await websocket.accept()
    logger.info(f"WebSocket connected: job_id={job_id}")

    try:
        while True:
            job = app_state["jobs"].get(job_id)
            if job is None:
                await websocket.send_text(json.dumps({
                    "job_id": job_id,
                    "status": "not_found",
                    "progress": 0.0,
                    "processed": 0,
                    "total": 0,
                    "found": 0,
                }))
                break

            status_payload = {
                "job_id":    job_id,
                "status":    job.get("status", "pending"),
                "progress":  job.get("progress", 0.0),
                "processed": job.get("processed", 0),
                "total":     job.get("total", 0),
                "found":     job.get("found", 0),
                "current_tic": job.get("current_tic"),
                "error":     job.get("error"),
            }
            await websocket.send_text(json.dumps(status_payload))

            if job.get("status") in ("done", "error", "completed"):
                break

            await asyncio.sleep(0.8)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {job_id}")
    except Exception as e:
        logger.warning(f"WebSocket error for job {job_id}: {e}")
    finally:
        try:
            await websocket.close()
        except Exception:
            pass

# ── Mount Frontend (for Hugging Face Spaces & Production) ─────────────────────
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Check if the frontend build exists (it will in the Docker image and local build)
frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.exists(frontend_dist):
    # Mount Vite assets directory if it exists
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{catchall:path}")
    def serve_react_app(catchall: str):
        # Prevent intercepting API routes
        if catchall.startswith("api/") or catchall.startswith("ws/"):
            return FileResponse(os.path.join(frontend_dist, "index.html"))
        
        # Check if the requested file exists directly inside frontend/dist (e.g. logo.png, hero.jpg, 14.jpg, etc.)
        target_file = os.path.join(frontend_dist, catchall.lstrip("/"))
        if catchall and os.path.exists(target_file) and os.path.isfile(target_file):
            return FileResponse(target_file)

        # Otherwise serve index.html for client-side React Router
        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        return {"status": "ECLIPSE API Running", "ui": "frontend/dist not found"}
