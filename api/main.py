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
    if ckpt:
        try:
            from src.utils.checkpoint import load_checkpoint
            load_checkpoint(ckpt, model, device=device)
            logger.info(f"Model loaded from {ckpt}")
        except Exception as e:
            logger.warning(f"Checkpoint load failed: {e} — using random weights")
    else:
        logger.warning(
            f"No checkpoint found in '{DEFAULT_CONFIG.api.checkpoint_dir}'. "
            "API running with untrained model. "
            "Run notebooks/02_train_4class_colab.ipynb to train."
        )

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
