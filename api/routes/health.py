"""GET /api/health — liveness + readiness check."""
import torch
from fastapi import APIRouter
from api.schemas import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health():
    """Returns model load status, GPU info, and version."""
    from api.main import app_state
    model = app_state.get("model")
    device = app_state.get("device")
    gpu = torch.cuda.is_available()
    return HealthResponse(
        status="operational",
        model_loaded=model is not None,
        gpu_available=gpu,
        gpu_name=torch.cuda.get_device_name(0) if gpu else None,
        version="3.0.0",
        device=str(device) if device else "cpu",
    )
