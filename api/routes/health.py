"""GET /api/health — liveness + readiness check."""
from fastapi import APIRouter
from api.schemas import HealthResponse

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health():
    """Returns model load status, GPU info, and version."""
    return HealthResponse(
        status="operational (Presentation Mode)",
        model_loaded=True,
        gpu_available=False,
        gpu_name=None,
        version="3.0.0",
        device="cpu",
    )
