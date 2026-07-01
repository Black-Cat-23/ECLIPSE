"""
Pydantic v2 schemas for the ECLIPSE API.
Matches the exact JSON contract the frontend expects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── Request schemas ───────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    tic_id: int = Field(..., description="TESS Input Catalog ID")
    sector: int = Field(1, ge=1, le=99, description="TESS sector number")
    model_path: Optional[str] = Field(None, description="Override checkpoint path")


class SectorProcessRequest(BaseModel):
    sector: int = Field(..., ge=1, le=99)
    max_tic: int = Field(1000, ge=1, le=30000, description="Max TIC IDs to process")
    model_path: Optional[str] = None


# ── Sub-schemas ───────────────────────────────────────────────────────────────

class ClassProbabilities(BaseModel):
    TRANSIT: float
    EB: float
    BLEND: float
    OTHER: float


class StellarParams(BaseModel):
    host_name:       Optional[str]   = None
    teff:            Optional[float] = None
    logg:            Optional[float] = None
    stellar_mass:    Optional[float] = None   # M_sun
    stellar_radius:  Optional[float] = None   # R_sun
    tmag:            Optional[float] = None
    ra:              Optional[float] = None
    dec:             Optional[float] = None
    distance_pc:     Optional[float] = None
    luminosity_lsun: Optional[float] = None


class SHAPFeature(BaseModel):
    name:        str
    value:       float
    shap_value:  float


class XAIResult(BaseModel):
    top_shap_features:  List[SHAPFeature] = []
    attention_map_b64:  Optional[str] = None


class HabitabilityResult(BaseModel):
    esi_score:            float
    hz_class:             str    # CONSERVATIVE / INNER / OUTER / NONE
    priority_score:       float
    tier:                 int
    rv_amplitude_ms:      Optional[float] = None
    in_confirmed_catalog: bool = False


class TransitParams(BaseModel):
    period_days:    float
    period_err:     float
    duration_hrs:   float
    duration_err:   float
    depth_ppm:      float
    depth_err:      float
    snr:            float
    n_transits:     int
    rp_rearth:      Optional[float] = None
    t_eq_kelvin:    Optional[float] = None


# ── Full pipeline result ──────────────────────────────────────────────────────

class PredictResponse(BaseModel):
    """Full per-candidate result, returned by POST /api/predict and GET /api/candidate/{tic_id}."""
    tic_id:         int
    sector:         int

    # Classification
    predicted_class:       str
    class_probs:           ClassProbabilities
    confidence:            float
    conformal_class_set:   Optional[List[str]]  = None
    in_conformal_90:       bool = False
    in_conformal_95:       bool = False

    # Transit parameters (None if not TRANSIT)
    period:           Optional[float] = None
    period_err:       Optional[float] = None
    duration_days:    Optional[float] = None
    duration_hrs:     Optional[float] = None
    duration_err:     Optional[float] = None
    depth:            Optional[float] = None
    depth_err:        Optional[float] = None
    depth_ppm:        Optional[float] = None
    snr_tls:          Optional[float] = None
    snr_photometric:  Optional[float] = None
    centroid_ratio:   Optional[float] = None
    n_transits:       Optional[int]   = None
    odd_even_mismatch: Optional[float] = None
    rp_rearth:        Optional[float] = None
    t_eq_kelvin:      Optional[float] = None

    # Stellar
    stellar: Optional[StellarParams] = None

    # Habitability (TRANSIT only, high-conf)
    habitability: Optional[HabitabilityResult] = None

    # XAI
    xai: Optional[XAIResult] = None

    # Phase fold arrays for frontend light-curve charts
    phase_fold_global: Optional[List[float]] = None   # 2001 values
    phase_fold_local:  Optional[List[float]] = None   # 201 values
    batman_model:      Optional[List[float]] = None   # 201 values (batman fit)
    centroid_map_b64:  Optional[str] = None           # Base64 centroid validation plot

    # Meta
    processing_time_s: float = 0.0
    error:             Optional[str] = None


# ── Candidate list (summary) ──────────────────────────────────────────────────

class CandidateOut(BaseModel):
    id:             int
    tic_id:         int
    sector:         int
    predicted_class: str
    prob_transit:   Optional[float] = None
    prob_eb:        Optional[float] = None
    prob_blend:     Optional[float] = None
    prob_other:     Optional[float] = None
    confidence:     Optional[float] = None
    period:         Optional[float] = None
    period_err:     Optional[float] = None
    duration:       Optional[float] = None
    depth:          Optional[float] = None
    depth_ppm:      Optional[float] = None
    snr_tls:        Optional[float] = None
    rp_rearth:      Optional[float] = None
    t_eq_kelvin:    Optional[float] = None
    esi_score:      Optional[float] = None
    hz_class:       Optional[str]   = None
    tier:           Optional[int]   = None
    priority_score: Optional[float] = None
    ra:             Optional[float] = None
    dec:            Optional[float] = None
    host_name:      Optional[str]   = None
    tmag:           Optional[float] = None
    in_confirmed_catalog: Optional[bool] = None

    class Config:
        from_attributes = True


class CandidateListResponse(BaseModel):
    total:      int
    candidates: List[CandidateOut]


# ── Sector / Job schemas ──────────────────────────────────────────────────────

class SectorProcessResponse(BaseModel):
    job_id:  str
    sector:  int
    status:  str
    message: str


class PipelineStatus(BaseModel):
    job_id:       str
    status:       str         # pending, running, done, error
    progress:     float       # 0.0 to 1.0
    current_tic:  Optional[int] = None
    processed:    int
    total:        int
    found:        int = 0
    message:      Optional[str] = None
    error:        Optional[str] = None


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status:        str
    model_loaded:  bool
    gpu_available: bool
    gpu_name:      Optional[str] = None
    version:       str = "3.0.0"
    device:        str = "cpu"
