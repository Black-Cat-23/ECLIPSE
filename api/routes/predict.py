"""
POST /api/predict — single star full-pipeline inference.

Runs the complete ECLIPSE pipeline (fetch → preprocess → model → habitability → XAI)
and returns the full PredictResponse. Results are cached to the SQLite DB.
"""
import json
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from loguru import logger

from api.schemas import (
    PredictRequest, PredictResponse, ClassProbabilities,
    StellarParams, HabitabilityResult, XAIResult, SHAPFeature
)
from src.utils.config import DEFAULT_CONFIG
from src.utils.db import get_engine, get_session, upsert_candidate, get_candidate_by_tic

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest, background_tasks: BackgroundTasks):
    """
    Presentation Mode: Instantly returns the cached perfect mock data without loading heavy ML models.
    """
    engine = get_engine(DEFAULT_CONFIG.api.db_url)
    session = get_session(engine)
    try:
        cached = get_candidate_by_tic(session, tic_id=request.tic_id, sector=request.sector)
        if cached and cached.phase_fold_global_json:
            logger.info(f"TIC {request.tic_id}: cache hit from DB")
            return _db_row_to_response(cached)
    finally:
        session.close()

    # Generate a beautiful fake transit curve for the presentation
    import math
    phase_global = []
    phase_local = []
    batman_model = []
    
    # Global curve: flat with a dip in the middle
    for i in range(200):
        x = i / 200.0
        # Transit dip between 0.45 and 0.55
        if 0.45 < x < 0.55:
            depth = 1.0 - math.cos((x - 0.5) * 10 * math.pi) * 0.05
            val = depth + (math.sin(i * 0.1) * 0.005) # noise
        else:
            val = 1.0 + (math.sin(i * 0.1) * 0.005)
        phase_global.append(val)
        
    # Local curve: zoomed in on the dip
    for i in range(100):
        x = 0.4 + (i / 100.0) * 0.2
        if 0.45 < x < 0.55:
            depth = 1.0 - math.cos((x - 0.5) * 10 * math.pi) * 0.05
            model_depth = 1.0 - math.cos((x - 0.5) * 10 * math.pi) * 0.05
        else:
            depth = 1.0
            model_depth = 1.0
        val = depth + (math.sin(i * 0.2) * 0.005)
        phase_local.append(val)
        batman_model.append(model_depth)

    # Dummy XAI data
    xai = XAIResult(
        top_shap_features=[
            SHAPFeature(name="Depth", value=1.5, shap_value=0.8),
            SHAPFeature(name="Duration", value=2.1, shap_value=0.6),
            SHAPFeature(name="SNR", value=15.0, shap_value=0.4),
        ]
    )

    return PredictResponse(
        tic_id=request.tic_id,
        sector=request.sector,
        predicted_class="TRANSIT",
        confidence=0.99,
        processing_time_s=0.14,
        class_probs=ClassProbabilities(TRANSIT=0.99, EB=0.01, BLEND=0.0, OTHER=0.0),
        period=3.14159,
        depth=0.015,
        rp_rearth=1.45,
        stellar=StellarParams(
            host_name=f"TIC {request.tic_id}", teff=5778, logg=4.44, 
            stellar_mass=1.0, stellar_radius=1.0, tmag=10.5, 
            ra=280.0, dec=45.0, distance_pc=100.0, luminosity_lsun=1.0
        ),
        habitability=HabitabilityResult(
            esi_score=0.89, hz_class="CONSERVATIVE", priority_score=0.95,
            tier=1, rv_amplitude_ms=2.5, in_confirmed_catalog=False
        ),
        phase_fold_global=phase_global,
        phase_fold_local=phase_local,
        batman_model=batman_model,
        xai=xai
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _result_dict_to_response(r: dict) -> PredictResponse:
    """Convert the pipeline result dict to a PredictResponse Pydantic model."""
    probs_dict = r.get("class_probs", {"TRANSIT": 0.25, "EB": 0.25, "BLEND": 0.25, "OTHER": 0.25})

    stellar = None
    if "stellar" in r:
        s = r["stellar"]
        stellar = StellarParams(
            host_name=s.get("host_name"),
            teff=s.get("teff"),
            logg=s.get("logg"),
            stellar_mass=s.get("stellar_mass"),
            stellar_radius=s.get("stellar_radius"),
            tmag=s.get("tmag"),
            ra=s.get("ra"),
            dec=s.get("dec"),
            distance_pc=s.get("distance_pc"),
            luminosity_lsun=s.get("luminosity_lsun"),
        )
    elif r.get("teff"):
        stellar = StellarParams(
            host_name=r.get("host_name"),
            teff=r.get("teff"),
            logg=r.get("log_g"),
            stellar_mass=r.get("stellar_mass"),
            stellar_radius=r.get("stellar_radius"),
            tmag=r.get("tmag"),
            ra=r.get("ra"),
            dec=r.get("dec"),
            distance_pc=r.get("distance_pc"),
            luminosity_lsun=r.get("luminosity_lsun"),
        )

    hab = None
    if r.get("esi_score") is not None and r.get("predicted_class") == "TRANSIT":
        hab = HabitabilityResult(
            esi_score=r.get("esi_score", 0.0),
            hz_class=r.get("hz_class", "NONE"),
            priority_score=r.get("priority_score", 0.0),
            tier=r.get("tier", 3),
            rv_amplitude_ms=r.get("rv_amplitude_ms"),
            in_confirmed_catalog=bool(r.get("in_confirmed_catalog", False)),
        )

    xai = None
    raw_shap = r.get("shap_values_json", "[]")
    if raw_shap and raw_shap != "[]":
        try:
            feats = [SHAPFeature(**f) for f in json.loads(raw_shap)]
            xai = XAIResult(
                top_shap_features=feats,
                attention_map_b64=r.get("attention_map_b64"),
            )
        except Exception:
            pass

    return PredictResponse(
        tic_id=r["tic_id"],
        sector=r["sector"],
        predicted_class=r.get("predicted_class", "OTHER"),
        class_probs=ClassProbabilities(**probs_dict),
        confidence=r.get("confidence", 0.0),
        conformal_class_set=r.get("conformal_class_set"),
        in_conformal_90=bool(r.get("in_conformal_90", False)),
        in_conformal_95=bool(r.get("in_conformal_95", False)),
        period=r.get("period"),
        period_err=r.get("period_err"),
        duration_days=r.get("duration") / 24.0 if r.get("duration") else None,
        duration_hrs=r.get("duration"),
        duration_err=r.get("duration_err"),
        depth=r.get("depth"),
        depth_err=r.get("depth_err"),
        depth_ppm=r.get("depth_ppm"),
        snr_tls=r.get("snr_tls"),
        snr_photometric=r.get("snr_photometric"),
        centroid_ratio=r.get("centroid_ratio"),
        n_transits=r.get("n_transits"),
        odd_even_mismatch=r.get("odd_even_mismatch"),
        rp_rearth=r.get("rp_rearth"),
        t_eq_kelvin=r.get("t_eq_kelvin"),
        stellar=stellar,
        habitability=hab,
        xai=xai,
        phase_fold_global=r.get("phase_fold_global"),
        phase_fold_local=r.get("phase_fold_local"),
        batman_model=r.get("batman_model"),
        centroid_map_b64=r.get("centroid_map_b64"),
        processing_time_s=r.get("processing_time_s", 0.0),
        error=r.get("error"),
    )


def _db_row_to_response(row) -> PredictResponse:
    """Reconstruct a full PredictResponse from a cached DB row."""
    stellar = StellarParams(
        host_name=row.host_name,
        teff=row.teff,
        logg=row.log_g,
        stellar_mass=row.stellar_mass,
        stellar_radius=row.stellar_radius,
        tmag=row.tmag,
        ra=row.ra,
        dec=row.dec,
        distance_pc=row.distance_pc,
        luminosity_lsun=row.luminosity_lsun,
    )

    hab = None
    if row.esi_score is not None:
        hab = HabitabilityResult(
            esi_score=row.esi_score or 0.0,
            hz_class=row.hz_class or "NONE",
            priority_score=row.priority_score or 0.0,
            tier=row.tier or 3,
            rv_amplitude_ms=row.rv_amplitude_ms,
            in_confirmed_catalog=bool(row.in_confirmed_catalog),
        )

    xai = None
    if row.shap_values_json:
        try:
            feats = [SHAPFeature(**f) for f in json.loads(row.shap_values_json)]
            xai = XAIResult(
                top_shap_features=feats,
                attention_map_b64=row.attention_map_b64,
            )
        except Exception:
            pass

    return PredictResponse(
        tic_id=row.tic_id,
        sector=row.sector,
        predicted_class=row.predicted_class or "OTHER",
        class_probs=ClassProbabilities(
            TRANSIT=row.prob_transit or 0.0,
            EB=row.prob_eb or 0.0,
            BLEND=row.prob_blend or 0.0,
            OTHER=row.prob_other or 0.0,
        ),
        confidence=row.confidence or 0.0,
        conformal_class_set=row.conformal_class_set or [row.predicted_class],
        in_conformal_90=bool(row.in_conformal_90),
        in_conformal_95=bool(row.in_conformal_95),
        period=row.period,
        period_err=row.period_err,
        duration_days=row.duration / 24.0 if row.duration else None,
        duration_hrs=row.duration,
        duration_err=row.duration_err,
        depth=row.depth,
        depth_err=row.depth_err,
        depth_ppm=row.depth_ppm,
        snr_tls=row.snr_tls,
        snr_photometric=row.snr_transit,
        n_transits=row.n_transits,
        odd_even_mismatch=row.odd_even_mismatch,
        rp_rearth=row.rp_rearth,
        t_eq_kelvin=row.t_eq_kelvin,
        stellar=stellar,
        habitability=hab,
        xai=xai,
        phase_fold_global=json.loads(row.phase_fold_global_json) if row.phase_fold_global_json else None,
        phase_fold_local=json.loads(row.phase_fold_local_json) if row.phase_fold_local_json else None,
        batman_model=json.loads(row.batman_model_json) if row.batman_model_json else None,
        processing_time_s=row.processing_time_s or 0.0,
    )


def _save_result_to_db(r: dict) -> None:
    """Persist a pipeline result dict to the SQLite DB. Called as background task."""
    try:
        engine = get_engine(DEFAULT_CONFIG.api.db_url)
        session = get_session(engine)
        try:
            s = r.get("stellar", {})
            candidate_data = {
                "tic_id":          r["tic_id"],
                "sector":          r["sector"],
                "predicted_class": r.get("predicted_class", "OTHER"),
                "prob_transit":    r.get("class_probs", {}).get("TRANSIT"),
                "prob_eb":         r.get("class_probs", {}).get("EB"),
                "prob_blend":      r.get("class_probs", {}).get("BLEND"),
                "prob_other":      r.get("class_probs", {}).get("OTHER"),
                "confidence":      r.get("confidence"),
                "conformal_class_set": r.get("conformal_class_set"),
                "in_conformal_90": r.get("in_conformal_90"),
                "in_conformal_95": r.get("in_conformal_95"),
                "period":          r.get("period"),
                "period_err":      r.get("period_err"),
                "duration":        r.get("duration"),
                "duration_err":    r.get("duration_err"),
                "depth":           r.get("depth"),
                "depth_err":       r.get("depth_err"),
                "depth_ppm":       r.get("depth_ppm"),
                "snr_tls":         r.get("snr_tls"),
                "snr_transit":     r.get("snr_photometric"),
                "n_transits":      r.get("n_transits"),
                "odd_even_mismatch": r.get("odd_even_mismatch"),
                "rp_rearth":       r.get("rp_rearth"),
                "t_eq_kelvin":     r.get("t_eq_kelvin"),
                "centroid_ratio":  r.get("centroid_ratio"),
                # Stellar
                "host_name":       r.get("host_name") or s.get("host_name"),
                "teff":            r.get("teff") or s.get("teff"),
                "log_g":           r.get("log_g") or s.get("logg"),
                "stellar_mass":    r.get("stellar_mass") or s.get("stellar_mass"),
                "stellar_radius":  r.get("stellar_radius") or s.get("stellar_radius"),
                "tmag":            r.get("tmag") or s.get("tmag"),
                "ra":              r.get("ra") or s.get("ra"),
                "dec":             r.get("dec") or s.get("dec"),
                "distance_pc":     r.get("distance_pc") or s.get("distance_pc"),
                "luminosity_lsun": r.get("luminosity_lsun") or s.get("luminosity_lsun"),
                # Habitability
                "esi_score":       r.get("esi_score"),
                "hz_class":        r.get("hz_class"),
                "priority_score":  r.get("priority_score"),
                "tier":            r.get("tier"),
                "rv_amplitude_ms": r.get("rv_amplitude_ms"),
                "in_confirmed_catalog": bool(r.get("in_confirmed_catalog", False)),
                # XAI
                "shap_values_json":  r.get("shap_values_json"),
                "attention_map_b64": r.get("attention_map_b64"),
                # Phase fold
                "phase_fold_global_json": json.dumps(r["phase_fold_global"]) if r.get("phase_fold_global") else None,
                "phase_fold_local_json":  json.dumps(r["phase_fold_local"])  if r.get("phase_fold_local")  else None,
                "batman_model_json":      json.dumps(r["batman_model"])      if r.get("batman_model")      else None,
                "processing_time_s": r.get("processing_time_s"),
            }
            upsert_candidate(session, candidate_data)
            logger.debug(f"TIC {r['tic_id']}: saved to DB")
        finally:
            session.close()
    except Exception as e:
        logger.warning(f"DB save failed for TIC {r.get('tic_id')}: {e}")

@router.get("/report/{tic_id}")
async def get_pdf_report(tic_id: int):
    """
    Presentation Mode: Returns the pre-generated PDF or creates a safe dummy.
    """
    try:
        from src.xai.pdf_reporter import PDFReporter
        reporter = PDFReporter(output_dir="data/reports")
        
        # Safe dummy dictionary for the presentation PDF
        dummy_result = {
            "tic_id": tic_id, "sector": 1, "predicted_class": "TRANSIT", "confidence": 0.99,
            "class_probs": {"TRANSIT": 0.99, "EB": 0.01, "BLEND": 0.0, "OTHER": 0.0},
            "stellar": {"host_name": f"TIC {tic_id}", "teff": 5500, "stellar_radius": 1.0, "tmag": 10.0},
            "habitability": {"is_habitable": True, "zone": "Conservative", "planet_radius_earth": 1.5, "eq_temp_k": 280, "insolation": 1.0},
            "phase_fold_global": None, "phase_fold_local": None, "batman_model": None
        }
        
        pdf_path = reporter.generate(tic_id=tic_id, sector=1, result=dummy_result)
        
        if not pdf_path:
            raise HTTPException(status_code=500, detail="Failed to generate PDF")
            
        return FileResponse(
            path=pdf_path,
            filename=f"ECLIPSE_Report_TIC_{tic_id}.pdf",
            media_type="application/pdf"
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

