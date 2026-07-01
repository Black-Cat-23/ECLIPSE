import json
import logging
import asyncio
from src.inference.pipeline import ECLIPSEInferencePipeline
from src.utils.config import DEFAULT_CONFIG
from api.routes.predict import _save_result_to_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_pipeline")

def seed_db():
    pipe = ECLIPSEInferencePipeline(sector=1, config=DEFAULT_CONFIG, run_xai=True)
    
    # 307210830 (TOI-700), 261136679 (WASP-18), 149603524
    test_tics = [307210830, 261136679, 149603524]

    for tic in test_tics:
        logger.info(f"Running pipeline for TIC {tic}...")
        result = pipe.run(tic_id=tic)
        
        if result.get("error"):
            logger.error(f"Pipeline failed for TIC {tic}: {result['error']}")
            continue
            
        # For UI demonstration purposes while we have random weights, 
        # force TOI-700 and WASP-18 to be TRANSIT so we can see habitability and XAI.
        if tic in [307210830, 261136679]:
            result["predicted_class"] = "TRANSIT"
            result["confidence"] = 0.98
            result["class_probs"] = {"TRANSIT": 0.98, "EB": 0.01, "BLEND": 0.01, "OTHER": 0.0}
            
            # Re-run XAI and Habitability manually to force it since we bypassed the gating
            from src.habitability import compute_esi, classify_habitable_zone, score_priority, assign_tier, estimate_rv_amplitude
            from src.xai import explain_prediction, render_attention_heatmap
            import numpy as np
            
            rp = result.get("rp_rearth") or 1.0
            teq = result.get("t_eq_kelvin") or 288.0
            esi = compute_esi(rp, teq)
            
            hz = classify_habitable_zone(
                result.get("period", 365.25), 
                result.get("stellar", {}).get("stellar_mass") or 1.0,
                result.get("stellar", {}).get("teff") or 5778.0,
                result.get("stellar", {}).get("luminosity_lsun") or 1.0
            )
            
            snr = result.get("snr_tls", 15.0)
            tmag = result.get("stellar", {}).get("tmag", 10.0)
            
            p = score_priority(esi, hz, snr, tmag)
            tier = assign_tier("TRANSIT", esi, hz, snr, 0.98)
            rv = estimate_rv_amplitude(result.get("period", 365.25), rp, result.get("stellar", {}).get("stellar_mass") or 1.0)
            
            result.update({
                "esi_score": esi,
                "hz_class": hz,
                "priority_score": p,
                "tier": tier,
                "rv_amplitude_ms": rv
            })
            
            if pipe.run_xai:
                logger.info(f"Forcing SHAP run for TIC {tic}...")
                feats = np.array([
                    result.get("period", 1.0),
                    result.get("duration", 2.0),
                    result.get("depth_ppm", 1000.0),
                    snr,
                    0.1, 0.01
                ], dtype=np.float32)
                shap_json = explain_prediction(pipe.model, np.zeros((10,), dtype=np.float32), feats, pipe.device)
                result["shap_values_json"] = shap_json
                
        _save_result_to_db(result)
        logger.info(f"TIC {tic} seeded to DB.")

if __name__ == "__main__":
    seed_db()
