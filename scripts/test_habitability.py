from src.habitability.esi_calculator import compute_esi
from src.habitability.habitable_zone import classify_habitable_zone
from src.habitability.priority_scorer import score_priority, assign_tier
from src.habitability.rv_estimator import estimate_rv_amplitude

esi = compute_esi(rp_rearth=1.0, t_eq_kelvin=288.0)
hz  = classify_habitable_zone(period_days=365.25, stellar_mass_msun=1.0, t_eff=5778.0, luminosity_lsun=1.0)
p   = score_priority(esi, hz, snr=15.0, tmag=9.0)
t   = assign_tier("TRANSIT", esi, hz, snr=15.0, confidence=0.95)
rv  = estimate_rv_amplitude(365.25, 1.0, 1.0)

print(f"ESI (Earth):      {esi:.4f}  (expect ~1.0)")
print(f"HZ class:         {hz}  (expect CONSERVATIVE)")
print(f"Priority score:   {p:.4f}")
print(f"Tier:             {t}  (expect 1)")
print(f"RV amplitude K:   {rv:.4f} m/s  (expect ~0.089 m/s)")
