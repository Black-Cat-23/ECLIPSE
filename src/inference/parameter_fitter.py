"""
Batman MCMC Parameter Fitter.
For high-confidence TRANSIT candidates (prob > 0.8), refines P, τ, δ, impact
parameter b using MCMC (emcee) to obtain full posterior distributions.

Used after initial ECLIPSE-PRIME prediction for precise parameter reporting.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from loguru import logger


class BatmanMCMCFitter:
    """
    Refine transit parameters using batman + emcee MCMC.

    Priors: centered on ECLIPSE-PRIME predictions ± 3σ.
    Likelihood: Gaussian on phase-folded local view.

    Usage:
        fitter = BatmanMCMCFitter()
        result = fitter.fit(local_view, period_init, duration_init, depth_init)
    """

    def __init__(self, n_walkers: int = 32, n_steps: int = 2000, burn_in: int = 500):
        self.n_walkers = n_walkers
        self.n_steps = n_steps
        self.burn_in = burn_in

    def fit(
        self,
        local_view: np.ndarray,
        period_init: float,
        duration_init: float,
        depth_init: float,
        period_sigma: float = 0.5,
        duration_sigma: float = 0.02,
        depth_sigma: float = 0.001
    ) -> Dict[str, float]:
        """
        Run MCMC to refine transit parameters.

        Args:
            local_view:      (201,) phase-folded local view
            period_init:     Initial period estimate (days) from ECLIPSE-PRIME
            duration_init:   Initial duration estimate (days)
            depth_init:      Initial depth estimate (fractional)
            *_sigma:         Prior widths (1σ) centered on init values

        Returns:
            dict with period_median, period_std, duration_median, duration_std,
            depth_median, depth_std, impact_b_median, impact_b_std, acceptance_rate
        """
        try:
            import emcee
            import batman
        except ImportError:
            logger.warning("emcee or batman not installed. Returning init values.")
            return {
                "period_median": period_init, "period_std": period_sigma,
                "duration_median": duration_init, "duration_std": duration_sigma,
                "depth_median": depth_init, "depth_std": depth_sigma,
                "impact_b_median": 0.0, "impact_b_std": 0.1,
                "acceptance_rate": 0.0
            }

        n_pts = len(local_view)
        # Phase grid matching local view
        half_dur = 2.0 * duration_init / period_init
        phase_grid = np.linspace(-half_dur, half_dur, n_pts)
        t_grid = phase_grid * period_init

        flux_obs = local_view.astype(np.float64)
        # Estimate noise from out-of-transit bins
        outer = np.abs(phase_grid) > 1.5 * (duration_init / period_init)
        sigma_obs = float(np.std(flux_obs[outer])) if outer.sum() > 10 else 0.001

        def log_prior(theta):
            p, dur, dep, b = theta
            if (period_init - 3*period_sigma < p < period_init + 3*period_sigma and
                0.005 < dur < period_init * 0.2 and
                1e-5 < dep < 0.5 and
                0.0 <= b < 1.0):
                return 0.0
            return -np.inf

        def log_likelihood(theta):
            p, dur, dep, b = theta
            rp_rs = np.sqrt(dep)
            a_rs = max(((p / 365.25) ** (2.0/3.0)) * 215.0, 2.0)
            inc = np.degrees(np.arccos(b / a_rs))

            params = batman.TransitParams()
            params.t0 = 0.0
            params.per = p
            params.rp = min(rp_rs, 0.5)
            params.a = a_rs
            params.inc = inc
            params.ecc = 0.0
            params.w = 90.0
            params.u = [0.4804, 0.1867]
            params.limb_dark = "quadratic"

            try:
                m = batman.TransitModel(params, t_grid)
                model = m.light_curve(params) - 1.0
                residuals = flux_obs - model
                return -0.5 * np.sum((residuals / sigma_obs) ** 2)
            except Exception:
                return -np.inf

        def log_prob(theta):
            lp = log_prior(theta)
            if not np.isfinite(lp):
                return -np.inf
            return lp + log_likelihood(theta)

        # Initialize walkers near initial values
        init = np.array([period_init, duration_init, depth_init, 0.1])
        scale = np.array([period_sigma * 0.1, duration_sigma * 0.1, depth_sigma * 0.1, 0.05])
        pos = init + scale * np.random.randn(self.n_walkers, 4)

        try:
            sampler = emcee.EnsembleSampler(self.n_walkers, 4, log_prob)
            sampler.run_mcmc(pos, self.n_steps, progress=False)
            flat_samples = sampler.get_chain(discard=self.burn_in, flat=True)
            acceptance = float(np.mean(sampler.acceptance_fraction))

            percentiles = np.percentile(flat_samples, [16, 50, 84], axis=0)
            return {
                "period_median": float(percentiles[1, 0]),
                "period_std": float((percentiles[2, 0] - percentiles[0, 0]) / 2),
                "duration_median": float(percentiles[1, 1]),
                "duration_std": float((percentiles[2, 1] - percentiles[0, 1]) / 2),
                "depth_median": float(percentiles[1, 2]),
                "depth_std": float((percentiles[2, 2] - percentiles[0, 2]) / 2),
                "impact_b_median": float(percentiles[1, 3]),
                "impact_b_std": float((percentiles[2, 3] - percentiles[0, 3]) / 2),
                "acceptance_rate": acceptance
            }
        except Exception as e:
            logger.warning(f"MCMC failed: {e}. Returning init values.")
            return {
                "period_median": period_init, "period_std": period_sigma,
                "duration_median": duration_init, "duration_std": duration_sigma,
                "depth_median": depth_init, "depth_std": depth_sigma,
                "impact_b_median": 0.0, "impact_b_std": 0.1,
                "acceptance_rate": 0.0
            }
