import numpy as np


def _box_transit(time, t0, period, depth, duration):
    """Simple box-car transit model."""
    phase = ((time - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    sig = np.zeros_like(time)
    half = duration / period / 2
    sig[np.abs(phase) < half] = -depth
    return sig


def synthetic_lc(kind: str = "transit", n_pts: int = 18000, seed: int = 42) -> dict:
    """Generate a realistic synthetic TESS-like light curve.

    kind ∈ {'transit', 'eclipse', 'blend', 'starspot', 'noise'}
    """
    rng = np.random.default_rng(seed)
    dt = 2 / 1440  # 2-min cadence in days
    time = np.arange(n_pts) * dt

    # --- stellar variability (low-freq) ---
    P_var = rng.uniform(5, 25)  # stellar rotation period
    amp_var = rng.uniform(0.0005, 0.002)
    flux = 1.0 + amp_var * np.sin(2 * np.pi * time / P_var + rng.uniform(0, 2 * np.pi))

    # --- correlated (red) noise ---
    sigma_red = rng.uniform(3e-4, 8e-4)
    red_noise = np.cumsum(rng.normal(0, sigma_red / np.sqrt(200), n_pts))
    red_noise -= red_noise.mean()
    red_noise *= 1e-3
    flux += red_noise

    # --- white noise ---
    sigma_white = rng.uniform(3e-4, 6e-4)
    flux += rng.normal(0, sigma_white, n_pts)

    # --- astrophysical signal ---
    params = {}
    if kind == "transit":
        period = rng.uniform(1.5, 12.0)
        t0 = rng.uniform(0, period)
        depth = rng.uniform(0.002, 0.025)
        dur = rng.uniform(0.05, 0.15)
        params = {"period": period, "t0": t0, "depth": depth, "duration": dur}
        flux += _box_transit(time, t0, period, depth, dur)

    elif kind == "eclipse":
        period = rng.uniform(0.5, 5.0)
        t0 = rng.uniform(0, period)
        depth = rng.uniform(0.08, 0.5)  # deep eclipse
        dur = rng.uniform(0.05, 0.2)
        params = {"period": period, "t0": t0, "depth": depth, "duration": dur}
        flux += _box_transit(time, t0, period, depth, dur)
        # secondary eclipse at half phase
        flux += _box_transit(time, t0 + period / 2, period, depth * 0.4, dur)

    elif kind == "blend":
        period = rng.uniform(0.5, 3.0)
        t0 = rng.uniform(0, period)
        depth = rng.uniform(0.01, 0.06)
        dur = rng.uniform(0.05, 0.15)
        params = {"period": period, "t0": t0, "depth": depth, "duration": dur}
        flux += _box_transit(time, t0, period, depth, dur)
        # add background EB contribution
        flux += _box_transit(time, t0 + period / 2, period, depth * 0.3, dur * 0.8)
        flux += amp_var * 2 * np.sin(4 * np.pi * time / period)

    elif kind == "starspot":
        P_spot = rng.uniform(8, 20)
        amp_sp = rng.uniform(0.003, 0.015)
        flux += amp_sp * (np.sin(2 * np.pi * time / P_spot) ** 2 - 0.5)
        params = {"period": P_spot, "depth": amp_sp}

    # kind == noise -> no additional astrophysical signal

    flux_err = np.full_like(flux, sigma_white)
    return {
        "time": time,
        "flux": flux,
        "flux_err": flux_err,
        "kind": kind,
        "true_params": params,
    }

