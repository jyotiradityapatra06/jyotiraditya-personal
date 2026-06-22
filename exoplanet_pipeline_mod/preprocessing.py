import numpy as np
from scipy.signal import medfilt, savgol_filter
from scipy.stats import median_abs_deviation


def preprocess(lc: dict, window_hours: float = 12.0) -> dict:
    """Normalise → sigma-clip → detrend with Savitzky-Golay.

    Returns cleaned, normalised flux and retains trend/flux_raw.
    """
    t = lc["time"].copy()
    f = lc["flux"].copy()
    fe = lc["flux_err"].copy()

    # --- normalise ---
    med = np.median(f)
    f /= med
    fe /= med

    # --- sigma clipping: remove only upward spikes and NaN/Inf ---
    finite = np.isfinite(f)
    mu = np.median(f[finite])
    sig = 1.4826 * median_abs_deviation(f[finite])
    sig = max(sig, 1e-6)

    # Very deep eclipses need wider window — handled downstream by classifier/fit.
    mask = finite & (f - mu < max(20 * sig, 0.03))
    t, f, fe = t[mask], f[mask], fe[mask]

    # --- detrend with Savitzky-Golay ---
    cadence = np.median(np.diff(t))
    window_d = window_hours / 24.0
    wlen = max(int(window_d / cadence) | 1, 51)  # must be odd
    if wlen >= len(f):
        wlen = max(len(f) // 3 | 1, 51)

    try:
        trend = savgol_filter(f, window_length=wlen, polyorder=2)
    except Exception:
        trend = medfilt(f, kernel_size=min(wlen, 101) | 1)

    f_flat = f / trend

    lc_out = dict(lc)
    lc_out.update(
        {
            "time": t,
            "flux_raw": f,
            "flux": f_flat,
            "flux_err": fe,
            "trend": trend,
        }
    )
    return lc_out

