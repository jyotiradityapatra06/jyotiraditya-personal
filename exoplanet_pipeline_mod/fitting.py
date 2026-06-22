import numpy as np
from scipy.optimize import curve_fit


def trapezoid_model(t, t0, depth, T_total, T_flat):
    """Symmetric trapezoid transit model."""
    T_ingress = max((T_total - T_flat) / 2, 1e-6)
    f = np.ones_like(t)
    dt = np.abs(t - t0)

    # flat bottom
    f[dt <= T_flat / 2] = 1 - depth

    # ingress/egress
    mask = (dt > T_flat / 2) & (dt < T_total / 2)
    f[mask] = 1 - depth * (1 - (dt[mask] - T_flat / 2) / T_ingress)
    return f


def fit_transit(lc: dict, bls: dict) -> dict:
    """Fold at BLS period and fit a trapezoid model."""
    t = lc["time"]
    f = lc["flux"]
    fe = lc["flux_err"]

    P = bls.get("period", 1)
    t0 = bls.get("t0", t[0])
    dep = bls.get("depth", 0.01)
    dur = bls.get("duration", 0.1)

    # fold
    phase = ((t - t0) / P) % 1.0
    phase[phase > 0.5] -= 1.0
    sort = np.argsort(phase)

    ph = phase[sort]
    fl = f[sort]
    fle = fe[sort]

    half = dur / P * 3
    in_tr = np.abs(ph) < half
    if in_tr.sum() < 10:
        out = {"status": "insufficient_data"}
        out.update(bls)
        return out

    ph_tr = ph[in_tr] * P
    fl_tr = fl[in_tr]
    fle_tr = fle[in_tr]

    try:
        p0 = [0, dep, dur, dur * 0.5]
        bnds = (
            [-dur, 0, 1e-4, 1e-5],
            [dur, 1, dur * 3, dur * 2],
        )

        popt, pcov = curve_fit(
            trapezoid_model,
            ph_tr,
            fl_tr,
            p0=p0,
            bounds=bnds,
            sigma=fle_tr,
            maxfev=5000,
        )
        perr = np.sqrt(np.diag(pcov))
        phase_folded = (ph[in_tr], fl[in_tr])
        x = np.linspace(-dur * 1.5, dur * 1.5, 300)
        fit_curve = (x, trapezoid_model(x, *popt))

        return {
            "t0_offset_hrs": round(float(popt[0]) * 24, 4),
            "depth": round(float(popt[1]), 6),
            "depth_ppm": round(float(popt[1]) * 1e6, 1),
            "duration_hrs": round(float(popt[2]) * 24, 3),
            "duration_err_hrs": round(float(perr[2]) * 24, 4),
            "period_days": round(float(P), 6),
            "t0_bjd": round(float(t0), 6),
            "ingress_egress_hrs": round(float(popt[2] - popt[3]) / 2 * 24, 3),
            "status": "converged",
            "phase_folded": phase_folded,
            "fit_curve": fit_curve,
        }
    except Exception as ex:
        return {
            "status": f"failed: {ex}",
            "depth": dep,
            "duration_hrs": dur * 24,
            "period_days": P,
        }

