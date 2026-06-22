import numpy as np
from scipy.stats import median_abs_deviation


def bls_search(
    lc: dict,
    p_min: float = 0.5,
    p_max: float = 15.0,
    n_per: int = 2000,
    n_phase: int = 300,
) -> dict:
    """Fast vectorised BLS periodogram using phase-binning."""
    t = lc["time"]
    f = lc["flux"]
    T = t[-1] - t[0]

    periods = np.exp(
        np.linspace(
            np.log(p_min),
            np.log(min(p_max, max(T * 0.9, p_min + 0.01))),
            n_per,
        )
    )

    dur_fracs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.10, 0.13]

    best_power = 0.0
    best = {
        "period": periods[0],
        "t0": t[0],
        "depth": 0.0,
        "duration": 0.01,
        "power": 0.0,
    }

    powers = np.zeros(n_per)
    global_mean = np.mean(f)

    bins = np.linspace(0, 1, n_phase + 1)
    bc = (bins[:-1] + bins[1:]) / 2

    for ip, P in enumerate(periods):
        phase = ((t - t[0]) / P) % 1.0

        cnt, _ = np.histogram(phase, bins=bins)
        fsum, _ = np.histogram(phase, bins=bins, weights=f)
        fmean = np.where(cnt > 0, fsum / np.maximum(cnt, 1), global_mean)

        best_p = 0.0

        for q in dur_fracs:
            w = max(int(q * n_phase), 1)

            # sliding box sum (tile for wrap-around)
            fm2 = np.tile(fmean, 2)
            ct2 = np.tile(cnt, 2)

            bsum = np.convolve(fm2, np.ones(w), mode="valid")[:n_phase]
            bcnt = np.convolve(ct2, np.ones(w, dtype=int), mode="valid")[:n_phase]

            box_mean = np.where(bcnt > 0, bsum / w, global_mean)
            depth_arr = global_mean - box_mean

            power_arr = np.where(
                (depth_arr > 0) & (bcnt > 3),
                (depth_arr**2) * bcnt / (len(t) * q * max(1 - q, 0.01)),
                0.0,
            )

            idx = int(np.argmax(power_arr))
            if power_arr[idx] > best_p:
                best_p = float(power_arr[idx])
                if power_arr[idx] > best_power:
                    best_power = float(power_arr[idx])
                    phi0 = bc[idx]
                    t0_est = t[0] + phi0 * P
                    best = {
                        "period": P,
                        "t0": t0_est,
                        "duration": q * P,
                        "depth": max(float(depth_arr[idx]) / w, 0.0),
                        "dur_frac": q,
                        "power": float(power_arr[idx]),
                    }

        powers[ip] = best_p

    noise = 1.4826 * float(median_abs_deviation(f))
    snr = best["depth"] / max(noise, 1e-9)
    best["snr"] = snr
    best["power_spectrum"] = (periods, powers)
    best["noise"] = noise
    return best

