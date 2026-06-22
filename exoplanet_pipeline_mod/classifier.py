import numpy as np

CLASSES = ["transit", "eclipse", "blend", "starspot", "noise"]


def classify(lc: dict, bls: dict) -> dict:
    """Rule-based classifier using BLS results + light-curve statistics."""
    f = lc["flux"]
    depth = bls.get("depth", 0)
    snr = bls.get("snr", 0)
    dur = bls.get("duration", 0)
    per = bls.get("period", 1)

    noise = bls.get("noise", 1)

    # --- feature extraction ---
    skewness = float(np.mean((f - np.mean(f)) ** 3) / (np.std(f) + 1e-9) ** 3)
    kurtosis = float(np.mean((f - np.mean(f)) ** 4) / (np.std(f) + 1e-9) ** 4)
    rms = float(np.std(f))
    duty_cycle = dur / per if per > 0 else 0

    feats = {
        "depth": round(depth, 6),
        "snr": round(snr, 2),
        "duration_hrs": round(dur * 24, 2),
        "period_days": round(per, 4),
        "duty_cycle": round(duty_cycle, 4),
        "rms": round(rms, 6),
        "skewness": round(skewness, 3),
        "kurtosis": round(kurtosis, 3),
    }

    scores = {c: 0.0 for c in CLASSES}

    if snr < 5 or depth < 1e-4:
        scores["noise"] += 60
    else:
        scores["noise"] -= 20

    # transit
    if 0.0005 < depth < 0.03:
        scores["transit"] += 40
    if 1.0 < dur * 24 < 8.0:
        scores["transit"] += 20
    if snr > 7:
        scores["transit"] += 15
    if duty_cycle < 0.10:
        scores["transit"] += 10

    # eclipse
    if depth > 0.05:
        scores["eclipse"] += 60
    if depth > 0.1:
        scores["eclipse"] += 30

    # blend
    if 0.01 < depth < 0.08 and rms > 0.003:
        scores["blend"] += 35
    if abs(skewness) > 0.5:
        scores["blend"] += 20

    # starspot
    if kurtosis < 2.5 and rms > 0.002:
        scores["starspot"] += 40
    if depth < 0.005 and rms > 0.001:
        scores["starspot"] += 20

    total = sum(scores.values()) + 1e-9
    probs = {c: max(s, 0) / max(sum(max(v, 0) for v in scores.values()), 1e-9) for c, s in scores.items()}

    label = max(probs, key=probs.get)
    conf = probs[label]

    return {
        "label": label,
        "confidence": round(conf, 3),
        "probabilities": {k: round(v, 3) for k, v in probs.items()},
        "features": feats,
    }

