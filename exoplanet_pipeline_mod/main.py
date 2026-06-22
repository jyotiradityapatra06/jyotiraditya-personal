import json
import os
import numpy as np

from exoplanet_pipeline_mod.config import OUT, PLOT_DIR
from exoplanet_pipeline_mod.synthetic import synthetic_lc
from exoplanet_pipeline_mod.preprocessing import preprocess
from exoplanet_pipeline_mod.bls import bls_search
from exoplanet_pipeline_mod.classifier import classify
from exoplanet_pipeline_mod.fitting import fit_transit
from exoplanet_pipeline_mod.visualization import plot_target, plot_summary_grid
from exoplanet_pipeline_mod.report import generate_report


def run_pipeline():
    print("=" * 65)
    print("  EXOPLANET DETECTION PIPELINE  — Problem Statement 7")
    print("=" * 65)

    # Demo targets (synthetic generation)
    targets = [
        {"name": "WASP-18b (transit)", "kind": "transit", "seed": 1},
        {"name": "TIC-Eclipse-Binary", "kind": "eclipse", "seed": 2},
        {"name": "TIC-Blend-EB", "kind": "blend", "seed": 3},
        {"name": "Starspot-Variable", "kind": "starspot", "seed": 4},
        {"name": "TOI-270 (transit)", "kind": "transit", "seed": 5},
        {"name": "Noisy-Baseline", "kind": "noise", "seed": 6},
    ]

    results = []
    for tgt in targets:
        name = tgt["name"]
        kind = tgt["kind"]

        print(f"\n── {name} ──")

        print("  [data] generating synthetic TESS light curve …")
        lc_raw = synthetic_lc(kind=kind, n_pts=18000, seed=tgt["seed"])

        print("  [preprocess] detrending …")
        lc = preprocess(lc_raw)

        print("  [BLS] searching for periodic dips …")
        bls = bls_search(lc)
        print(
            f"  [BLS] best period = {bls.get('period',0):.4f} d, "
            f"depth = {bls.get('depth',0)*1e6:.0f} ppm, SNR = {bls.get('snr',0):.1f}σ"
        )

        clf = classify(lc, bls)
        print(f"  [clf] → {clf['label'].upper()} (conf={clf['confidence']:.1%})")

        print("  [fit] fitting trapezoid transit model …")
        fit = fit_transit(lc, bls)
        print(
            f"  [fit] status={fit.get('status','?')}, "
            f"depth={fit.get('depth_ppm',0):.0f} ppm, "
            f"dur={fit.get('duration_hrs',0):.2f} hr"
        )

        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        plot_path = os.path.join(PLOT_DIR, f"{safe_name}.png")
        plot_target(name, lc, bls, clf, fit, plot_path)

        results.append(
            {
                "name": name,
                "kind": kind,
                "lc": lc,
                "bls": bls,
                "clf": clf,
                "fit": fit,
                "plot": plot_path,
            }
        )

    print("\n── Generating summary grid …")
    summary_img = os.path.join(OUT, "summary_grid.png")
    plot_summary_grid(results, summary_img)

    json_path = os.path.join(OUT, "pipeline_results.json")
    json_out = []
    for res in results:
        fit_out = {k: v for k, v in res["fit"].items() if not isinstance(v, tuple)}
        json_out.append(
            {
                "name": res["name"],
                "truth_class": res["kind"],
                "predicted_class": res["clf"]["label"],
                "confidence": res["clf"]["confidence"],
                "probabilities": res["clf"]["probabilities"],
                "features": res["clf"]["features"],
                "bls": {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in res["bls"].items()
                    if not isinstance(v, tuple)
                },
                "fit_params": {
                    k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                    for k, v in fit_out.items()
                },
            }
        )

    with open(json_path, "w") as fh:
        json.dump(json_out, fh, indent=2)
    print(f"  [json] results saved {json_path}")

    print("\n── Generating PDF report …")
    report_path = os.path.join(OUT, "exoplanet_pipeline_report.pdf")
    generate_report(results, summary_img, report_path)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print(f"  Summary grid : {summary_img}")
    print(f"  JSON results : {json_path}")
    print(f"  PDF report   : {report_path}")
    print(f"  Plots dir    : {PLOT_DIR}")
    print("=" * 65)

    return results


if __name__ == "__main__":
    run_pipeline()

