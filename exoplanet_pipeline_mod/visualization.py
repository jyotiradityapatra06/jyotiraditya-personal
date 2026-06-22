import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

PALETTE = {
    "bg": "#0d0d0d",
    "panel": "#1a1a2e",
    "accent": "#f0a500",
    "transit": "#00d4ff",
    "eclipse": "#ff6b6b",
    "blend": "#a29bfe",
    "starspot": "#55efc4",
    "noise": "#636e72",
    "text": "#e8e8e8",
    "grid": "#2d2d44",
}

CLASS_COLORS = {
    "transit": PALETTE["transit"],
    "eclipse": PALETTE["eclipse"],
    "blend": PALETTE["blend"],
    "starspot": PALETTE["starspot"],
    "noise": PALETTE["noise"],
}

plt.rcParams.update(
    {
        "figure.facecolor": PALETTE["bg"],
        "axes.facecolor": PALETTE["panel"],
        "axes.edgecolor": PALETTE["grid"],
        "axes.labelcolor": PALETTE["text"],
        "xtick.color": PALETTE["text"],
        "ytick.color": PALETTE["text"],
        "text.color": PALETTE["text"],
        "grid.color": PALETTE["grid"],
        "grid.alpha": 0.5,
        "font.family": "DejaVu Sans",
        "font.size": 9,
    }
)


def plot_target(name: str, lc: dict, bls: dict, clf: dict, fit: dict, out_path: str):
    """Full 6-panel diagnostic figure for one target."""
    fig = plt.figure(figsize=(16, 12))
    fig.patch.set_facecolor(PALETTE["bg"])

    label = clf["label"]
    conf = clf["confidence"]
    color = CLASS_COLORS.get(label, PALETTE["accent"])
    snr = bls.get("snr", 0)
    period = bls.get("period", 1)

    gs = gridspec.GridSpec(
        3,
        3,
        figure=fig,
        hspace=0.45,
        wspace=0.35,
        left=0.07,
        right=0.97,
        top=0.88,
        bottom=0.07,
    )

    fig.text(
        0.5,
        0.95,
        f"Exoplanet Pipeline — {name}",
        ha="center",
        fontsize=15,
        fontweight="bold",
        color=PALETTE["accent"],
    )
    cls_txt = f"Classification: {label.upper()}  |  Confidence: {conf:.1%}  |  SNR: {snr:.1f}σ"
    fig.text(0.5, 0.91, cls_txt, ha="center", fontsize=10, color=color)

    t = lc["time"]
    fr = lc.get("flux_raw", lc["flux"])
    ff = lc["flux"]

    # Panel 1
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(t, fr, ".", ms=0.8, alpha=0.6, color="#aaaaaa", rasterized=True)
    ax1.plot(t, lc.get("trend", np.ones_like(t)), color=PALETTE["accent"], lw=1.2, label="SG trend")
    ax1.set_xlabel("Time (days)")
    ax1.set_ylabel("Norm. Flux")
    ax1.set_title("Raw + Detrended", color=PALETTE["text"], fontsize=9)
    ax1.legend(fontsize=7)
    ax1.grid(True)

    # Panel 2
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(t, ff, ".", ms=0.8, alpha=0.6, color=color, rasterized=True)
    ax2.axhline(1.0, color="#555", lw=0.7, ls="--")
    ax2.set_xlabel("Time (days)")
    ax2.set_ylabel("Detrended Flux")
    ax2.set_title("Flattened Light Curve", color=PALETTE["text"], fontsize=9)
    ax2.grid(True)

    # Panel 3
    ax3 = fig.add_subplot(gs[2, :2])
    per_arr, pow_arr = bls.get("power_spectrum", (np.array([1]), np.array([0])))
    ax3.plot(per_arr, pow_arr, color=PALETTE["accent"], lw=0.8)
    ax3.axvline(period, color=color, lw=1.5, ls="--", label=f"P = {period:.4f} d")
    ax3.set_xlabel("Period (days)")
    ax3.set_ylabel("BLS Power")
    ax3.set_title("BLS Periodogram", color=PALETTE["text"], fontsize=9)
    ax3.legend(fontsize=7)
    ax3.grid(True)

    # Panel 4
    ax4 = fig.add_subplot(gs[0, 2])
    t0 = bls.get("t0", t[0])
    phase = ((t - t0) / period) % 1.0
    phase[phase > 0.5] -= 1.0
    sort = np.argsort(phase)
    ax4.plot(phase[sort], ff[sort], ".", ms=1.0, alpha=0.4, color=color, rasterized=True)

    bins = np.linspace(-0.5, 0.5, 150)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_flux = []
    for i in range(len(bins) - 1):
        m = (phase[sort] >= bins[i]) & (phase[sort] < bins[i + 1])
        bin_flux.append(ff[sort][m].mean() if m.sum() > 0 else np.nan)
    ax4.plot(bin_centers, bin_flux, "-", color="white", lw=1.0, alpha=0.9)
    ax4.axhline(1.0, color="#555", lw=0.6, ls="--")
    ax4.set_xlabel("Phase")
    ax4.set_ylabel("Norm. Flux")
    ax4.set_title(f"Phase-Folded (P={period:.3f}d)", fontsize=9)
    ax4.grid(True)

    # Panel 5
    ax5 = fig.add_subplot(gs[1, 2])
    ph_data, fl_data = fit.get("phase_folded", (np.array([]), np.array([])))
    ph_fit_x, fl_fit_y = fit.get("fit_curve", (np.array([]), np.array([])))
    if len(ph_data) > 0:
        ax5.plot(ph_data * 24, fl_data, ".", ms=2.5, alpha=0.5, color=color, label="data")
        ax5.plot(ph_fit_x * 24, fl_fit_y, "-", color="white", lw=1.5, label="trapezoid fit")
        ax5.set_xlabel("Phase (hrs)")
    else:
        ax5.text(0.5, 0.5, "Fit not available", transform=ax5.transAxes, ha="center", va="center")
    ax5.set_ylabel("Norm. Flux")
    ax5.set_title("Transit Fit (Zoom)", fontsize=9)
    ax5.legend(fontsize=6)
    ax5.grid(True)

    # Panel 6
    ax6 = fig.add_subplot(gs[2, 2])
    probs = clf.get("probabilities", {})
    cats = list(probs.keys())
    vals = [probs[c] for c in cats]
    cols = [CLASS_COLORS.get(c, "#888") for c in cats]
    bars = ax6.barh(cats, vals, color=cols, edgecolor="#333", height=0.6)
    ax6.set_xlim(0, 1)
    ax6.set_xlabel("Probability")
    ax6.set_title("Classification Scores", fontsize=9)
    for bar, val in zip(bars, vals):
        ax6.text(min(val + 0.02, 0.95), bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=7)
    ax6.grid(axis="x", alpha=0.4)

    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [plot] saved {out_path}")


def plot_summary_grid(results: list, out_path: str):
    """Grid of phase-folded curves for all targets."""
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(14, rows * 3.5))
    fig.patch.set_facecolor(PALETTE["bg"])
    axes = axes.flatten()

    for i, res in enumerate(results):
        ax = axes[i]
        ax.set_facecolor(PALETTE["panel"])

        lc = res["lc"]
        bls = res["bls"]
        clf = res["clf"]

        label = clf["label"]
        color = CLASS_COLORS.get(label, PALETTE["accent"])

        t = lc["time"]
        ff = lc["flux"]
        t0 = bls.get("t0", t[0])
        P = bls.get("period", 1)

        phase = ((t - t0) / P) % 1.0
        phase[phase > 0.5] -= 1.0
        sort = np.argsort(phase)

        ax.plot(phase[sort], ff[sort], ".", ms=0.8, alpha=0.3, color=color, rasterized=True)

        bins = np.linspace(-0.5, 0.5, 100)
        bc = (bins[:-1] + bins[1:]) / 2
        bf = []
        for j in range(len(bins) - 1):
            m = (phase[sort] >= bins[j]) & (phase[sort] < bins[j + 1])
            bf.append(ff[sort][m].mean() if m.sum() > 0 else np.nan)
        ax.plot(bc, bf, "-", color="white", lw=1.2)

        ax.axhline(1.0, color="#555", lw=0.5, ls="--")
        ax.set_title(f"{res['name']}\n[{label.upper()}]  SNR={bls['snr']:.1f}σ", fontsize=8, color=color)
        ax.tick_params(labelsize=7)
        ax.set_xlabel("Phase", fontsize=7)
        ax.set_ylabel("Flux", fontsize=7)
        for sp in ax.spines.values():
            sp.set_color(PALETTE["grid"])

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(
        "Exoplanet Pipeline — All Targets Summary",
        fontsize=13,
        color=PALETTE["accent"],
        y=0.98,
    )
    plt.savefig(out_path, dpi=130, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"  [plot] summary grid saved {out_path}")

