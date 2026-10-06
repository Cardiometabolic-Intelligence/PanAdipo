"""Per-study PNG report: PanAdipo score and CT-calibrated EAT estimates vs. the CT reference cohort."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .eat import EATCalibration  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, REF_FILL = "#fcfcfb", "#d9d8d3"
ACCENT, ACCENT_SOFT, ACCENT_FAINT = "#2a78d6", "#a9c8ef", "#dbe8f8"


def _density_from_quantiles(q: np.ndarray, log: bool, n_grid: int = 300):
    """Smooth density of the reference cohort reconstructed from its 0..100th percentiles."""
    x = np.log10(q) if log else q
    grid = np.linspace(x[1], x[-2], n_grid)
    cdf = np.interp(grid, x, np.linspace(0, 1, len(x)))
    dens = np.gradient(cdf, grid)
    k = np.exp(-0.5 * (np.arange(-30, 31) / 10) ** 2)
    dens = np.convolve(np.pad(dens, 30, mode="edge"), k / k.sum(), mode="valid")
    return (10 ** grid if log else grid), dens / dens.max()


def _panel(ax, title, value_txt, sub_txt, ref_q, est, pi80=None, pi95=None, log=False, xlabel=""):
    gx, gd = _density_from_quantiles(np.asarray(ref_q), log)
    ax.fill_between(gx, 0, gd, color=REF_FILL, lw=0, zorder=1)
    if pi95 is not None:
        ax.axvspan(*pi95, color=ACCENT_FAINT, lw=0, zorder=0)
    if pi80 is not None:
        ax.axvspan(*pi80, color=ACCENT_SOFT, alpha=0.6, lw=0, zorder=0)
    ax.axvline(est, color=ACCENT, lw=2, zorder=3)
    ax.plot([est], [1.04], marker="v", ms=9, color=ACCENT, mec=SURFACE, mew=2, zorder=4, clip_on=False)
    if log:
        ax.set_xscale("log")
        ticks = [t for t in (0.003, 0.01, 0.03, 0.1, 0.3, 1) if gx.min() * 0.7 <= t <= max(gx.max(), est) * 1.3]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.minorticks_off()
    ax.set_facecolor(SURFACE)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)
    ax.text(0, 1.62, title, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.text(0, 1.27, value_txt, transform=ax.transAxes, fontsize=22, color=INK, va="bottom", weight="bold")
    ax.text(0, 1.14, sub_txt, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")


def study_report(row, out_path, calib: EATCalibration = None):
    """row: one line of study_predictions.csv (dict / Series)."""
    calib = calib or EATCalibration.load()
    spec = calib.spec
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), facecolor=SURFACE)
    fig.subplots_adjust(left=0.04, right=0.98, top=0.60, bottom=0.21, wspace=0.18)

    s = float(row["panadipo_score"])
    _panel(axes[0], "PanAdipo score", f"{s:.3f}",
           f"{row['score_percentile']:.0f}th percentile  |  P(high EAT) {row['p_high_eat']:.0%}",
           spec["reference_score_quantiles"], s, log=True, xlabel="PanAdipo score (log scale)")
    _panel(axes[1], "Estimated EAT / heart volume", f"{row['eat_index']:.2f}",
           f"80% PI {row['eat_index_pi80_lo']:.2f}–{row['eat_index_pi80_hi']:.2f}",
           spec["reference_quantiles"]["eat_index"], row["eat_index"],
           (row["eat_index_pi80_lo"], row["eat_index_pi80_hi"]),
           (row["eat_index_pi95_lo"], row["eat_index_pi95_hi"]), xlabel="EAT volume / total heart volume")
    _panel(axes[2], "Estimated EAT volume", f"{row['eat_volume_ml']:.0f} mL",
           f"80% PI {row['eat_volume_ml_pi80_lo']:.0f}–{row['eat_volume_ml_pi80_hi']:.0f} mL",
           spec["reference_quantiles"]["eat_volume_ml"], row["eat_volume_ml"],
           (row["eat_volume_ml_pi80_lo"], row["eat_volume_ml_pi80_hi"]),
           (row["eat_volume_ml_pi95_lo"], row["eat_volume_ml_pi95_hi"]), xlabel="EAT volume (mL)")

    fig.text(0.04, 0.95, f"PanAdipo  ·  study {row['study_id']}  ·  {int(row['n_clips'])} clips",
             fontsize=12, color=INK, weight="bold")
    fig.text(0.04, 0.025,
             f"Grey: CT reference cohort (YNHHS–ORFAN, n={spec['n_patients']:,}). Blue line: this study; "
             "bands: 80% / 95% prediction intervals.\n"
             f"EAT / heart volume = {calib.formula('eat_index')};  EAT (mL) = {calib.formula('eat_volume_ml')}.  "
             "Research use only — not a CT measurement.",
             fontsize=7, color=MUTED)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path
