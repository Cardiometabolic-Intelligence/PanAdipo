"""Per-study PNG report: PanAdipo score, CT-calibrated EAT estimates, and the age-sex nomogram."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .eat import EATCalibration  # noqa: E402

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
SURFACE, REF_FILL, GRID = "#fcfcfb", "#d9d8d3", "#ecebe7"
ACCENT = "#2a78d6"
SCORE_TICKS = (0.003, 0.01, 0.03, 0.1, 0.3, 1)


def _density_from_quantiles(q, log: bool, n_grid: int = 300):
    """Smooth reference density reconstructed from evenly spaced quantiles."""
    q = np.asarray(q, float)
    x = np.log10(q) if log else q
    lv = np.linspace(0, 1, len(x))
    keep = np.r_[True, np.diff(x) > 0]          # drop tied (clamped) quantiles -> no edge spikes
    x, lv = x[keep], lv[keep]
    grid = np.linspace(x[1], x[-2], n_grid)
    cdf = np.interp(grid, x, lv)
    dens = np.gradient(cdf, grid)
    k = np.exp(-0.5 * (np.arange(-30, 31) / 10) ** 2)
    dens = np.convolve(np.pad(dens, 30, mode="edge"), k / k.sum(), mode="valid")
    return (10 ** grid if log else grid), dens / dens.max()


def _style(ax, xlabel):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.set_xlabel(xlabel, fontsize=8, color=INK2)


def _log_score_axis(ax, lo, hi):
    ax.set_xscale("log")
    ticks = [t for t in SCORE_TICKS if lo * 0.7 <= t <= hi * 1.3]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t:g}" for t in ticks])
    ax.minorticks_off()


def _header(ax, title, value_txt, sub_txt):
    ax.text(0, 1.62, title, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")
    ax.text(0, 1.27, value_txt, transform=ax.transAxes, fontsize=22, color=INK, va="bottom", weight="bold")
    ax.text(0, 1.14, sub_txt, transform=ax.transAxes, fontsize=8, color=INK2, va="bottom")


def _hist_panel(ax, ref_q, est, log=False, xlabel="", threshold=None):
    gx, gd = _density_from_quantiles(ref_q, log)
    ax.fill_between(gx, 0, gd, color=REF_FILL, lw=0, zorder=1)
    if threshold is not None:
        ax.axvline(threshold, color=INK2, lw=1, ls=(0, (3, 2)), zorder=2)
        ax.text(threshold, 1.0, f" high EAT\n ≥ {threshold:.2f}", fontsize=7, color=INK2, va="top", ha="left")
    ax.axvline(est, ymax=1.0 / 1.12, color=ACCENT, lw=2, zorder=3)
    ax.plot([est], [1.04], marker="v", ms=9, color=ACCENT, mec=SURFACE, mew=2, zorder=4, clip_on=False)
    if log:
        _log_score_axis(ax, gx.min(), max(gx.max(), est))
    ax.set_ylim(0, 1.12)
    ax.set_yticks([])
    _style(ax, xlabel)


def _nomogram_panel(ax, calib: EATCalibration, group: dict, score: float):
    thr = group["paper_score_threshold"]["threshold"]
    ax.axvline(thr, color=INK2, lw=1, ls=(0, (3, 2)), zorder=1)
    pt = group["paper_score_threshold"]
    ax.text(thr, 1.0, f"threshold {thr:.3f}  \nsens {pt['sensitivity']}% · spec {pt['specificity']}%  ",
            fontsize=7, color=INK2, va="top", ha="right" if thr > 0.15 else "left")
    xs = np.logspace(np.log10(0.002), 0, 200)
    lx = np.log10(xs)
    p = 1 / (1 + np.exp(-(group["logit_b0"] + group["logit_b1"] * lx)))
    ax.plot(xs, p, color=INK2, lw=2, zorder=2)
    ps = 1 / (1 + np.exp(-(group["logit_b0"] + group["logit_b1"] * np.log10(score))))
    ax.plot([score], [ps], "o", ms=9, color=ACCENT, mec=SURFACE, mew=2, zorder=4)
    _log_score_axis(ax, 0.003, 1)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
    ax.spines["left"].set_visible(False)
    _style(ax, "PanAdipo score (log scale)")
    ax.spines["left"].set_visible(False)


def study_report(row, out_path, calib: EATCalibration = None):
    """row: one line of study_predictions.csv (dict / Series)."""
    calib = calib or EATCalibration.load()
    spec = calib.spec
    g = calib.age_sex_group(row.get("age"), row.get("sex"))
    s = float(row["panadipo_score"])

    fig, axes = plt.subplots(1, 4 if g else 3, figsize=(14.5 if g else 11, 3.7), facecolor=SURFACE)
    fig.subplots_adjust(left=0.035, right=0.985, top=0.60, bottom=0.21, wspace=0.22)

    if g:
        grp = f"{g['sex']}, {g['age_group'].replace('>=', '≥')} y"
        ref = {"score": g["score_quantiles"], "eat_index": g["eat_index_quantiles"],
               "eat_volume_ml": g["eat_volume_ml_quantiles"]}
        pct, p_high, thr = row["score_percentile_age_sex"], row["p_high_eat_age_sex"], g["high_eat_threshold_eat_index"]
        ref_txt = f"Grey: CT reference patients of the same age group and sex ({grp}; n={g['n']:,})."
    else:
        grp = "all ages, both sexes"
        ref = {"score": spec["reference_score_quantiles"], **spec["reference_quantiles"]}
        pct, p_high, thr = row["score_percentile"], row["p_high_eat"], None
        ref_txt = (f"Grey: CT reference cohort (n={spec['n_patients']:,}). "
                   "Give age and sex for the age-sex-specific nomogram.")

    _header(axes[0], "PanAdipo score", f"{s:.3f}", f"{pct:.0f}th percentile ({grp})")
    _hist_panel(axes[0], ref["score"], s, log=True, xlabel="PanAdipo score (log scale)")

    _header(axes[1], "Estimated EAT / heart volume", f"{row['eat_index']:.2f}",
            f"80% PI {row['eat_index_pi80_lo']:.2f}–{row['eat_index_pi80_hi']:.2f}")
    _hist_panel(axes[1], ref["eat_index"], row["eat_index"], xlabel="EAT volume / total heart volume",
                threshold=thr)

    _header(axes[2], "Estimated EAT volume", f"{row['eat_volume_ml']:.0f} mL",
            f"80% PI {row['eat_volume_ml_pi80_lo']:.0f}–{row['eat_volume_ml_pi80_hi']:.0f} mL")
    _hist_panel(axes[2], ref["eat_volume_ml"], row["eat_volume_ml"], xlabel="EAT volume (mL)")

    if g:
        _header(axes[3], f"Probability of high EAT ({grp})", f"{p_high:.0%}", str(row.get("eat_flag", "")))
        _nomogram_panel(axes[3], calib, g, s)

    bmi = row.get("bmi")
    demo = "" if not g else f"  ·  {g['sex']}, age {row['age']:.0f}"
    if bmi is not None and np.isfinite(bmi):
        demo += f"  ·  BMI {bmi:.1f}"
    fig.text(0.035, 0.95, f"PanAdipo  ·  study {row['study_id']}  ·  {int(row['n_clips'])} clips{demo}",
             fontsize=12, color=INK, weight="bold")
    fig.text(0.035, 0.025,
             f"{ref_txt} Blue: this study."
             + (f" Dashed (right): age-sex PanAdipo threshold for high EAT (EAT / heart volume ≥ {thr:.2f})."
                if g else "") + "\n"
             f"EAT / heart volume = {calib.formula('eat_index')};  EAT (mL) = {calib.formula('eat_volume_ml')}.  "
             "Research use only — not a CT measurement.", fontsize=7, color=MUTED)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out_path
