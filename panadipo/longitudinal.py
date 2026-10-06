"""Track PanAdipo and EAT estimates for a patient across serial echocardiograms."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .report import ACCENT, GRID, INK, INK2, MUTED, SURFACE  # noqa: E402


def _serial(studies: pd.DataFrame) -> pd.DataFrame:
    d = studies.dropna(subset=["patient_id", "study_date", "panadipo_score"]).copy()
    d["patient_id"] = d["patient_id"].astype(str)
    d["study_date"] = pd.to_datetime(d["study_date"], errors="coerce")
    return d.dropna(subset=["study_date"]).sort_values(["patient_id", "study_date"])


def trajectories(studies: pd.DataFrame) -> pd.DataFrame:
    """One row per patient with >= 2 dated studies: first -> last change and annualised change."""
    rows = []
    for pid, g in _serial(studies).groupby("patient_id"):
        if len(g) < 2:
            continue
        a, b = g.iloc[0], g.iloc[-1]
        yrs = max((b.study_date - a.study_date).days / 365.25, 1e-9)
        flags = g.get("eat_flag", pd.Series(dtype=object)).fillna("")
        rows.append({
            "patient_id": pid, "n_studies": len(g),
            "first_date": a.study_date.date(), "last_date": b.study_date.date(), "years": round(yrs, 2),
            "score_first": a.panadipo_score, "score_last": b.panadipo_score,
            "eat_index_change": b.eat_index - a.eat_index,
            "eat_ml_change": b.eat_volume_ml - a.eat_volume_ml,
            "eat_ml_change_per_year": (b.eat_volume_ml - a.eat_volume_ml) / yrs,
            "screen_positive_ever": bool(flags.str.startswith("screen-positive").any()),
            "occult_ever": bool(flags.str.contains("occult").any()),
        })
    return pd.DataFrame(rows)


def plot_trajectory(studies: pd.DataFrame, patient_id: str):
    """Two panels over time: PanAdipo score (with the age-sex threshold) and estimated EAT / heart volume."""
    g = _serial(studies)
    g = g[g.patient_id == str(patient_id)]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURFACE)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.78, bottom=0.24, wspace=0.25)
    if g.empty:
        fig.text(0.5, 0.5, "No dated studies for this patient", ha="center", color=MUTED)
        return fig

    panels = [("PanAdipo score", "panadipo_score", "score_threshold_age_sex", "age-sex threshold", True),
              ("Estimated EAT / heart volume", "eat_index", "high_eat_threshold", "high-EAT cutoff", False)]
    for ax, (title, col, thr_col, thr_lab, log) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        if col == "eat_index" and {"eat_index_pi80_lo", "eat_index_pi80_hi"} <= set(g):
            ax.vlines(g.study_date, g.eat_index_pi80_lo, g.eat_index_pi80_hi, color=ACCENT, alpha=0.35, lw=6,
                      zorder=1, label="80% prediction interval")
        if thr_col in g and g[thr_col].notna().any():
            ax.step(g.study_date, g[thr_col], where="post", color=INK2, lw=1, ls=(0, (3, 2)), zorder=2,
                    label=thr_lab)
        ax.plot(g.study_date, g[col], color=ACCENT, lw=2, marker="o", ms=8, mec=SURFACE, mew=2, zorder=3)
        occ = g[g.get("occult_adiposity", pd.Series(False, index=g.index)).fillna(False).astype(bool)]
        if len(occ):
            ax.plot(occ.study_date, occ[col], ls="none", marker="o", ms=14, mfc="none", mec=INK, mew=1.2,
                    zorder=4, label="occult (screen-positive, BMI <30)")
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 1]))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
            ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.spines["bottom"].set_color(MUTED)
        ax.tick_params(colors=INK2, labelsize=8, length=3)
        span = (g.study_date.max() - g.study_date.min()).days / 365.25
        ax.xaxis.set_major_locator(mdates.YearLocator(base=max(1, int(np.ceil(span / 6)))))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        pad = pd.Timedelta(days=max(60, int(span * 365.25 * 0.06)))
        ax.set_xlim(g.study_date.min() - pad, g.study_date.max() + pad)
    handles, labels = {}, []
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lab, h)
    if handles:
        fig.legend(handles.values(), handles.keys(), loc="lower center", ncol=len(handles), fontsize=7.5,
                   frameon=False, labelcolor=INK2, bbox_to_anchor=(0.5, 0.0))
    yrs = (g.study_date.iloc[-1] - g.study_date.iloc[0]).days / 365.25
    fig.text(0.07, 0.93, f"Patient {patient_id}  ·  {len(g)} studies over {yrs:.1f} years",
             fontsize=12, color=INK, weight="bold")
    return fig


def demo_series(seed: int = 0) -> pd.DataFrame:
    """Synthetic serial studies (no real data) for the app's demo mode."""
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(["2016-04-02", "2018-09-14", "2021-02-20", "2024-07-08"])
    score = np.array([0.09, 0.14, 0.22, 0.31]) * np.exp(rng.normal(0, 0.05, 4))
    return pd.DataFrame({"study_id": [f"DEMO{i + 1}" for i in range(4)], "n_clips": [55, 62, 48, 60],
                         "panadipo_score": score, "patient_id": "DEMO-PATIENT",
                         "study_date": dates.strftime("%Y-%m-%d"), "age": [61.0, 63.4, 65.9, 69.3],
                         "sex": "female", "bmi": [27.1, 27.8, 28.4, 28.9]})
