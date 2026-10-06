"""
CT-calibrated EAT estimates from the PanAdipo score.

PanAdipo is a continuous, *uncalibrated* score (trained with a positive-class
weight). To put it on a physical scale we use the YNHHS–ORFAN paired cohort
held out from model development (n = 5,081 patients with a CCTA within one
year of the TTE; BOA segmentation, EAT = fat -190 to -30 HU inside the
pericardium) and the manuscript's log-linear fit (Figure 3A):

    EAT index  (EAT / total heart volume) = a · log10(score) + b
    EAT volume (mL)                       = a · log10(score) + b

with coefficients, residual-based 80% / 95% prediction intervals, and a binned
P(high EAT) stored in assets/eat_calibration.json (aggregate values only).

Age x sex nomogram (optional; manuscript definition): high EAT = EAT index
>= 95th percentile within age (<50, 50-64, >=65 y) x sex. For each of the six
strata the file stores that threshold, reference quantiles, and a logistic
model P(high EAT) = sigmoid(b0 + b1 * log10(score)), plus the published age-sex
PanAdipo thresholds for high EAT (Supplemental Table 6).

Occult adiposity (optional, needs BMI): PanAdipo >= the age-sex threshold in a
patient without obesity (BMI < 30 kg/m2): screen-positive for high EAT where
BMI alone would not raise suspicion. The thresholds favour sensitivity
(PPV 16-31% at 5% prevalence), so a positive screen is not a diagnosis.

These are population-level estimates for research use; they are not a
substitute for CT measurement.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CALIB = Path(__file__).resolve().parent.parent / "assets" / "eat_calibration.json"


class EATCalibration:
    def __init__(self, spec: dict):
        self.spec = spec

    @classmethod
    def load(cls, path=DEFAULT_CALIB):
        with open(path) as f:
            return cls(json.load(f))

    def formula(self, target: str) -> str:
        f = self.spec["formula"][target]
        return f"{f['a']:.3f} * log10(score) + {f['b']:.3f}"

    def score_percentile(self, score) -> np.ndarray:
        """Percentile of the score within the CT reference cohort (0-100)."""
        q = self.spec["reference_score_quantiles"]
        return np.interp(np.asarray(score, float), q, np.linspace(0, 100, len(q)))

    # ── age x sex nomogram ────────────────────────────────────────────────────
    def age_sex_group(self, age, sex):
        """Stratum dict for (age in years, sex 'male'/'female'/'M'/'F'), or None if unknown."""
        sex = normalize_sex(sex)
        if sex is None or age is None or not np.isfinite(age):
            return None
        a = self.spec["age_sex"]
        label = a["age_labels"][int(np.searchsorted(a["age_bins"], age, side="right"))]
        return next(g for g in a["groups"] if g["age_group"] == label and g["sex"] == sex)

    def _age_sex_row(self, s: float, g: dict, bmi: float) -> dict:
        lx = np.log10(np.clip(s, 1e-6, 1.0))
        lv = g["quantile_levels"]
        thr = g["paper_score_threshold"]["threshold"]
        above = bool(s >= thr)
        cut = self.spec["occult_adiposity_definition"]["bmi_cutoff"]
        has_bmi = bmi is not None and np.isfinite(bmi)
        if not above:
            occult, flag = False, "screen-negative"
        elif not has_bmi:
            occult, flag = None, "screen-positive (BMI unavailable)"
        elif bmi < cut:
            occult, flag = True, f"screen-positive · occult (BMI <{cut:g})"
        else:
            occult, flag = False, "screen-positive · with obesity"
        return {
            "age_sex_group": f"{g['sex']} {g['age_group']}",
            "high_eat_threshold": g["high_eat_threshold_eat_index"],
            "p_high_eat_age_sex": 1.0 / (1.0 + np.exp(-(g["logit_b0"] + g["logit_b1"] * lx))),
            "score_percentile_age_sex": float(np.interp(s, g["score_quantiles"], lv, left=lv[0] / 2,
                                                        right=(100 + lv[-1]) / 2)),
            "score_threshold_age_sex": thr,
            "above_score_threshold": above,
            "occult_adiposity": occult,
            "eat_flag": flag,
        }

    def predict(self, score, age=None, sex=None, bmi=None) -> pd.DataFrame:
        """EAT estimates for each score.

        With age (years) and sex, adds the age-sex nomogram: P(high EAT), the age-sex PanAdipo threshold from
        Supplemental Table 6 and whether the score is above it. With BMI as well, flags occult adiposity
        (above threshold and BMI < 30 kg/m2). Columns are present but empty when inputs are missing.
        """
        s = np.asarray(score, float)
        n = len(s)
        lx = np.log10(np.clip(s, 1e-6, 1.0))
        out = {"score_percentile": self.score_percentile(s)}
        for t, f in self.spec["formula"].items():
            mu = f["a"] * lx + f["b"]
            rq = f["resid_quantiles"]
            out[t]              = np.maximum(mu, f["floor"])
            out[f"{t}_pi80_lo"] = np.maximum(mu + rq["q10"], 0)
            out[f"{t}_pi80_hi"] = mu + rq["q90"]
            out[f"{t}_pi95_lo"] = np.maximum(mu + rq["q02.5"], 0)
            out[f"{t}_pi95_hi"] = mu + rq["q97.5"]
        hb = self.spec["high_eat_bins"]
        out["p_high_eat"] = np.interp(lx, np.log10(hb["score_median"]), hb["rate"])
        df = pd.DataFrame(out)
        df.loc[~np.isfinite(s)] = np.nan

        def vec(x, dtype):
            return np.broadcast_to(np.asarray(np.nan if x is None else x, dtype=dtype), (n,))

        rows = []
        for si, ai, xi, bi in zip(s, vec(age, float), vec(sex, object), vec(bmi, float)):
            g = self.age_sex_group(ai, xi)
            if g is None or not np.isfinite(si):
                rows.append({"eat_flag": "age/sex unavailable" if np.isfinite(si) else None})
            else:
                rows.append(self._age_sex_row(si, g, bi))
        return pd.concat([df, pd.DataFrame(rows, columns=AGE_SEX_COLS)], axis=1)


AGE_SEX_COLS = ["age_sex_group", "high_eat_threshold", "p_high_eat_age_sex", "score_percentile_age_sex",
                "score_threshold_age_sex", "above_score_threshold", "occult_adiposity", "eat_flag"]


def normalize_sex(sex):
    """'M'/'male'/'F'/'female' (any case) -> 'male'/'female'; anything else -> None."""
    if sex is None or (isinstance(sex, float) and np.isnan(sex)):
        return None
    x = str(sex).strip().lower()
    return {"m": "male", "male": "male", "f": "female", "female": "female"}.get(x)
