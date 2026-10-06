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

    def predict(self, score) -> pd.DataFrame:
        s = np.asarray(score, float)
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
        return df
