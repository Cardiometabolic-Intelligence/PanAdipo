"""
Step 2 — PanAdipo inference: clip scores -> per-study score (+ CT-calibrated EAT estimate).

Inputs (pick one):
  --videos_csv CSV       e.g. videos.csv from preprocess_dicom.py (needs `fpath`; `study_id` optional)
  --video_dir  DIR       every *.avi in DIR
  --videos     A.avi ... explicit list

The study_id is taken from the CSV, or else parsed from the filename
`<study_id>_<clip_id>.avi` (the preprocess_dicom.py naming convention).

Outputs in --out_dir:
  clip_predictions.csv    fpath, study_id, panadipo_prob
  study_predictions.csv   study_id, n_clips, panadipo_score (mean of clip probabilities),
                          reference percentile, estimated CT EAT volume / EAT index (+ 80% / 95% PIs),
                          and — if age and sex are given — the age-sex nomogram columns

Age / sex (optional): --age/--sex (applied to every study) or --demographics CSV
(study_id, age, sex). run_pipeline.py also reads them from the DICOM header.

Usage:
    python infer.py --videos_csv work/videos.csv --out_dir work/
    python infer.py --videos clip1.avi clip2.avi --out_dir out/ --age 58 --sex F
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from panadipo.eat import EATCalibration, normalize_sex
from panadipo.model import DEFAULT_CKPT, load_model
from panadipo.video import ClipDataset

PROB_COL = "panadipo_prob"


def collect_clips(args) -> pd.DataFrame:
    if args.videos_csv:
        df = pd.read_csv(args.videos_csv)
    elif args.video_dir:
        df = pd.DataFrame({"fpath": sorted(map(str, Path(args.video_dir).glob("*.avi")))})
    else:
        df = pd.DataFrame({"fpath": args.videos})
    if df.empty:
        raise SystemExit("No clips found.")
    if "study_id" not in df.columns:
        df["study_id"] = [Path(f).stem.rsplit("_", 1)[0] for f in df.fpath]
    return df[["fpath", "study_id"]].astype(str).reset_index(drop=True)


@torch.no_grad()
def predict_clips(model, fpaths, clip_len, device, batch_size=32, num_workers=4) -> np.ndarray:
    loader = torch.utils.data.DataLoader(ClipDataset(fpaths, clip_len), batch_size=batch_size,
                                         shuffle=False, num_workers=num_workers,
                                         pin_memory=device.type == "cuda")
    probs = np.full(len(fpaths), np.nan)
    for x, idx, ok in tqdm(loader, desc="PanAdipo", unit="batch"):
        p = torch.sigmoid(model(x.to(device))).float().cpu().numpy()
        probs[idx.numpy()] = np.where(ok.numpy(), p, np.nan)
    return probs


def aggregate_studies(clips: pd.DataFrame) -> pd.DataFrame:
    """Per-study PanAdipo score = mean probability over all readable clips (manuscript definition)."""
    return (clips.dropna(subset=[PROB_COL]).groupby("study_id", as_index=False)
            .agg(n_clips=(PROB_COL, "size"), panadipo_score=(PROB_COL, "mean")))


STUDY_COLS = ["study_id", "n_clips", "panadipo_score", "score_percentile", "eat_volume_ml",
              "eat_index", "p_high_eat", "age", "sex", "age_sex_group", "high_eat_threshold",
              "p_high_eat_age_sex", "score_percentile_age_sex"]


def add_demographics(studies: pd.DataFrame, demographics=None, age=None, sex=None) -> pd.DataFrame:
    """Attach age / sex columns from a (study_id, age, sex) table and/or scalar overrides."""
    studies = studies.copy()
    studies["age"], studies["sex"] = np.nan, None
    if demographics is not None and len(demographics):
        dem = demographics.copy()
        dem["study_id"] = dem["study_id"].astype(str)
        dem = dem.set_index("study_id")
        studies["age"] = studies.study_id.map(pd.to_numeric(dem.get("age"), errors="coerce")) \
            if "age" in dem else np.nan
        studies["sex"] = studies.study_id.map(dem["sex"]) if "sex" in dem else None
    if age is not None:
        studies["age"] = float(age)
    if sex is not None:
        studies["sex"] = sex
    studies["sex"] = [normalize_sex(x) for x in studies["sex"]]
    return studies


def run_inference(clips: pd.DataFrame, out_dir, checkpoint=DEFAULT_CKPT, device=None,
                  batch_size=32, num_workers=4, eat=True, demographics=None, age=None, sex=None):
    """Score clips (fpath, study_id) -> (clip table, study table); both written to out_dir."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(clips):,} clips from {clips.study_id.nunique():,} studies | device={device}")

    model, meta = load_model(checkpoint, device)
    clips = clips.copy()
    clips[PROB_COL] = predict_clips(model, clips.fpath.tolist(), meta["clip_len"], device,
                                    batch_size, num_workers)
    n_bad = int(clips[PROB_COL].isna().sum())
    if n_bad:
        print(f"  {n_bad} clip(s) could not be decoded (panadipo_prob = NaN)")
    clips.to_csv(out_dir / "clip_predictions.csv", index=False)

    studies = add_demographics(aggregate_studies(clips), demographics, age, sex)
    if eat:
        est = EATCalibration.load().predict(studies.panadipo_score, age=studies.age.values,
                                            sex=studies.sex.values)
        studies = pd.concat([studies, est], axis=1)
        n_as = int(studies["age_sex_group"].notna().sum())
        print(f"  age-sex nomogram applied to {n_as}/{len(studies)} studies"
              + ("" if n_as == len(studies) else " (others: age or sex unavailable)"))
    studies.to_csv(out_dir / "study_predictions.csv", index=False)
    return clips, studies


def print_studies(studies: pd.DataFrame, n=20):
    cols = [c for c in STUDY_COLS if c in studies.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(studies[cols].head(n).to_string(index=False))


def add_demographic_args(p):
    g = p.add_argument_group("age / sex (optional; enables the age-sex nomogram)")
    g.add_argument("--age", type=float, help="age in years (applied to all studies)")
    g.add_argument("--sex", help="M / F (applied to all studies)")
    g.add_argument("--demographics", help="CSV with study_id, age, sex")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos_csv")
    src.add_argument("--video_dir")
    src.add_argument("--videos", nargs="+")
    p.add_argument("--out_dir", default="panadipo_out")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: auto)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no_eat", action="store_true", help="skip the CT-calibrated EAT estimate")
    add_demographic_args(p)
    args = p.parse_args()

    dem = pd.read_csv(args.demographics) if args.demographics else None
    _, studies = run_inference(collect_clips(args), args.out_dir, args.checkpoint, args.device,
                               args.batch_size, args.num_workers, eat=not args.no_eat,
                               demographics=dem, age=args.age, sex=args.sex)
    print_studies(studies)
    print(f"\n-> {Path(args.out_dir) / 'clip_predictions.csv'}\n-> {Path(args.out_dir) / 'study_predictions.csv'}")


if __name__ == "__main__":
    main()
