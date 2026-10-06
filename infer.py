"""
Step 2 — PanAdipo inference: clip scores -> per-study score -> EAT estimates and age-sex nomogram.

Inputs (pick one):
  --videos_csv CSV          e.g. videos.csv from preprocess_dicom.py (needs `fpath`; `study_id` optional)
  --video_dir  DIR          every *.avi in DIR
  --videos     A.avi ...    explicit list
  --study_predictions CSV   an existing study_predictions.csv: re-apply the EAT estimates / nomogram with
                            new age, sex or BMI (e.g. a filled-in missing_demographics.csv) without
                            re-running the model

The study_id is taken from the CSV, or else parsed from the filename `<study_id>_<clip_id>.avi`.

Study information (all optional; command line > --demographics CSV > DICOM header > existing columns):
  --age / --sex / --bmi     applied to every study
  --demographics CSV        study_id plus any of patient_id, study_date, age, sex, bmi
  age + sex enable the age-sex nomogram; BMI additionally enables the occult-adiposity flag;
  patient_id + study_date enable tracking a patient across studies (app.py / docs viewer).

Outputs in --out_dir:
  clip_predictions.csv       fpath, study_id, panadipo_prob
  study_predictions.csv      one row per study (see README for columns)
  missing_demographics.csv   written when any study lacks age, sex or BMI: fill in and pass back with
                             --study_predictions results/study_predictions.csv --demographics <file>

Usage:
    python infer.py --videos_csv work/videos.csv --out_dir work/
    python infer.py --videos clip1.avi clip2.avi --out_dir out/ --age 58 --sex F --bmi 27
    python infer.py --study_predictions work/study_predictions.csv --demographics filled.csv --out_dir work/
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from panadipo.eat import AGE_SEX_COLS, EATCalibration, normalize_sex
from panadipo.model import DEFAULT_CKPT, load_model
from panadipo.video import ClipDataset

PROB_COL = "panadipo_prob"
INFO_COLS = ["patient_id", "study_date", "age", "sex", "bmi"]
STUDY_COLS = ["study_id", "n_clips", "panadipo_score", "score_percentile", "eat_volume_ml", "eat_index",
              "age", "sex", "bmi", "age_sex_group", "p_high_eat_age_sex", "above_score_threshold",
              "eat_flag"]


# ── clips -> study score ─────────────────────────────────────────────────────
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


# ── study information (age / sex / BMI / patient / date) ─────────────────────
def _table(df):
    if df is None or len(df) == 0:
        return None
    df = df.copy()
    df["study_id"] = df["study_id"].astype(str)
    return df.drop_duplicates("study_id").set_index("study_id")


def merge_study_info(studies, dicom_info=None, csv_info=None, age=None, sex=None, bmi=None) -> pd.DataFrame:
    """Fill INFO_COLS per study with priority command line > CSV > DICOM header, recording the source."""
    given = [c for c in INFO_COLS if c in studies]
    st = studies.drop(columns=given + [c for c in ("age_sex_source", "bmi_source") if c in studies]).copy()
    st["study_id"] = st["study_id"].astype(str)
    inline = studies[["study_id"] + given].assign(study_id=st["study_id"]) if given else None
    sources = [("input", _table(inline)), ("dicom", _table(dicom_info)), ("csv", _table(csv_info))]
    cli = {"age": age, "sex": sex, "bmi": bmi}
    src = {}
    for c in INFO_COLS:
        vals = pd.Series([None] * len(st), index=st.index, dtype=object)
        s = pd.Series("missing", index=st.index, dtype=object)
        for name, tab in sources:
            if tab is not None and c in tab:
                v = st.study_id.map(tab[c])
                ok = v.notna() & (v.astype(str).str.strip() != "")
                vals[ok], s[ok] = v[ok], name
        if cli.get(c) is not None:
            vals[:], s[:] = cli[c], "command line"
        st[c], src[c] = vals, s
    st["age"] = pd.to_numeric(st["age"], errors="coerce")
    st["bmi"] = pd.to_numeric(st["bmi"], errors="coerce")
    st["sex"] = [normalize_sex(x) for x in st["sex"]]
    ok_as = st["age"].notna() & st["sex"].notna()
    st["age_sex_source"] = [(a if a == x else f"age: {a}, sex: {x}") if ok else "missing"
                            for ok, a, x in zip(ok_as, src["age"], src["sex"])]
    st["bmi_source"] = [b if ok else "missing" for ok, b in zip(st["bmi"].notna(), src["bmi"])]
    return st


def add_estimates(studies: pd.DataFrame) -> pd.DataFrame:
    calib = EATCalibration.load()
    est_cols = set(calib.predict([0.1]).columns) | set(AGE_SEX_COLS)
    st = studies[[c for c in studies.columns if c not in est_cols]].reset_index(drop=True)
    est = calib.predict(st.panadipo_score, age=st.age.values, sex=st.sex.values, bmi=st.bmi.values)
    return pd.concat([st, est], axis=1)


def report_missing(studies: pd.DataFrame, out_dir: Path) -> None:
    """Warn about studies without age/sex (no nomogram) or BMI (no occult flag); write a fill-in template."""
    no_as = studies.index[studies.age.isna() | studies.sex.isna()]
    no_bmi = studies.index[studies.bmi.isna()]
    n, either = len(studies), no_as.union(no_bmi)
    print(f"  age-sex nomogram: {n - len(no_as)}/{n} studies | occult-adiposity flag (needs BMI): "
          f"{n - len(either)}/{n} studies")
    path = out_dir / "missing_demographics.csv"
    if len(either) == 0:
        path.unlink(missing_ok=True)
        return
    studies.loc[either, ["study_id"] + INFO_COLS].to_csv(path, index=False)

    def ids(idx):
        s = studies.loc[idx, "study_id"].tolist()
        return ", ".join(s[:10]) + (f" ... (+{len(s) - 10})" if len(s) > 10 else "")

    print("\n  ! Missing study information")
    if len(no_as):
        print(f"    no age/sex -> pooled estimates only, no nomogram: {ids(no_as)}")
    if len(no_bmi):
        print(f"    no BMI     -> no occult-adiposity flag:           {ids(no_bmi)}")
    print(f"    Fill in {path}, then re-apply (no re-inference needed):\n"
          f"      python infer.py --study_predictions {out_dir / 'study_predictions.csv'} "
          f"--demographics {path} --out_dir {out_dir}\n")


# ── main entry points ────────────────────────────────────────────────────────
def finalize(studies, out_dir, dicom_info=None, csv_info=None, age=None, sex=None, bmi=None):
    """Attach study info + EAT estimates + nomogram, report what is missing, write study_predictions.csv."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    studies = add_estimates(merge_study_info(studies, dicom_info, csv_info, age, sex, bmi))
    report_missing(studies, out_dir)
    studies.to_csv(out_dir / "study_predictions.csv", index=False)
    return studies


def run_inference(clips: pd.DataFrame, out_dir, checkpoint=DEFAULT_CKPT, device=None, batch_size=32,
                  num_workers=4, dicom_info=None, csv_info=None, age=None, sex=None, bmi=None):
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

    studies = finalize(aggregate_studies(clips), out_dir, dicom_info, csv_info, age, sex, bmi)
    return clips, studies


def print_studies(studies: pd.DataFrame, n=20):
    cols = [c for c in STUDY_COLS if c in studies.columns]
    with pd.option_context("display.width", 250, "display.max_columns", 20,
                           "display.float_format", "{:.3f}".format):
        print(studies[cols].head(n).to_string(index=False))


def add_demographic_args(p):
    g = p.add_argument_group("study information (optional)")
    g.add_argument("--age", type=float, help="age in years (applied to all studies)")
    g.add_argument("--sex", help="M / F (applied to all studies)")
    g.add_argument("--bmi", type=float, help="BMI in kg/m2 (applied to all studies)")
    g.add_argument("--demographics", help="CSV with study_id and any of patient_id, study_date, age, sex, bmi")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--videos_csv")
    src.add_argument("--video_dir")
    src.add_argument("--videos", nargs="+")
    src.add_argument("--study_predictions", help="existing study_predictions.csv (skip the model)")
    p.add_argument("--out_dir", default="panadipo_out")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--device", default=None, help="cuda / cuda:0 / cpu (default: auto)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    add_demographic_args(p)
    args = p.parse_args()

    csv_info = pd.read_csv(args.demographics, dtype={"study_id": str, "patient_id": str}) \
        if args.demographics else None
    if args.study_predictions:
        prev = pd.read_csv(args.study_predictions, dtype={"study_id": str, "patient_id": str})
        prev_info = prev[["study_id"] + [c for c in INFO_COLS if c in prev]]   # keep earlier values
        studies = finalize(prev[["study_id", "n_clips", "panadipo_score"]], args.out_dir,
                           dicom_info=prev_info, csv_info=csv_info, age=args.age, sex=args.sex, bmi=args.bmi)
    else:
        _, studies = run_inference(collect_clips(args), args.out_dir, args.checkpoint, args.device,
                                   args.batch_size, args.num_workers, csv_info=csv_info,
                                   age=args.age, sex=args.sex, bmi=args.bmi)
    print_studies(studies)
    print(f"\n-> {Path(args.out_dir) / 'study_predictions.csv'}")


if __name__ == "__main__":
    main()
