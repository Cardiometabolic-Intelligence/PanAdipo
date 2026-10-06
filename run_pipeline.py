"""
PanAdipo end-to-end: echo DICOM study -> PanAdipo score -> estimated EAT volume / indexed EAT.

    DICOM study folder(s)
      └─ 1. preprocess  : motion-mask, crop, 256x256 .avi per cine clip   (preprocess_dicom.py)
      └─ 2. infer       : PanAdipo probability per clip (16 frames, 224x224) (infer.py)
      └─ 3. aggregate   : study score = mean over all clips
      └─ 4. EAT         : CT-calibrated EAT volume (mL) and EAT / heart volume, with 80/95% PIs
                          + age-sex nomogram and occult-adiposity flag (age / sex / BMI from the
                            DICOM header, or --age/--sex/--bmi/--demographics)
      └─ 5. report      : reports/<study_id>.png

Usage:
    python run_pipeline.py --study_dir /data/STUDY001 --out_dir results/
    python run_pipeline.py --studies_root /data/echo_dicoms --out_dir results/ --n_workers 16
    python run_pipeline.py --study_dir /data/STUDY001 --out_dir results/ --age 58 --sex F --bmi 27
"""

import argparse
from pathlib import Path

import pandas as pd

from infer import add_demographic_args, print_studies, run_inference
from panadipo.dicom import read_study_info
from panadipo.eat import EATCalibration
from panadipo.model import DEFAULT_CKPT
from panadipo.report import study_report
from preprocess_dicom import add_input_args, build_jobs, run_preprocess


def read_headers(jobs: pd.DataFrame) -> pd.DataFrame:
    """patient_id / study_date / age / sex / bmi per study from the DICOM headers."""
    info = pd.DataFrame([{"study_id": sid, **read_study_info(g.dicom_path)} for sid, g in jobs.groupby("study_id")])
    found = {c: int(info[c].notna().sum()) for c in ("age", "sex", "bmi", "patient_id", "study_date")}
    print(f"  DICOM header ({len(info)} studies): " + ", ".join(f"{k} {v}" for k, v in found.items()))
    return info


def run_studies(jobs, out_dir, n_workers=4, checkpoint=DEFAULT_CKPT, device=None, batch_size=32,
                csv_info=None, age=None, sex=None, bmi=None, dicom_info=None, report=True):
    """DICOM jobs (study_id, dicom_path) -> study table (+ reports). Used by the CLI and app.py."""
    out = Path(out_dir)
    print("── 1/3  DICOM -> AVI")
    videos = run_preprocess(jobs, out, n_workers=n_workers)
    if videos.empty:
        raise SystemExit("No cine clips could be extracted (only stills / unreadable DICOMs).")
    if dicom_info is None:
        dicom_info = read_headers(jobs)

    print("── 2/3  PanAdipo inference")
    _, studies = run_inference(videos, out, checkpoint, device, batch_size, num_workers=min(n_workers, 8),
                               dicom_info=dicom_info, csv_info=csv_info, age=age, sex=sex, bmi=bmi)

    print("── 3/3  Reports")
    if report:
        calib = EATCalibration.load()
        for _, row in studies.iterrows():
            study_report(row, out / "reports" / f"{row['study_id']}.png", calib)
        print(f"  {len(studies)} report(s) -> {out / 'reports'}")
    return studies


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_workers", type=int, default=4, help="DICOM decoding processes")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--device", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--no_report", action="store_true")
    add_demographic_args(p)
    p.add_argument("--no_dicom_header", action="store_true",
                   help="do not read patient ID / date / age / sex / BMI from the DICOM header")
    args = p.parse_args()

    jobs = build_jobs(args)
    if jobs.empty:
        raise SystemExit("No DICOM files found.")
    csv_info = pd.read_csv(args.demographics, dtype={"study_id": str, "patient_id": str}) \
        if args.demographics else None
    dicom_info = pd.DataFrame({"study_id": jobs.study_id.unique()}) if args.no_dicom_header else None
    studies = run_studies(jobs, args.out_dir, args.n_workers, args.checkpoint, args.device, args.batch_size,
                          csv_info, args.age, args.sex, args.bmi, dicom_info, report=not args.no_report)
    print()
    print_studies(studies)
    print(f"\n-> {Path(args.out_dir) / 'study_predictions.csv'}")


if __name__ == "__main__":
    main()
