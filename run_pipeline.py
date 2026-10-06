"""
PanAdipo end-to-end: echo DICOM study -> PanAdipo score -> estimated EAT volume / indexed EAT.

    DICOM study folder(s)
      └─ 1. preprocess  : motion-mask, crop, 256x256 .avi per cine clip   (preprocess_dicom.py)
      └─ 2. infer       : PanAdipo probability per clip (16 frames, 224x224) (infer.py)
      └─ 3. aggregate   : study score = mean over all clips
      └─ 4. EAT         : CT-calibrated EAT volume (mL) and EAT / heart volume, with 80/95% PIs
                          + age-sex nomogram (age / sex from the DICOM header, or --age/--sex)
      └─ 5. report      : reports/<study_id>.png

Usage:
    python run_pipeline.py --study_dir /data/STUDY001 --out_dir results/
    python run_pipeline.py --studies_root /data/echo_dicoms --out_dir results/ --n_workers 16
    python run_pipeline.py --study_dir /data/STUDY001 --out_dir results/ --age 58 --sex F
"""

import argparse
from pathlib import Path

import pandas as pd

from infer import add_demographic_args, print_studies, run_inference
from panadipo.dicom import read_demographics
from panadipo.eat import EATCalibration
from panadipo.model import DEFAULT_CKPT
from panadipo.report import study_report
from preprocess_dicom import add_input_args, build_jobs, run_preprocess


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
    p.add_argument("--no_dicom_demographics", action="store_true",
                   help="do not read age / sex from the DICOM header")
    args = p.parse_args()
    out = Path(args.out_dir)

    print("── 1/3  DICOM -> AVI")
    jobs = build_jobs(args)
    if jobs.empty:
        raise SystemExit("No DICOM files found.")
    videos = run_preprocess(jobs, out, n_workers=args.n_workers)
    if videos.empty:
        raise SystemExit("No cine clips could be extracted (only stills / unreadable DICOMs).")

    dem = None
    if args.demographics:
        dem = pd.read_csv(args.demographics)
    elif not args.no_dicom_demographics:
        dem = pd.DataFrame([{"study_id": sid, **read_demographics(g.dicom_path)}
                            for sid, g in jobs.groupby("study_id")])
        print(f"  age / sex from DICOM header: {int(dem[['age', 'sex']].notna().all(axis=1).sum())}"
              f"/{len(dem)} studies")

    print("── 2/3  PanAdipo inference")
    _, studies = run_inference(videos, out, args.checkpoint, args.device, args.batch_size,
                               num_workers=min(args.n_workers, 8), demographics=dem,
                               age=args.age, sex=args.sex)

    print("── 3/3  Reports")
    if not args.no_report:
        calib = EATCalibration.load()
        for _, row in studies.iterrows():
            study_report(row, out / "reports" / f"{row['study_id']}.png", calib)
        print(f"  {len(studies)} report(s) -> {out / 'reports'}")

    print()
    print_studies(studies)
    print(f"\n-> {out / 'study_predictions.csv'}")


if __name__ == "__main__":
    main()
