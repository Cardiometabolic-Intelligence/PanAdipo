"""
Step 1 — DICOM -> 256x256 .avi clips (PanAdipo preprocessing).

Inputs (pick one):
  --study_dir    DIR [DIR ...]   one folder per echocardiogram (study_id = folder name)
  --studies_root DIR             a folder whose sub-folders are studies
  --manifest     CSV             columns: study_id, local_path (one row per DICOM)

Outputs in --out_dir:
  videos/<study_id>_<dicom_stem>.avi
  videos.csv                     fpath, study_id, video_id   (input to infer.py)
  preprocess_log.csv             one row per DICOM with status (video / still / *_failed)

Existing AVIs are skipped unless --overwrite is given. Input DICOMs are only read.

Usage:
    python preprocess_dicom.py --studies_root /data/echo_dicoms --out_dir work/ --n_workers 8
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from panadipo.dicom import dicom_to_avi, find_dicoms


def build_jobs(args) -> pd.DataFrame:
    if args.manifest:
        df = pd.read_csv(args.manifest, usecols=["study_id", "local_path"]).dropna()
        return df.rename(columns={"local_path": "dicom_path"}).astype(str)
    study_dirs = ([Path(p) for p in args.study_dir] if args.study_dir
                  else sorted(p for p in Path(args.studies_root).iterdir() if p.is_dir()))
    rows = [{"study_id": d.name, "dicom_path": str(f)} for d in study_dirs for f in find_dicoms(d)]
    return pd.DataFrame(rows, columns=["study_id", "dicom_path"])


def _run(job):
    study_id, dicom_path, out_path, img_res, buffer, overwrite = job
    if Path(out_path).exists() and not overwrite:
        return "exists"
    return dicom_to_avi(dicom_path, out_path, img_res=img_res, buffer=buffer)


def run_preprocess(jobs: pd.DataFrame, out_dir, n_workers=4, img_res=256, buffer=32,
                   overwrite=False) -> pd.DataFrame:
    """Convert DICOMs listed in jobs (study_id, dicom_path) -> AVIs. Returns videos table."""
    out_dir = Path(out_dir)
    (out_dir / "videos").mkdir(parents=True, exist_ok=True)
    jobs = jobs.copy()
    jobs["fpath"] = [str((out_dir / "videos" / f"{s}_{Path(d).stem}.avi").resolve())
                     for s, d in zip(jobs.study_id, jobs.dicom_path)]
    print(f"{len(jobs):,} DICOMs from {jobs.study_id.nunique():,} studies -> {out_dir}")

    payload = [(s, d, f, img_res, buffer, overwrite)
               for s, d, f in zip(jobs.study_id, jobs.dicom_path, jobs.fpath)]
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        jobs["status"] = list(tqdm(ex.map(_run, payload, chunksize=1), total=len(payload),
                                   desc="DICOM -> AVI"))

    jobs.to_csv(out_dir / "preprocess_log.csv", index=False)
    videos = jobs[jobs.status.isin(["video", "exists"])].copy()
    videos["video_id"] = [Path(d).stem for d in videos.dicom_path]
    videos = videos[["fpath", "study_id", "video_id"]].reset_index(drop=True)
    videos.to_csv(out_dir / "videos.csv", index=False)
    print("  " + ", ".join(f"{k}: {v}" for k, v in jobs.status.value_counts().items()))
    return videos


def add_input_args(p):
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--study_dir", nargs="+", help="one folder per echocardiogram")
    src.add_argument("--studies_root", help="folder whose sub-folders are studies")
    src.add_argument("--manifest", help="CSV with study_id, local_path")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_input_args(p)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--img_res", type=int, default=256)
    p.add_argument("--buffer", type=int, default=32)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    jobs = build_jobs(args)
    if jobs.empty:
        raise SystemExit("No DICOM files found.")
    videos = run_preprocess(jobs, args.out_dir, args.n_workers, args.img_res, args.buffer, args.overwrite)
    print(f"videos.csv -> {Path(args.out_dir) / 'videos.csv'}  ({len(videos):,} clips)")


if __name__ == "__main__":
    main()
