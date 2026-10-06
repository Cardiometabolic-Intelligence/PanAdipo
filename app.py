"""
PanAdipo local app — a browser interface to the full pipeline, running on your own machine.

    python app.py                      # opens http://127.0.0.1:7860
    python app.py --port 8080
    python app.py --demo               # start with a synthetic demo patient loaded (no data needed)

Tabs
  1 Load & run        folder (or zip) of echo DICOM studies -> review / edit patient ID, date, age, sex,
                      BMI -> run PanAdipo -> scores, EAT estimates, age-sex nomogram, occult-adiposity
                      flags, per-study reports, CSV download
  2 Track over time   serial studies per patient (also merges earlier study_predictions.csv files)
  3 Nomogram          score + age + sex (+ BMI) -> EAT estimates and nomogram, no images needed
  4 About

Everything runs locally; nothing is uploaded. The server binds to 127.0.0.1 (this computer only)
unless you pass --host. Research use only.
"""

import argparse
import tempfile
import zipfile
from pathlib import Path

import gradio as gr
import pandas as pd

from infer import INFO_COLS, finalize
from panadipo.dicom import find_dicoms, read_study_info
from panadipo.eat import EATCalibration
from panadipo.longitudinal import demo_series, plot_trajectory, trajectories
from panadipo.model import DEFAULT_CKPT
from panadipo.report import study_report
from run_pipeline import run_studies

CALIB = EATCalibration.load()
CHECKPOINT = DEFAULT_CKPT
EDIT_COLS = ["study_id"] + INFO_COLS + ["n_dicoms"]
SHOW_COLS = ["study_id", "patient_id", "study_date", "n_clips", "panadipo_score", "eat_volume_ml", "eat_index",
             "age", "sex", "bmi", "p_high_eat_age_sex", "eat_flag"]
FMT = {"panadipo_score": 3, "eat_volume_ml": 0, "eat_index": 3, "age": 1, "bmi": 1, "p_high_eat_age_sex": 3}


def _rounded(df, cols):
    d = df[[c for c in cols if c in df]].copy()
    for c, k in FMT.items():
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce").round(k)
    return d


# ── 1 Load & run ─────────────────────────────────────────────────────────────
def scan(folder, zip_file, single_study):
    """Find studies, read their DICOM headers, return an editable table + the job list."""
    if zip_file is not None:
        root = Path(tempfile.mkdtemp(prefix="panadipo_upload_"))
        with zipfile.ZipFile(zip_file) as z:
            z.extractall(root)
        subdirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("__MACOSX")]
        if len(subdirs) == 1 and not any(f.parent == root for f in find_dicoms(root)):
            root = subdirs[0]          # zip wrapped everything in one top-level folder
    elif folder and Path(folder).expanduser().is_dir():
        root = Path(folder).expanduser()
    else:
        raise gr.Error("Give a folder path or upload a .zip of DICOM studies.")

    study_dirs = [root] if single_study else sorted(p for p in root.iterdir() if p.is_dir())
    jobs, rows = [], []
    for d in study_dirs:
        files = find_dicoms(d)
        if not files:
            continue
        jobs += [{"study_id": d.name, "dicom_path": str(f)} for f in files]
        rows.append({"study_id": d.name, **read_study_info(files), "n_dicoms": len(files)})
    if not rows:
        raise gr.Error(f"No DICOM files found under {root} (tick 'single study' if the folder is one study).")
    table = pd.DataFrame(rows, columns=EDIT_COLS)
    table["age"] = pd.to_numeric(table["age"], errors="coerce").round(1)
    table["bmi"] = pd.to_numeric(table["bmi"], errors="coerce").round(1)
    n_miss = int((table.age.isna() | table.sex.isna()).sum())
    msg = (f"Found **{len(table)} studies** ({len(jobs):,} DICOM files) in `{root}`. Values were read from the "
           f"DICOM headers — edit any cell, then press **Run PanAdipo**.")
    if n_miss:
        msg += f"\n\n⚠️ {n_miss} study/studies have no age or sex: fill them in to get the age-sex nomogram."
    return table, pd.DataFrame(jobs), msg


def run(table, jobs, out_dir, n_workers, device, progress=gr.Progress(track_tqdm=True)):
    if jobs is None or len(jobs) == 0:
        raise gr.Error("Scan a folder first.")
    table = pd.DataFrame(table)
    keep = set(table.study_id.astype(str))
    jobs = jobs[jobs.study_id.astype(str).isin(keep)]
    info = table[["study_id"] + INFO_COLS].replace("", None)
    out = Path(out_dir).expanduser()
    studies = run_studies(jobs, out, n_workers=int(n_workers), checkpoint=CHECKPOINT,
                          device=device or None, csv_info=info, dicom_info=info.iloc[0:0])
    reports = [str(out / "reports" / f"{s}.png") for s in studies.study_id]
    return (studies, _rounded(studies, SHOW_COLS), summary(studies), reports,
            str(out / "study_predictions.csv"), *refresh_patients(studies))


def summary(st: pd.DataFrame) -> str:
    n = len(st)
    pos = st.eat_flag.fillna("").str.startswith("screen-positive")
    occ = st.occult_adiposity.fillna(False).astype(bool) if "occult_adiposity" in st else pos & False
    no_as = st.age_sex_group.isna().sum()
    no_bmi = (pos & st.bmi.isna()).sum()
    lines = [f"### {n} studies scored",
             f"- **Screen-positive for high EAT** (score ≥ age-sex threshold): **{int(pos.sum())}**",
             f"- **Occult adiposity** (screen-positive and BMI < 30): **{int(occ.sum())}**"]
    if occ.any():
        lines.append("  - " + ", ".join(st.loc[occ, "study_id"].astype(str)))
    if no_as:
        lines.append(f"- {int(no_as)} without age/sex → pooled estimates only (no threshold / nomogram)")
    if no_bmi:
        lines.append(f"- {int(no_bmi)} screen-positive without BMI → occult status unknown")
    lines.append("\n_Thresholds favour sensitivity (PPV 16–31% at 5% prevalence): a positive screen is not a "
                 "diagnosis. Research use only._")
    return "\n".join(lines)


# ── 2 Track over time ────────────────────────────────────────────────────────
def refresh_patients(st):
    if st is None or len(st) == 0 or "patient_id" not in st:
        return gr.update(choices=[], value=None), pd.DataFrame()
    tr = trajectories(st)
    serial = list(tr.patient_id) if len(tr) else []
    others = sorted(set(st.patient_id.dropna().astype(str)) - set(serial))
    choices = serial + others
    return gr.update(choices=choices, value=choices[0] if choices else None), _round_traj(tr)


def _round_traj(tr):
    return tr.round({"score_first": 3, "score_last": 3, "eat_index_change": 3, "eat_ml_change": 1,
                     "eat_ml_change_per_year": 1}) if len(tr) else tr


def merge_files(st, files):
    """Add earlier study_predictions.csv files (re-estimated with the current calibration)."""
    frames = [] if st is None else [st]
    for f in files or []:
        frames.append(pd.read_csv(f, dtype={"study_id": str, "patient_id": str}))
    if not frames:
        raise gr.Error("Upload one or more study_predictions.csv files.")
    allst = pd.concat(frames, ignore_index=True).drop_duplicates("study_id", keep="last")
    allst = finalize(allst[["study_id", "n_clips", "panadipo_score"] + [c for c in INFO_COLS if c in allst]],
                     tempfile.mkdtemp(prefix="panadipo_merge_"))
    return (allst, _rounded(allst, SHOW_COLS), summary(allst), *refresh_patients(allst))


def show_patient(st, pid):
    if st is None or not pid:
        return None
    return plot_trajectory(st, pid)


def load_demo():
    st = finalize(demo_series(), tempfile.mkdtemp(prefix="panadipo_demo_"))
    return (st, _rounded(st, SHOW_COLS), summary(st), *refresh_patients(st))


# ── 3 Nomogram calculator ────────────────────────────────────────────────────
def calculate(score, age, sex, bmi):
    if not score or score <= 0 or score > 1:
        raise gr.Error("PanAdipo score must be in (0, 1].")
    st = finalize(pd.DataFrame({"study_id": ["calculator"], "n_clips": [0], "panadipo_score": [float(score)],
                                "age": [age], "sex": [sex], "bmi": [bmi]}),
                  tempfile.mkdtemp(prefix="panadipo_calc_"))
    r = st.iloc[0]
    png = study_report(r, Path(tempfile.mkdtemp()) / "calculator.png", CALIB)
    md = [f"**EAT / heart volume:** {r.eat_index:.2f} (80% PI {r.eat_index_pi80_lo:.2f}–{r.eat_index_pi80_hi:.2f})",
          f"**EAT volume:** {r.eat_volume_ml:.0f} mL (80% PI {r.eat_volume_ml_pi80_lo:.0f}–{r.eat_volume_ml_pi80_hi:.0f})"]
    if pd.notna(r.get("age_sex_group")):
        md += [f"**Age-sex group:** {r.age_sex_group} · score percentile {r.score_percentile_age_sex:.0f}",
               f"**P(high EAT):** {r.p_high_eat_age_sex:.0%} · threshold {r.score_threshold_age_sex:.3f} → "
               f"**{r.eat_flag}**"]
    else:
        md.append("Add age and sex for the age-sex nomogram.")
    return "\n\n".join(md), str(png)


# ── UI ───────────────────────────────────────────────────────────────────────
ABOUT = f"""
**PanAdipo** scores prominent epicardial adiposity from routine echocardiograms (any 2D view, TTE or POCUS),
and maps the score to CT-measured epicardial adipose tissue (EAT) using {CALIB.spec['n_patients']:,} patients
with a CCTA within one year of the echo.

- **EAT / heart volume** = {CALIB.formula('eat_index')} · **EAT (mL)** = {CALIB.formula('eat_volume_ml')}
- **High EAT**: EAT / heart volume ≥ 95th percentile within age group (<50, 50–64, ≥65 y) and sex
- **Age-sex threshold**: PanAdipo score cut-offs for high EAT (Supplemental Table 6; AUROC 0.91–0.95)
- **Occult adiposity**: screen-positive (score ≥ threshold) in a patient without obesity (BMI < 30)

Estimates are population-level and carry wide individual prediction intervals; they are not a CT measurement.
Research use only — not a medical device.

Cite: Aminorroaya A, …, Oikonomou EK\\*. AI-Enabled Echocardiography Identifies Prominent Epicardial Adiposity
Associated with an Adverse Cardiometabolic Profile. *Under review*, 2026. Contact: evangelos.oikonomou@yale.edu
"""


def build(demo=False):
    with gr.Blocks(title="PanAdipo", theme=gr.themes.Soft(primary_hue="blue")) as ui:
        studies = gr.State(None)
        jobs = gr.State(None)
        gr.Markdown("# PanAdipo\nEcho DICOM studies → PanAdipo score → CT-calibrated EAT, age-sex nomogram, "
                    "occult adiposity, tracking over time. _Runs locally; nothing leaves this computer._")

        with gr.Tab("1 · Load & run"):
            with gr.Row():
                folder = gr.Textbox(label="Folder of studies (one sub-folder per echo study)",
                                    placeholder="/path/to/echo_dicoms", scale=3)
                zip_up = gr.File(label="…or upload a .zip", file_types=[".zip"], scale=1)
            with gr.Row():
                single = gr.Checkbox(label="This folder is a single study", value=False)
                out_dir = gr.Textbox(label="Output folder", value="panadipo_results")
                n_workers = gr.Slider(1, 32, value=4, step=1, label="DICOM decoding workers")
                device = gr.Dropdown(["", "cuda", "cpu"], value="", label="Device (blank = auto)")
            scan_btn = gr.Button("Scan studies", variant="secondary")
            scan_msg = gr.Markdown()
            table = gr.Dataframe(headers=EDIT_COLS, interactive=True, wrap=True,
                                 label="Studies (edit patient ID, date, age, sex M/F, BMI)")
            run_btn = gr.Button("Run PanAdipo", variant="primary")
            summary_md = gr.Markdown()
            results = gr.Dataframe(label="Results", wrap=True)
            gallery = gr.Gallery(label="Reports", columns=1, height="auto")
            csv_out = gr.File(label="study_predictions.csv")

        with gr.Tab("2 · Track over time"):
            gr.Markdown("Patients with ≥ 2 dated studies are listed first. Add earlier `study_predictions.csv` "
                        "files to follow patients across runs.")
            with gr.Row():
                prior = gr.File(label="Earlier study_predictions.csv", file_count="multiple", file_types=[".csv"])
                merge_btn = gr.Button("Add to tracker")
                demo_btn = gr.Button("Load synthetic demo patient")
            patient = gr.Dropdown(label="Patient", choices=[])
            traj_plot = gr.Plot()
            traj_table = gr.Dataframe(label="Change from first to last study", wrap=True)

        with gr.Tab("3 · Nomogram calculator"):
            with gr.Row():
                c_score = gr.Number(label="PanAdipo score", value=0.25, minimum=0.0005, maximum=1)
                c_age = gr.Number(label="Age (years)", value=58)
                c_sex = gr.Radio(["F", "M"], label="Sex", value="F")
                c_bmi = gr.Number(label="BMI (kg/m², optional)", value=None)
            calc_btn = gr.Button("Calculate", variant="primary")
            calc_md = gr.Markdown()
            calc_img = gr.Image(label="Report", type="filepath")

        with gr.Tab("4 · About"):
            gr.Markdown(ABOUT)
            gr.Image("docs/figures/nomogram.png", show_label=False)

        scan_btn.click(scan, [folder, zip_up, single], [table, jobs, scan_msg])
        run_btn.click(run, [table, jobs, out_dir, n_workers, device],
                      [studies, results, summary_md, gallery, csv_out, patient, traj_table])
        merge_btn.click(merge_files, [studies, prior], [studies, results, summary_md, patient, traj_table])
        demo_btn.click(load_demo, None, [studies, results, summary_md, patient, traj_table])
        patient.change(show_patient, [studies, patient], traj_plot)
        calc_btn.click(calculate, [c_score, c_age, c_sex, c_bmi], [calc_md, calc_img])
        if demo:
            ui.load(load_demo, None, [studies, results, summary_md, patient, traj_table])
    return ui


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only")
    p.add_argument("--port", type=int, default=7860)
    p.add_argument("--demo", action="store_true", help="start with a synthetic demo patient")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    args = p.parse_args()
    global CHECKPOINT
    CHECKPOINT = args.checkpoint
    build(args.demo).queue().launch(server_name=args.host, server_port=args.port, share=False)


if __name__ == "__main__":
    main()
