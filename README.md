<h1 align="center">PanAdipo</h1>
<p align="center"><b>Epicardial adiposity from any routine echocardiogram</b><br>
Echo DICOM study ➜ PanAdipo score ➜ CT-calibrated EAT volume (mL) and EAT / heart volume</p>

<p align="center">
<img alt="python" src="https://img.shields.io/badge/python-3.10-3776AB">
<img alt="pytorch" src="https://img.shields.io/badge/PyTorch-2.1-EE4C2C">
<img alt="reproducibility" src="https://img.shields.io/badge/manuscript%20reproduction-bit--exact-1baf7a">
<img alt="use" src="https://img.shields.io/badge/use-academic%20%2F%20non--commercial-8a8984">
<img alt="license" src="https://img.shields.io/badge/code-PolyForm%20Noncommercial-2a78d6">
<img alt="weights" src="https://img.shields.io/badge/weights-CC%20BY--NC--SA%204.0-4a3aa7">
</p>

<p align="center"><img src="docs/figures/central_illustration.jpg" width="820"></p>

PanAdipo is a video model fine-tuned from the [PanEcho](https://github.com/CarDS-Yale/PanEcho) echocardiography
foundation model. It scores **prominent epicardial adiposity** from any standard 2D B-mode clip (any view,
TTE or POCUS) and averages the clip scores into a single score per study. In 5,081 patients with a CCTA
within a year of the echo, the score tracked CT-measured epicardial adipose tissue (EAT): **Spearman ρ = 0.76**
with EAT indexed to heart volume, **AUROC 0.90** for age- and sex-specific high EAT. This repository holds the
end-to-end inference code: DICOM preprocessing, the model, the CT-calibrated EAT estimate, and Grad-CAM++.

---

## Quick start: one echo study, one command

```bash
conda env create -f envs/panadipo.yml && conda activate panadipo
./copy_weights.sh                       # downloads weights/best_model.pt from the v1.0 release

python run_pipeline.py --study_dir /path/to/STUDY001 --out_dir results/
```

```
# illustrative output (synthetic study ID, score = 0.25)
── 1/3  DICOM -> AVI
92 DICOMs from 1 studies -> results
  video: 57, still: 33, pixel_failed: 2
── 2/3  PanAdipo inference
57 clips from 1 studies | device=cuda
── 3/3  Reports

  study_id    n_clips  panadipo_score  score_percentile  eat_volume_ml  eat_volume_ml_pi80_lo  eat_volume_ml_pi80_hi  eat_index  eat_index_pi80_lo  eat_index_pi80_hi  p_high_eat
  STUDY001       57           0.250            88.8        141.9                   97.7                  190.3      0.273              0.197              0.354       0.164
```

Each study also gets a one-page report, `results/reports/<study_id>.png`:

<p align="center"><img src="docs/figures/example_report.png" width="900"><br>
<sub>Example report (synthetic score = 0.25). Grey: CT reference cohort. Blue: this study, with 80% / 95% prediction intervals.</sub></p>

For many studies, point it at a folder of study folders (`--studies_root`) or at a manifest CSV
(`--manifest`, columns `study_id,local_path`). A GPU is recommended; CPU works but is slow.

---

## What you get

| Output | Meaning |
|---|---|
| `panadipo_score` | Mean PanAdipo probability over all cine clips in the study (manuscript definition). A continuous score (0–1), **not a calibrated probability**: training used a positive-class weight. |
| `score_percentile` | Percentile of the score in the CT reference cohort (n = 5,081). |
| `eat_index` (+ `_pi80_*`, `_pi95_*`) | Estimated **CT EAT volume / total heart volume**, with 80% / 95% prediction intervals. |
| `eat_volume_ml` (+ PIs) | Estimated **CT EAT volume in mL**. |
| `p_high_eat` | Observed rate of high EAT (indexed EAT ≥ 95th age-sex percentile) among reference patients with a similar score. |
| `clip_predictions.csv` | Score for every clip, if you want to filter by view yourself. |

### How EAT is estimated

<img align="right" src="docs/figures/fig3a_eat_formula.jpg" width="360">

The score is mapped to CT EAT with the log-linear fit from the manuscript (Figure 3A). The fit used YNHHS
patients held out from model development who had a CCTA within 365 days of the echo (ORFAN cohort, one pair
per patient). EAT was segmented with Body & Organ Analysis (BOA) as fat between −190 and −30 HU inside the
pericardium.

```
EAT / heart volume = 0.151 · log10(score) + 0.364      R² = 0.53
EAT volume (mL)    = 73.3  · log10(score) + 186.0      R² = 0.44
```

Heart volume is the sum of the LV myocardium, LV, RV, LA, and RA. Prediction intervals are the empirical
residual quantiles. Patient-grouped 5-fold cross-validation:

| Target | CV R² | CV MAE | 80% PI coverage | 95% PI coverage |
|---|---|---|---|---|
| EAT / heart volume | 0.53 | 0.050 | 80.0% | 94.9% |
| EAT volume (mL) | 0.44 | 29 mL | 80.0% | 95.0% |

Only the coefficients and aggregate summaries ship, in [`assets/eat_calibration.json`](assets/eat_calibration.json).
No patient-level data is included. These are **population-level estimates for research**. They are not a
substitute for CT, and they were derived in one health system on a mostly Philips fleet.

<br clear="right">

---

## Pipeline

```mermaid
flowchart LR
    A[Echo study<br/>DICOM folder] -->|preprocess_dicom.py| B[Cine clips<br/>256×256 .avi]
    B -->|infer.py| C[Clip scores<br/>16 frames · 224×224]
    C -->|mean over clips| D[PanAdipo<br/>study score]
    D -->|assets/eat_calibration.json| E[EAT mL ·<br/>EAT / heart volume]
    D --> F[reports/*.png]
    B -->|gradcam_panadipo.py| G[Grad-CAM++<br/>panels · videos]
```

Each step also runs on its own:

```bash
# 1. DICOM -> AVI   (stills and <16-frame objects are skipped; inputs are only read)
python preprocess_dicom.py --studies_root /data/echo_dicoms --out_dir work/ --n_workers 16

# 2. AVI -> clip + study scores (+ EAT estimates)
python infer.py --videos_csv work/videos.csv --out_dir work/
python infer.py --video_dir  my_avis/       --out_dir out/     # files named <study_id>_<clip>.avi
python infer.py --videos a.avi b.avi        --out_dir out/

# 3. Grad-CAM++ explainability (manuscript settings are the defaults)
python gradcam_panadipo.py --fpaths work/videos/STUDY001_12.avi --out_dir gradcam_out/
```

<p align="center"><img src="docs/figures/fig2a_echo_ct.jpg" width="820"><br>
<sub>Grad-CAM++ attention over the epicardium (left) and the paired CCTA BOA segmentation used as the reference standard (right).</sub></p>

**Preprocessing** is a local-file port of the PanEcho pipeline. A motion mask built from frame-to-frame
changes keeps only the ultrasound sector and drops burned-in text, ECG traces, and calipers. Frames are cropped
to the mask, resized to 224 px with `INTER_AREA`, and zero-padded to 256 × 256. Clips are named
`<study_id>_<dicom_stem>.avi`.

**Inference** samples 16 evenly spaced frames, resizes them to 224 × 224 with `INTER_AREA`, and applies
ImageNet normalisation, exactly as in the manuscript evaluation. The **Grad-CAM++** script uses the per-frame
ConvNeXt encoder (penultimate stage) with unsharp-mask sharpening and a 0.25 threshold. It writes a
cardiac-cycle panel PNG plus overlay and side-by-side MP4s.

---

## Reproducibility

Before release, the code was checked against the stored manuscript predictions (summary below; the internal reference data are not distributed).

| Check | Data | Result |
|---|---|---|
| Clip scores from preprocessed AVIs | 1,000 random YNHHS TTE test clips | **bit-exact**, max \|Δ\| = 3×10⁻⁸ |
| Clip scores from preprocessed AVIs | 1,000 random clips from the 5,081 CT-paired studies | **bit-exact**, max \|Δ\| = 6×10⁻⁸ |
| Same, in a fresh env built from `envs/panadipo.yml` | 300 CT-paired clips | **bit-exact** |
| Same, in the original training env (`panecho`) | 2 × 1,000 clips | **bit-exact** |
| End to end from raw DICOM (`run_pipeline.py`) | 8 random CT-paired studies, 517 clips | clip counts 8/8 identical; study score \|Δ\| ≤ 0.003 (median 0.0014) |
| EAT calibration (manuscript Fig. 3A) | 5,081 held-out CT pairs | ρ = 0.76, R² = 0.53, AUROC (high EAT) = 0.90, matching the manuscript |

<details>
<summary>Notes</summary>

- The previous version of this repository resized 256 px clips with `INTER_LINEAR` rather than `INTER_AREA`. That
  shifted clip scores by a median of 6% (max |Δ| 0.078). It is fixed here.
- DICOM decoding needs **pydicom ≥ 3**, which converts YBR_FULL_422 JPEG to RGB, as in the original preprocessing.
  pydicom 2.x with pylibjpeg returns raw YBR and gives wrong colours (mean pixel error ≈ 26 / 255).
- The small end-to-end differences come from JPEG decoder implementations. The motion-mask threshold can
  occasionally move the crop box by one pixel. Both effects are tiny at the study level.
</details>

---

## Model weights

| | |
|---|---|
| File | `weights/best_model.pt` (≈ 170 MB; model weights only, no optimizer state or training logs) |
| Version | v1.0 (manuscript model) |
| MD5 | `1aa0ce1989506553dca91f3982a199cf` |
| Architecture | PanEcho backbone (ConvNeXt-T per frame → 4-layer temporal Transformer → 768-d) + Dropout(0.3) → Linear |
| Input | 16 frames × 224 × 224 RGB, ImageNet normalisation |

Weights are not stored in git. They are published as a release asset (`v1.0`), and `copy_weights.sh` fetches them:

```bash
./copy_weights.sh                          # download from the release (uses gh, or GH_TOKEN for a private repo)
./copy_weights.sh /path/to/best_model.pt   # or copy a local file
md5sum weights/best_model.pt               # expect 1aa0ce1989506553dca91f3982a199cf
```

The script also prints a sanity check (head type, clip_len, MD5). On first run, `torch.hub` downloads the PanEcho architecture
code (not its weights) from GitHub. That step needs internet once and is then cached in `~/.cache/torch/hub`.

## Repository layout

```
run_pipeline.py          DICOM study -> score -> EAT -> report         (one command)
preprocess_dicom.py      step 1: DICOM -> 256x256 AVI
infer.py                 step 2: AVI -> clip/study scores + EAT estimates
gradcam_panadipo.py      Grad-CAM++ panels and videos
copy_weights.sh          download / stage weights/best_model.pt
panadipo/
  model.py               PanEcho backbone + head, checkpoint loading
  video.py               clip loading (identical to the manuscript evaluation)
  dicom.py               motion-mask crop / resize / AVI writing
  eat.py                 CT-calibrated EAT estimates
  report.py              per-study PNG report
assets/eat_calibration.json   formula coefficients + aggregate reference quantiles
envs/panadipo.yml        single environment for everything
docs/figures/            README figures (from the manuscript)
```

## Intended use and limitations

- **Research use only.** PanAdipo is not a medical device and has not been prospectively validated.
- The training label was a report-derived "prominent epicardial fat depot" (weak supervision). The score is
  continuous and has no diagnostic threshold.
- EAT estimates carry wide individual prediction intervals (R² ≈ 0.5). Use them to rank or stratify cohorts, not
  to replace CT measurement in an individual patient.
- Developed at Yale-New Haven Health. Performance on other vendors, sites, and paediatric studies has not been
  established.

## License

**Academic and non-commercial use only. Commercial use is not permitted.**

- **Code**: [PolyForm Noncommercial License 1.0.0](LICENSE). Use for research, teaching, and other
  non-commercial purposes by individuals and by academic, non-profit, and government institutions.
- **Model weights** (`best_model.pt` release asset): [CC BY-NC-SA 4.0](WEIGHTS_LICENSE.md), which follows the
  upstream PanEcho weights.

Copyright © 2026 Yale University. For commercial licensing, or to tell us how you are using PanAdipo, contact
**evangelos.oikonomou@yale.edu**.

## Citation

If you use PanAdipo, please cite:

> Aminorroaya A, …, Oikonomou EK\*. PanAdipo: AI-enabled echocardiography identifies prominent epicardial
> adiposity. *Under review*, 2026.
>
> \*Corresponding author: Evangelos K. Oikonomou, MD, DPhil (evangelos.oikonomou@yale.edu)

PanAdipo is built on the [PanEcho](https://github.com/CarDS-Yale/PanEcho) echocardiography foundation model:
Holste G, *et al.* Complete AI-enabled echocardiography interpretation with multitask deep learning. *JAMA* 2025.

## Contact

Questions, bug reports, and collaboration requests: open a GitHub issue or email **evangelos.oikonomou@yale.edu**.
