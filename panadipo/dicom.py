"""
DICOM -> AVI preprocessing (local-file port of the PanEcho/PanAdipo pipeline,
Echo-pipeline/PanEcho/preprocess/preprocess_tte_eko.py).

Per multi-frame DICOM:
  1. read pixel data (single-frame / <16-frame objects are skipped as stills)
  2. build a motion mask from frame-to-frame changes of channel 0, which removes
     static burned-in text, ECG traces, and other annotations outside the sector
  3. mask every frame, crop to the mask bounding box, resize to
     (img_res - buffer)^2 with INTER_AREA, zero-pad by buffer/2 on each side
  4. write an MJPG .avi at the DICOM frame rate (default 256x256, 32-px border)

Output videos are named <study_id>_<dicom_stem>.avi. The study_id (folder name)
is what infer.py groups on to produce one score per echocardiogram.
"""

from pathlib import Path

import cv2
import numpy as np
import pydicom

DICOM_SUFFIXES = {".dcm", ".dicom"}


def fill_small_holes(mask: np.ndarray) -> np.ndarray:
    mask_uint8 = (mask * 255).astype(np.uint8)
    height, width = mask.shape
    padded = np.pad(mask_uint8, pad_width=1, mode="constant", constant_values=0)
    floodfill_mask = np.zeros([height + 4, width + 4], np.uint8)
    filled = padded.copy()
    cv2.floodFill(filled, floodfill_mask, (0, 0), 255)
    filled = cv2.bitwise_not(filled)
    result = cv2.bitwise_or(mask_uint8, filled[1:-1, 1:-1])
    return (result / 255.0).astype(mask.dtype)


def create_mask(gray_frames: np.ndarray):
    """Motion mask of the ultrasound sector. Returns (eroded full-size mask, bbox).

    NB: frame differences are taken in the native (uint8) dtype, as in the
    original pipeline, so they wrap around; kept as-is for exact reproducibility.
    """
    diffs = np.abs(gray_frames[1:] - gray_frames[:-1])
    changes = np.sum(diffs, axis=0)
    changes_frequency = np.sum(diffs > 0, axis=0)

    changes[changes_frequency < 10] = 0
    if changes.max() > 0:
        changes = (changes - changes.min()) / (changes.max() - changes.min())

    binary_mask = (changes > 0).astype(np.uint8)
    kernel = np.ones((5, 5), np.int32)
    eroded = cv2.erode(binary_mask, kernel, iterations=1)
    eroded = fill_small_holes(eroded)
    mask = np.where(eroded, binary_mask, 0)

    coords = np.column_stack(np.nonzero(mask))
    if coords.size == 0:
        raise ValueError("No moving pixels found (empty motion mask).")
    (min_x, min_y), (max_x, max_y) = coords.min(axis=0), coords.max(axis=0)
    bbox = dict(min_x=int(min_x), min_y=int(min_y), max_x=int(max_x), max_y=int(max_y))
    return eroded, bbox


def resolve_fps(ds) -> float:
    try:
        return float(ds[(0x0018, 0x0040)].value)          # CineRate
    except Exception:
        pass
    try:
        frame_time_ms = float(ds[(0x0018, 0x1063)].value)  # FrameTime
        if frame_time_ms > 0:
            return 1000.0 / frame_time_ms
    except Exception:
        pass
    return 30.0


def load_echo_array(ds) -> np.ndarray:
    """Return (T, H, W, 3) frames or raise ValueError('STILL_IMAGE')."""
    echo = ds.pixel_array
    if echo.ndim == 4:
        return echo
    if echo.ndim == 3 and echo.shape[-1] != 3 and echo.shape[0] >= 16:
        return np.repeat(echo[..., None], 3, -1)
    raise ValueError("STILL_IMAGE")


def dicom_to_avi(dicom_path, out_path, img_res: int = 256, buffer: int = 32) -> str:
    """Convert one DICOM to an .avi. Returns a status string ('video', 'still', or '<reason>_failed')."""
    try:
        ds = pydicom.dcmread(str(dicom_path), force=True)
    except Exception:
        return "read_failed"
    try:
        echo = load_echo_array(ds)
    except ValueError as e:
        return "still" if str(e) == "STILL_IMAGE" else "pixel_failed"
    except Exception:
        return "pixel_failed"

    try:
        mask, bbox = create_mask(echo[:, :, :, 0].copy())
    except Exception:
        return "mask_failed"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"MJPG"),
                             resolve_fps(ds), (img_res, img_res), True)
    half = buffer // 2
    try:
        for frame in echo:
            masked = np.where(mask[..., None], frame, 0)
            cropped = masked[bbox["min_x"]:bbox["max_x"], bbox["min_y"]:bbox["max_y"]]
            out = cv2.resize(cropped, (img_res - buffer, img_res - buffer),
                             interpolation=cv2.INTER_AREA)
            out = np.pad(out, ((half, half), (half, half), (0, 0)), constant_values=0)
            writer.write(out)
    except Exception:
        writer.release()
        out_path.unlink(missing_ok=True)
        return "video_write_failed"
    writer.release()
    return "video"


def find_dicoms(root):
    """All DICOM files under root (by suffix; falls back to extension-less files with a DICM preamble)."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in DICOM_SUFFIXES)
    if files:
        return files
    out = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix == "":
            try:
                with open(p, "rb") as f:
                    f.seek(128)
                    if f.read(4) == b"DICM":
                        out.append(p)
            except OSError:
                pass
    return out


def read_demographics(dicom_paths, max_tries: int = 5) -> dict:
    """Age (years) and sex from the first readable DICOM header of a study (pixel data not read).

    Uses PatientAge (e.g. '063Y'), else StudyDate - PatientBirthDate; PatientSex 'M'/'F'.
    Returns {'age': float or None, 'sex': 'male'/'female' or None}.
    """
    for p in list(dicom_paths)[:max_tries]:
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True, force=True)
        except Exception:
            continue
        age = None
        pa = str(getattr(ds, "PatientAge", "") or "").strip().upper()
        if len(pa) >= 2 and pa[:-1].isdigit():
            age = float(pa[:-1]) * {"Y": 1, "M": 1 / 12, "W": 1 / 52, "D": 1 / 365.25}.get(pa[-1], np.nan)
        if age is None or not np.isfinite(age):
            bd, sd = str(getattr(ds, "PatientBirthDate", "") or ""), str(getattr(ds, "StudyDate", "") or "")
            try:
                from datetime import datetime
                age = (datetime.strptime(sd[:8], "%Y%m%d") - datetime.strptime(bd[:8], "%Y%m%d")).days / 365.25
            except ValueError:
                age = None
        sex = {"M": "male", "F": "female"}.get(str(getattr(ds, "PatientSex", "") or "").strip().upper())
        if age is not None or sex is not None:
            return {"age": age, "sex": sex}
    return {"age": None, "sex": None}
