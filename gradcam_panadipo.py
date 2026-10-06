"""
GradCAM visualisation for PanAdipo (epicardial-fat) on 224x224 .avi echo clips.

Per video it writes, into --out_dir:
  • <name>_panel.png       cardiac-cycle panel: anatomy / CAM overlay / dAttention
  • <name>_overlay.mp4      CAM overlay video
  • <name>_sidebyside.mp4   anatomy | CAM overlay

GradCAM operates on the per-frame ConvNeXt image encoder + classification head
(the temporal Transformer is bypassed for spatial attribution), matching how the
model attends within each frame.

Usage:
    python gradcam_panadipo.py \
        --fpaths clip1.avi clip2.avi \
        --checkpoint weights/best_model.pt \
        --out_dir gradcam_out \
        --cam_method gradcam++ \
        --layer_depth -2 \
        --sharpen --threshold 0.25 \
        --n_display 8

Defaults (Grad-CAM++, penultimate ConvNeXt stage, sharpen, threshold 0.25)
match the manuscript figures.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, HiResCAM, EigenGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from panadipo.model import DEFAULT_CKPT, load_model
from panadipo.video import load_clip

CAM_METHODS = {
    "gradcam":      GradCAM,
    "gradcam++":    GradCAMPlusPlus,
    "hirescam":     HiResCAM,
    "eigengradcam": EigenGradCAM,
}


# ── Heatmap post-processing ──────────────────────────────────────────────────
def sharpen_cam(grayscale: np.ndarray, sigma: float = 3.0,
                strength: float = 2.0, threshold: float = 0.0) -> np.ndarray:
    """Unsharp-mask sharpening + optional hard threshold, renormalised to [0,1]."""
    blurred   = gaussian_filter(grayscale, sigma=sigma)
    sharpened = np.clip(grayscale + strength * (grayscale - blurred), 0, 1)
    if threshold > 0:
        sharpened[sharpened < threshold] = 0
    vmax = sharpened.max()
    if vmax > 0:
        sharpened /= vmax
    return sharpened


# ── Frame-level wrapper for GradCAM (image encoder + head, no Transformer) ────
class FrameModel(nn.Module):
    def __init__(self, image_enc: nn.Module, head: nn.Module):
        super().__init__()
        self.image_enc = image_enc
        self.head      = head

    def forward(self, x):
        return self.head(self.image_enc(x)).squeeze(1)  # (B,)


class BinaryTarget:
    """Backprop the raw logit (gradient flows toward higher fat-depot score)."""
    def __call__(self, output):
        return output


# ── Cardiac phase estimation ─────────────────────────────────────────────────
def estimate_phase_labels(n_frames: int) -> list:
    labels = []
    for i in range(n_frames):
        frac = i / max(n_frames - 1, 1)
        if   frac < 0.10: labels.append("ED")
        elif frac < 0.30: labels.append("early-sys")
        elif frac < 0.45: labels.append("mid-sys")
        elif frac < 0.55: labels.append("ES")
        elif frac < 0.70: labels.append("early-dia")
        elif frac < 0.85: labels.append("mid-dia")
        else:             labels.append("late-dia")
    return labels


# ── Save helpers ─────────────────────────────────────────────────────────────
def save_overlay_video(overlays: list, out_path: Path, fps: int = 10) -> Path:
    out_path = out_path.with_suffix(".mp4")
    H, W = overlays[0].shape[:2]
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    for frame in overlays:
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()
    return out_path


def save_side_by_side_video(frames_uint8: np.ndarray, overlays: list,
                            out_path: Path, fps: int = 10) -> Path:
    out_path = out_path.with_suffix(".mp4")
    H, W = frames_uint8[0].shape[:2]
    gap = 4
    canvas_w = 2 * W + gap
    writer = cv2.VideoWriter(str(out_path),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_w, H))
    for t in range(len(overlays)):
        row = np.ones((H, canvas_w, 3), dtype=np.uint8) * 30
        row[:, :W]       = cv2.cvtColor(frames_uint8[t], cv2.COLOR_RGB2BGR)
        row[:, W + gap:] = cv2.cvtColor(overlays[t], cv2.COLOR_RGB2BGR)
        writer.write(row)
    writer.release()
    return out_path


def make_colorbar(height: int, width: int = 20) -> np.ndarray:
    gradient = np.linspace(0, 1, height).astype(np.float32)[::-1]
    gradient = np.stack([gradient] * width, axis=1)
    return cv2.applyColorMap((gradient * 255).astype(np.uint8), cv2.COLORMAP_JET)


def compute_frame_diff(grayscales: list, t: int, smooth_sigma: float = 5.0) -> np.ndarray:
    """|CAM(t) - CAM(t-1)|, smoothed and normalised to [0,1]."""
    if t > 0:
        diff = np.abs(grayscales[t].astype(np.float32) - grayscales[t - 1].astype(np.float32))
    elif len(grayscales) > 1:
        diff = np.abs(grayscales[0].astype(np.float32) - grayscales[1].astype(np.float32))
    else:
        return np.zeros_like(grayscales[0])
    if smooth_sigma > 0:
        diff = gaussian_filter(diff, sigma=smooth_sigma)
    vmax = diff.max()
    if vmax > 0:
        diff /= vmax
    return diff


def save_cardiac_cycle_panel(frames_uint8, overlays, grayscales, out_path,
                             prob, fpath, label, n_display=8,
                             cam_method="gradcam", show_diff=True,
                             diff_smooth_sigma=5.0):
    """Row1 anatomy | Row2 CAM overlay | Row3 (optional) frame-to-frame dAttention."""
    T = len(overlays)
    indices = np.linspace(0, T - 1, n_display, dtype=int)
    phases  = estimate_phase_labels(T)

    H, W = frames_uint8[0].shape[:2]
    pad = 4
    n_rows = 3 if show_diff else 2

    cbar_w, cbar_gap = 24, 8
    cbar_total = cbar_gap + cbar_w + 30
    col_w    = W + pad
    canvas_w = pad + n_display * col_w + cbar_total
    title_h, label_h = 44, 22
    canvas_h = title_h + n_rows * (H + pad + label_h) + pad + 20
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    title = (f"{cam_method.upper()} | prob={prob:.3f}  label={label} | "
             f"{Path(fpath).name}")
    cv2.putText(canvas, title, (pad, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (30, 30, 30), 1, cv2.LINE_AA)

    row_titles = ["Anatomy", f"{cam_method.upper()} overlay"]
    if show_diff:
        row_titles.append("Frame-to-frame \u0394attention")

    for row_idx, row_title in enumerate(row_titles):
        y_label = title_h + row_idx * (H + pad + label_h)
        cv2.putText(canvas, row_title, (pad, y_label + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 0, 180) if row_idx > 0 else (0, 0, 0), 1, cv2.LINE_AA)
        y_img = y_label + label_h

        for col, t in enumerate(indices):
            x0 = pad + col * col_w
            if row_idx == 0:
                img = cv2.cvtColor(frames_uint8[t], cv2.COLOR_RGB2BGR)
            elif row_idx == 1:
                img = cv2.cvtColor(overlays[t], cv2.COLOR_RGB2BGR)
            else:
                diff = compute_frame_diff(grayscales, t, smooth_sigma=diff_smooth_sigma)
                rgb_float = frames_uint8[t].astype(np.float32) / 255.0
                img = cv2.cvtColor(show_cam_on_image(rgb_float, diff, use_rgb=True),
                                   cv2.COLOR_RGB2BGR)
            canvas[y_img:y_img + H, x0:x0 + W] = img

            if row_idx == n_rows - 1:
                y_phase = y_img + H + 14
                cv2.putText(canvas, f"F{t} {phases[t]}", (x0 + 2, y_phase),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)

    cbar_x = pad + n_display * col_w + cbar_gap
    cbar_y = title_h + label_h
    canvas[cbar_y:cbar_y + H, cbar_x:cbar_x + cbar_w] = make_colorbar(H, cbar_w)
    for txt, yy in [("1.0", cbar_y + 12), ("0.5", cbar_y + H // 2 + 4), ("0.0", cbar_y + H - 2)]:
        cv2.putText(canvas, txt, (cbar_x + cbar_w + 3, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 60), 1)

    cv2.imwrite(str(out_path), canvas)


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="GradCAM for PanAdipo epicardial-fat model")
    p.add_argument("--fpaths", nargs="+", required=True, help="One or more .avi clips")
    p.add_argument("--checkpoint", default=str(DEFAULT_CKPT))
    p.add_argument("--out_dir", default="gradcam_out")
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")

    p.add_argument("--cam_method", default="gradcam++", choices=list(CAM_METHODS.keys()))
    p.add_argument("--layer_depth", type=int, default=-2,
                   help="ConvNeXt stage index: -1 = last (7x7), -2 = penultimate (14x14)")
    p.add_argument("--sharpen", action="store_true", default=True)
    p.add_argument("--no_sharpen", dest="sharpen", action="store_false")
    p.add_argument("--sharpen_sigma", type=float, default=3.0)
    p.add_argument("--sharpen_strength", type=float, default=2.0)
    p.add_argument("--threshold", type=float, default=0.25,
                   help="Zero-out CAM activations below this value (0-1)")

    p.add_argument("--n_display", type=int, default=8)
    p.add_argument("--show_diff", action="store_true", default=True)
    p.add_argument("--no_show_diff", dest="show_diff", action="store_false")
    p.add_argument("--diff_smooth_sigma", type=float, default=5.0)
    return p.parse_args()


def main():
    args    = parse_args()
    device  = torch.device(
        args.device if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Device     : {device}")
    print(f"Checkpoint : {args.checkpoint}")
    model, meta = load_model(args.checkpoint, device)
    clip_len = meta["clip_len"]
    print(f"  head={meta['head']}  clip_len={clip_len}")

    # Locate the ConvNeXt image encoder inside the PanEcho backbone
    image_encoder = None
    for name, module in model.backbone.named_modules():
        if hasattr(module, "features") and hasattr(module, "avgpool"):
            image_encoder = (name, module)
    if image_encoder is None:
        raise RuntimeError("Could not find ConvNeXt image encoder in backbone.")
    enc_name, enc_module = image_encoder

    target_layer = enc_module.features[args.layer_depth]
    print(f"  CAM method   : {args.cam_method}")
    print(f"  Target layer : {enc_name}.features[{args.layer_depth}] "
          f"({type(target_layer).__name__})")
    if args.sharpen:
        print(f"  Sharpening   : sigma={args.sharpen_sigma} "
              f"strength={args.sharpen_strength} threshold={args.threshold}")

    CamClass    = CAM_METHODS[args.cam_method]
    frame_model = FrameModel(enc_module, model.head).to(device).eval()

    for rank, fpath in enumerate(args.fpaths, 1):
        stem = Path(fpath).stem
        try:
            tensor, frames_uint8 = load_clip(fpath, clip_len)
            tensor = tensor.unsqueeze(0)
        except Exception as e:
            print(f"[{rank}/{len(args.fpaths)}] {stem}  SKIP (load error: {e})")
            continue
        tensor = tensor.to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        print(f"\n[{rank}/{len(args.fpaths)}] {stem}  prob={prob:.3f}")

        T, overlays, grayscales = tensor.shape[2], [], []
        with CamClass(model=frame_model,
                      target_layers=[frame_model.image_enc.features[args.layer_depth]]) as cam:
            for t in range(T):
                frame_t   = tensor[:, :, t, :, :].contiguous()
                grayscale = cam(input_tensor=frame_t, targets=[BinaryTarget()])[0]
                if args.sharpen:
                    grayscale = sharpen_cam(grayscale, sigma=args.sharpen_sigma,
                                            strength=args.sharpen_strength,
                                            threshold=args.threshold)
                grayscales.append(grayscale)
                rgb_float = frames_uint8[t].astype(np.float32) / 255.0
                overlays.append(show_cam_on_image(rgb_float, grayscale, use_rgb=True))

        base = f"{rank:02d}_{stem}_prob{prob:.3f}"
        print(f"  Overlay video → {save_overlay_video(overlays, out_dir / f'{base}_overlay.mp4')}")
        print(f"  Side-by-side  → {save_side_by_side_video(frames_uint8, overlays, out_dir / f'{base}_sidebyside.mp4')}")
        panel_path = out_dir / f"{base}_panel.png"
        save_cardiac_cycle_panel(frames_uint8, overlays, grayscales, panel_path,
                                 prob=prob, fpath=fpath, label=-1,
                                 n_display=args.n_display, cam_method=args.cam_method,
                                 show_diff=args.show_diff,
                                 diff_smooth_sigma=args.diff_smooth_sigma)
        print(f"  Panel PNG     → {panel_path}")

    print(f"\nDone. Outputs in: {out_dir}")


if __name__ == "__main__":
    main()
