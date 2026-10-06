"""
Video loading — identical to the manuscript evaluation (2b_evaluate_panadipo.py).

  1. decode all frames (BGR -> RGB), up to 1000
  2. sample `clip_len` frames evenly (np.linspace, int); shorter clips repeat the last frame
  3. resize each sampled frame to 224x224 with cv2.INTER_AREA
     (preprocessed AVIs are 256x256; INTER_AREA matters — INTER_LINEAR shifts
      scores by ~6% relative)
  4. scale to [0,1] and ImageNet-normalise
"""

import cv2
import numpy as np
import torch

TARGET_SIZE   = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def read_frames(fpath, max_frames: int = 1000) -> np.ndarray:
    """Decode an .avi into (T, H, W, 3) uint8 RGB."""
    cap = cv2.VideoCapture(str(fpath))
    frames = []
    for _ in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise IOError(f"No frames decoded from: {fpath}")
    return np.stack(frames)


def sample_indices(n_frames: int, clip_len: int) -> np.ndarray:
    if n_frames >= clip_len:
        return np.linspace(0, n_frames - 1, clip_len, dtype=int)
    return np.array(list(range(n_frames)) + [n_frames - 1] * (clip_len - n_frames))


def load_clip(fpath, clip_len: int = 16):
    """Return (tensor (3, T, 224, 224) float32 normalised, frames (T, 224, 224, 3) uint8 RGB)."""
    raw = read_frames(fpath)
    frames = np.stack([
        cv2.resize(raw[i], (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_AREA)
        for i in sample_indices(len(raw), clip_len)
    ])
    norm = (frames.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(norm).permute(3, 0, 1, 2).float(), frames


class ClipDataset(torch.utils.data.Dataset):
    """Clip paths -> (tensor, index, ok). Unreadable clips return zeros with ok=False."""

    def __init__(self, fpaths, clip_len: int = 16):
        self.fpaths   = list(map(str, fpaths))
        self.clip_len = clip_len

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, i):
        try:
            x, _ = load_clip(self.fpaths[i], self.clip_len)
            return x, i, True
        except Exception:
            return torch.zeros(3, self.clip_len, TARGET_SIZE, TARGET_SIZE), i, False
