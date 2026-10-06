"""
PanAdipo model.

PanAdipo = PanEcho video backbone (per-frame ConvNeXt-Tiny -> 4-layer temporal
Transformer -> mean pool -> 768-d) + a binary head (Dropout 0.3 -> Linear) that
scores a prominent epicardial fat depot from a single 16-frame echo clip.

The fine-tuned checkpoint contains the FULL model state (backbone + head), so the
PanEcho hub backbone is built WITHOUT its own pretrained weights and then
overwritten by the checkpoint.

torch.hub fetches the PanEcho repo code from GitHub on first use and caches it
under ~/.cache/torch/hub, so internet is needed once.
"""

from pathlib import Path

import torch
import torch.nn as nn

LABEL_NAME       = "prominent_epicardial_fat_depot"
DEFAULT_CLIP_LEN = 16
DEFAULT_CKPT     = Path(__file__).resolve().parent.parent / "weights" / "best_model.pt"

# MD5 of the released weights-only checkpoint (v1.0 release asset best_model.pt)
CKPT_MD5 = "1aa0ce1989506553dca91f3982a199cf"


class PanEchoClassifier(nn.Module):
    """PanEcho video backbone + binary head."""

    def __init__(self, clip_len: int = DEFAULT_CLIP_LEN, dropout: float = 0.3,
                 simple_head: bool = True, pretrained_backbone: bool = False):
        super().__init__()
        self.backbone = torch.hub.load(
            "CarDS-Yale/PanEcho", "PanEcho",
            pretrained=pretrained_backbone,   # checkpoint supplies all weights
            backbone_only=True,
            clip_len=clip_len,
            force_reload=False,
            trust_repo=True,
        )
        if simple_head:   # released model
            self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 1))
        else:             # experimental 4-layer head (kept for older checkpoints)
            self.head = nn.Sequential(
                nn.LayerNorm(768), nn.Linear(768, 256),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(256, 1),
            )

    def forward(self, x):
        # x: (B, 3, clip_len, 224, 224) -> (B,) logit
        return self.head(self.backbone(x)).squeeze(1)


def load_model(checkpoint=DEFAULT_CKPT, device="cpu"):
    """Load a PanAdipo checkpoint (head type and clip_len auto-detected).

    Returns (model.eval(), meta) where meta holds clip_len / head.
    """
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Download best_model.pt into weights/ (see README) or pass --checkpoint."
        )
    ckpt  = torch.load(checkpoint, map_location=device, weights_only=True)   # tensors + config only
    state = ckpt["model_state"]
    cfg   = ckpt.get("args", {}) or {}

    clip_len    = cfg.get("clip_len", DEFAULT_CLIP_LEN)
    simple_head = "head.4.weight" not in state

    model = PanEchoClassifier(clip_len=clip_len, dropout=cfg.get("dropout", 0.3),
                              simple_head=simple_head).to(device)
    model.load_state_dict(state)
    model.eval()

    meta = {
        "clip_len":  clip_len,
        "head":      "simple" if simple_head else "4-layer",
    }
    return model, meta
