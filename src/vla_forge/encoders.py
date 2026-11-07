"""Frozen pretrained encoders of the VLA: DINOv2-small for vision, SigLIP2 (base) text tower for language.

Both run once per frame / once per instruction; their outputs feed the trainable action transformer.
Vision tokens: the 12 x 12 DINOv2 patch grid of a 168 px image, average-pooled 2 x 2 to 6 x 6, plus CLS
(37 tokens x 384 per camera).
"""

from __future__ import annotations

import threading
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

DINO_ID = "facebook/dinov2-small"
SIGLIP_ID = "google/siglip2-base-patch16-224"
VIS_TOKENS = 37
VIS_DIM = 384
TXT_DIM = 768

_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
_lock = threading.Lock()


def device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def _dtype(dev: str) -> torch.dtype:
    return (
        torch.float16 if dev == "mps" else torch.float32
    )  # fp16 is 3.5x faster on MPS


@lru_cache(maxsize=2)
def _dino(dev: str):
    from transformers import Dinov2Model

    m = Dinov2Model.from_pretrained(DINO_ID).eval().to(dev, _dtype(dev))
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@lru_cache(maxsize=1)
def _siglip():
    from transformers import AutoModel, AutoTokenizer

    m = AutoModel.from_pretrained(SIGLIP_ID).eval()
    tok = AutoTokenizer.from_pretrained(SIGLIP_ID)
    return m, tok


@torch.no_grad()
def encode_images(
    imgs: np.ndarray | torch.Tensor, dev: torch.device | None = None, batch: int = 256
) -> torch.Tensor:
    """uint8 [N, H, W, 3] -> float16 [N, 37, 384] (CLS + 6x6 pooled patches), on CPU."""
    dev = dev or device()
    model = _dino(str(dev))
    out = []
    x_all = torch.as_tensor(imgs)
    with _lock:
        for i in range(0, len(x_all), batch):
            x = x_all[i : i + batch].to(dev).permute(0, 3, 1, 2).float() / 255.0
            x = ((x - _MEAN.to(dev)) / _STD.to(dev)).to(_dtype(str(dev)))
            h = model(pixel_values=x).last_hidden_state  # [B, 1 + 144, 384]
            cls, patches = h[:, :1], h[:, 1:]
            g = round(patches.shape[1] ** 0.5)
            patches = patches.transpose(1, 2).reshape(-1, VIS_DIM, g, g)
            patches = F.avg_pool2d(patches, 2).flatten(2).transpose(1, 2)
            out.append(torch.cat([cls, patches], 1).half().cpu())
    return torch.cat(out)


@torch.no_grad()
def encode_text(texts: list[str]) -> torch.Tensor:
    """Instruction -> float32 [N, 768] SigLIP2 text embedding (L2 normalised)."""
    model, tok = _siglip()
    with _lock:
        t = tok(
            [s.lower() for s in texts],
            padding="max_length",
            max_length=64,
            truncation=True,
            return_tensors="pt",
        )
        f = model.get_text_features(**t)
        if not isinstance(f, torch.Tensor):
            f = f.pooler_output
    return F.normalize(f.float(), dim=-1)


def param_counts() -> dict[str, int]:
    d = sum(p.numel() for p in _dino("cpu").parameters())
    s = sum(p.numel() for p in _siglip()[0].text_model.parameters())
    return {"vision_dinov2_small": d, "text_siglip2": s}
