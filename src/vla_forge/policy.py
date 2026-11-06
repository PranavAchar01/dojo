"""The VLA: frozen DINOv2 vision + frozen SigLIP2 text + a trainable action transformer (ACT-style chunking).

Inputs per step: the front and wrist camera images (168 px), the instruction, and the 6 joint positions.
Output: the next K = 10 joint-position targets (1 s at 10 Hz) for the 5 arm joints and the gripper.

Vision tokens are FiLM-modulated by the instruction embedding, so the same image tokens read differently for
"the red block" and "the blue block"; the instruction also enters as its own token. Learned action queries
read the scene through a pre-norm transformer encoder and are decoded to normalised joint targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

from .encoders import TXT_DIM, VIS_DIM, VIS_TOKENS


@dataclass
class HeadConfig:
    d: int = 256
    layers: int = 4
    heads: int = 8
    chunk: int = 10
    act_dim: int = 6
    state_dim: int = 6
    cams: int = 2
    aux_dim: int = (
        5  # auxiliary: target block xyz + goal xy, predicted from the image tokens
    )
    latents: int = 8  # learned queries per camera that summarise its 37 DINOv2 tokens
    dropout: float = 0.1
    use_state: bool = True  # False: the arm's joint readings are hidden; actions come from the images alone
    ground: bool = False  # ablation (neutral at 400 demos: 24/30 vs 25/30). True: the first half of the layers locates the blocks; the action queries read that estimate


class ActionHead(nn.Module):
    def __init__(self, cfg: HeadConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d
        self.vis_proj = nn.Sequential(nn.LayerNorm(VIS_DIM), nn.Linear(VIS_DIM, d))
        self.vis_pos = nn.Parameter(torch.randn(cfg.cams, VIS_TOKENS, d) * 0.02)
        self.lang_proj = nn.Linear(TXT_DIM, d)
        self.film = nn.Linear(TXT_DIM, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        # per-camera latent queries (Perceiver-style): one cross-attention read of the camera's tokens
        self.lat = nn.Parameter(torch.randn(cfg.cams, cfg.latents, d) * 0.02)
        self.lat_lang = nn.Linear(
            TXT_DIM, d
        )  # the instruction shifts what the latents look for
        self.xattn = nn.MultiheadAttention(
            d, cfg.heads, dropout=cfg.dropout, batch_first=True
        )
        self.xnorm_q = nn.LayerNorm(d)
        self.xnorm_kv = nn.LayerNorm(d)
        self.xmlp = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.state_proj = nn.Sequential(
            nn.Linear(cfg.state_dim, d), nn.GELU(), nn.Linear(d, d)
        )
        self.queries = nn.Parameter(
            torch.randn(cfg.chunk + 1, d) * 0.02
        )  # K action queries + 1 aux query
        self.type_emb = nn.Parameter(
            torch.randn(4, d) * 0.02
        )  # lang, state, vision latents, action queries
        layer = nn.TransformerEncoderLayer(
            d,
            cfg.heads,
            4 * d,
            cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        if cfg.ground:
            # spatial grounding: layers 1..L/2 predict where the target block and goal are (supervised by the
            # aux loss); that estimate becomes a token that layers L/2+1..L turn into joint targets
            n1 = cfg.layers // 2
            self.enc1 = nn.TransformerEncoder(layer, n1, enable_nested_tensor=False)
            self.enc = nn.TransformerEncoder(
                layer, cfg.layers - n1, enable_nested_tensor=False
            )
            self.norm1 = nn.LayerNorm(d)
            self.ground_proj = nn.Sequential(
                nn.Linear(cfg.aux_dim, d), nn.GELU(), nn.Linear(d, d)
            )
            self.ground_type = nn.Parameter(torch.randn(d) * 0.02)
        else:
            self.enc = nn.TransformerEncoder(
                layer, cfg.layers, enable_nested_tensor=False
            )
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, cfg.act_dim)
        self.aux_out = nn.Linear(d, cfg.aux_dim)

    def forward(
        self, vis: torch.Tensor, lang: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        """vis [B, cams, 37, 384], lang [B, 768], state [B, 6] (normalised) -> [B, K, 6] (normalised)."""
        b, cams = vis.shape[:2]
        if not self.cfg.use_state:
            state = torch.zeros_like(state)
        v = self.vis_proj(vis.to(self.lat.dtype)) + self.vis_pos
        g, beta = self.film(lang).chunk(2, -1)
        v = v * (1 + g[:, None, None]) + beta[:, None, None]
        q = (self.lat[None] + self.lat_lang(lang)[:, None, None]).flatten(
            0, 1
        )  # [B*cams, L, d]
        kv = self.xnorm_kv(v.flatten(0, 1))  # [B*cams, 37, d]
        z = q + self.xattn(self.xnorm_q(q), kv, kv, need_weights=False)[0]
        z = (z + self.xmlp(z)).view(b, cams * self.cfg.latents, -1)
        tokens = torch.cat(
            [
                (self.lang_proj(lang) + self.type_emb[0])[:, None],
                (self.state_proj(state) + self.type_emb[1])[:, None],
                z + self.type_emb[2],
                (self.queries + self.type_emb[3]).expand(b, -1, -1),
            ],
            1,
        )
        if self.cfg.ground:
            h1 = self.enc1(tokens)
            self.last_aux = self.aux_out(
                self.norm1(h1[:, -1])
            )  # read by the trainer only
            gtok = self.ground_proj(self.last_aux) + self.ground_type
            h = self.norm(self.enc(torch.cat([h1, gtok[:, None]], 1)))
            return self.out(h[:, -self.cfg.chunk - 2 : -2])
        h = self.norm(self.enc(tokens))
        self.last_aux = self.aux_out(h[:, -1])  # read by the trainer only
        return self.out(h[:, -self.cfg.chunk - 1 : -1])


@dataclass
class Norm:
    mean: list[float]
    std: list[float]


class VLA(nn.Module):
    """Action head plus the joint-space normalisation; the frozen encoders live in encoders.py."""

    def __init__(self, cfg: HeadConfig | None = None, norm: Norm | None = None):
        super().__init__()
        self.cfg = cfg or HeadConfig()
        self.head = ActionHead(self.cfg)
        n = norm or Norm([0.0] * 6, [1.0] * 6)
        self.register_buffer("mu", torch.tensor(n.mean, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(n.std, dtype=torch.float32))

    def norm_q(self, q):
        return (q - self.mu) / self.sd

    def denorm_q(self, q):
        return q * self.sd + self.mu

    @torch.no_grad()
    def act(self, vis, lang, state) -> torch.Tensor:
        """Raw joint state [B, 6] -> raw joint targets [B, K, 6]."""
        return self.denorm_q(self.head(vis, lang, self.norm_q(state)))

    def save(self, path: Path, meta: dict | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"cfg": asdict(self.cfg), "state": self.state_dict(), "meta": meta or {}},
            path,
        )

    @staticmethod
    def load(path: Path, device: torch.device | str = "cpu") -> tuple[VLA, dict]:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        m = VLA(HeadConfig(**ck["cfg"]))
        m.load_state_dict(ck["state"])
        return m.to(device), ck.get("meta", {})

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
