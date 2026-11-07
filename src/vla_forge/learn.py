"""Datasets of encoded demonstrations, and time-budgeted training of the action head on MPS."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .encoders import device, encode_images, encode_text
from .policy import VLA, Norm


@dataclass
class Demos:
    """Encoded demonstrations: per-step vision tokens, state, instruction index and the K-step action chunk."""

    chunk: int = 10
    vis: list[torch.Tensor] = field(default_factory=list)  # each [T, 2, 37, 384] fp16
    states: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    aux: list[np.ndarray] = field(default_factory=list)
    lang_idx: list[int] = field(default_factory=list)
    instructions: list[str] = field(default_factory=list)
    _packed: dict | None = None

    def add_episode(self, npz_path: str, instruction: str) -> int:
        z = np.load(npz_path)
        imgs = z["images"]  # [T, 2, H, W, 3]
        t = imgs.shape[0]
        feats = encode_images(imgs.reshape(-1, *imgs.shape[2:])).view(
            t, 2, *encode_shape()
        )
        if instruction not in self.instructions:
            self.instructions.append(instruction)
        self.vis.append(feats)
        self.states.append(z["states"])
        self.actions.append(z["actions"])
        self.aux.append(z["aux"])
        self.lang_idx.append(self.instructions.index(instruction))
        self._packed = None
        return t

    def add_encoded(
        self, feats: torch.Tensor, states, actions, aux, instruction: str
    ) -> None:
        if instruction not in self.instructions:
            self.instructions.append(instruction)
        self.vis.append(feats)
        self.states.append(states)
        self.actions.append(actions)
        self.aux.append(aux)
        self.lang_idx.append(self.instructions.index(instruction))
        self._packed = None

    @property
    def episodes(self) -> int:
        return len(self.vis)

    @property
    def frames(self) -> int:
        return sum(len(s) for s in self.states)

    def norm(self) -> Norm:
        a = np.concatenate(self.actions + self.states)
        return Norm(a.mean(0).tolist(), np.maximum(a.std(0), 1e-3).tolist())

    def pack(self, dev: torch.device) -> dict:
        if self._packed is not None and self._packed["dev"] == dev:
            return self._packed
        k = self.chunk
        chunks, ep_lang = [], []
        for acts, li in zip(self.actions, self.lang_idx, strict=True):
            t = len(acts)
            idx = np.minimum(np.arange(t)[:, None] + np.arange(k)[None], t - 1)
            chunks.append(acts[idx])
            ep_lang.append(np.full(t, li))
        self._packed = {
            "dev": dev,
            "vis": torch.cat(self.vis).to(dev),
            "state": torch.from_numpy(np.concatenate(self.states)).float().to(dev),
            "chunk": torch.from_numpy(np.concatenate(chunks)).float().to(dev),
            "aux": torch.from_numpy((np.concatenate(self.aux) - AUX_MU) / AUX_SD)
            .float()
            .to(dev),
            "lang_i": torch.from_numpy(np.concatenate(ep_lang)).long().to(dev),
            "lang": encode_text(self.instructions).to(dev),
        }
        return self._packed


AUX_MU = np.array([0.2, 0.0, 0.03, 0.2, 0.0], np.float32)
AUX_SD = np.array([0.06, 0.1, 0.03, 0.06, 0.1], np.float32)
AUX_WEIGHT = 0.5


def encode_shape() -> tuple[int, int]:
    from .encoders import VIS_DIM, VIS_TOKENS

    return VIS_TOKENS, VIS_DIM


def _batch(p: dict, n: int, b: int, dev: torch.device):
    idx = torch.randint(0, n, (b,), device=dev)
    return (
        p["vis"][idx],
        p["lang"][p["lang_i"][idx]],
        p["state"][idx],
        p["chunk"][idx],
        p["aux"][idx],
    )


# which layers a fine-tune updates: the perception half reads the cameras and the instruction, the action half
# is the transformer, the queries and the output layers that turn that reading into joint targets
SCOPES = {
    "all": None,
    "action_layers": (
        "enc",
        "norm",
        "out",
        "aux_out",
        "queries",
        "state_proj",
        "type_emb",
    ),
    "perception_layers": (
        "vis_proj",
        "vis_pos",
        "lang_proj",
        "film",
        "lat",
        "lat_lang",
        "xattn",
        "xnorm_q",
        "xnorm_kv",
        "xmlp",
    ),
}


def scoped_params(model: VLA, scope: str) -> list[torch.nn.Parameter]:
    """The parameters a fine-tune of this scope updates (head module names, split on the first dot)."""
    prefixes = SCOPES[scope]
    out = []
    for name, p in model.head.named_parameters():
        top = name.split(".")[0]
        keep = prefixes is None or top in prefixes
        p.requires_grad_(keep)
        if keep:
            out.append(p)
    return out


def quiet_frozen_layers(model: VLA) -> None:
    """A frozen layer does not drop activations (and MPS attention has no dropout path without gradients)."""
    for mod in model.head.modules():
        own = list(mod.parameters(recurse=False))
        if own and not any(q.requires_grad for q in own):
            mod.eval()


def train(
    model: VLA,
    data: Demos,
    seconds: float,
    lr: float = 3e-4,
    batch: int = 256,
    state_noise: float = 0.03,
    on_log: Callable[[dict], None] | None = None,
    log_every: float = 0.5,
    should_stop: Callable[[], bool] | None = None,
    replay: Demos | None = None,
    replay_frac: float = 0.5,
    weight_decay: float = 1e-4,
    scope: str = "all",
) -> dict:
    """Train for a wall-clock budget. L1 on normalised joint targets, AdamW, cosine decay over the budget.

    `scope` picks which layers learn: see SCOPES."""
    dev = device()
    model.to(dev).train()
    params = scoped_params(model, scope)
    for mod in (
        model.head.modules()
    ):  # a frozen layer does not drop activations (MPS attention has no dropout path)
        own = list(mod.parameters(recurse=False))
        if own and not any(q.requires_grad for q in own):
            mod.eval()
    p = data.pack(dev)
    n = p["vis"].shape[0]
    r = replay.pack(dev) if replay is not None and replay.episodes else None
    nr = int(batch * replay_frac)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    t0 = time.time()
    last = t0
    step, ema = 0, None
    while True:
        el = time.time() - t0
        if el >= seconds or (should_stop and should_stop()):
            break
        for g in opt.param_groups:
            g["lr"] = lr * (
                0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(el / seconds, 1.0)))
            )
        vis, lang, state, chunk, aux = _batch(
            p, n, batch if r is None else batch - nr, dev
        )
        if (
            r is not None
        ):  # replay of earlier skills keeps the library from being forgotten
            rb = _batch(r, r["vis"].shape[0], nr, dev)
            vis, lang, state, chunk, aux = (
                torch.cat(x)
                for x in zip((vis, lang, state, chunk, aux), rb, strict=True)
            )
        state = model.norm_q(state)
        state = state + torch.randn_like(state) * state_noise / model.sd
        target = model.norm_q(chunk)
        with torch.autocast(dev.type, dtype=torch.bfloat16, enabled=dev.type == "mps"):
            pred = model.head(vis, lang, state)
        loss_act = F.l1_loss(pred.float(), target)
        loss = loss_act + AUX_WEIGHT * F.l1_loss(model.head.last_aux.float(), aux)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        step += 1
        if time.time() - last >= log_every:
            lv = float(loss_act.item())
            ema = lv if ema is None else 0.8 * ema + 0.2 * lv
            last = time.time()
            if on_log:
                on_log(
                    {
                        "t": round(last - t0, 2),
                        "step": step,
                        "loss": round(lv, 4),
                        "ema": round(ema, 4),
                    }
                )
    model.eval()
    if dev.type == "mps":
        torch.mps.synchronize()
    return {
        "steps": step,
        "seconds": round(time.time() - t0, 2),
        "loss": ema,
        "frames": n,
    }


def save_shard(d: Demos, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "vis": d.vis,
            "states": d.states,
            "actions": d.actions,
            "aux": d.aux,
            "instr": [d.instructions[i] for i in d.lang_idx],
        },
        path.with_suffix(".tmp"),
    )
    path.with_suffix(".tmp").replace(path)


def load_shards(paths: list[Path]) -> Demos:
    d = Demos()
    for p in paths:
        z = torch.load(p, weights_only=False)
        for v, s, a, x, ins in zip(
            z["vis"], z["states"], z["actions"], z["aux"], z["instr"], strict=True
        ):
            d.add_encoded(v, s, a, x, ins)
    return d
