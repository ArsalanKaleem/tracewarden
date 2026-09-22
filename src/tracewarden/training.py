"""Training / inference loops for DriftNet (post-hoc) and StreamGuard (pre-dispatch).

Everything runs on a laptop CPU: the trunk has ~1-2M parameters and inputs are cached embeddings.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from .featurize import ACTION, EventItem, StepItem
from .models import DriftNet, StreamGuard
from .schema import LABEL2ID

B, I, H, F = (LABEL2ID[k] for k in ("benign", "injection_point", "hijacked", "failed_injection"))


@dataclass
class TrainConfig:
    epochs: int = 40
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup: float = 0.1
    patience: int = 6
    seed: int = 42
    # DriftNet loss weights (as reported in the paper)
    traj_pos_weight: float = 1.27
    step_weights: tuple[float, float, float, float] = (0.08, 0.76, 0.34, 2.81)
    # StreamGuard loss weights
    hijack_pos_weight: float = 3.0
    poison_pos_weight: float = 4.0
    poison_loss_weight: float = 0.5
    device: str = "cpu"
    verbose: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ------------------------------------------------------------------------------------------ batching
def collate_events(items: list[EventItem]):
    E = max(len(it.etype) for it in items)
    D = items[0].x.shape[1]
    x = torch.zeros(len(items), E, D)
    et = torch.zeros(len(items), E, dtype=torch.long)
    m = torch.zeros(len(items), E, dtype=torch.bool)
    yh = torch.full((len(items), E), -1, dtype=torch.long)
    yp = torch.full((len(items), E), -1, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it.etype)
        x[b, :n] = torch.from_numpy(it.x.astype(np.float32))
        et[b, :n] = torch.from_numpy(it.etype)
        m[b, :n] = True
        yh[b, :n] = torch.from_numpy(it.y_hijack)
        yp[b, :n] = torch.from_numpy(it.y_poison)
    return x, et, m, yh, yp


def collate_steps(items: list[StepItem]):
    T = max(len(it.step_labels) for it in items)
    D = items[0].x.shape[1]
    x = torch.zeros(len(items), T, D)
    m = torch.zeros(len(items), T, dtype=torch.bool)
    ys = torch.full((len(items), T), -100, dtype=torch.long)
    for b, it in enumerate(items):
        n = len(it.step_labels)
        x[b, :n] = torch.from_numpy(it.x.astype(np.float32))
        m[b, :n] = True
        ys[b, :n] = torch.from_numpy(it.step_labels)
    yt = torch.tensor([it.y_traj for it in items], dtype=torch.float32)
    return x, m, yt, ys


def _batches(items, bs, shuffle, rng):
    idx = list(range(len(items)))
    if shuffle:
        rng.shuffle(idx)
    for i in range(0, len(idx), bs):
        yield [items[j] for j in idx[i:i + bs]]


def _schedule(opt, total, warm):
    return torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min((s + 1) / max(warm, 1), max(0.0, (total - s) / max(total - warm, 1))))


# ------------------------------------------------------------------------------------------ StreamGuard
def streamguard_loss(model, batch, cfg: TrainConfig):
    x, et, m, yh, yp = (t.to(cfg.device) for t in batch)
    lh, lp = model(x, et, m)
    ah, ap = yh >= 0, yp >= 0
    bce_h = nn.functional.binary_cross_entropy_with_logits(
        lh[ah], yh[ah].float(), pos_weight=torch.tensor(cfg.hijack_pos_weight, device=cfg.device))
    bce_p = nn.functional.binary_cross_entropy_with_logits(
        lp[ap], yp[ap].float(), pos_weight=torch.tensor(cfg.poison_pos_weight, device=cfg.device))
    return bce_h + cfg.poison_loss_weight * bce_p


@torch.no_grad()
def predict_events(model: StreamGuard, items: list[EventItem], bs: int = 64, device: str = "cpu"):
    """Returns list of (hijack_prob per step (T,), poison_prob per step (T,))."""
    model.eval().to(device)
    out = []
    for batch in _batches(items, bs, False, None):
        x, et, m, _, _ = collate_events(batch)
        lh, lp = model(x.to(device), et.to(device), m.to(device))
        ph, pp = torch.sigmoid(lh).cpu().numpy(), torch.sigmoid(lp).cpu().numpy()
        for b, it in enumerate(batch):
            n = len(it.etype)
            act = it.etype == ACTION
            out.append((ph[b, :n][act], pp[b, :n][~act]))
    return out


def decode_events(hij: np.ndarray, poi: np.ndarray, t_hijack: float = 0.5, t_poison: float = 0.5) -> np.ndarray:
    """Streaming scores -> AgentDrift 4-way step labels, so both models share one metric suite."""
    lab = np.full(len(hij), B)
    hflag = hij >= t_hijack
    lab[hflag] = H
    first_h = int(np.argmax(hflag)) if hflag.any() else None
    for i in np.where((poi >= t_poison) & ~hflag)[0]:
        lab[i] = I if (first_h is not None and i < first_h) else F
    return lab


# ------------------------------------------------------------------------------------------ DriftNet
def driftnet_loss(model, batch, cfg: TrainConfig):
    x, m, yt, ys = (t.to(cfg.device) for t in batch)
    lt, ls = model(x, m)
    bce = nn.functional.binary_cross_entropy_with_logits(
        lt, yt, pos_weight=torch.tensor(cfg.traj_pos_weight, device=cfg.device))
    ce = nn.functional.cross_entropy(ls.reshape(-1, 4), ys.reshape(-1),
                                     weight=torch.tensor(cfg.step_weights, device=cfg.device), ignore_index=-100)
    return bce + ce


@torch.no_grad()
def predict_steps(model: DriftNet, items: list[StepItem], bs: int = 64, device: str = "cpu"):
    """Returns list of (traj_prob, step_label_pred (T,), step_probs (T,4))."""
    model.eval().to(device)
    out = []
    for batch in _batches(items, bs, False, None):
        x, m, _, _ = collate_steps(batch)
        lt, ls = model(x.to(device), m.to(device))
        pt, ps = torch.sigmoid(lt).cpu().numpy(), torch.softmax(ls, -1).cpu().numpy()
        for b, it in enumerate(batch):
            n = len(it.step_labels)
            out.append((float(pt[b]), ps[b, :n].argmax(-1), ps[b, :n]))
    return out


# ------------------------------------------------------------------------------------------ generic fit
@torch.no_grad()
def _val_loss(model, items, collate, loss_fn, cfg) -> float:
    model.eval()
    tot, n = 0.0, 0
    for batch in _batches(items, 64, False, None):
        tot += loss_fn(model, collate(batch), cfg).item() * len(batch)
        n += len(batch)
    return tot / max(n, 1)


def fit(model: nn.Module, train_items, val_items, cfg: TrainConfig, kind: str):
    """kind: 'events' (StreamGuard) or 'steps' (DriftNet). Early-stops on validation trajectory F1."""
    from .metrics import trajectory_prf

    set_seed(cfg.seed)
    rng = random.Random(cfg.seed)
    model.to(cfg.device)
    collate = collate_events if kind == "events" else collate_steps
    loss_fn = streamguard_loss if kind == "events" else driftnet_loss
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = (len(train_items) + cfg.batch_size - 1) // cfg.batch_size
    total = cfg.epochs * steps_per_epoch
    sched = _schedule(opt, total, int(cfg.warmup * total))
    best, best_state, bad, hist = -1.0, None, 0, []
    for ep in range(cfg.epochs):
        model.train()
        t0, tot = time.time(), 0.0
        for batch in _batches(train_items, cfg.batch_size, True, rng):
            loss = loss_fn(model, collate(batch), cfg)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        if kind == "events":
            preds = [float(h.max()) if len(h) else 0.0 for h, _ in predict_events(model, val_items, device=cfg.device)]
        else:
            preds = [p for p, _, _ in predict_steps(model, val_items, device=cfg.device)]
        f1 = trajectory_prf([it.y_traj for it in val_items], [int(p >= 0.5) for p in preds])["f1"]
        vloss = _val_loss(model, val_items, collate, loss_fn, cfg)
        hist.append({"epoch": ep + 1, "loss": tot / steps_per_epoch, "val_loss": vloss, "val_f1": f1,
                     "sec": time.time() - t0})
        if cfg.verbose:
            print(f"epoch {ep + 1:02d}  loss {tot / steps_per_epoch:.4f}  val loss {vloss:.4f}  "
                  f"val traj-F1 {f1:.4f}  ({time.time() - t0:.0f}s)")
        key = (round(f1, 4), -vloss)  # F1 first, validation loss breaks ties
        if best == -1.0 or key > best:
            best, best_state, bad = key, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, hist


# ------------------------------------------------------------------------------------------ checkpoints
def save_checkpoint(path: str, model: nn.Module, kind: str, encoder: str, groups=None, thresholds=None,
                    train_cfg: TrainConfig | None = None, extra: dict | None = None) -> None:
    from . import __version__

    torch.save({
        "kind": kind, "config": model.config, "state_dict": model.state_dict(), "encoder": encoder,
        "groups": sorted(groups) if groups else None, "thresholds": thresholds or {},
        "train_cfg": asdict(train_cfg) if train_cfg else None, "tracewarden_version": __version__,
        **(extra or {}),
    }, path)


def load_checkpoint(path: str, device: str = "cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model = (StreamGuard if ck["kind"] == "streamguard" else DriftNet)(**ck["config"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck
