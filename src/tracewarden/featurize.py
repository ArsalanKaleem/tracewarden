"""Turn trajectories into model inputs. The SAME functions are used offline (training) and online
(runtime Session), which guarantees the guard sees exactly the features it was trained on.

Streaming layout (N1): every step becomes two events in time order,
    ACTION_i = (tool, thought, args)   -> known BEFORE the call executes (decision point)
    OBS_i    = observation returned     -> known AFTER the call executes
Each event vector = [embedding (D) | world 4 | provenance 7 | anchor 4]   for ACTION
                  = [embedding (D) | obs 3   | zeros 12               ]   for OBS
Feature groups can be switched off (zeroed) for ablations without changing the width.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from .features.anchor import N_ANCHOR, anchor_feats
from .features.obsfeat import N_OBS, obs_feats
from .features.provenance import N_PROV, provenance_feats
from .features.world import N_WORLD, known_sets, world_feats
from .schema import LABEL2ID, Trajectory

N_EXTRA = N_WORLD + N_PROV + N_ANCHOR  # 15
ALL_GROUPS = frozenset({"world", "prov", "anchor", "obs"})
ACTION, OBS = 0, 1


def action_extra(args: dict, user_task: str, world: dict, prior_obs: list[str], emb_task: np.ndarray,
                 emb_action: np.ndarray, prior_action_embs: list[np.ndarray], prior_obs_embs: list[np.ndarray],
                 groups: frozenset = ALL_GROUPS) -> np.ndarray:
    known, domains = known_sets(world)
    w = world_feats(args, known, domains) if "world" in groups else [0.0] * N_WORLD
    p = provenance_feats(args, user_task, world, prior_obs) if "prov" in groups else [0.0] * N_PROV
    a = anchor_feats(emb_task, emb_action, prior_action_embs, prior_obs_embs) if "anchor" in groups \
        else [0.0] * N_ANCHOR
    return np.asarray(w + p + a, dtype=np.float32)


def obs_extra(obs: str, user_task: str, world: dict, groups: frozenset = ALL_GROUPS) -> np.ndarray:
    v = obs_feats(obs, user_task + " " + json.dumps(world, default=str)) if "obs" in groups else [0.0] * N_OBS
    return np.asarray(v + [0.0] * (N_EXTRA - N_OBS), dtype=np.float32)


@dataclass
class EventItem:
    id: str
    x: np.ndarray            # (E, D + 15) float16
    etype: np.ndarray        # (E,) 0 = ACTION, 1 = OBS
    y_hijack: np.ndarray     # (E,) 1 at hijacked ACTION events, -1 at OBS events (ignored)
    y_poison: np.ndarray     # (E,) 1 at poisoned OBS events,   -1 at ACTION events (ignored)
    step_labels: np.ndarray  # (T,) label ids, -100 if unknown
    y_traj: int
    category: str = ""
    meta: dict = field(default_factory=dict)


@dataclass
class StepItem:
    id: str
    x: np.ndarray            # (T, D + 4) float16
    step_labels: np.ndarray  # (T,)
    y_traj: int
    category: str = ""
    meta: dict = field(default_factory=dict)


def build_events(t: Trajectory, emb_action: np.ndarray, emb_obs: np.ndarray, emb_task: np.ndarray,
                 groups: frozenset = ALL_GROUPS) -> EventItem:
    rows, etype, yh, yp = [], [], [], []
    prior_obs_txt: list[str] = []
    prior_a: list[np.ndarray] = []
    prior_o: list[np.ndarray] = []
    for i, s in enumerate(t.steps):
        ex = action_extra(s.args, t.user_task, t.world, prior_obs_txt, emb_task, emb_action[i], prior_a, prior_o, groups)
        rows.append(np.concatenate([emb_action[i], ex]))
        etype.append(ACTION)
        yh.append(int(s.label == "hijacked") if s.label is not None else -1)
        yp.append(-1)
        rows.append(np.concatenate([emb_obs[i], obs_extra(s.obs, t.user_task, t.world, groups)]))
        etype.append(OBS)
        yh.append(-1)
        yp.append(int(s.label in ("injection_point", "failed_injection")) if s.label is not None else -1)
        prior_obs_txt.append(s.obs)
        prior_a.append(emb_action[i])
        prior_o.append(emb_obs[i])
    return EventItem(
        id=t.id, x=np.stack(rows).astype(np.float16), etype=np.asarray(etype, np.int64),
        y_hijack=np.asarray(yh, np.int64), y_poison=np.asarray(yp, np.int64),
        step_labels=np.asarray([LABEL2ID[s.label] if s.label else -100 for s in t.steps], np.int64),
        y_traj=t.compromised, category=t.category,
        meta={"compliance": t.meta.get("compliance"), "domain": t.domain, "source": t.source},
    )


def build_steps(t: Trajectory, emb_full: np.ndarray, use_world: bool = True) -> StepItem:
    known, domains = known_sets(t.world)
    extra = np.asarray([world_feats(s.args, known, domains) if use_world else [0.0] * N_WORLD
                        for s in t.steps], np.float32)
    return StepItem(
        id=t.id, x=np.concatenate([emb_full, extra], 1).astype(np.float16),
        step_labels=np.asarray([LABEL2ID[s.label] if s.label else -100 for s in t.steps], np.int64),
        y_traj=t.compromised, category=t.category,
        meta={"compliance": t.meta.get("compliance"), "domain": t.domain, "source": t.source},
    )


def build_split(trajs: list[Trajectory], cache, kind: str = "events", groups: frozenset = ALL_GROUPS,
                use_world: bool = True):
    """cache: encoders.EmbeddingCache holding all `trajs`."""
    out = []
    for t in trajs:
        if kind == "events":
            out.append(build_events(t, cache.get(t.id, "action"), cache.get(t.id, "obs"), cache.get_task(t.id), groups))
        else:
            out.append(build_steps(t, cache.get(t.id, "full"), use_world))
    return out
