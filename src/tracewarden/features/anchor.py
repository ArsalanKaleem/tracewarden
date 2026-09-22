"""Goal-anchoring drift features - novel addition N3. Inputs are L2-normalized embeddings."""
from __future__ import annotations

import numpy as np


def anchor_feats(task_e: np.ndarray, action_e: np.ndarray, prior_action_es: list[np.ndarray],
                 prior_obs_es: list[np.ndarray]) -> list[float]:
    sim_task = float(action_e @ task_e)
    sim_obs = float(max((action_e @ o for o in prior_obs_es), default=0.0))
    prev = [float(a @ task_e) for a in prior_action_es]
    drop = (float(np.mean(prev)) - sim_task) if prev else 0.0
    return [sim_task, sim_obs, sim_obs - sim_task, drop]


N_ANCHOR = 4
ANCHOR_NAMES = ["sim_task", "sim_obs", "obs_minus_task", "task_drop"]
