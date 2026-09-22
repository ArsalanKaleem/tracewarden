"""Day 15 - publish AgentDojo-Steps to the Hugging Face Hub (one row per step, Parquet, with a card).

python scripts/publish_dataset.py --data data/agentdojo_steps --repo <you>/agentdojo-steps --private
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from tracewarden.schema import load_jsonl

CARD = """---
license: mit
task_categories: [text-classification, token-classification]
tags: [prompt-injection, llm-agents, security, agentdojo]
---
# AgentDojo-Steps

Real tool-call trajectories of open-weight LLM agents in AgentDojo environments, with every step labeled
benign / injection_point / hijacked / failed_injection by an automatic rule checked against AgentDojo's
ground-truth security oracle (see TraceWarden, `tracewarden/io/agentdojo.py`).

* Agent models: {models}
* Trajectories: {n_traj} | steps: {n_steps}
* Hand audit: {audit}
* Splits: {splits}

Environments, user tasks and attacker goals come from AgentDojo (Debenedetti et al., NeurIPS 2024, MIT license).
All data is from fictional sandbox environments. Intended use: training and evaluating injection detectors.
"""

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    root = Path(a.data)
    api = HfApi()
    api.create_repo(a.repo, repo_type="dataset", private=a.private, exist_ok=True)
    splits, n_traj, n_steps, models, audits = {}, 0, 0, set(), []
    for f in sorted(root.glob("*.jsonl")):
        T = load_jsonl(str(f))
        rows = [{"trajectory_id": t.id, "step": i, "source": t.source, "suite": t.domain, "category": t.category,
                 "user_task": t.user_task, "tool": s.tool, "args": json.dumps(s.args, ensure_ascii=False),
                 "thought": s.thought, "observation": s.obs, "label": s.label,
                 "attack": t.meta.get("attack"), "injection_task_id": t.meta.get("injection_task_id")}
                for t in T for i, s in enumerate(t.steps)]
        pq = root / f"{f.stem}.parquet"
        pd.DataFrame(rows).to_parquet(pq, index=False)
        api.upload_file(path_or_fileobj=str(pq), path_in_repo=f"data/{f.stem}.parquet", repo_id=a.repo,
                        repo_type="dataset")
        splits[f.stem] = len(T)
        n_traj += len(T)
        n_steps += len(rows)
        models |= {t.source for t in T}
        au = f.with_suffix(".audit.json")
        if au.exists():
            d = json.loads(au.read_text())
            audits.append(f"{f.stem}: {d['agreement']:.1%} of {d['n']}")
    card = CARD.format(models=", ".join(sorted(models)), n_traj=n_traj, n_steps=n_steps,
                       audit="; ".join(audits) or "TODO", splits=json.dumps(splits))
    api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=a.repo, repo_type="dataset")
    print("published", a.repo, splits)
