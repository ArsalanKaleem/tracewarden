"""Collect every number, figure and provenance detail needed to draft the paper into one zip.

    python scripts/collect_paper_assets.py --out paper/assets

Gathers:
  * every results table and JSON under paper/results/
  * every figure under paper/figures/
  * dataset statistics computed live from data/* (counts, label mix, suites, step-length distribution)
  * checkpoint metadata (architecture, encoder, feature groups, thresholds, training config)
  * label-audit numbers, both hand and automated
  * environment provenance (git commit, package versions, platform)
  * MANIFEST.md: every headline number in one place, ready to quote

Nothing here recomputes a model, so it is safe to run any time and cheap to re-run before submission.
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

DATA_SETS = {
    "agentdrift_test": "data/processed/test.jsonl",
    "agentdrift_train": "data/processed/train.jsonl",
    "real_qwen7b": "data/real_qwen/test.jsonl",
    "real_qwen14b": "data/real_qwen14b/test.jsonl",
    "real_multistep": "data/real_multistep/test.jsonl",
}
CHECKPOINTS = ["runs/sg.pt", "runs/sg_mpnet.pt", "runs/driftnet.pt", "models/tracewarden-demo-hashing.pt"]


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:
        return ""


def dataset_stats(path: Path) -> dict:
    from tracewarden.schema import load_jsonl

    T = load_jsonl(str(path))
    steps = [s for t in T for s in t.steps]
    lens = sorted(len(t.steps) for t in T)
    worlds = sum(1 for t in T if (t.world or {}).get("known_emails") or (t.world or {}).get("accounts"))
    return {
        "file": str(path),
        "trajectories": len(T),
        "steps": len(steps),
        "categories": dict(Counter(t.category for t in T)),
        "step_labels": dict(Counter(s.label for s in steps)),
        "suites": dict(Counter(t.domain for t in T)),
        "sources": dict(Counter(t.source for t in T)),
        "steps_per_trajectory": {
            "min": lens[0] if lens else 0, "max": lens[-1] if lens else 0,
            "median": lens[len(lens) // 2] if lens else 0,
            "mean": round(sum(lens) / len(lens), 2) if lens else 0,
        },
        "trajectories_with_known_world": worlds,
        "attack_success_rate": round(
            sum(1 for t in T if t.category == "attacked")
            / max(sum(1 for t in T if t.category in ("attacked", "failed_attack")), 1), 4),
    }


def checkpoint_meta(path: Path) -> dict:
    import torch

    ck = torch.load(str(path), map_location="cpu", weights_only=False)
    n = sum(v.numel() for v in ck["state_dict"].values())
    return {"file": str(path), "kind": ck.get("kind"), "config": ck.get("config"),
            "encoder": ck.get("encoder"), "feature_groups": ck.get("groups"),
            "thresholds": ck.get("thresholds"), "train_config": ck.get("train_cfg"),
            "parameters": int(n), "tracewarden_version": ck.get("tracewarden_version")}


def manifest(assets: dict) -> str:
    L = ["# Paper assets manifest",
         f"\nGenerated {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC · commit "
         f"`{assets['environment'].get('git_commit', '?')[:8]}`\n",
         "Every number below is machine-generated from the artifacts in this archive. Quote from here.\n"]

    L.append("## Datasets\n")
    L.append("| dataset | trajectories | steps | attacked | failed | benign | attack success | known world |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, d in assets["datasets"].items():
        c = d["categories"]
        L.append(f"| {name} | {d['trajectories']} | {d['steps']} | {c.get('attacked', 0)} | "
                 f"{c.get('failed_attack', 0)} | {c.get('benign', 0) + c.get('hard_negative', 0)} | "
                 f"{d['attack_success_rate']:.1%} | {d['trajectories_with_known_world']} |")

    L.append("\n### Step labels\n")
    L.append("| dataset | benign | injection_point | hijacked | failed_injection | steps/traj (min/med/max) |")
    L.append("|---|---|---|---|---|---|")
    for name, d in assets["datasets"].items():
        s, sp = d["step_labels"], d["steps_per_trajectory"]
        L.append(f"| {name} | {s.get('benign', 0)} | {s.get('injection_point', 0)} | {s.get('hijacked', 0)} | "
                 f"{s.get('failed_injection', 0)} | {sp['min']}/{sp['median']}/{sp['max']} |")

    if assets["results"]:
        L.append("\n## Result tables\n")
        for name in sorted(assets["results"]):
            L.append(f"- `results/{name}`")

    if assets["checkpoints"]:
        L.append("\n## Models\n")
        L.append("| checkpoint | kind | params | encoder | features | hold threshold |")
        L.append("|---|---|---|---|---|---|")
        for c in assets["checkpoints"]:
            th = (c.get("thresholds") or {}).get("hold")
            L.append(f"| {Path(c['file']).name} | {c['kind']} | {c['parameters']:,} | {c['encoder']} | "
                     f"{','.join(c['feature_groups'] or []) or 'none'} | "
                     f"{f'{th:.4f}' if isinstance(th, float) else '-'} |")

    if assets["audits"]:
        L.append("\n## Label audits\n")
        for name, a in assets["audits"].items():
            if "agreement" in a:
                L.append(f"- **{name}** (hand): {a['agreement']:.1%} agreement on {a['n']} trajectories")
            else:
                L.append(f"- **{name}** (automatic): {a.get('n_flagged')}/{a.get('n_trajectories')} flagged; "
                         f"{a.get('counts')}")

    L.append("\n## Figures\n")
    for f in sorted(assets["figures"]):
        L.append(f"- `figures/{f}`")

    e = assets["environment"]
    L.append("\n## Environment\n")
    L.append(f"- commit `{e.get('git_commit', '?')}` ({e.get('git_status', 'unknown')})")
    L.append(f"- python {e.get('python')} on {e.get('platform')}")
    L.append(f"- packages: {', '.join(f'{k} {v}' for k, v in e.get('packages', {}).items())}")
    L.append("\n> Reproduce every table with `docs/reproduce.md`.")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/assets")
    ap.add_argument("--zip", default=None, help="zip path (default <out>.zip)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "results").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    assets: dict = {"datasets": {}, "results": [], "figures": [], "checkpoints": [], "audits": {},
                    "environment": {}}

    # results and figures
    for src in sorted(Path("paper/results").glob("*")) if Path("paper/results").exists() else []:
        if src.is_file():
            shutil.copy(src, out / "results" / src.name)
            assets["results"].append(src.name)
            if src.name.endswith(".audit.json") or src.name.startswith("audit"):
                try:
                    assets["audits"][src.stem] = json.loads(src.read_text())
                except json.JSONDecodeError:
                    pass
    for src in sorted(Path("paper/figures").glob("*")) if Path("paper/figures").exists() else []:
        if src.is_file():
            shutil.copy(src, out / "figures" / src.name)
            assets["figures"].append(src.name)

    # hand audits live next to the data
    for src in Path("data").rglob("*.audit.json") if Path("data").exists() else []:
        try:
            assets["audits"][src.stem] = json.loads(src.read_text())
        except json.JSONDecodeError:
            pass

    # datasets
    for name, path in DATA_SETS.items():
        p = Path(path)
        if p.exists():
            try:
                assets["datasets"][name] = dataset_stats(p)
            except Exception as e:  # noqa: BLE001 - never fail the whole collection on one file
                assets["datasets"][name] = {"file": path, "error": str(e)}

    # checkpoints
    for c in CHECKPOINTS:
        p = Path(c)
        if p.exists():
            try:
                assets["checkpoints"].append(checkpoint_meta(p))
            except Exception as e:  # noqa: BLE001
                assets["checkpoints"].append({"file": c, "error": str(e)})

    # environment
    pkgs = {}
    for mod in ("torch", "numpy", "sklearn", "sentence_transformers", "agentdojo", "vllm"):
        try:
            pkgs[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            pass
    assets["environment"] = {
        "git_commit": _run(["git", "rev-parse", "HEAD"]),
        "git_status": "clean" if not _run(["git", "status", "--porcelain"]) else "dirty",
        "python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    (out / "assets.json").write_text(json.dumps(assets, indent=1, default=str))
    (out / "MANIFEST.md").write_text(manifest(assets))
    for extra in ("LOG.md", "docs/reproduce.md", "docs/claims.md", "docs/threat_model.md", "README.md"):
        if Path(extra).exists():
            shutil.copy(extra, out / Path(extra).name)

    zpath = Path(a.zip or f"{out}.zip")
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob("*")):
            if f.is_file():
                z.write(f, f.relative_to(out))
    print(f"{len(assets['datasets'])} datasets · {len(assets['results'])} result files · "
          f"{len(assets['figures'])} figures · {len(assets['checkpoints'])} checkpoints")
    print("manifest:", out / "MANIFEST.md")
    print("archive :", zpath, f"({zpath.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()