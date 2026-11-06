"""Multi-seed transfer study for the Computers & Security manuscript (paper/cose, Table "seeds").

For each encoder and training seed: encode every test set (once per encoder), train StreamGuard on
anonymized AgentDrift, calibrate at a 1% budget, then evaluate the unchanged checkpoint on AgentDrift
test, AgentDojo-Steps 7B/14B (overall and per suite), the stress sets, and every SWE-Inject run (learned
guard alone and with the provenance rule). Finally writes mean / sd / min / max over seeds.

Kaggle (see kaggle/README.md):
    python scripts/multiseed.py --encoders hashing-768 --seeds 0,1,2,3,4 --out results/seeds
    python scripts/multiseed.py --encoders sentence-transformers/all-mpnet-base-v2 --device cuda \
        --seeds 0,1,2,3,4 --out results/seeds
    python scripts/multiseed.py --aggregate-only --out results/seeds
Every step is skipped if its output already exists, so a timed-out session can be resumed.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

# name -> processed folder holding <split>.jsonl
SETS = {
    "agentdrift": "data/processed",
    "real7B": "data/real_qwen",
    "real14B": "data/real_qwen14b",
    "stress_zero_width": "data/stress/zero_width",
    "stress_homoglyph": "data/stress/homoglyph",
    "stress_base64": "data/stress/base64",
    "stress_split": "data/stress/split",
}
SWE_RUNS = ["coder7b", "coder14b", "coder32b", "urlfree_coder14b", "urlfree_coder32b"]


def tag(encoder: str) -> str:
    return "hashing" if encoder.startswith("hashing") else encoder.split("/")[-1]


def run(cmd: list[str], env: dict | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env={**os.environ, **(env or {})})


def tw(*args: str) -> list[str]:
    return [sys.executable, "-m", "tracewarden.cli", *args]


def encode_all(encoder: str, device: str | None, cache_root: Path) -> dict[str, Path]:
    caches = {}
    for name, folder in SETS.items():
        out = cache_root / tag(encoder) / name
        caches[name] = out
        if not Path(folder).exists():
            print(f"[skip] {name}: {folder} missing", flush=True)
            continue
        if out.exists() and any(out.iterdir()):
            continue
        cmd = tw("encode", "--data", folder, "--encoder", encoder, "--out", str(out))
        if device:
            cmd += ["--device", device]
        run(cmd)
    return caches


def one_seed(encoder: str, seed: int, caches: dict[str, Path], swe_dir: Path, out: Path, device: str) -> None:
    t = tag(encoder)
    ck = out / "checkpoints" / f"{t}_s{seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    if not ck.exists():
        run(tw("train", "--model", "streamguard", "--data", SETS["agentdrift"], "--cache", str(caches["agentdrift"]),
               "--out", str(ck), "--epochs", "15", "--layers", "2", "--patience", "6", "--seed", str(seed)))
        run(tw("calibrate", "--checkpoint", str(ck), "--data", SETS["agentdrift"],
               "--cache", str(caches["agentdrift"]), "--alpha", "0.01"))

    transfer = out / f"{t}_s{seed}_transfer.md"
    if not transfer.with_suffix(".json").exists():
        evals = []
        for name, folder in SETS.items():
            if Path(folder).exists():
                evals += ["--eval", f"{name}:{folder}:test:{caches[name]}"]
        run([sys.executable, "scripts/transfer_eval.py", "--checkpoint", str(ck), *evals, "--out", str(transfer)])

    # same metrics at a common threshold of 0.5, for the encoder comparison
    transfer05 = out / f"{t}_s{seed}_transfer_t05.md"
    if not transfer05.with_suffix(".json").exists():
        evals = []
        for name in ("agentdrift", "real7B", "real14B"):
            evals += ["--eval", f"{name}:{SETS[name]}:test:{caches[name]}"]
        _transfer_at(ck, evals, transfer05, 0.5)

    suites = out / f"{t}_s{seed}_per_suite.json"
    if not suites.exists():
        run([sys.executable, "scripts/per_suite_eval.py", str(ck), str(suites),
             f"real7B:{SETS['real7B']}:{caches['real7B']}", f"real14B:{SETS['real14B']}:{caches['real14B']}"])

    for r in SWE_RUNS:
        traj = swe_dir / f"{r}_goalonly.jsonl"
        if not traj.exists():
            print(f"[skip] {traj} missing", flush=True)
            continue
        for policy in (None, "hold"):
            dest = out / f"{t}_s{seed}_swe_{r}{'_taint' if policy else ''}.json"
            if dest.exists():
                continue
            cmd = [sys.executable, "scripts/eval_swe_inject.py", "--trajectories", str(traj),
                   "--checkpoint", str(ck), "--device", device, "--out", str(dest)]
            if policy:
                cmd += ["--taint-policy", policy]
            run(cmd)


def _transfer_at(ck: Path, evals: list[str], dest: Path, threshold: float) -> None:
    """transfer_eval.py with the checkpoint's hold threshold replaced (written to a temporary copy)."""
    import torch
    tmp = dest.with_suffix(".tmp.pt")
    state = torch.load(ck, map_location="cpu", weights_only=False)
    state.setdefault("thresholds", {})["hold"] = threshold
    torch.save(state, tmp)
    try:
        run([sys.executable, "scripts/transfer_eval.py", "--checkpoint", str(tmp), *evals, "--out", str(dest)])
    finally:
        tmp.unlink(missing_ok=True)


def summarize(vals: list[float]) -> dict:
    vals = [v for v in vals if v is not None]
    if not vals:
        return {}
    return {"mean": statistics.fmean(vals), "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals), "n": len(vals), "values": vals}


def aggregate(out: Path) -> None:
    rows: dict[str, dict[str, list[float]]] = {}

    def add(t: str, key: str, v):
        rows.setdefault(t, {}).setdefault(key, []).append(v)

    for f in sorted(out.glob("*_s*_transfer.json")):
        t = f.name.split("_s")[0]
        for r in json.loads(f.read_text()):
            for m in ("traj/f1", "injection_em", "prevent/first_hijack_caught", "prevent/benign_interruption",
                      "conformal/benign_hold_or_block"):
                if m in r:
                    add(t, f"{r['eval']} {m}", r[m])
    for f in sorted(out.glob("*_s*_transfer_t05.json")):
        t = f.name.split("_s")[0]
        for r in json.loads(f.read_text()):
            add(t, f"{r['eval']} traj/f1 @0.5", r["traj/f1"])
    for f in sorted(out.glob("*_s*_swe_coder*.json")) + sorted(out.glob("*_s*_swe_urlfree*.json")):
        t = f.name.split("_s")[0]
        rep = json.loads(f.read_text())
        run_name = f.stem.split("_swe_")[1]
        add(t, f"swe {run_name} auroc", rep.get("scores/auroc"))
        add(t, f"swe {run_name} blocked", rep.get("prevent/blocked_before_execution", 0) * rep.get("n_attacked", 0))
    # hijacks blocked across the three main runs (of 48), learned guard and with the rule
    for t in {f.name.split("_s")[0] for f in out.glob("*_s*_swe_*.json")}:
        seeds = sorted({f.name.split("_s")[1].split("_")[0] for f in out.glob(f"{t}_s*_swe_*.json")})
        for suffix, label in (("", "learned"), ("_taint", "with rule")):
            for s in seeds:
                total = 0.0
                for r in ("coder7b", "coder14b", "coder32b"):
                    p = out / f"{t}_s{s}_swe_{r}{suffix}.json"
                    if p.exists():
                        rep = json.loads(p.read_text())
                        total += rep.get("prevent/blocked_before_execution", 0) * rep.get("n_attacked", 0)
                add(t, f"swe main hijacks blocked of 48 ({label})", round(total))

    summary = {t: {k: summarize(v) for k, v in m.items()} for t, m in rows.items()}
    (out / "seed_summary.json").write_text(json.dumps(summary, indent=1))
    lines = ["| encoder | metric | mean | sd | min | max | n |", "|---|---|---|---|---|---|---|"]
    for t, m in summary.items():
        for k, s in sorted(m.items()):
            if s:
                lines.append(f"| {t} | {k} | {s['mean']:.3f} | {s['sd']:.3f} | {s['min']:.3f} | {s['max']:.3f} | {s['n']} |")
    (out / "seed_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", default="hashing-768", help="comma list, e.g. hashing-768,sentence-transformers/all-mpnet-base-v2")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cpu", help="for encoding and SWE-Inject replay (cuda on a GPU session)")
    ap.add_argument("--swe-dir", default="data/swe_inject_final", help="folder with <run>_goalonly.jsonl files")
    ap.add_argument("--cache-root", default="cache_seeds")
    ap.add_argument("--out", default="results/seeds")
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.aggregate_only:
        for enc in a.encoders.split(","):
            caches = encode_all(enc, a.device if a.device != "cpu" else None, Path(a.cache_root))
            for s in a.seeds.split(","):
                one_seed(enc, int(s), caches, Path(a.swe_dir), out, a.device)
    aggregate(out)


if __name__ == "__main__":
    main()
