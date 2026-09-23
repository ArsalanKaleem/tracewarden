"""tracewarden command line.

  tracewarden data agentdrift --root AgentDrift/data_taskdisjoint --out data/processed --identity anonymize
  tracewarden stats   --data data/processed
  tracewarden audit   --data data/processed
  tracewarden encode  --data data/processed --encoder sentence-transformers/all-mpnet-base-v2 --out cache/mpnet
  tracewarden train   --model streamguard --data data/processed --cache cache/mpnet --out runs/sg.pt
  tracewarden calibrate --checkpoint runs/sg.pt --data data/processed --cache cache/mpnet --alpha 0.01
  tracewarden evaluate  --checkpoint runs/sg.pt --data data/processed --cache cache/mpnet --split test
  tracewarden scan    --checkpoint runs/sg.pt --log some_trajectory.json
"""
from __future__ import annotations

import argparse
import json
import sys
import zlib
from collections import Counter
from pathlib import Path


def _cmd_data(a):
    from .features.anonymize import anonymize, identity_shuffle
    from .io import agentdojo, agentdrift
    from .schema import save_jsonl

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.source == "agentdrift":
        parts = agentdrift.load(a.root)
    else:
        trajs = agentdojo.convert_dir(a.root, source=a.name or "agentdojo")
        parts = {a.split or "test": trajs}
    for p, trajs in parts.items():
        if a.identity == "anonymize":
            trajs = [anonymize(t) for t in trajs]
        save_jsonl(trajs, str(out / f"{p}.jsonl"))
        print(f"{p}: {len(trajs)} trajectories -> {out / f'{p}.jsonl'}")
        if p == "train" and a.shuffle_copies:
            aug = []
            for k in range(a.shuffle_copies):
                for t in parts[p]:
                    s = identity_shuffle(t, seed=zlib.crc32(f"{t.id}#{k}".encode()))
                    s.id = f"{t.id}#shuf{k}"
                    aug.append(s)
            save_jsonl(aug, str(out / "train_shuf.jsonl"))
            print(f"train_shuf: {len(aug)} identity-shuffled copies")


def _cmd_stats(a):
    from .experiments import load_processed

    for p, T in load_processed(a.data).items():
        steps = [s for t in T for s in t.steps]
        print(f"== {p}: {len(T)} trajectories, {len(steps)} steps")
        print("   categories:", dict(Counter(t.category for t in T)))
        print("   labels:    ", dict(Counter(s.label for s in steps)))
        print("   domains:   ", dict(Counter(t.domain for t in T)))
        print("   steps/traj: min %d max %d" % (min(len(t.steps) for t in T), max(len(t.steps) for t in T)))


def _cmd_audit(a):
    from .experiments import dump, load_processed, shortcut_audit

    D = load_processed(a.data)
    res = shortcut_audit(D["train"], D[a.split])
    print(json.dumps(res, indent=2))
    if a.out:
        dump(res, a.out)


def _cmd_encode(a):
    from .encoders import encode_corpus, get_encoder
    from .schema import load_jsonl

    files = sorted(Path(a.data).glob("*.jsonl")) if Path(a.data).is_dir() else [Path(a.data)]
    trajs, seen = [], set()
    for f in files:
        for t in load_jsonl(str(f)):
            if t.id not in seen:
                seen.add(t.id)
                trajs.append(t)
    enc = get_encoder(a.encoder, device=a.device)
    print(f"encoding {len(trajs)} trajectories / {sum(len(t.steps) for t in trajs)} steps with {enc.name}")
    encode_corpus(trajs, enc, a.out)
    print("written to", a.out)


def _cmd_train(a):
    from .encoders import EmbeddingCache
    from .experiments import dump, load_processed, parse_groups, train
    from .schema import load_jsonl
    from .training import TrainConfig, save_checkpoint

    D = load_processed(a.data)
    tr = D["train"]
    if a.augment:
        tr = tr + load_jsonl(a.augment)
    if a.limit:
        tr = tr[: a.limit]
    cache = EmbeddingCache(a.cache)
    groups = parse_groups(a.groups)
    cfg = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr, seed=a.seed, patience=a.patience)
    model, hist = train(a.model, tr, D["val"], cache, groups, cfg, {"layers": a.layers})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(a.out, model, a.model, cache.encoder, groups, train_cfg=cfg)
    dump(hist, Path(a.out).with_suffix(".history.json"))
    print("saved", a.out)


def _load(a):
    from .encoders import EmbeddingCache
    from .experiments import load_processed
    from .featurize import ALL_GROUPS
    from .training import load_checkpoint

    model, ck = load_checkpoint(a.checkpoint)
    groups = frozenset(ck.get("groups") or ALL_GROUPS)
    return model, ck, groups, EmbeddingCache(a.cache), load_processed(a.data)


def _cmd_calibrate(a):
    import torch

    from .experiments import calibrate_on, rates, score

    model, ck, groups, cache, D = _load(a)
    if ck["kind"] != "streamguard":
        sys.exit("calibration applies to StreamGuard checkpoints")
    s = score("streamguard", model, D[a.split], cache, groups)
    th, curve = calibrate_on(s, a.alpha)
    th["poison"] = a.poison
    ck["thresholds"] = th
    torch.save(ck, a.checkpoint)
    print(json.dumps({"thresholds": th, "realized_on_calibration_split": rates(s, th)}, indent=2))
    print("risk-coverage:", json.dumps(curve, indent=1))


def _cmd_evaluate(a):
    from .experiments import dump, evaluate, rates, score
    from .metrics import format_report

    model, ck, groups, cache, D = _load(a)
    kind = ck["kind"]
    th = ck.get("thresholds") or {}
    t = a.threshold if a.threshold is not None else th.get("hold", 0.5)
    s = score(kind, model, D[a.split], cache, groups, t_hijack=t, t_poison=th.get("poison", 0.5))
    rep, cis = evaluate(s, threshold=t, reps=a.reps)
    if th:
        rep.update({f"conformal/{k}": v for k, v in rates(s, th).items()})
    md = f"# {kind} on {a.split} (threshold {t:.4f})\n\n" + format_report(rep, cis)
    print(md)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(md)
        dump({"report": rep, "ci": cis}, Path(a.out).with_suffix(".json"))


def _cmd_scan(a):
    from .guard import Guard
    from .io.agentdojo import convert_log
    from .io.agentdrift import record_to_traj
    from .schema import Trajectory

    guard = Guard.load(a.checkpoint, encoder=a.encoder)
    rec = json.loads(Path(a.log).read_text(encoding="utf-8"))
    if "messages" in rec:
        t = convert_log(rec, "agentdojo")
    elif "task" in rec and "agent" in rec:
        t = record_to_traj(rec)
    else:
        t = Trajectory.from_dict(rec)
    rep = guard.scan(t)
    print(rep.to_json() if a.json else rep.to_markdown())


def main(argv=None) -> None:
    p = argparse.ArgumentParser("tracewarden")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("data", help="convert a source dataset into TraceWarden jsonl")
    s.add_argument("source", choices=["agentdrift", "agentdojo"])
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--identity", choices=["keep", "anonymize"], default="anonymize")
    s.add_argument("--shuffle-copies", type=int, default=0)
    s.add_argument("--name", default=None, help="source name for agentdojo, e.g. agentdojo-qwen")
    s.add_argument("--split", default=None)
    s.set_defaults(fn=_cmd_data)

    s = sub.add_parser("stats")
    s.add_argument("--data", required=True)
    s.set_defaults(fn=_cmd_stats)

    s = sub.add_parser("audit", help="shortcut audit (N4)")
    s.add_argument("--data", required=True)
    s.add_argument("--split", default="test")
    s.add_argument("--out", default=None)
    s.set_defaults(fn=_cmd_audit)

    s = sub.add_parser("encode")
    s.add_argument("--data", required=True, help="folder of jsonl files or one jsonl")
    s.add_argument("--encoder", default="sentence-transformers/all-mpnet-base-v2")
    s.add_argument("--device", default=None)
    s.add_argument("--out", required=True)
    s.set_defaults(fn=_cmd_encode)

    s = sub.add_parser("train")
    s.add_argument("--model", choices=["streamguard", "driftnet"], default="streamguard")
    s.add_argument("--data", required=True)
    s.add_argument("--cache", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--groups", default="all", help="all | none | comma list of world,prov,anchor,obs")
    s.add_argument("--augment", default=None, help="extra training jsonl, e.g. train_shuf.jsonl")
    s.add_argument("--epochs", type=int, default=40)
    s.add_argument("--batch-size", type=int, default=32)
    s.add_argument("--lr", type=float, default=3e-4)
    s.add_argument("--layers", type=int, default=3)
    s.add_argument("--patience", type=int, default=6)
    s.add_argument("--seed", type=int, default=42)
    s.add_argument("--limit", type=int, default=0)
    s.set_defaults(fn=_cmd_train)

    for name, fn in (("calibrate", _cmd_calibrate), ("evaluate", _cmd_evaluate)):
        s = sub.add_parser(name)
        s.add_argument("--checkpoint", required=True)
        s.add_argument("--data", required=True)
        s.add_argument("--cache", required=True)
        s.add_argument("--split", default="val" if name == "calibrate" else "test")
        if name == "calibrate":
            s.add_argument("--alpha", type=float, default=0.01)
            s.add_argument("--poison", type=float, default=0.5)
        else:
            s.add_argument("--threshold", type=float, default=None)
            s.add_argument("--reps", type=int, default=500)
            s.add_argument("--out", default=None)
        s.set_defaults(fn=fn)

    s = sub.add_parser("scan", help="analyze one finished trajectory / AgentDojo log")
    s.add_argument("--checkpoint", required=True)
    s.add_argument("--log", required=True)
    s.add_argument("--encoder", default=None)
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=_cmd_scan)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
