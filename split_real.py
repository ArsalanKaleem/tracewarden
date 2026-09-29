# split_real.py
import random
from tracewarden.schema import load_jsonl, save_jsonl
from pathlib import Path

for name in ["real_qwen", "real_qwen14b"]:
    T = load_jsonl(f"data/{name}/test.jsonl")
    random.Random(0).shuffle(T)
    n = len(T)
    parts = {"train": T[:int(.5*n)], "val": T[int(.5*n):int(.65*n)], "test": T[int(.65*n):]}
    d = Path(f"data/{name}_split"); d.mkdir(exist_ok=True)
    for k, v in parts.items():
        save_jsonl(v, str(d / f"{k}.jsonl"))
    print(name, {k: len(v) for k, v in parts.items()})