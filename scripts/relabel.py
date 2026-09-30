from pathlib import Path

from tracewarden.io.agentdojo import convert_dir
from tracewarden.schema import save_jsonl

for tag in ["qwen", "qwen14b"]:
    for kind, attacked in [("ii", True), ("benign", False)]:
        logs = Path(f"data/agentdojo_raw/logs_{tag}_{kind}")
        if not logs.exists():
            print("missing:", logs)
            continue
        t = convert_dir(str(logs), source=f"agentdojo-{tag}")
        t = [x for x in t if bool(x.meta.get("attack")) == attacked]
        out = f"data/agentdojo_raw/dojo_{tag}_{kind}_v2.jsonl"
        save_jsonl(t, out)
        print(tag, kind, len(t), "->", out)