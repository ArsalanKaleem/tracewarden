"""Guard a live agent run step by step.  python examples/quickstart.py

Uses the bundled DEMO model (offline hashing encoder, trained on anonymized AgentDrift). For real use,
train with all-mpnet-base-v2 embeddings (see README) and load that checkpoint instead.
"""
import json
from pathlib import Path

from tracewarden import Guard

HERE = Path(__file__).parent
guard = Guard.load(str(HERE.parent / "models" / "tracewarden-demo-hashing.pt"))
rec = json.loads((HERE / "trajectories" / "attacked_bank_transfer.json").read_text())

session = guard.session(user_task=rec["task"], world=rec["world"])
for step in rec["steps"]:                       # replay what the agent proposed, one call at a time
    d = session.check(step["tool"], step["args"], step["thought"])
    print(f"{step['tool']:18s} -> {d.action:5s} (score {d.score:.2f})  {d.reason}")
    if d.allowed:
        session.observe(step["obs"])            # the tool ran; the guard sees its output
    else:
        session.skip()                          # held/blocked: the tool never ran

print()
print(session.report().to_markdown())
