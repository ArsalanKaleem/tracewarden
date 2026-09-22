"""Train a tiny StreamGuard on the fixtures and run a live guarded session end to end."""
import json

from tracewarden.encoders import EmbeddingCache, HashingEncoder, encode_corpus
from tracewarden.experiments import train
from tracewarden.guard import Guard
from tracewarden.integrations.openai_loop import guarded_dispatch
from tracewarden.training import TrainConfig, save_checkpoint


def test_guard_end_to_end(trajs, tmp_path):
    enc = HashingEncoder(64)
    encode_corpus(trajs, enc, tmp_path / "cache")
    cache = EmbeddingCache(tmp_path / "cache")
    cfg = TrainConfig(epochs=60, batch_size=3, lr=3e-3, patience=60, verbose=False, warmup=0.0)
    model, _ = train("streamguard", trajs * 4, trajs, cache, cfg=cfg, model_kw=dict(d=32, layers=1, heads=2, ff=64))
    ck = tmp_path / "g.pt"
    save_checkpoint(str(ck), model, "streamguard", enc.name, thresholds={"hold": 0.5, "block": 0.9, "poison": 0.5})

    g = Guard.load(str(ck))
    att = trajs[0]
    sess = g.session(att.user_task, att.world, "t")
    decisions = []
    for s in att.steps:
        d = sess.check(s.tool, s.args, s.thought)
        decisions.append(d)
        (sess.observe(s.obs) if d.allowed else sess.skip())
    assert decisions[0].allowed and decisions[1].allowed
    assert decisions[2].action in ("hold", "block")          # the hijacked call is stopped before it runs
    assert "audit@ext-audit.com" in decisions[2].tainted_values
    rep = sess.report()
    assert rep.verdict == "compromised" and rep.hijacked_steps[0] == 2
    assert json.loads(rep.to_json())["trajectory_id"] == "t"

    # the OpenAI-loop adapter on the benign trajectory runs every tool
    ben = trajs[1]
    s2 = g.session(ben.user_task, ben.world)
    outs = [guarded_dispatch(s2, st.tool, st.args, lambda n, a, o=st.obs: o) for st in ben.steps]
    assert outs == [st.obs for st in ben.steps]
