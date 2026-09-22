import json

from tracewarden.io import agentdojo
from tracewarden.schema import LABELS, Trajectory, load_jsonl, save_jsonl

from .conftest import FIX


def test_agentdrift_loader(trajs):
    assert len(trajs) == 3
    assert all(s.label in LABELS for t in trajs for s in t.steps)
    att = trajs[0]
    assert att.label_string == "BIHH" and att.compromised == 1
    assert att.injection_indices == [1] and att.hijacked_indices == [2, 3]
    assert att.meta["compliance"] == "full_hijack"
    assert trajs[2].label_string == "FBB" and trajs[2].compromised == 0


def test_jsonl_roundtrip(trajs, tmp_path):
    p = tmp_path / "x.jsonl"
    save_jsonl(trajs, str(p))
    back = load_jsonl(str(p))
    assert [t.to_dict() for t in back] == [t.to_dict() for t in trajs]
    assert isinstance(Trajectory.from_dict(trajs[0].to_dict()), Trajectory)


def test_agentdojo_autolabel():
    rec = json.loads((FIX / "agentdojo_log.json").read_text())
    t = agentdojo.convert_log(rec, "agentdojo-test")
    assert [s.tool for s in t.steps] == ["read_file", "send_money", "send_money"]
    assert t.label_string == "IBH"          # paying the real bill stays benign; the attacker IBAN is hijacked
    assert t.category == "attacked"
    assert "oracle_disagrees" not in t.meta   # agrees with AgentDojo's security verdict


def test_agentdojo_failed_injection():
    rec = json.loads((FIX / "agentdojo_log.json").read_text())
    rec["messages"] = rec["messages"][:6] + [{"role": "assistant", "content": "Done."}]
    rec["security"] = False
    t = agentdojo.convert_log(rec, "agentdojo-test")
    assert t.label_string == "FB" and t.category == "failed_attack"
