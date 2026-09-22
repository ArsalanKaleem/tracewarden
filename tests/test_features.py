import copy

import numpy as np

from tracewarden.features.anonymize import anonymize, identity_shuffle, world_fingerprint
from tracewarden.features.normalize import normalize
from tracewarden.features.provenance import extract_values, provenance_feats
from tracewarden.features.world import known_sets, world_feats
from tracewarden.featurize import build_events


def test_world_feats(trajs):
    known, dom = known_sets(trajs[0].world)
    assert "acme.com" in dom
    assert world_feats({"to": "audit@ext-audit.com"}, known, dom) == [1.0, 1.0, 1.0, 0.0]
    assert world_feats({"to": "omar.siddiqui@acme.com"}, known, dom) == [1.0, 0.0, 0.0, 0.0]


def test_provenance_taint(trajs):
    t = trajs[0]
    prior = [s.obs for s in t.steps[:2]]
    f = provenance_feats(t.steps[2].args, t.user_task, t.world, prior)
    assert f[4] == 1.0            # any_tainted: the address came from a tool output
    benign = trajs[1]
    f2 = provenance_feats(benign.steps[1].args, benign.user_task, benign.world, [benign.steps[0].obs])
    assert f2[4] == 0.0           # known contact -> not tainted


def test_extract_values():
    v = extract_values({"a": ["x@y.com", {"b": "https://evil.io/p"}], "n": 1234567, "flag": True})
    assert "x@y.com" in v and "https://evil.io/p" in v and "1234567" in v


def test_normalize_defeats_obfuscation():
    assert normalize("ign\u200bore prev\u0456ous") .startswith("ignore")
    assert normalize("\u0441\u043epy") == "copy"


def test_no_future_leakage(trajs, encoder):
    """Features of step t must not change when later steps are altered (pre-dispatch validity)."""
    from tracewarden.features import serialize as S

    t = trajs[0]
    def items(tr):
        ea = encoder.encode([S.action(s) for s in tr.steps])
        eo = encoder.encode([S.obs(s) for s in tr.steps])
        return build_events(tr, ea, eo, encoder.encode([S.task(tr)])[0])
    a = items(t)
    t2 = copy.deepcopy(t)
    t2.steps[3].obs = "totally different output with new@addr.com"
    t2.steps[3].args = {"to": "someone@else.org"}
    b = items(t2)
    upto = 2 * 3  # events of steps 0..2
    np.testing.assert_array_equal(a.x[:upto], b.x[:upto])
    assert a.x.shape[0] == 2 * len(t.steps)


def test_anonymize_keeps_relations(trajs):
    t = trajs[0]
    a = anonymize(t)
    assert a.world["email"].endswith("@org1.example")
    hij = a.steps[2].args["email"]
    assert hij.endswith(".example") and not hij.endswith("org1.example")   # external stays external
    assert "Lena" not in a.to_json() and "acme" not in a.to_json().lower()
    s1, s2 = identity_shuffle(t, 1), identity_shuffle(t, 2)
    assert world_fingerprint(s1) != world_fingerprint(s2)
    known, dom = known_sets(s1.world)
    assert s1.steps[3].args["to"].split("@")[1] not in dom            # external stays external
    assert s1.steps[0].label == t.steps[0].label
