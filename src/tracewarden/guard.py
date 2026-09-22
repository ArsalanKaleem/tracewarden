"""The runtime API.

    from tracewarden import Guard
    guard = Guard.load("tracewarden-core.pt")
    sess = guard.session(user_task="Pay the March invoice", world={"email": "me@acme.com", "contacts": [...]})
    sess.observe(...)                                   # optional: tool outputs the agent saw before any call
    d = sess.check("send_money", {"iban": "...", "amount": 5000})   # BEFORE executing
    if d.action == "allow":
        result = run_tool(...)
        sess.observe(result)                            # after executing
    report = sess.report()                              # rollback / incident report
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import torch

from .calibrate import decide
from .encoders import Encoder, get_encoder
from .features import serialize as S
from .features.provenance import RX, extract_values
from .featurize import ACTION, ALL_GROUPS, OBS, action_extra, build_events, obs_extra
from .report import Report, build_report
from .schema import Step, Trajectory
from .training import decode_events, load_checkpoint, predict_events

DEFAULT_THRESHOLDS = {"hold": 0.5, "block": 0.85, "poison": 0.5}
NOT_EXECUTED = "[not executed: stopped by TraceWarden]"
ADDRESS = re.compile("|".join(RX[k].pattern for k in ("email", "url", "iban", "phone", "domain")))


@dataclass
class Decision:
    action: str                       # allow | hold | block
    score: float                      # hijack probability of the proposed call
    step: int                         # index of the proposed call in the session
    reason: str
    injection_sources: list[int] = field(default_factory=list)   # earlier steps whose outputs look poisoned
    tainted_values: list[str] = field(default_factory=list)      # argument values introduced by tool outputs

    @property
    def allowed(self) -> bool:
        return self.action == "allow"


class Guard:
    def __init__(self, model, encoder: Encoder, thresholds: dict | None = None, groups=ALL_GROUPS,
                 device: str = "cpu", taint_policy: str | None = None):
        """taint_policy: None | "hold" | "block". Optional deterministic layer on top of the learned model:
        any call whose arguments contain an entity (address, URL, account...) that only an untrusted tool
        output introduced is escalated to at least this action. Defense in depth for out-of-distribution
        attacks; costs some false holds when agents legitimately reuse values they read."""
        if taint_policy not in (None, "hold", "block"):
            raise ValueError("taint_policy must be None, 'hold' or 'block'")
        self.taint_policy = taint_policy
        self.model = model.eval().to(device)
        self.encoder = encoder
        self.thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.groups = frozenset(groups)
        self.device = device

    @classmethod
    def load(cls, checkpoint: str, encoder: str | Encoder | None = None, device: str = "cpu",
             thresholds: dict | None = None, taint_policy: str | None = None) -> Guard:
        model, ck = load_checkpoint(checkpoint, device)
        if ck["kind"] != "streamguard":
            raise ValueError("Guard needs a StreamGuard checkpoint (pre-dispatch model)")
        enc = encoder if encoder is not None and not isinstance(encoder, str) else get_encoder(encoder or ck["encoder"])
        if enc.dim + 15 != ck["config"]["d_in"]:
            raise ValueError(f"encoder dim {enc.dim} does not match checkpoint input {ck['config']['d_in']}")
        return cls(model, enc, {**ck.get("thresholds", {}), **(thresholds or {})},
                   ck.get("groups") or ALL_GROUPS, device, taint_policy)

    def session(self, user_task: str, world: dict | None = None, session_id: str = "session") -> Session:
        return Session(self, user_task, world or {}, session_id)

    # ---------------------------------------------------------------- post-hoc analysis of a finished log
    def scan(self, t: Trajectory) -> Report:
        if not t.steps:
            return build_report(t, [], [], [])
        ea = self.encoder.encode([S.action(s) for s in t.steps])
        eo = self.encoder.encode([S.obs(s) for s in t.steps])
        et = self.encoder.encode([S.task(t)])[0]
        item = build_events(t, ea, eo, et, self.groups)
        (h, p), = predict_events(self.model, [item], device=self.device)
        labels = decode_events(h, p, self.thresholds["hold"], self.thresholds["poison"])
        executed = [s.obs != NOT_EXECUTED for s in t.steps]
        return build_report(t, labels.tolist(), h.tolist(), p.tolist(), executed)


class Session:
    """One live agent run. Call check() before every tool call and observe() after it runs."""

    def __init__(self, guard: Guard, user_task: str, world: dict, session_id: str):
        self.g = guard
        self.traj = Trajectory(id=session_id, steps=[], user_task=user_task, world=world, source="runtime")
        self.e_task = guard.encoder.encode([S.task(self.traj)])[0]
        self.rows: list[np.ndarray] = []
        self.etypes: list[int] = []
        self.a_embs: list[np.ndarray] = []
        self.o_embs: list[np.ndarray] = []
        self.decisions: list[Decision] = []
        self._pending = False
        self._last_poison: np.ndarray = np.zeros(0)

    # ------------------------------------------------------------------ internals
    def _score(self) -> tuple[float, np.ndarray]:
        x = torch.from_numpy(np.stack(self.rows)).float().unsqueeze(0).to(self.g.device)
        et = torch.tensor([self.etypes], dtype=torch.long, device=self.g.device)
        m = torch.ones_like(et, dtype=torch.bool)
        with torch.no_grad():
            lh, lp = self.g.model(x, et, m)
        ph, pp = torch.sigmoid(lh)[0].cpu().numpy(), torch.sigmoid(lp)[0].cpu().numpy()
        obs_idx = [i for i, e in enumerate(self.etypes) if e == OBS]
        return float(ph[-1]), pp[obs_idx]

    def _close_pending(self, output: str) -> None:
        s = self.traj.steps[-1]
        s.obs = "" if output is None else str(output)
        e = self.g.encoder.encode([S.obs(s)])[0]
        self.rows.append(np.concatenate([e, obs_extra(s.obs, self.traj.user_task, self.traj.world, self.g.groups)]))
        self.etypes.append(OBS)
        self.o_embs.append(e)
        self._pending = False

    # ------------------------------------------------------------------ public API
    def check(self, tool: str, args: dict | None = None, thought: str = "") -> Decision:
        """Score a PROPOSED tool call before it executes."""
        if self._pending:  # previous call never reported back: record it as not executed
            self._close_pending(NOT_EXECUTED)
        args = dict(args or {})
        step = Step(tool=tool, args=args, thought=thought or "")
        prior_obs = [s.obs for s in self.traj.steps]
        e = self.g.encoder.encode([S.action(step)])[0]
        ex = action_extra(args, self.traj.user_task, self.traj.world, prior_obs, self.e_task, e,
                          self.a_embs, self.o_embs, self.g.groups)
        self.traj.steps.append(step)
        self.rows.append(np.concatenate([e, ex]))
        self.etypes.append(ACTION)
        self.a_embs.append(e)
        self._pending = True
        score, poison = self._score()
        self._last_poison = poison
        verdict = decide(score, self.g.thresholds)
        sources = [i for i, p in enumerate(poison) if p >= self.g.thresholds["poison"]]
        tainted = self._tainted(args, prior_obs)
        reason = self._reason(verdict, score, sources, tainted)
        rank = {"allow": 0, "hold": 1, "block": 2}
        addr = [v for v in tainted if ADDRESS.search(v)]  # destinations only: emails, URLs, domains, IBANs, phones
        if self.g.taint_policy and addr and rank[verdict] < rank[self.g.taint_policy]:
            verdict = self.g.taint_policy
            reason = f"taint policy: destination(s) {', '.join(addr[:3])} came only from tool output; " + reason
        d = Decision(verdict, score, len(self.traj.steps) - 1, reason, sources, tainted)
        self.decisions.append(d)
        return d

    def observe(self, output: str) -> None:
        """Report the output of the call that was last checked (and executed)."""
        if not self._pending:
            raise RuntimeError("observe() must follow check() for the same tool call")
        self._close_pending(output)

    def skip(self) -> None:
        """The last checked call was not executed (held or blocked)."""
        if self._pending:
            self._close_pending(NOT_EXECUTED)

    def report(self) -> Report:
        if self._pending:
            self._close_pending(NOT_EXECUTED)
        return self.g.scan(self.traj)

    # ------------------------------------------------------------------ evidence
    def _tainted(self, args: dict, prior_obs: list[str]) -> list[str]:
        ctx = (self.traj.user_task + " " + str(self.traj.world)).lower()
        obs = [o.lower() for o in prior_obs]
        return [v for v in extract_values(args, entities_only=True) if v not in ctx and any(v in o for o in obs)]

    @staticmethod
    def _reason(verdict, score, sources, tainted) -> str:
        if verdict == "allow":
            return f"no hijack evidence (score {score:.2f})"
        bits = [f"hijack score {score:.2f}"]
        if tainted:
            bits.append("arguments contain values introduced by tool output: " + ", ".join(tainted[:3]))
        if sources:
            bits.append(f"suspected injection in output of step(s) {sources}")
        return "; ".join(bits)
