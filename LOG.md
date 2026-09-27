# TraceWarden — working log

One entry per working day. Three things every time: **what worked**, **what broke**, **what's next**.
Numbers go in the day they are produced, with the exact command that produced them. This file becomes the
method section of the paper and the raw material for the launch post.

Conventions:
- **Split**: AgentDrift task-disjoint (9,081 / 1,733 / 1,722), test evaluated at the calibrated hold
  threshold (0.814) unless stated.
- **Identity**: `anonymize` = placeholder names with internal/external address relations preserved.
- **Encoder**: `hashing-768` = offline bag-of-n-grams; `mpnet` = all-mpnet-base-v2 (768-d, frozen).
- **Trunk size**: CLI defaults to 3 layers; earlier 2-layer runs are marked. Never mix them in one table.

---

## Where the project stands (updated Day 12)

| Phase | Days | Status |
|---|---|---|
| 1 · Foundation | 1–5 | **done** |
| 2 · Novel detector | 6–10 | **done** |
| 3 · Real-world data | 11–15 | **done** — both models collected, labels audited by hand and by rule |
| 4 · Experiments | 16–19 | **transfer done**; adaptation curve, stress sets, latency outstanding |
| 5 · Product | 20–23 | code exists; integrations untested, no release |
| 6 · Paper and launch | 24–25 | all headline numbers in hand; drafting not started |

Roughly **65% complete**. The scientific core is finished: four measured findings, two datasets, and a
result that survived a labeling audit in both error directions.

---

## Day 0–10 — summary (details in earlier revisions of this file)

- Repo, CI, 18 tests; DriftNet reproduced at 0.981 F1 on the anonymized task-disjoint split.
- **Shortcut audit (N4):** world-identity lookup predicts compromise at 85.9% (93–98% in four domains,
  48% in email); after anonymization 55.5%, i.e. the majority rate.
- **Leakage-controlled evaluation:** anonymizing costs 0.0008 F1 — the detector never used the identity
  shortcut. This answers DriftNet's first Future Work item.
- **StreamGuard (N1–N3):** pre-dispatch matches post-hoc on detection (0.981 vs 0.981) and loses ~10 points
  of injection exact match. Provenance + anchoring cut benign false alarms fivefold (1.90% → 0.38%).
- **Conformal (N5):** 1% budget honored at 0.53% realized, 97.6% of attacks caught.
- **Encoder comparison (3 seeds):** hashing 0.981/0.984/0.983 vs mpnet 0.968/0.968/0.948 — hashing wins
  in-distribution, well outside seed noise.

---

## Day 11 — real agent trajectories

Working configuration: `--model VLLM_PARSED --model-id <hf id>`, vLLM with `--enable-auto-tool-choice
--tool-call-parser hermes`, `--max-model-len 8192`, single T4, ready in ~220 s.

**Model A** Qwen2.5-7B-Instruct-AWQ · **Model B** Qwen2.5-14B-Instruct-AWQ.
Llama-3.1-8B was tried first and **does not work with AgentDojo**: its chat template rejects the parallel
tool calls AgentDojo agents emit (`This model only supports single tool-calls at once!`); neither
`llama3_json` nor `pythonic` avoids it. Documented limitation — the cross-model comparison is therefore a
scale comparison within one family, not cross-vendor.

| suite | model A attack success | model B attack success |
|---|---|---|
| banking | 22.9% | 45.1% |
| slack | 54.3% | 53.3% |

**Finding:** the 14B model is twice as easy to hijack in banking while only ~3 points better at the task.
Capability and vulnerability move together. The effect is environment-dependent: Slack is equally
vulnerable for both, and is the most vulnerable environment overall.

**Excluded:** workspace (prompts exceed 8k and then 16k tokens — a T4 cannot serve a larger KV cache
alongside the weights); travel is partial for the same reason.

---

## Day 12 — labeling audit, transfer result, encoder reversal

### Three labeler bugs, all found before they reached a result

1. **Escaped payloads.** Travel tools return a stringified dict, so payload newlines arrive as the two
   characters `\n`. Whitespace normalization missed them, the signature never matched, and an entire
   suite's poison went undetected. Fixed by folding escape sequences before matching.
2. **Named attacker targets.** "Invite user Fred with the email fred9246@gmail.com … add him to the
   general channel" — the email matched, `add_user_to_channel({"user": "Fred"})` did not, because the bare
   name is below the entity threshold. Fixed by harvesting capitalized names within 90 characters of an
   attacker entity (excluding the user's own world and prompt boilerplate).
3. **Digit collisions.** A fallback matched attacker values as substrings of the alphanumeric form of an
   observation, so a transaction list (a wall of digits) could be marked poisoned by coincidence. Fixed by
   restricting the fallback to address-like entities compared entity-to-entity.

### Hand audit (the gold standard)

| dataset | agreement |
|---|---|
| model A (7B) | **96.7%** (29/30) |
| model B (14B) | **86.7%** (26/30) — all four errors were bug 2 |
| multistep (N7) | **100%** (20/20) |

### Bulk audit (`scripts/audit_labels.py`)

Six rule-based consistency checks over a whole dataset, so only flagged trajectories need reading.
`missed_hijack` 56 → 19 and `oracle_says_attacked` 14 → 4 after the fixes. `attacker_value_benign` stays
high (~70) by design: it flags benign steps reusing values that arrived from tool outputs, which is usually
legitimate — and is exactly the ambiguity the N2 provenance features exist to resolve.

### The headline result — synthetic to real (`paper/results/transfer_v3.md`)

| test set | n | traj F1 | macro step F1 | injection EM | blocked before execution | benign interrupted |
|---|---|---|---|---|---|---|
| AgentDrift (synthetic) | 1,722 | **0.984** | 0.957 | 0.883 | 0.985 | 0.5% |
| real, Qwen 7B | 450 | **0.327** | 0.258 | 0.029 | 0.093 | 13.1% |
| real, Qwen 14B | 396 | **0.436** | 0.272 | 0.052 | 0.152 | 19.4% |

Detection falls to a third. **Localization collapses to near zero** (0.883 → 0.03). The conformal guarantee
is violated by 25×.

**Recalibrating on real benign data** restores the promise but not the utility: at a 1% budget the
threshold rises to 0.998 and only **3.6%** of attacks are caught; at a 10% budget, 21.6%. The score
distributions of attacked and benign real trajectories have merged. The conformal machinery behaves
correctly throughout — at very small alpha it returns an infinite threshold rather than guessing.

### Ablation: world features available vs absent

The AgentDojo converter originally left `world` empty, disabling the world and provenance channels.
`world_from_log` now reconstructs known addresses and accounts from the environment's own tool outputs,
excluding anything that appears in a payload. Result: **171/373 trajectories have a non-empty world**
(against 100% in AgentDrift), and transfer F1 moves 0.333 → 0.327 (A) and 0.428 → 0.436 (B) — **no effect**.
The collapse is not caused by disabled features.

### Encoder reversal (`paper/results/transfer_mpnet.md`)

| encoder | AgentDrift | real 7B | real 14B |
|---|---|---|---|
| hashing-768 | **0.984** | 0.327 | 0.436 |
| mpnet | 0.965 | **0.501** | **0.544** |

Hashing wins in-distribution; mpnet wins on real agents by 17 and 11 points of F1. **Benchmark performance
and transfer performance select opposite representations** — tuning on AgentDrift alone picks the encoder
that transfers worst.

**But at matched false-alarm budgets the advantage mostly disappears:**

| budget | hashing catches | mpnet catches |
|---|---|---|
| 1% | 3.6% | 9.4% |
| 5% | 16.5% | 15.9% |
| 10% | 21.6% | 21.7% |

mpnet's F1 advantage is largely a threshold effect: it scores everything higher, which helps at strict
operating points and washes out by 5%. Honest claim: **mpnet transfers better at strict budgets, but
neither representation is usable at any budget an operator would accept.**

### Broke

- **`tracewarden calibrate` silently rewrites thresholds inside the checkpoint.** A calibration run on real
  data left hold = 0.998 in `runs/sg.pt`; the next transfer evaluation used it everywhere and produced a
  plausible table with the synthetic baseline 7 points low (0.918 instead of 0.984). Caught only because
  the synthetic row should never move. **Fix pending:** write to a copy unless `--in-place` is passed, and
  print the threshold in the evaluation table header.
- Real trajectories run to 51 steps against AgentDrift's maximum of 11 — a 5× length shift that is itself
  part of the distribution gap.

---

## The four findings (paper skeleton)

1. Step-level injection detectors trained on synthetic trajectories **do not transfer** to real agents:
   0.984 → 0.33 trajectory F1, measured against hand- and rule-audited labels.
2. **Localization degrades far worse than detection** (injection exact match 0.88 → 0.03): knowing an
   attack happened is easier than knowing where it entered.
3. **Distribution shift breaks the conformal guarantee** (25× budget violation), and recalibration restores
   the promise only by reducing the detector to 3.6% recall.
4. **Representation choice determines transfer, and the benchmark selects the wrong one** — with the
   caveat that the advantage is operating-point dependent.

Supporting: a released real-agent dataset (846 trajectories, both models); the identity-leakage result;
multi-step decomposition reducing attack success (18% vs 31%); model scale increasing vulnerability.

---

## Open items

- [ ] `calibrate` must not mutate checkpoints in place; `transfer_eval` should print its threshold.
- [ ] Regenerate all figures from v3 results; add `figures.py transfer` and `figures.py budget`.
- [ ] Adaptation curve (Day 17): how many real trajectories recover the gap?
- [ ] Multilingual / obfuscated stress sets (Day 18); ONNX latency (Day 19).
- [ ] Dataset card + HF release for AgentDojo-Steps, with both audit numbers.
- [ ] Check arXiv for DriftNet follow-ups (last checked: Day 1).

## Next actions

1. Commit Day 12 work; regenerate figures from v3 and mpnet results.
2. Draft the paper — every headline number is in hand.
3. Adaptation curve, then the stress sets if time allows.

---

## Template — copy for each new day

## Day N — <title>

**Command(s)**
```
```

**Worked** · **Broke** · **Numbers** · **Reading** (what claim this does and does not support) · **Next**
