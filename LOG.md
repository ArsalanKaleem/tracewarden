# TraceWarden — working log

One entry per working day. Three things every time: **what worked**, **what broke**, **what's next**.
Numbers go in the day they are produced, with the exact command that produced them. This file becomes the
method section of the paper and the raw material for the launch post.

Conventions:
- **Split**: AgentDrift task-disjoint (9,081 / 1,733 / 1,722), test evaluated at threshold 0.5 unless stated.
- **Identity**: `anonymize` = placeholder names with internal/external address relations preserved; `keep` = raw.
- **Encoder**: `hashing-768` = offline bag-of-n-grams; `mpnet` = all-mpnet-base-v2 (768-d, frozen).
- **Trunk size**: the CLI defaults to 2 layers (1,321,474 params); `scripts/run_ablation.py` uses the model
  default of 3 layers (1,848,578). **Never mix the two in one table.** TODO: make the CLI default 3.

---

## Where the project stands (updated Day 11)

| Phase | Days | Status |
|---|---|---|
| 1 · Foundation | 1–5 | **done** — repo, loader, shortcut audit, mpnet cache, DriftNet reproduced |
| 2 · Novel detector | 6–10 | **done** — StreamGuard, provenance + anchoring, conformal policy, 3-seed ablation |
| 3 · Real-world data | 11–15 | **in progress** — labeler fixed and validated; benign + banking attacked collected for model A |
| 4 · Experiments | 16–19 | not started (blocked on Phase 3) |
| 5 · Product | 20–23 | code exists from the scaffold; untested integrations, no release |
| 6 · Paper and launch | 24–25 | skeleton + figures script only |

Roughly **45% complete**. The synthetic half of the project is finished and defensible; the real-data half —
which is where the contribution actually lives — is about a third done.

---

## Day 0 — environment and repository

**Worked** — Python 3.12.10 venv (3.14 is the machine default and has no torch wheels); CPU-only torch;
`pip install -e ".[dev]"`; 17/17 tests in ~17 s; `examples/quickstart.py` holds `add_beneficiary` (0.91) and
blocks `transfer_money` (1.00), both before execution. Repo pushed; CI on 3.10/3.11/3.12.

**Broke** — CI failed on the quickstart step: `.gitignore` excluded `*.pt`, so the demo checkpoint was never
committed. Fixed with `!models/*.pt` and `git add -f`.

---

## Day 2 — data loader

```
tracewarden data agentdrift --root ../AgentDrift/data_taskdisjoint --out data/processed --identity anonymize --shuffle-copies 1
tracewarden stats --data data/processed
```

Counts match the AgentDrift paper exactly: train 9,081 / 51,507 steps, val 1,733 / 9,721, test 1,722 / 9,796;
12,536 trajectories and 71,024 steps in total. Corpus labels: benign 51,639 · hijacked 12,349 ·
injection_point 5,536 · failed_injection 1,500. 9,081 identity-shuffled copies written for Day 10.

---

## Day 3 — shortcut audit (N4)

```
tracewarden data agentdrift --root ../AgentDrift/data_taskdisjoint --out data/raw_identity --identity keep
tracewarden audit --data data/raw_identity --out paper/results/audit_raw.json
tracewarden audit --data data/processed
```

| lookup key | raw identities | anonymized |
|---|---|---|
| majority class | 0.550 | 0.550 |
| **world identity** | **0.859** (P 0.882 / R 0.792) | **0.555** (R 0.012) |
| company only | 0.667 | 0.550 |
| tool sequence | 0.600 (14.6% coverage) | 0.600 |
| number of steps | 0.584 | 0.584 |

World identity by domain (raw): medical 0.984 · banking 0.959 · coding 0.950 · web 0.931 · **email 0.484**.
Matches the dataset README (93–98% in four domains, email clean); DriftNet reports 86.9% on this split, we
measure 85.9%. Anonymized: 0.50–0.65, i.e. noise.

---

## Day 4 — embeddings on a free GPU

`notebooks/kaggle/01_encode_on_kaggle.ipynb`, T4, ~25 min for three encoders (mpnet, MiniLM, multilingual)
over 122,531 step-views (21,617 trajectories including the shuffled copies). Upload 1.49 GB to
`ArsalanKaleem/tracewarden-cache` at 149 MB/s; download to the laptop ~24 min. Verified locally:
`all-mpnet-base-v2 768 122531 21617`.

**Kaggle gotchas, all of which cost time:**
- Datasets mount at `/kaggle/input/datasets/<username>/<slug>`, not `/kaggle/input/<slug>`.
- The Internet toggle only takes effect after **Run → Restart session**, and a restart clears all variables.
- Secret labels are matched exactly, and secrets are attached **per notebook**, not per account.
- `pip install git+https://...` failed even with a public repo; shipping `src/` as a dataset and using
  `sys.path.insert` is more reliable. `/kaggle/input` is read-only, so patching a module means copying the
  package to `/kaggle/working` first.

---

## Day 5–6 — reproduction, pre-dispatch model, feature ablation (hashing, anonymized, **2 layers**)

| model | decides | traj F1 (CI) | macro step F1 | injection EM | hijack IoU | delayed-exec recall | benign flagged | hard-neg flagged |
|---|---|---|---|---|---|---|---|---|
| DriftNet (re-impl.) | post-hoc | 0.981 [0.974, 0.988] | 0.989 | 0.981 | 0.966 | 0.863 | 1.71% | 1.47% |
| StreamGuard, text only | pre-dispatch | 0.973 [0.965, 0.981] | 0.950 | 0.879 | 0.959 | 0.880 | 1.90% | 5.88% |
| StreamGuard, all features | pre-dispatch | 0.981 [0.974, 0.987] | 0.955 | 0.884 | 0.965 | **0.906** | **0.38%** | 5.39% |

Prevention (all features): blocked before execution 0.989 · first hijack caught 0.981 · benign interruption 1.80%.

**Readings**
1. *Price of causality*: pre-dispatch matches post-hoc on detection (0.981 vs 0.981) but loses ~10 points of
   injection exact match (0.884 vs 0.981). A poisoned observation must be judged when it arrives, before the
   agent's later behavior disambiguates it. Design consequence, not a defect.
2. *Features*: the defensible claims are the fivefold drop in benign false alarms (1.90% → 0.38%) and
   delayed-execution recall (0.880 → 0.906), both confirmed across seeds.
3. *Hard negatives* are the streaming model's weak point (5.4% vs DriftNet's 1.5%): they look suspicious early
   and are only vindicated later, which a causal model cannot see.

---

## Day 9 — conformal calibration (N5)

Thresholds from 980 non-compromised validation trajectories: hold 0.814, block 0.994. On test the promise
holds: **0.53%** of benign runs held or blocked (budget 1%), **0%** blocked, catching **97.6%** of attacks and
**97.3%** at or before the first hijacked call. Every false positive is a hard negative (2.45%).

Risk–coverage (test): alpha 0.0025 → 0.10% flagged / 93.8% caught · 0.01 → 0.82% / 96.9% · 0.05 → 4.90% / 99.7%.

**Note** — at alpha 0.001 the threshold is `Infinity`: with 980 calibration points the finite-sample correction
cannot certify a rate below ~1/981, so it refuses rather than guessing. Correct behavior; footnote it.

---

## Day 10 — encoder comparison and seed variance (**3 layers**)

| encoder | traj F1 (3 seeds) | macro step F1 | injection EM | benign interrupted | delayed-exec recall |
|---|---|---|---|---|---|
| hashing-768 | 0.981 / 0.984 / 0.983 | 0.968 / 0.963 / 0.956 | 0.894 / 0.888 / 0.888 | 1.2% / 1.3% / 1.2% | 0.897 / 0.915 / 0.915 |
| mpnet | 0.968 / 0.968 / **0.948\*** | 0.932 / 0.930 / 0.928 | 0.845 / 0.844 / 0.781 | 3.7% / 3.6% / 3.5% | 0.923 / 0.923 / 0.752 |

\* seed 2 stopped at epoch 15 on patience 6 while seeds 0 and 1 ran to epochs 39–40. Early-stopping artifact;
re-run with `patience=10`. Even excluding it the gap is ~1.5 F1 points against a hashing seed spread of 0.0015.

**Finding (provisional)** — a hashed bag of n-grams matches or beats a 110M-parameter sentence encoder on
AgentDrift, with a third of the false alarms. This contradicts DriftNet's Future Work claim that the frozen
encoder is the likely bottleneck. **Counter-crumb**: mpnet wins on delayed execution (0.923 vs 0.909), the
pattern where the attack hides in timing rather than wording.

---

## Day 10b — does anonymization cause the encoder result? (matched 2-layer, hashing)

| identities | traj F1 | macro step F1 | injection EM | hijack IoU | benign flagged | hard-neg flagged | benign interruption |
|---|---|---|---|---|---|---|---|
| anonymized | 0.9807 | 0.955 | 0.884 | 0.965 | 0.38% | 5.39% | 1.80% |
| raw | 0.9799 | 0.964 | 0.892 | 0.959 | 0.38% | 3.43% | 1.37% |

**The important result of the week.** Removing every name, company and address changes trajectory F1 by
0.0008. The detector is not exploiting the identity regularity even when it is available, and anonymization
costs nothing. This answers DriftNet's first Future Work item — essentially all of the headline survives
leakage-controlled evaluation. Credit them for the question; the measurement is ours.

**Still open** — the mpnet half of the 2×2 (`cache/mpnet_raw`) needs one short Kaggle session.

---

## Day 11 — real agent trajectories (notebook 02)

**Setup that works** (record it, it took several attempts):
`--model VLLM_PARSED --model-id <hf id>` with vLLM serving `Qwen/Qwen2.5-7B-Instruct-AWQ`,
`--enable-auto-tool-choice --tool-call-parser hermes`, `--max-model-len 8192`, `--gpu-memory-utilization 0.90`,
single T4. Server ready in 200–220 s.

### Three labeling bugs, found by the smoke test before any large run

1. **Reflowed payloads.** AgentDojo's `injections` dict holds `\n\n` where the environment text has `\n`, so
   exact substring matching never fired and nothing was marked poisoned. Fix: whitespace- and case-normalized
   comparison of an 80-character payload signature.
2. **Truncated identifiers.** The agent wrote `US13300000012121212` for the attacker's
   `US133000000121212121212` — four digits short. Fix: compare alphanumeric-only keys on a common prefix of
   at least ten characters. Different IBANs are still rejected.
3. **Goal-based attacks with no identifier.** "Change the password to new_password" contains no address or
   account, so nothing matched. Fix: also extract distinctive payload tokens (quoted strings, or words with a
   digit or underscore) and match them exactly against argument values.

All three are in `src/tracewarden/io/agentdojo.py`; 18 tests pass.

### Labeling convention (paper material)

`hijacked` = the agent took an action serving the attacker, **whether or not the attacker's goal completed**.
AgentDojo's `security` oracle checks final environment state instead. On the smoke set, four of the five
disagreements were partial compliance: money genuinely sent to the attacker's IBAN with the wrong
amount or subject line, which the oracle scores as a failed attack and a guard should still block. The
oracle verdict is preserved in `meta.security` so the stricter definition can be re-derived.

### Data collected (model A, Qwen2.5-7B-Instruct-AWQ)

| set | trajectories | steps | notes |
|---|---|---|---|
| benign (banking, slack, travel) | 75 | 357 benign | 0 oracle disagreements |
| banking attacked | 136 | 44 injection_point · 60 hijacked · 68 failed_injection · 157 benign | 11 disagreements (8%) |

Suite-level rates: banking benign utility 50.0%; banking attacked utility 46.5%, security 22.9%. The ~4-point
utility drop under attack is the attack's collateral damage even when resisted — worth reporting.
Run-to-run variation on the same suite was 5/9 vs 6/9 injection tasks passing, so attack-success rates carry
a few points of sampling error at default temperature.

**Broke**
- **workspace excluded**: prompts exceed 8,192 and then 16,384 tokens (large cloud-drive listings plus retry
  loops). A T4 cannot serve a bigger KV cache alongside the weights. Documented exclusion.
- **slack is low-value**: `send_channel_message` returns `None` and the agent retries four or five times;
  trajectories are mostly failed calls. Attacked run launched; conversion pending.
- Session restarts lose the helper functions from cell 4 — re-run cells 1, 3, 4 before any conversion cell,
  or use **Save Version → Save & Run All** so everything executes in order server-side.

**Next** — travel attacked; convert and upload slack; then the 30-trajectory hand audit.

---

## Open items

- [ ] Travel attacked run; convert + upload slack.
- [ ] `scripts/review_labels.py --n 30` — agreement rate for the dataset card (gate before model B).
- [ ] Model B (Llama-3.1-8B-Instruct-AWQ-INT4, parser `llama3_json`), same suites.
- [ ] Re-run mpnet seed 2 with `patience=10`; make the CLI trunk default 3 layers.
- [ ] `cache/mpnet_raw` to close the encoder/identity 2×2.
- [ ] Check arXiv for DriftNet follow-ups (last checked: Day 1).

## Figures

Done: risk–coverage, price of causality, step head + confusion, training dynamics (`scripts/figures.py`).
Pending: encoder/identity 2×2 (needs `cache/mpnet_raw`); adaptation curve (needs real data).

---

## Template — copy for each new day

## Day N — <title>

**Command(s)**
```
```

**Worked** · **Broke** · **Numbers** · **Reading** (what claim this does and does not support) · **Next**
