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

## Day 0 — environment and repository

**Worked** — Python 3.12.10 venv (3.14 is the machine default and has no torch wheels); CPU-only torch;
`pip install -e ".[dev]"`; 17/17 tests in ~17 s; `examples/quickstart.py` holds `add_beneficiary` (0.91) and
blocks `transfer_money` (1.00), both before execution. Repo pushed; CI on 3.10/3.11/3.12.

**Broke** — CI failed on the quickstart step: `.gitignore` excluded `*.pt`, so the demo checkpoint was never
committed. Fixed with `!models/*.pt` and `git add -f`.

**Next** — replace remaining `<you>` placeholders; confirm the name is free on PyPI and HF.

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
- Secret labels are matched exactly (`tracewarden-token-huggingface`).
- `pip install git+https://...` failed even with a public repo; shipping `src/` as a second dataset
  (`tw-src`) and using `sys.path.insert` is more reliable.

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
2. *Features*: the F1 difference is inside the CIs on one seed; the defensible claims are the fivefold drop in
   benign false alarms (1.90% → 0.38%) and delayed-execution recall (0.880 → 0.906). Confirmed by seeds below.
3. *Hard negatives* are the streaming model's weak point (5.4% vs DriftNet's 1.5%): they look suspicious early
   and are only vindicated later, which a causal model cannot see.

---

## Day 9 — conformal calibration (N5)

```
tracewarden calibrate --checkpoint runs/sg.pt --data data/processed --cache cache/hashing --alpha 0.01
```

Thresholds from 980 non-compromised validation trajectories: hold 0.814, block 0.994. On test the promise
holds: **0.53%** of benign runs held or blocked (budget 1%), **0%** blocked, catching **97.6%** of attacks and
**97.3%** at or before the first hijacked call. Zero false alarms on benign and resisted attacks; every false
positive is a hard negative (2.45%).

Risk–coverage (test): alpha 0.0025 → 0.10% flagged / 93.8% caught · 0.01 → 0.82% / 96.9% · 0.05 → 4.90% / 99.7%.

**Note** — at alpha 0.001 the threshold is `Infinity`: with 980 calibration points the finite-sample correction
cannot certify a rate below ~1/981, so it refuses rather than guessing. Correct behavior; footnote it.

---

## Day 10 — encoder comparison and seed variance (**3 layers**, `run_ablation.py --only 5 --seeds 3`)

| encoder | traj F1 (3 seeds) | macro step F1 | injection EM | benign interrupted | delayed-exec recall |
|---|---|---|---|---|---|
| hashing-768 | 0.981 / 0.984 / 0.983 | 0.968 / 0.963 / 0.956 | 0.894 / 0.888 / 0.888 | 1.2% / 1.3% / 1.2% | 0.897 / 0.915 / 0.915 |
| mpnet | 0.968 / 0.968 / **0.948\*** | 0.932 / 0.930 / 0.928 | 0.845 / 0.844 / 0.781 | 3.7% / 3.6% / 3.5% | 0.923 / 0.923 / 0.752 |

\* seed 2 stopped at epoch 15 on patience 6 while seeds 0 and 1 ran to epochs 39–40. **Early-stopping artifact,
not a result** — re-run with `patience=10` before reporting. Even excluding it, the gap is ~1.5 points of F1
against a hashing seed spread of 0.0015, so it is far outside seed noise.

Single-run mpnet checks: 15 epochs → 0.965 F1 / 0.795 EM; 40 epochs → 0.969 / 0.861. Converged (training loss
0.023, validation plateaued from epoch 27), so undertraining is ruled out.

**Finding (provisional)** — a hashed bag of n-grams matches or beats a 110M-parameter sentence encoder on
AgentDrift, with a third of the false alarms. This contradicts DriftNet's Future Work claim that the frozen
encoder is the likely bottleneck and that fine-tuning it is "the most direct route to gains".

**Counter-crumb** — mpnet wins on delayed execution (0.923 vs 0.909 in both good seeds), the one pattern where
the attack hides in timing rather than wording. Plausible story: surface cues carry the easy cases, semantics
help on the stealthy ones.

---

## Day 10b — does anonymization cause the encoder result? (matched 2-layer comparison, hashing)

```
tracewarden encode --data data/raw_identity --encoder hashing-768 --out cache/hashing_raw
tracewarden train --model streamguard --data data/raw_identity --cache cache/hashing_raw --out runs/sg_raw_hash.pt --epochs 40 --patience 10
tracewarden evaluate --checkpoint runs/sg_raw_hash.pt --data data/raw_identity --cache cache/hashing_raw --threshold 0.5
```

| identities | traj F1 | macro step F1 | injection EM | hijack IoU | benign flagged | hard-neg flagged | benign interruption |
|---|---|---|---|---|---|---|---|
| anonymized | 0.9807 | 0.955 | 0.884 | 0.965 | 0.38% | 5.39% | 1.80% |
| raw | 0.9799 | 0.964 | 0.892 | 0.959 | 0.38% | 3.43% | 1.37% |

**This is the important result of the week.** Removing every name, company and address changes trajectory F1 by
0.0008 — nothing. The detector is **not** exploiting the identity regularity even when it is available, and
anonymization costs nothing. This answers DriftNet's first Future Work item ("leakage-controlled evaluation
would bound how much of the trajectory-level headline survives"): essentially all of it survives. Credit them
for the question; the measurement is ours.

Raw identities do help slightly on hard negatives (3.4% vs 5.4% flagged), which makes sense: knowing a contact
is real makes a suspicious-looking but legitimate action easier to accept.

**Still open** — the mpnet half of the 2×2 (`cache/mpnet_raw`) needs one short Kaggle session. If the
hashing-over-mpnet gap persists on raw identities, the finding is about the benchmark; if it closes,
anonymization interacts with the encoder and the claim must be narrowed.

---

## Figures to produce (paper/figures/)

Nothing was plotted until now because every number was provisional. These are stable enough to draw:

1. **Risk–coverage curve** (data in hand): attacks caught vs benign flagged, log x-axis —
   `python scripts/make_figures.py --checkpoint runs/sg.pt --data data/processed --cache cache/hashing`.
2. **Post-hoc vs pre-dispatch** (data in hand): the price-of-causality figure — equal trajectory F1, lower
   injection exact match.
3. **Encoder / identity 2×2** (needs `cache/mpnet_raw`): grouped bars, traj F1 and benign interruption.
4. **Adaptation curve** (Day 17, after real data): real trajectories vs recovered performance.

---

## Open items

- [ ] Re-run mpnet seed 2 with `patience=10`; report the corrected seed table.
- [ ] Make the CLI trunk default 3 layers so all runs are comparable.
- [ ] `cache/mpnet_raw` on Kaggle to close the 2×2.
- [ ] Check arXiv for DriftNet follow-ups (last checked: Day 1).

## Next actions (in order)

1. Figures 1 and 2 from existing results; commit to `paper/figures/`.
2. Notebook 02: vLLM + AgentDojo, model A. Run `--help` on the installed AgentDojo first; smoke-test one suite
   and hand-read three converted trajectories before generating hundreds.
3. `scripts/review_labels.py` on 30 trajectories; record the agreement rate for the dataset card.
4. Model B (Llama-3.1-8B-AWQ), then the transfer table.

---

## Template — copy for each new day

## Day N — <title>

**Command(s)**
```
```

**Worked** · **Broke** · **Numbers** · **Reading** (what claim this does and does not support) · **Next**
