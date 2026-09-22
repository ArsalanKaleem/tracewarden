# TraceWarden — working log

One entry per working day. Three things every time: **what worked**, **what broke**, **what's next**.
Numbers go in here the day they are produced, with the exact command that produced them. This file becomes
the method section of the paper and the raw material for the launch post, so write it for someone else to read.

Conventions used below:
- **Split**: AgentDrift task-disjoint (9,081 / 1,733 / 1,722), unless stated otherwise.
- **Identity**: "anonymized" = `--identity anonymize` (placeholder names, in/out-of-organization addresses preserved).
- **Encoder**: `hashing-768` is the offline bag-of-n-grams stand-in; `mpnet` is `all-mpnet-base-v2`.
- Every test-set number is reported once, with a 95% bootstrap CI where the tooling provides one.

---

## Day 0 — environment and repository

**Worked**
- Python 3.12.10 venv (3.14 is the machine default and has no torch wheels yet; do not use it).
- CPU-only torch, `pip install -e ".[dev]"`, 17/17 tests pass in ~17 s.
- `examples/quickstart.py` behaves as designed: two benign calls allowed, `add_beneficiary` held (0.91),
  `transfer_money` blocked (1.00), both stopped before execution; report names the poisoned `get_bills` output.
- Repo initialized and pushed to GitHub; CI configured (ruff + pytest + quickstart on 3.10/3.11/3.12).

**Broke / notes**
- `py -3.11` is not installed on this machine; 3.12 is the working interpreter.
- Prompt shows both `(.venv)` and `(python-3.14)`; check `python -V` if anything behaves oddly.

**Next**
- Replace remaining `<you>` / `<Your Name>` placeholders (pyproject, CITATION.cff, LICENSE, README, notebooks).
- Confirm the `tracewarden` name is free on PyPI and Hugging Face; create an HF write token for Kaggle.

---

## Day 2 — data loader

**Command**
```
tracewarden data agentdrift --root ../AgentDrift/data_taskdisjoint --out data/processed --identity anonymize --shuffle-copies 1
tracewarden stats --data data/processed
```

**Worked** — counts match the AgentDrift paper exactly.

| part | trajectories | steps |
|---|---|---|
| train | 9,081 | 51,507 |
| val | 1,733 | 9,721 |
| test | 1,722 | 9,796 |
| total | 12,536 | 71,024 |

Label distribution across the corpus: benign 51,639 · hijacked 12,349 · injection_point 5,536 ·
failed_injection 1,500. Domains per the paper, banking under-represented (1,541 train) as documented.
9,081 identity-shuffled training copies written to `train_shuf.jsonl` for Day 10.

**Next** — shortcut audit before any model is trained.

---

## Day 3 — shortcut audit (N4)

**Command**
```
tracewarden data agentdrift --root ../AgentDrift/data_taskdisjoint --out data/raw_identity --identity keep
tracewarden audit --data data/raw_identity --out paper/results/audit_raw.json
tracewarden audit --data data/processed
```

**Worked** — the world-identity leak reproduces, and anonymization removes it.

| lookup key | raw identities | anonymized |
|---|---|---|
| majority class | 0.550 | 0.550 |
| **world identity** | **0.859** (P 0.882 / R 0.792) | **0.555** (R 0.012) |
| company only | 0.667 | 0.550 |
| tool sequence | 0.600 (14.6% coverage) | 0.600 |
| number of steps | 0.584 | 0.584 |

Per domain, world identity on raw data: medical 0.984 · banking 0.959 · coding 0.950 · web 0.931 · **email 0.484**.
Matches the dataset README (93–98% in four domains, email clean). Anonymized: 0.50–0.65, i.e. noise.

**Reading** — a model that reads names or addresses as text can reach ~86% on this benchmark without
learning anything about injection. All training from here uses the anonymized corpus; this table is the
justification and belongs in the paper's data section.

**Next** — train the detectors on the anonymized corpus only.

---

## Day 5–6 — reproduction, pre-dispatch model, and first feature ablation

**Commands**
```
tracewarden encode --data data/processed --encoder hashing-768 --out cache/hashing
tracewarden train --model driftnet    --data data/processed --cache cache/hashing --out runs/driftnet.pt   --epochs 15
tracewarden train --model streamguard --data data/processed --cache cache/hashing --out runs/sg_textonly.pt --groups none --epochs 15
tracewarden train --model streamguard --data data/processed --cache cache/hashing --out runs/sg.pt          --epochs 15
tracewarden evaluate --checkpoint <ckpt> --data data/processed --cache cache/hashing --threshold 0.5 --out paper/results/<name>.md
```
Encoding 122,531 step-views: ~20 s. Training: 9–22 s per epoch on one CPU core. Single seed (42).

**Results — anonymized task-disjoint test, `hashing-768` encoder, common threshold 0.5**

| model | decides | traj F1 (95% CI) | macro step F1 | injection EM | hijack IoU | delayed-exec recall | benign flagged | hard-neg flagged |
|---|---|---|---|---|---|---|---|---|
| DriftNet (re-impl.) | after the run | 0.981 [0.974, 0.988] | 0.989 | 0.981 | 0.966 | 0.863 | 1.71% | 1.47% |
| StreamGuard, text only | before each call | 0.973 [0.965, 0.981] | 0.950 | 0.879 | 0.959 | 0.880 | 1.90% | 5.88% |
| StreamGuard, all features | before each call | 0.981 [0.974, 0.987] | 0.955 | 0.884 | 0.965 | **0.906** | **0.38%** | 5.39% |

Prevention metrics (StreamGuard, all features, threshold 0.5): blocked before execution 0.989 ·
first hijack caught 0.981 · benign interruption 1.80%.

**Reading**
1. *Price of causality*: pre-dispatch matches post-hoc on whether an attack is caught (0.981 vs 0.981) but
   loses ~10 points of injection exact match (0.884 vs 0.981). A poisoned observation must be judged when it
   arrives, before the agent's later behavior disambiguates it. Design consequence, not a defect — state it.
2. *Features*: trajectory F1 gain (0.973 → 0.981) is **not** significant on one seed — the CIs overlap. The
   defensible claims are the fivefold drop in benign false alarms (1.90% → 0.38%) and delayed-execution
   recall (0.880 → 0.906). Re-run with `--seeds 3` before claiming more.
3. *Hard negatives* are the streaming model's weak point (5.4% vs DriftNet's 1.5%): they look suspicious early
   and are only vindicated later, which a causal model cannot see. Honest, mechanistic limitation.

**Open question raised (important)** — DriftNet reproduces at 0.981 F1 / 0.981 injection EM using a
*bag-of-n-grams* encoder, essentially matching the published 0.983 / 0.987 with all-mpnet. If semantic
embeddings barely beat word hashing, AgentDrift may be lexically separable (surface cues such as
"SECURITY ALERT", `ext-audit.com`). Test directly once the mpnet cache exists: if mpnet ≈ hashing, that is a
finding about the benchmark and it strengthens the case for real data.

---

## Day 9 — conformal calibration (N5)

**Command**
```
tracewarden calibrate --checkpoint runs/sg.pt --data data/processed --cache cache/hashing --alpha 0.01
```

**Worked** — thresholds from 980 non-compromised validation trajectories: hold 0.814, block 0.994.
On the test split the promise holds: **0.53%** of benign runs held or blocked (budget 1%), **0%** blocked,
while catching **97.6%** of attacks and **97.3%** at or before the first hijacked call. Zero false alarms on
benign and on resisted attacks; every false positive is a hard negative (2.45%).

Risk–coverage (test): alpha 0.0025 → 0.10% benign flagged / 93.8% caught · alpha 0.01 → 0.82% / 96.9% ·
alpha 0.05 → 4.90% / 99.7%.

**Note** — at alpha 0.001 the threshold is `Infinity`: with 980 calibration points the finite-sample
correction cannot certify a rate below ~1/981, so it refuses rather than guessing. Correct behavior, and a
good footnote on conformal calibration in practice. A tighter budget needs more calibration data.

---

## Day 1 (still open) — literature lock

- [ ] Read DriftNet (arXiv 2609.10892) **Limitations** and **Future Work**; mark overlap with N1–N10 in `docs/claims.md`.
- [ ] Skim AgentDrift Sections VIII and XI (known regularities) and StepJack's decomposition pipeline.
- [ ] Check arXiv listings for the DriftNet authors — a streaming or real-data follow-up changes the framing.

## Next actions (in order)

1. Upload `data/processed` to Kaggle as a private dataset; run `notebooks/kaggle/01_encode_on_kaggle.ipynb`
   to build `cache/mpnet`, `cache/minilm`, `cache/multilingual`; pull them back with `snapshot_download`.
2. Re-run the three models against `cache/mpnet` and answer the lexical-separability question above.
3. `python scripts/run_ablation.py --data data/processed --cache cache/mpnet --seeds 3` for the full table
   with seed variance, including the identity-shuffle-augmented variant.
4. Notebook 02: vLLM + AgentDojo, model A trajectories. Check `--help` of the installed AgentDojo first.

---

## Template — copy for each new day

## Day N — <one-line title>

**Command(s)**
```
<exact commands, so the result is reproducible>
```

**Worked**
-

**Broke / surprised me**
-

**Numbers**
| metric | value |
|---|---|

**Reading** — what this means for RQ1–RQ5, and what claim it does or does not support.

**Next**
-
