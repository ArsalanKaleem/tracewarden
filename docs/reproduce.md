# Reproduction guide (maps to the 25-day plan)

| Day | Command / notebook | Output |
|---|---|---|
| 2 | `tracewarden data agentdrift --root ../AgentDrift/data_taskdisjoint --out data/processed --identity anonymize --shuffle-copies 1` then `tracewarden stats --data data/processed` | processed jsonl; counts 9081/1733/1722, 71,024 steps |
| 3 | `tracewarden data agentdrift ... --out data/raw_identity --identity keep`; `tracewarden audit --data data/raw_identity --out paper/results/audit.json` | shortcut audit |
| 4 | `notebooks/kaggle/01_encode_on_kaggle.ipynb` | cache/mpnet, cache/minilm, cache/multilingual |
| 5 | `tracewarden train --model driftnet --groups world ...` + `evaluate` | DriftNet reproduction |
| 6-8 | `tracewarden train --model streamguard --groups none|world|world,prov|all ...` | StreamGuard variants |
| 9 | `tracewarden calibrate --alpha 0.01`; `scripts/make_figures.py --checkpoint ...` | thresholds, risk-coverage |
| 10 | `scripts/run_ablation.py --seeds 3` (or notebook 03) | ablation table |
| 11-14 | `notebooks/kaggle/02_generate_agentdojo_trajectories.ipynb` (uses `scripts/agentdojo_attacks.py`) | raw logs + labeled jsonl |
| 12, 15 | `scripts/review_labels.py`; `scripts/publish_dataset.py` | audit agreement; AgentDojo-Steps on the Hub |
| 16 | `tracewarden encode` the real data with the SAME encoder; `scripts/transfer_eval.py` | transfer table |
| 17 | `scripts/adaptation_curve.py`; `scripts/make_figures.py --adaptation ...` | adaptation curve |
| 18 | `scripts/make_stress_set.py` (+ `--langs` with a vLLM endpoint); evaluate with the multilingual cache | robustness table |
| 19 | `scripts/export_onnx.py`; `scripts/benchmark_latency.py` | ONNX int8, latency |
| 20-22 | `examples/`, `tracewarden.integrations`, `app/app.py` | product |

Rules: evaluate every test set once; seeds 42, 43, 44; always state the split (task-disjoint) and identity
handling (anonymized) in tables.
