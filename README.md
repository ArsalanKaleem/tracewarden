# TraceWarden

**A pre-dispatch firewall for AI-agent tool calls.** TraceWarden watches an agent's tool loop and, *before*
each tool call executes, decides **allow / hold / block**, points to the tool output the injection came from,
and produces a rollback report of what to undo.

```python
from tracewarden import Guard

guard = Guard.load("models/tracewarden-demo-hashing.pt")
sess = guard.session(user_task="Pay the electricity bill", world={"email": "me@acme.com", "contacts": [...]})

d = sess.check("get_bills", {"type": "electricity"})     # before the call runs
sess.observe(bill_text)                                   # after it runs
d = sess.check("add_beneficiary", {"email": "audit@ext-audit.com"})
# Decision(action='hold', score=0.91, reason='hijack score 0.91; arguments contain values introduced by
#          tool output: audit@ext-audit.com; suspected injection in output of step(s) [1]')
print(sess.report().to_markdown())                        # what to roll back, which source to quarantine
```

## Why

Detectors such as DriftNet ([arXiv 2609.10892](https://arxiv.org/abs/2609.10892)) read *finished* agent logs,
are trained on one synthetic corpus (AgentDrift, [arXiv 2609.06972](https://arxiv.org/abs/2609.06972)), and
assume one injection point. TraceWarden asks whether step-level detection works **before execution**, on **real
agents**, against **multi-step, multilingual and obfuscated** attacks.

| | What it adds |
|---|---|
| N1 | Causal StreamGuard: scores each *proposed* call using only the past |
| N2 | Argument provenance (taint) features: where each argument value first appeared |
| N3 | Goal-anchoring drift features |
| N4 | Shortcut audit + anonymization / identity-shuffle training |
| N5 | Conformal thresholds: a stated false-alarm budget, three-way decision |
| N6 | AgentDojo-Steps: real, auto-labeled trajectories from two model families |
| N7 | Multi-step decomposed injections and a multi-injection label grammar |
| N8 | Multilingual (Urdu, Roman Urdu, Arabic, Hindi) and obfuscated attacks |
| N9 | Rollback / incident report |
| N10 | CPU-first deployment: int8 ONNX trunk, 1.7 MB |

## First results (already reproducible from this repo)

AgentDrift task-disjoint test split (1,722 trajectories), **identities anonymized**, offline hashing encoder
(no GPU, no downloads). These are baselines; the paper numbers use all-mpnet-base-v2.

**Shortcut audit (N4).** A lookup on the world identity alone predicts compromise with **85.9%** accuracy
(93-98% in banking, coding, medical, web; 48% in email). After anonymization: **55.5%**, the majority rate.

| model (threshold 0.5) | decides | traj F1 | macro step F1 | injection EM | hijack IoU | delayed-exec recall | first hijack caught | benign runs interrupted |
|---|---|---|---|---|---|---|---|---|
| DriftNet re-implementation | after the run | 0.982 | 0.988 | 0.974 | 0.959 | 0.863 | n/a | n/a |
| StreamGuard, text only | before each call | 0.949 | 0.910 | 0.814 | 0.932 | 0.812 | 0.952 | 4.6% |
| StreamGuard + world/provenance/anchor | before each call | 0.977 | 0.953 | 0.888 | 0.966 | 0.897 | 0.981 | 2.3% |

With conformal calibration at a 1% budget (N5) the full StreamGuard interrupts **0.95%** of benign test
trajectories and still catches **97.3%** of attacks at or before the first hijacked call. For comparison,
the deterministic taint rule alone (`taint_policy="hold"`) catches 65% of attacks but holds 22.5% of benign runs:
the learned model is doing the work. Numbers are single-seed; see `paper/results/`.

**CPU cost (N10):** int8 ONNX trunk 1.7 MB, 0.26 ms per 12-event forward; `check()` 2 ms with the hashing
encoder (the sentence encoder dominates with all-mpnet; measure with `scripts/benchmark_latency.py`).

## Install

```bash
pip install -e ".[dev]"            # core + tests
pip install -e ".[st]"             # + sentence-transformers (real encoder)
pip install -e ".[app,onnx,mcp]"   # dashboard, ONNX export, MCP proxy
```
On a CPU-only machine install PyTorch first: `pip install torch --index-url https://download.pytorch.org/whl/cpu`.

## Reproduce the pipeline

```bash
git clone https://github.com/Asif-0209/AgentDrift ../AgentDrift
tracewarden data agentdrift --root ../AgentDrift/data_taskdisjoint --out data/processed --identity anonymize --shuffle-copies 1
tracewarden audit --data data/processed                     # shortcut audit (use --identity keep to see the leak)
tracewarden encode --data data/processed --encoder hashing-768 --out cache/hashing          # laptop, seconds
#   or on Kaggle: notebooks/kaggle/01_encode_on_kaggle.ipynb  (all-mpnet-base-v2 -> cache/mpnet)
tracewarden train --model streamguard --data data/processed --cache cache/hashing --out runs/sg.pt
tracewarden calibrate --checkpoint runs/sg.pt --data data/processed --cache cache/hashing --alpha 0.01
tracewarden evaluate  --checkpoint runs/sg.pt --data data/processed --cache cache/hashing --out paper/results/sg.md
python scripts/run_ablation.py --data data/processed --cache cache/hashing
```
Real trajectories: `notebooks/kaggle/02_generate_agentdojo_trajectories.ipynb`, then
`scripts/transfer_eval.py`, `scripts/adaptation_curve.py`, `scripts/make_stress_set.py`. Full guide:
[`docs/reproduce.md`](docs/reproduce.md).

## Integrations

* **OpenAI-compatible tool loops** (OpenAI, vLLM, Ollama, Groq): `tracewarden.integrations.openai_loop`
  (see `examples/openai_agent_demo.py`).
* **Plain functions / LangChain / LangGraph tools**: `tracewarden.integrations.wrap`.
* **MCP** (experimental): `python -m tracewarden.integrations.mcp_proxy --checkpoint runs/sg.pt -- <upstream server cmd>`.
* **Dashboard**: `TW_CHECKPOINT=models/tracewarden-demo-hashing.pt python app/app.py`.

## Repository

```
src/tracewarden/   schema, io/ (agentdrift, agentdojo), features/, models/ (driftnet, streaming),
                   featurize, encoders, training, metrics, calibrate, guard, report, cli, integrations/
scripts/           ablation, transfer, adaptation, stress sets, AgentDojo attacks, ONNX, latency, dataset release
notebooks/kaggle/  GPU encoding, AgentDojo + vLLM trajectory generation, ablation sweep
app/               Gradio dashboard (Hugging Face Space)
models/            demo checkpoint (hashing encoder; not the paper model)
docs/ paper/ tests/ examples/
```

## Limits

TraceWarden is a monitor, not a guarantee. Read [`docs/threat_model.md`](docs/threat_model.md). The demo model
was trained on synthetic data with an offline encoder and will miss attacks unlike its training data.

## Citation

See `CITATION.cff`. Please also cite AgentDrift, DriftNet and AgentDojo, whose data and ideas this builds on.
