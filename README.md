# TraceWarden

**Pre-dispatch detection and localization of indirect prompt injection in LLM agent tool calls.**

[![CI](https://github.com/ArsalanKaleem/tracewarden/actions/workflows/ci.yml/badge.svg)](https://github.com/ArsalanKaleem/tracewarden/actions/workflows/ci.yml)![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)![License](https://img.shields.io/badge/license-MIT-green)

An LLM agent that reads email, web pages, or files and then calls tools can be redirected by instructions hidden in that content. TraceWarden sits in the agent's tool loop and scores each proposed tool call **before it executes**, using only what has happened so far in the run. It returns allow, hold, or block, points to the tool output the injection came from, and produces a rollback report of what to undo.

```python
from tracewarden import Guard

guard = Guard.load("models/tracewarden-demo-hashing.pt")
session = guard.session(user_task="Pay the electricity bill",
                        world={"email": "me@acme.com", "contacts": [...]})

decision = session.check("get_bills", {"type": "electricity"})   # before the call runs
session.observe(bill_text)                                        # after it runs

decision = session.check("add_beneficiary", {"email": "audit@ext-audit.com"})
print(decision.action, decision.reason)
# hold  hijack score 0.91; arguments contain values introduced by tool output:
#       audit@ext-audit.com; suspected injection in output of step(s) [1]

print(session.report().to_markdown())   # what to roll back, which source to quarantine
```

## What is in this repository


| Component            | Description                                                                                         |
| -------------------- | --------------------------------------------------------------------------------------------------- |
| `tracewarden`library | Causal pre-dispatch detector, feature extraction, conformal calibration, runtime`Guard`API          |
| AgentDojo converter  | Turns AgentDojo benchmark logs into step-labeled trajectories automatically                         |
| Label audit tools    | Rule-based consistency checks and a hand-review CLI for auto-labeled data                           |
| Integrations         | Adapters for OpenAI-compatible tool loops, LangChain/LangGraph tools, and an experimental MCP proxy |
| Dashboard            | Gradio app that renders a trajectory as a timeline with per-step decisions                          |
| Experiment scripts   | Transfer evaluation, ablations, adaptation curves, stress tests, ONNX export, latency benchmarks    |

## How it works

Each agent step is split into two events in time order: an **action** event (tool name, reasoning, arguments), known before the call executes, and an **observation** event, known after. A small causal transformer (about 1.3M trainable parameters over a frozen sentence encoder) reads the events left to right. At every action event it scores whether the proposed call serves an attacker, which is the point where a guard can still refuse it.

Each event embedding is combined with three groups of hand-designed features:

* **World features** check whether addresses in the arguments belong to the user's known contacts.
* **Provenance features** record where each argument value first appeared: the user's task, the known world, or an untrusted tool output. This is a learned form of taint tracking.
* **Goal-anchoring features** measure how far the proposed action has drifted from the user's task.

Decision thresholds come from split conformal calibration on benign runs, so an operator can set a budget such as "hold at most 1% of normal runs" and get thresholds that respect it on data like the calibration set.

## Results

Trained on the synthetic AgentDrift benchmark and evaluated on real agent trajectories collected in AgentDojo (Qwen2.5-7B and Qwen2.5-14B agents), with labels checked by hand review (96.7% and 86.7% agreement) and by automated consistency audits:


| Metric                                  | Synthetic (AgentDrift) | Real, 7B agent | Real, 14B agent |
| --------------------------------------- | ---------------------- | -------------- | --------------- |
| Trajectory F1                           | 0.984                  | 0.327          | 0.436           |
| Injection point exact match             | 0.883                  | 0.029          | 0.052           |
| Hijacked calls blocked before execution | 0.985                  | 0.093          | 0.152           |
| Benign runs interrupted                 | 0.5%                   | 13.1%          | 19.4%           |

Performance on synthetic data does not carry over to real agents. Detection drops to about a third and localization almost disappears. A sentence encoder transfers better than the bag-of-n-grams encoder that wins in-distribution (0.50 vs 0.33 F1 on the 7B agent), though the gap narrows at matched false-alarm rates. Text normalization removes zero-width and homoglyph obfuscation completely, while attacks split across several observations halve detection.

If you use TraceWarden to evaluate a defense, evaluate it on real agent traffic as well as on synthetic benchmarks. The two can disagree by a wide margin.

**Cost.** With the default encoder a full decision takes a median of 1.8 ms on a laptop CPU. The transformer trunk exports to a 1.7 MB int8 ONNX model that runs in 0.45 ms.

## Installation

```bash
git clone https://github.com/ArsalanKaleem/tracewarden.git
cd tracewarden
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
pytest -q
```

Optional extras:

```bash
pip install -e ".[st]"     # sentence-transformers encoders
pip install -e ".[app]"    # Gradio dashboard
pip install -e ".[onnx]"   # ONNX export and runtime
pip install -e ".[mcp]"    # MCP proxy
```

Python 3.10 to 3.12 is supported. Use a CPU build of PyTorch unless you plan to train on a GPU.

## Quickstart

Run the bundled demo, which replays an attacked banking trajectory through the guard:

```bash
python examples/quickstart.py
python examples/openai_agent_demo.py
```

Launch the dashboard:

```bash
TW_CHECKPOINT=models/tracewarden-demo-hashing.pt python app/app.py
```

The demo checkpoint uses the offline hashing encoder and was trained on synthetic data. It is meant for trying the API, not for protecting a production agent.

## Training and evaluation

```bash
# 1. Prepare AgentDrift (anonymized, with identity-shuffled training copies)
git clone https://github.com/Asif-0209/AgentDrift ../AgentDrift
tracewarden data agentdrift --root ../AgentDrift/data_taskdisjoint --out data/processed \
    --identity anonymize --shuffle-copies 1

# 2. Check the dataset for identity shortcuts
tracewarden audit --data data/processed

# 3. Encode, train, calibrate, evaluate
tracewarden encode   --data data/processed --encoder hashing-768 --out cache/hashing
tracewarden train    --model streamguard --data data/processed --cache cache/hashing --out runs/sg.pt
tracewarden calibrate --checkpoint runs/sg.pt --data data/processed --cache cache/hashing --alpha 0.01
tracewarden evaluate --checkpoint runs/sg.pt --data data/processed --cache cache/hashing
```

`tracewarden calibrate` writes thresholds into the checkpoint. Calibrate on the distribution you intend to evaluate, and recalibrate before switching datasets.

### Real agent trajectories

`notebooks/kaggle/` contains notebooks that serve an open-weight model with vLLM on a free GPU, run it inside AgentDojo, and convert the logs. To convert and audit logs locally:

```bash
tracewarden data agentdojo --root <agentdojo-logdir> --out data/real --name agentdojo-qwen --identity keep
python scripts/audit_labels.py --data data/real/test.jsonl
python scripts/review_labels.py --data data/real/test.jsonl --n 30
```

### Experiments


| Script                         | Purpose                                                |
| ------------------------------ | ------------------------------------------------------ |
| `scripts/transfer_eval.py`     | Evaluate one checkpoint across several test sets       |
| `scripts/run_ablation.py`      | Feature-group ablation with multiple seeds             |
| `scripts/adaptation_curve.py`  | Fine-tune on k real trajectories and measure recovery  |
| `scripts/make_stress_set.py`   | Obfuscated, split, and translated injection variants   |
| `scripts/export_onnx.py`       | Export and quantize the trunk to ONNX                  |
| `scripts/benchmark_latency.py` | Per-decision latency and memory on the current machine |
| `scripts/figures.py`           | Publication figures from result files                  |

## Integrations

```python
# OpenAI-compatible tool loop (OpenAI, vLLM, Ollama, Groq)
from tracewarden.integrations.openai_loop import run_tool_calls
tool_messages = run_tool_calls(session, assistant_message, tools, on_hold=ask_a_human)

# Plain functions or LangChain / LangGraph tools
from tracewarden.integrations.wrap import guard_tool, wrap_langchain_tool
```

The MCP proxy is experimental:

```bash
tracewarden-mcp --checkpoint runs/sg.pt --task "what the user asked" -- python -m your_mcp_server
```

## Repository layout

```
src/tracewarden/     library: schema, io, features, models, training, calibration, guard, integrations
scripts/             experiments, audits, figures, export
notebooks/kaggle/    GPU notebooks for encoding and trajectory generation
app/                 Gradio dashboard
examples/            runnable demos and sample trajectories
models/              demo checkpoint
tests/               unit and end-to-end tests
docs/                threat model and reproduction notes
```

## Limitations

TraceWarden is a monitor, not a guarantee. An attacker who knows the detector can likely evade it, and a model trained on synthetic data performs poorly on real agents, as the results above show. Use it as one layer alongside least-privilege tools, human confirmation for high-impact actions, and input sanitization. See [`docs/threat_model.md`](https://claude.ai/chat/docs/threat_model.md) for what it does and does not defend against.

## Acknowledgements

This project builds on [AgentDojo](https://github.com/ethz-spylab/agentdojo) for agent environments and ground-truth checkers, and on the [AgentDrift](https://github.com/Asif-0209/AgentDrift) benchmark and the DriftNet detector for the synthetic training corpus and baseline architecture.

## License

MIT. See [`LICENSE`](https://claude.ai/chat/LICENSE).
