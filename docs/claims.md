# Novelty ledger

Fill in on Day 1 after reading DriftNet's Limitations and Future Work, AgentDrift, StepJack, and the neighbors
DriftNet cites (StepGuard, StepShield, TraceSafe, MELON). Update whenever a new paper appears.

| # | Closest prior work | What they did | What we add | Status |
|---|---|---|---|---|
| N1 pre-dispatch | DriftNet | Bidirectional, post-hoc over finished logs | Causal ACTION/OBS event model; blocks before execution; prevention metrics | |
| N2 provenance | DriftNet world features; CaMeL data flow | Email-only external-address flags; hard data-flow rules | Learned taint features for any entity type | |
| N3 anchoring | TaskTracker-style drift work | Activation / embedding drift for single prompts | Step-level goal-drift features in tool trajectories | |
| N4 shortcut audit | AgentDrift README | Reports the world-identity leak | Anonymized training + identity-shuffle augmentation + measured effect | |
| N5 conformal | DriftNet | Single threshold | Budgeted allow/hold/block with finite-sample guarantee; shift test | |
| N6 real data | AgentDojo | Trajectory-level success oracles | Auto step labels, 2 model families, released dataset | |
| N7 multi-step | StepJack | Decomposed attacks on AWS web sandboxes | Text-environment decomposition + multi-injection grammar | |
| N8 multilingual | - | English-only benchmarks | Urdu / Roman Urdu / Arabic / Hindi + obfuscation | |
| N9 rollback | - | Scores only | Incident report: undo list + quarantine | |
| N10 CPU | - | GPU-oriented | int8 ONNX trunk, latency budget on 8 GB laptop | |
