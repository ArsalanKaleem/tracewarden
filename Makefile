AGENTDRIFT ?= ../AgentDrift/data_taskdisjoint
ENC ?= hashing-768
CACHE ?= cache/hashing

.PHONY: install test lint data audit encode train calibrate evaluate ablation demo app
install:
	pip install -e ".[dev]"
test:
	pytest
lint:
	ruff check src tests scripts
data:
	tracewarden data agentdrift --root $(AGENTDRIFT) --out data/processed --identity anonymize --shuffle-copies 1
audit:
	tracewarden data agentdrift --root $(AGENTDRIFT) --out data/raw_identity --identity keep && tracewarden audit --data data/raw_identity --out paper/results/audit.json
encode:
	tracewarden encode --data data/processed --encoder $(ENC) --out $(CACHE)
train:
	tracewarden train --model streamguard --data data/processed --cache $(CACHE) --out runs/sg.pt
calibrate:
	tracewarden calibrate --checkpoint runs/sg.pt --data data/processed --cache $(CACHE) --alpha 0.01
evaluate:
	tracewarden evaluate --checkpoint runs/sg.pt --data data/processed --cache $(CACHE) --out paper/results/sg.md
ablation:
	python scripts/run_ablation.py --data data/processed --cache $(CACHE)
demo:
	python examples/quickstart.py && python examples/openai_agent_demo.py
app:
	TW_CHECKPOINT=models/tracewarden-demo-hashing.pt python app/app.py
