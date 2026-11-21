COMPOSE := docker compose -f deploy/compose.yaml
SIM := $(COMPOSE) -f deploy/compose.sim.yaml
VLLM := $(COMPOSE) -f deploy/compose.vllm.yaml
PROMTOOL := docker run --rm -v $(CURDIR)/deploy/prometheus:/p -w /p --entrypoint promtool prom/prometheus:v3.5.0

.PHONY: test lint up-sim up-vllm down smoke rules rules-check promtool dashboard

test:
	python -m pytest -q

lint:
	ruff check . && ruff format --check .

up-sim:
	$(SIM) up -d --build

up-vllm:
	$(VLLM) up -d --build

down:
	$(SIM) down -v --remove-orphans; $(VLLM) down -v --remove-orphans

smoke:
	python scripts/smoke_stack.py

# SLO rules are generated from slo/slo.yaml; CI fails if the committed file is stale.
rules:
	python -m inferobs slo gen --spec slo/slo.yaml --out deploy/prometheus/rules/slo.yaml

rules-check:
	python -m inferobs slo gen --spec slo/slo.yaml --out deploy/prometheus/rules/slo.yaml --check

promtool:
	$(PROMTOOL) check config prometheus.yml
	$(PROMTOOL) check rules rules/*.yaml
	$(PROMTOOL) test rules tests/*.yaml

dashboard:
	python scripts/build_dashboard.py

lab-rules:
	python -m inferobs slo gen --spec slo/slo-fast.yaml --out deploy/lab-rules/slo-fast.yaml
