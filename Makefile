.PHONY: install test lint q1 h1 h2 h3 homelab all clean

PYTHON ?= python3
VENV   ?= .venv
VPYTHON = $(VENV)/bin/python

install:
	$(PYTHON) -m venv $(VENV)
	$(VPYTHON) -m pip install -U pip
	$(VPYTHON) -m pip install -r requirements-dev.txt
	$(VPYTHON) -m pip install -e .

test:
	$(VPYTHON) -m pytest -q

lint:
	$(VPYTHON) -m ruff check hangar tests

q1:
	$(VPYTHON) experiments/abf_q1_main.py \
		--start 2010-01-01 \
		--end 2025-12-31 \
		--max-assets 75 \
		--output-dir results/abf_q1

h1:
	$(VPYTHON) experiments/h1_end_to_end_hrp_backtest.py \
		--start 2020-01-01 \
		--max-assets 50

h2:
	$(VPYTHON) experiments/h2_risk_engine_demo.py \
		--start 2015-01-01 \
		--max-assets 20 \
		--output-dir results/h2_risk

h3:
	$(VPYTHON) experiments/h3_agent_simulation_demo.py \
		--n-steps 500 \
		--output-dir results/agent_simulation

homelab:
	$(VPYTHON) -m hangar.homelab benchmarks/momentum_baseline.yaml
	$(VPYTHON) -m hangar.homelab benchmarks/mixed_population.yaml

all: install lint test q1 h1 h2 h3 homelab

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build
