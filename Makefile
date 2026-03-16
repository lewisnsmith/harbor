.PHONY: install test lint q1 h1 h2 all clean

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
	$(VPYTHON) -m ruff check harbor tests

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

all: install lint test q1 h1 h2

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build
