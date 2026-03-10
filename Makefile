.PHONY: install test lint q1 h1 h2 all clean

PYTHON ?= python3
VENV   ?= .venv

install:
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/python -m pip install -U pip
	$(VENV)/bin/python -m pip install -r requirements-dev.txt
	$(VENV)/bin/python -m pip install -e .

test:
	$(PYTHON) -m pytest -q

lint:
	$(PYTHON) -m ruff check harbor tests

q1:
	$(PYTHON) experiments/abf_q1_main.py \
		--start 2010-01-01 \
		--end 2025-12-31 \
		--max-assets 75 \
		--output-dir results/abf_q1

h1:
	$(PYTHON) experiments/h1_end_to_end_hrp_backtest.py \
		--start 2020-01-01 \
		--max-assets 50

h2:
	$(PYTHON) experiments/h2_risk_engine_demo.py \
		--start 2015-01-01 \
		--max-assets 20 \
		--output-dir results/h2_risk

all: install lint test q1 h1 h2

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build
