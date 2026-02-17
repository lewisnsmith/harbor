# Harbor Asset Management Algorithm (H.A.R.B.O.R.)

**H**ierarchical **A**daptive **R**isk‑**B**ased **O**ptimization **R**outine is a research-focused asset management framework aimed at giving retail traders institutional‑style risk discipline and portfolio construction tools.

This is a research and educational project/experiment, not financial advice.

## Goals
- Risk‑first portfolio optimization
- Systematic behavior testing under algorithmic market regimes
- Research into behavioral finance and algorithmic market structure
- Lightweight, retail‑scale portfolio automation

## Repository Structure
- `harbor/` Core package namespaces (risk, portfolio, signals, data, research, utils)
- `notebooks/` Research and experimentation notebooks
- `docs/` Project documentation and iteration history
- `experiments/` Experimental scripts and prototypes (to be populated)
- `research/` Research notes and references (to be populated)

## Docs
- `docs/iteration-history.md` Development history and evolution

## Notebooks
Start in `notebooks/` for the latest research iterations and experiments. Outputs are ignored by default to keep the repo clean.

## Quickstart (Local)
Prereqs: Python 3.9+ (3.11 recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements-dev.txt
python -m pip install -e .
```

Then open notebooks (example):

```bash
jupyter lab
```

## Disclaimer
This project is for research and educational purposes only and does not constitute financial advice.
