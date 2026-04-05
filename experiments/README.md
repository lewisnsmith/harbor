# Experiments

Temporary or exploratory scripts and prototypes live here. Promote stable work into `hangar/` or `notebooks/`.

> **Note on phase labels:** Scripts below use the old H1/H2/H3 phase naming from before the April 2026 restructure. These map to the 5-layer architecture as follows: H1 → Layer 3 (Portfolio/Risk), H2 → Layer 3, H3 → Layer 2 (Agent). The ABF Q1 pipeline has been removed; see `docs/abf-q1-research-summary.md` for historical results.

## Runnable Pipelines

### Phase H1 — Core Quant Stack
```bash
python experiments/h1_end_to_end_hrp_backtest.py --start 2020-01-01 --max-assets 50
# or: make h1
```

### Phase H2 — Advanced Risk & Stress Testing
```bash
python experiments/h2_risk_engine_demo.py \
    --start 2015-01-01 \
    --max-assets 20 \
    --output-dir results/h2_risk
# or: make h2
```
Options:
- `--simulation-method normal|student_t` — simulation distribution (default: normal)
- `--scenarios-config configs/risk/scenarios.json` — path to scenario definitions
- Outputs: `results/h2_risk/` (risk decomposition CSV, stress report JSON, scenario comparison)

### Phase H3 — Agent Simulation Core
```bash
python experiments/h3_agent_simulation_demo.py \
    --n-steps 500 \
    --output-dir results/agent_simulation
# or: make h3
```
- Runs 30 momentum agents in a synthetic market with price impact
- Compares agent-influenced dynamics to a baseline (no agents)
- Outputs: `results/agent_simulation/demo_figure.png` (4-panel comparison figure)

## Reproducibility

All pipelines can be run end-to-end in < 5 commands:
```bash
make install
make test
make h2
make h3
```
