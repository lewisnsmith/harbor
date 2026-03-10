# Experiments

Temporary or exploratory scripts and prototypes live here. Promote stable work into `harbor/` or `notebooks/`.

## Runnable Pipelines

### Phase H1 — Core Quant Stack
```bash
python experiments/h1_end_to_end_hrp_backtest.py --start 2020-01-01 --max-assets 50
# or: make h1
```

### Phase A2 — ABF Q1 Shock -> Persistence -> Reversal
```bash
python experiments/abf_q1_main.py \
    --start 2010-01-01 \
    --end 2025-12-31 \
    --max-assets 75 \
    --output-dir results/abf_q1
# or: make q1
```
Options:
- `--skip-robustness` — skip the robustness sweep for faster iteration
- Outputs: `results/abf_q1/` (coefficients JSON, CAR CSVs, robustness sweep CSV, figures)

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

## Reproducibility

All pipelines can be run end-to-end in < 5 commands:
```bash
make install
make test
make q1
make h2
```
