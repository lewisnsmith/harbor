# Notebooks

Research and experimentation notebooks. Organized by pipeline stage.

## Conventions
- Keep exploratory work in notebooks.
- Avoid committing large outputs; use `notebooks/outputs/` for local artifacts (gitignored).
- Use descriptive, versioned filenames that match the module they exercise.

---

## ABF Research Notebooks (Active)

Notebooks for the Artificial Behavior in Finance (ABF) research track.
See `docs/abf-prd.md` for the full research specification.

| Notebook | Phase | Description |
|---|---|---|
| `abf_q1_shock_definition.ipynb` | A2 | Pre-specify and validate volatility shock definitions |
| `abf_q1_persistence_reversal.ipynb` | A2 | Event study + local projection regressions for Q1 |

---

## HARBOR Paper Trading V1 (Active)

Current V1 paper trading strategy notebooks.

| Notebook | Description |
|---|---|
| `v1_HARBOR_Paper_Trading_Algorithm.ipynb` | Canonical V1 baseline — use for manual research runs and baseline behavior comparisons |
| `v1_guarded_Paper_Trading_Algorithm.ipynb` | Guarded V1 variant — automation-oriented; includes dependency-guard cell and stock-selection evaluation |

---

## Archive

Notebooks from earlier Wharton-competition-era and MVO/Monte Carlo experiments are preserved in `notebooks/archive/`. They are **not** part of the active pipeline.
