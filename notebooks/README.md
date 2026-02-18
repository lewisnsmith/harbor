# Notebooks

Research and experimentation notebooks live here. Filenames reflect the evolution of the HARBOR algorithm.

## Conventions
- Keep exploratory work in notebooks.
- Avoid committing large outputs; use `notebooks/outputs/` for local artifacts (ignored by git).
- Prefer descriptive, versioned filenames.

## Paper Trading V1 Variants
- `v1_HARBOR_Paper_Trading_Algorithm.ipynb`: Canonical V1 baseline. Use this for manual research runs and baseline behavior comparisons.
- `v1_guarded_Paper_Trading_Algorithm.ipynb`: Guarded V1 variant. Use this for automation-oriented runs; it includes a dependency-guard cell and an extra stock-selection evaluation cell.

## Which One To Run
- If you want the reference V1 strategy behavior, run `v1_HARBOR_Paper_Trading_Algorithm.ipynb`.
- If you want the guarded/automation workflow, run `v1_guarded_Paper_Trading_Algorithm.ipynb`.
