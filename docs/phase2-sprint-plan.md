# Phase 2 Sprint Plan (H2 + A2)

Date created: 2026-02-24
Source roadmap: `docs/plan.md`

## Scope

Phase 2 in the roadmap has two tracks:

1. H2 (Framework): Advanced Risk & Simulation
2. A2 (Research): Q1 Shock -> Persistence -> Reversal

Given current status ("A1/A2 scaffolded"), this plan prioritizes A2 completion first, then H2 engine work.

## What Is Still Missing

### A2 gaps (ABF Q1)

- `harbor.abf.q1` has only scaffolding; no executable analysis module yet.
- Q1 notebooks are placeholders with `NotImplementedError`.
- Local projection/event-study regressions are not implemented.
- Robustness checks and finalized figure outputs are not implemented.
- Reproducible raw-data-to-figure command path (<5 commands) is not implemented.
- A1 dependency still open: full WRDS/CRSP S&P 500 membership replacement.

### H2 gaps (HARBOR risk/simulation)

- No regime-aware covariance estimator.
- Monte Carlo is Gaussian-only; non-Gaussian/factor/copula engines not present.
- No config-driven stress scenario runner (vol spike, correlation spike, sector crash).
- No factor/cluster risk decomposition module.
- No ABF-facing pluggable risk-engine interface for scenario/risk outputs.

## Sprint Cadence

- Sprint length: 2 weeks
- Total: 4 sprints (8 weeks)

## Sprint 1 (2026-02-24 to 2026-03-09): A2 Core Pipeline

Primary outcome: executable Q1 baseline analysis code.

Backlog:
- Implement `harbor.abf.q1` package modules:
  - shock labeling wrapper(s)
  - event-window builder
  - persistence/reversal metrics API
  - local projection regression utility (HAC/Newey-West)
- Wire config loading from `configs/abf/q1_shock_definitions.json`.
- Replace notebook placeholder calls with real module calls.
- Add unit tests for event windows, CAR logic, and regression outputs.

Definition of done:
- Baseline Q1 run completes without `NotImplementedError`.
- At least 1 regression table and 1 CAR output generated programmatically.

## Sprint 2 (2026-03-10 to 2026-03-23): A2 Robustness + Reproducibility

Primary outcome: milestone-quality Q1 outputs and repeatable workflow.

Backlog:
- Implement pre-specified robustness checks:
  - alternative shock definitions
  - pre/post-2020 split
  - liquidity split
  - lag sensitivity for HAC
- Generate target figures and store outputs in `notebooks/outputs/`.
- Add one scripted runner in `experiments/` for "raw data -> Q1 figures/tables".
- Document exact reproducible commands in `experiments/README.md`.

Definition of done:
- Q1 outputs satisfy plan exit criteria: stable figures/tables with documented robustness.
- End-to-end run reproducible in <5 commands.

## Sprint 3 (2026-03-24 to 2026-04-06): H2 Risk Engine v1

Primary outcome: advanced risk primitives implemented.

Backlog:
- Add regime-aware covariance estimator(s) to `harbor.risk`.
- Add at least one robust Monte Carlo extension:
  - Student-t marginals or
  - factor-driven simulation engine.
- Add tests validating regime-switch behavior and simulation sanity.
- Expose unified risk-engine interface for downstream consumers.

Definition of done:
- Risk engine supports selectable covariance/simulation modes via config.
- Tests cover new branches and pass.

## Sprint 4 (2026-04-07 to 2026-04-20): H2 Scenarios + Decomposition + ABF Integration

Primary outcome: config-driven stress testing and ABF integration path.

Backlog:
- Create scenario runner for:
  - vol spike
  - correlation spike
  - sector crash
- Implement risk decomposition (factor and/or cluster attribution).
- Produce machine-readable stress-test report artifacts.
- Integrate pluggable risk engine in ABF experiments.
- Add docs and example experiment script.

Definition of done:
- Config-driven scenario runs produce decomposition + stress reports.
- ABF experiment can switch to pluggable risk engine path.

## Risks / Dependencies

- WRDS/CRSP constituent history is still a data dependency risk for A2 inference quality.
- If external data access slips, continue with documented fallback but flag results as provisional.
- Keep Q2/A3 stubs out of scope for this phase to prevent schedule creep.
