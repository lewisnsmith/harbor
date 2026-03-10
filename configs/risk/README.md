# Risk Configuration

This directory contains JSON configuration files for the HARBOR H2 pluggable risk engine.

## scenarios.json

Defines stress scenarios used by `RiskEngine.run_stress_test()`.

### Top-level keys

| Key                   | Description                                      |
|-----------------------|--------------------------------------------------|
| `version`             | Schema version string.                           |
| `description`         | Human-readable description.                      |
| `default_risk_config` | Default `RiskConfig` fields (covariance method, simulation method, etc.). |
| `scenarios`           | List of scenario objects.                        |

### Scenario object

| Key           | Description                                                    |
|---------------|----------------------------------------------------------------|
| `name`        | Unique identifier for the scenario.                            |
| `type`        | One of `vol_spike`, `correlation_spike`, or `sector_crash`.    |
| `description` | Human-readable summary.                                        |
| `params`      | Type-specific parameters (see below).                          |

### Scenario types

- **vol_spike** -- `params.multiplier` scales all volatilities.
- **correlation_spike** -- `params.target_corr` sets all off-diagonal correlations to the target value.
- **sector_crash** -- `params.crash_sector` names the affected sector; `params.crash_magnitude` is the return shock (e.g., -0.15 for a 15% drop). Requires a `sector_map` dict at runtime.

### Loading configs in Python

```python
from harbor.risk.engine import load_risk_config, load_scenarios_config

risk_cfg = load_risk_config("configs/risk/scenarios.json")  # reads default_risk_config
scenarios = load_scenarios_config("configs/risk/scenarios.json")
```
