"""hangar.homelab.__main__ — CLI entry point: python -m hangar.homelab experiment.yaml"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from hangar.homelab.config import ExperimentConfig
from hangar.homelab.runner import ExperimentRunner
from hangar.homelab.results.store import ResultStore


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="hangar.homelab",
        description="Run a Hangar homelab experiment from a YAML config.",
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to experiment YAML config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Directory to store results (default: results/).",
    )
    args = parser.parse_args(argv)

    if not args.config.exists():
        print(f"Error: config file not found: {args.config}", file=sys.stderr)
        sys.exit(1)

    # Load and run
    config = ExperimentConfig.from_yaml(args.config)
    print(f"Running experiment: {config.name}")
    print(f"  Seed: {config.seed}")
    print(f"  Agents: {sum(a.get('count', 1) for a in config.agents)}")

    runner = ExperimentRunner(config)
    result = runner.run()

    # Save results
    store = ResultStore(args.output_dir)
    exp_dir = store.save(
        name=config.name,
        config=config.to_dict(),
        metrics=result.metrics,
        prices=result.prices,
        returns=result.returns,
    )

    # Print summary
    print(f"\nExperiment complete in {result.elapsed_seconds:.2f}s")
    print(f"Results saved to: {exp_dir}")
    if result.metrics:
        print("\nMetrics:")
        for k, v in result.metrics.items():
            if isinstance(v, float):
                print(f"  {k}: {v:.6f}")
            else:
                print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
