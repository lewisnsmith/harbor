"""RL behavioral agent training script.

Trains 5 agent types with different reward shapers:
- rational: no reward shaper (baseline)
- loss_averse: LossAversionShaper only
- overconfident: OverconfidenceShaper only
- return_chaser: ReturnChasingShaper only
- disposition: DispositionEffectShaper only
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harbor.data import load_sp500_prices, load_sp500_tickers  # noqa: E402
from harbor.ml.behavior_agents import (  # noqa: E402
    AgentConfig,
    default_behavioral_shaper,
    train_agent,
)

AGENT_DEFINITIONS = {
    "rational": {
        "loss_aversion": False,
        "overconfidence": False,
        "return_chasing": False,
        "disposition_effect": False,
    },
    "loss_averse": {
        "loss_aversion": True,
        "overconfidence": False,
        "return_chasing": False,
        "disposition_effect": False,
    },
    "overconfident": {
        "loss_aversion": False,
        "overconfidence": True,
        "return_chasing": False,
        "disposition_effect": False,
    },
    "return_chaser": {
        "loss_aversion": False,
        "overconfidence": False,
        "return_chasing": True,
        "disposition_effect": False,
    },
    "disposition": {
        "loss_aversion": False,
        "overconfidence": False,
        "return_chasing": False,
        "disposition_effect": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RL behavioral portfolio agents.")
    parser.add_argument("--start", default="2010-01-01", help="Start date")
    parser.add_argument(
        "--end",
        default=pd.Timestamp.today().strftime("%Y-%m-%d"),
        help="End date",
    )
    parser.add_argument("--max-assets", type=int, default=20, help="Max tickers")
    parser.add_argument("--num-iterations", type=int, default=50, help="Training iterations")
    parser.add_argument(
        "--checkpoint-dir",
        default="data/models/rl_agents",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENT_DEFINITIONS.keys()),
        choices=list(AGENT_DEFINITIONS.keys()),
        help="Agent types to train",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_base = Path(args.checkpoint_dir)
    ckpt_base.mkdir(parents=True, exist_ok=True)

    # 1) Load data
    print("[1/3] Loading prices...")
    tickers = load_sp500_tickers(as_of=args.end)[: args.max_assets]
    prices = load_sp500_prices(tickers=tickers, start=args.start, end=args.end, adjusted=True)
    returns = prices.pct_change().dropna(how="all")
    min_obs = max(int(0.8 * len(returns)), 252)
    returns = returns.dropna(axis=1, thresh=min_obs).fillna(0.0)
    print(f"  {returns.shape[1]} assets, {returns.shape[0]} days")

    # 2) Train each agent
    all_history = {}
    for agent_name in args.agents:
        print(f"\n[2/3] Training agent: {agent_name}...")
        shaper_kwargs = AGENT_DEFINITIONS[agent_name]

        # rational agent has no shaper
        if agent_name == "rational":
            shaper = None
        else:
            shaper = default_behavioral_shaper(**shaper_kwargs)

        agent_ckpt_dir = str(ckpt_base / agent_name)

        agent_config = AgentConfig(
            checkpoint_dir=agent_ckpt_dir,
            num_workers=0,
        )

        result = train_agent(
            returns,
            agent_config=agent_config,
            reward_shaper=shaper,
            num_iterations=args.num_iterations,
            checkpoint_freq=max(1, args.num_iterations // 5),
            verbose=True,
        )

        all_history[agent_name] = {
            "final_metrics": result["final_metrics"],
            "checkpoint_path": result["checkpoint_path"],
            "training_history": result["training_history"],
        }
        print(f"  Checkpoint: {result['checkpoint_path']}")

    # 3) Save training history
    print("\n[3/3] Saving training history...")
    history_path = ckpt_base / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(all_history, f, indent=2, default=str)
    print(f"  Saved to {history_path}")

    print("Done.")


if __name__ == "__main__":
    main()
