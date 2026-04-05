"""hangar.ml.behavior_agents.agent — RLlib agent configuration and training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class AgentConfig:
    """Configuration for an RL portfolio agent."""

    algorithm: str = "PPO"
    hidden_sizes: Tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    train_batch_size: int = 4000
    sgd_minibatch_size: int = 128
    num_sgd_iter: int = 10
    clip_param: float = 0.2
    entropy_coeff: float = 0.01
    num_workers: int = 0
    num_envs_per_worker: int = 1
    framework: str = "torch"
    checkpoint_dir: Optional[str] = None


def build_rllib_config(
    returns: pd.DataFrame,
    *,
    env_config: Optional[Dict[str, Any]] = None,
    agent_config: Optional[AgentConfig] = None,
    reward_shaper: Optional[Any] = None,
) -> Any:
    """Build an RLlib AlgorithmConfig for portfolio optimization.

    Parameters
    ----------
    returns
        Daily return panel for the environment.
    env_config
        Overrides for ``EnvConfig`` fields (passed as dict).
    agent_config
        Agent hyperparameters.
    reward_shaper
        Optional ``CompositeRewardShaper`` to inject into the environment.

    Returns
    -------
    Any
        RLlib ``AlgorithmConfig`` instance.
    """
    from ray.rllib.algorithms.ppo import PPOConfig

    from hangar.ml.behavior_agents.environment import EnvConfig, PortfolioEnv

    if agent_config is None:
        agent_config = AgentConfig()

    # Build env config dict for RLlib
    env_kwargs = env_config or {}
    rllib_env_config = {
        "returns": returns,
        "config": EnvConfig(**env_kwargs) if isinstance(env_kwargs, dict) else env_kwargs,
        "reward_shaper": reward_shaper,
    }

    config = (
        PPOConfig()
        .environment(
            env=PortfolioEnv,
            env_config=rllib_env_config,
        )
        .framework(agent_config.framework)
        .training(
            lr=agent_config.learning_rate,
            gamma=agent_config.gamma,
            train_batch_size=agent_config.train_batch_size,
            sgd_minibatch_size=agent_config.sgd_minibatch_size,
            num_sgd_iter=agent_config.num_sgd_iter,
            clip_param=agent_config.clip_param,
            entropy_coeff=agent_config.entropy_coeff,
        )
        .env_runners(
            num_env_runners=agent_config.num_workers,
            num_envs_per_env_runner=agent_config.num_envs_per_worker,
        )
        .rl_module(
            model_config_dict={
                "fcnet_hiddens": list(agent_config.hidden_sizes),
                "fcnet_activation": "relu",
            },
        )
    )

    return config


def train_agent(
    returns: pd.DataFrame,
    *,
    env_config: Optional[Dict[str, Any]] = None,
    agent_config: Optional[AgentConfig] = None,
    reward_shaper: Optional[Any] = None,
    num_iterations: int = 100,
    checkpoint_freq: int = 10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Train an RL agent and return training results.

    Parameters
    ----------
    returns
        Daily return panel.
    env_config
        Environment configuration overrides.
    agent_config
        Agent hyperparameters.
    reward_shaper
        Behavioral reward shaper (optional).
    num_iterations
        Number of training iterations.
    checkpoint_freq
        Save checkpoint every N iterations.
    verbose
        Print progress during training.

    Returns
    -------
    Dict[str, Any]
        Keys: ``"final_metrics"``, ``"checkpoint_path"``, ``"training_history"``.
    """
    import ray

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    config = build_rllib_config(
        returns,
        env_config=env_config,
        agent_config=agent_config,
        reward_shaper=reward_shaper,
    )

    algo = config.build()

    if agent_config is None:
        agent_config = AgentConfig()
    ckpt_dir = agent_config.checkpoint_dir

    training_history: List[Dict[str, Any]] = []
    checkpoint_path: Optional[str] = None

    for i in range(num_iterations):
        result = algo.train()
        metrics = {
            "iteration": i,
            "episode_reward_mean": result.get("env_runners", {}).get(
                "episode_reward_mean", float("nan")
            ),
            "episode_len_mean": result.get("env_runners", {}).get(
                "episode_len_mean", float("nan")
            ),
        }
        training_history.append(metrics)

        if verbose and (i + 1) % max(1, checkpoint_freq) == 0:
            print(
                f"[iter {i + 1}/{num_iterations}] "
                f"reward={metrics['episode_reward_mean']:.4f}"
            )

        if checkpoint_freq > 0 and (i + 1) % checkpoint_freq == 0:
            save_result = algo.save(checkpoint_dir=ckpt_dir)
            checkpoint_path = str(save_result.checkpoint.path)

    # Final checkpoint
    save_result = algo.save(checkpoint_dir=ckpt_dir)
    checkpoint_path = str(save_result.checkpoint.path)

    final_metrics = training_history[-1] if training_history else {}
    algo.stop()

    return {
        "final_metrics": final_metrics,
        "checkpoint_path": checkpoint_path,
        "training_history": training_history,
    }


def agent_as_weight_func(
    checkpoint_path: str,
    returns: pd.DataFrame,
    *,
    env_config: Optional[Dict[str, Any]] = None,
) -> Callable[[pd.DataFrame, pd.Series], pd.Series]:
    """Load a trained RL agent and return a WeightFunction for backtesting.

    The returned function is compatible with
    ``hangar.backtest.run_cross_sectional_backtest``.

    Parameters
    ----------
    checkpoint_path
        Path to an RLlib checkpoint directory.
    returns
        Returns panel (used to reconstruct the environment for obs computation).
    env_config
        Environment configuration (must match training config).

    Returns
    -------
    WeightFunction
        ``Callable[[pd.DataFrame, pd.Series], pd.Series]`` compatible with
        the backtest engine.
    """
    import ray

    from hangar.ml.behavior_agents.environment import EnvConfig, PortfolioEnv

    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, log_to_driver=False)

    env_kwargs = env_config or {}
    env_cfg = EnvConfig(**env_kwargs) if isinstance(env_kwargs, dict) else env_kwargs

    # Create a temporary env for obs space / asset names
    temp_env = PortfolioEnv(returns, config=env_cfg)
    asset_names = temp_env.asset_names
    n_assets = temp_env.n_assets
    obs_window = env_cfg.obs_window

    config = build_rllib_config(returns, env_config=env_config)
    algo = config.build()
    algo.restore(checkpoint_path)

    def _weight_func(lookback: pd.DataFrame, current_weights: pd.Series) -> pd.Series:
        # Build observation from lookback returns and current weights
        ret_values = lookback.reindex(columns=asset_names, fill_value=0.0).values

        # Pad or trim to obs_window
        if len(ret_values) >= obs_window:
            window = ret_values[-obs_window:]
        else:
            pad = np.zeros((obs_window - len(ret_values), n_assets), dtype=np.float32)
            window = np.concatenate([pad, ret_values], axis=0)

        w = current_weights.reindex(asset_names, fill_value=0.0).values.astype(np.float32)
        if len(ret_values) >= 21:
            rv = np.std(ret_values[-21:], axis=0).astype(np.float32)
        else:
            rv = np.zeros(n_assets, dtype=np.float32)

        obs = {
            "returns_window": window.astype(np.float32),
            "current_weights": w,
            "rolling_vol": rv,
            "drawdown": np.array([0.0], dtype=np.float32),
        }

        action = algo.compute_single_action(obs)
        action = np.asarray(action, dtype=np.float32)

        # Normalize to valid weights
        action = np.maximum(action, 0.0)
        total = action.sum()
        if total > 0:
            action = action / total
        else:
            action = np.full(n_assets, 1.0 / n_assets, dtype=np.float32)

        return pd.Series(action, index=asset_names)

    return _weight_func
