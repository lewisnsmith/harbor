"""hangar.ml.behavior_agents.rewards — Reward shaping for behavioral biases."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Protocol, Tuple

import numpy as np


class RewardShaper(Protocol):
    """Protocol for reward shaping terms."""

    def compute(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        returns_history: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> float: ...


@dataclass
class LossAversionShaper:
    """Penalizes losses more heavily than equivalent gains.

    Implements prospect-theory-style asymmetric utility::

        If r_p >= 0: term = 0
        If r_p < 0:  term = -lambda_la * |r_p|

    Parameters
    ----------
    lambda_la
        Loss aversion coefficient (higher = more loss averse).
        Default 2.25 follows Kahneman & Tversky.
    """

    lambda_la: float = 2.25

    def compute(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        returns_history: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> float:
        if portfolio_return >= 0:
            return 0.0
        return -self.lambda_la * abs(portfolio_return)


@dataclass
class OverconfidenceShaper:
    """Penalizes high portfolio concentration (overconfidence in single bets).

    Uses the Herfindahl-Hirschman Index (HHI)::

        term = -lambda_oc * sum(w_i^2)

    HHI ranges from 1/n (perfectly diversified) to 1 (single asset).

    Parameters
    ----------
    lambda_oc
        Overconfidence penalty coefficient.
    """

    lambda_oc: float = 0.1

    def compute(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        returns_history: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> float:
        hhi = float(np.sum(weights**2))
        return -self.lambda_oc * hhi


@dataclass
class ReturnChasingShaper:
    """Penalizes chasing recent winners (momentum loading).

    Computes correlation between weight changes and recent asset returns.
    Positive correlation indicates return chasing behavior::

        term = -lambda_rc * max(0, corr(delta_w, recent_returns))

    Parameters
    ----------
    lambda_rc
        Return chasing penalty coefficient.
    lookback
        Number of trailing days to compute recent cumulative returns.
    """

    lambda_rc: float = 0.05
    lookback: int = 21

    def compute(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        returns_history: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> float:
        delta_w = weights - prev_weights
        if np.allclose(delta_w, 0):
            return 0.0

        # Recent cumulative returns per asset
        n_lookback = min(self.lookback, len(returns_history))
        if n_lookback < 2:
            return 0.0

        recent_returns = returns_history[-n_lookback:]
        cum_returns = np.sum(recent_returns, axis=0)

        # Correlation between weight changes and recent returns
        if np.std(delta_w) < 1e-10 or np.std(cum_returns) < 1e-10:
            return 0.0

        corr = float(np.corrcoef(delta_w, cum_returns)[0, 1])
        if np.isnan(corr):
            return 0.0

        return -self.lambda_rc * max(0.0, corr)


@dataclass
class DispositionEffectShaper:
    """Penalizes holding losers and selling winners (disposition effect).

    Detects the tendency to:
    - Sell assets with positive recent returns (selling winners)
    - Hold/buy assets with negative recent returns (holding losers)

    Penalty is proportional to the strength of this pattern::

        term = -lambda_de * disposition_score

    Parameters
    ----------
    lambda_de
        Disposition effect penalty coefficient.
    """

    lambda_de: float = 0.05

    def compute(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        returns_history: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> float:
        delta_w = weights - prev_weights
        if np.allclose(delta_w, 0) or len(returns_history) == 0:
            return 0.0

        # Recent asset returns (last period)
        recent_r = returns_history[-1]

        # Disposition score: selling winners (delta_w < 0 where r > 0)
        # and holding losers (delta_w >= 0 where r < 0)
        winners = recent_r > 0
        losers = recent_r < 0

        score = 0.0
        n = len(delta_w)
        if n == 0:
            return 0.0

        # Penalize selling winners
        if np.any(winners):
            selling_winners = np.sum(np.maximum(-delta_w[winners], 0))
            score += selling_winners

        # Penalize holding/buying losers
        if np.any(losers):
            holding_losers = np.sum(np.maximum(delta_w[losers], 0))
            score += holding_losers

        return -self.lambda_de * float(score)


@dataclass
class CompositeRewardShaper:
    """Combines multiple reward shapers with individual on/off toggles.

    Parameters
    ----------
    shapers
        List of ``(name, shaper, enabled)`` triples.
    """

    shapers: List[Tuple[str, RewardShaper, bool]] = field(default_factory=list)

    def add(self, name: str, shaper: RewardShaper, enabled: bool = True) -> None:
        """Register a reward shaper."""
        self.shapers.append((name, shaper, enabled))

    def toggle(self, name: str, enabled: bool) -> None:
        """Enable or disable a named shaper."""
        for i, (n, s, _) in enumerate(self.shapers):
            if n == name:
                self.shapers[i] = (n, s, enabled)
                return
        raise KeyError(f"No shaper named {name!r}")

    def compute(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        returns_history: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> float:
        """Sum all enabled shaper contributions."""
        total = 0.0
        for _name, shaper, enabled in self.shapers:
            if enabled:
                total += shaper.compute(
                    portfolio_return, weights, prev_weights,
                    returns_history, portfolio_value, peak_value,
                )
        return total

    def breakdown(
        self,
        portfolio_return: float,
        weights: np.ndarray,
        prev_weights: np.ndarray,
        returns_history: np.ndarray,
        portfolio_value: float,
        peak_value: float,
    ) -> Dict[str, float]:
        """Return per-shaper contributions for diagnostics."""
        result: Dict[str, float] = {}
        for name, shaper, enabled in self.shapers:
            if enabled:
                result[name] = shaper.compute(
                    portfolio_return, weights, prev_weights,
                    returns_history, portfolio_value, peak_value,
                )
            else:
                result[name] = 0.0
        return result


def default_behavioral_shaper(
    *,
    loss_aversion: bool = True,
    overconfidence: bool = True,
    return_chasing: bool = True,
    disposition_effect: bool = True,
) -> CompositeRewardShaper:
    """Create a CompositeRewardShaper with default behavioral biases.

    Parameters
    ----------
    loss_aversion, overconfidence, return_chasing, disposition_effect
        Toggle each bias on/off.

    Returns
    -------
    CompositeRewardShaper
        Ready-to-use composite shaper with default parameters.
    """
    shaper = CompositeRewardShaper()
    shaper.add("loss_aversion", LossAversionShaper(), enabled=loss_aversion)
    shaper.add("overconfidence", OverconfidenceShaper(), enabled=overconfidence)
    shaper.add("return_chasing", ReturnChasingShaper(), enabled=return_chasing)
    shaper.add("disposition_effect", DispositionEffectShaper(), enabled=disposition_effect)
    return shaper
