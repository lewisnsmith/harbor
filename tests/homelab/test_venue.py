"""Tests for harbor.homelab.venue — VenueSnapshot, Venue protocol, EquityVenue."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from harbor.homelab.venue.protocol import Venue, VenueSnapshot
from harbor.homelab.venue.equity import EquityVenue


class TestVenueSnapshot:
    def test_creation(self):
        snap = VenueSnapshot(
            timestamp=pd.Timestamp("2020-01-01"),
            step=0,
            assets=["A", "B"],
            prices=np.array([100.0, 200.0]),
            returns=np.array([0.0, 0.0]),
            volume=np.array([0.0, 0.0]),
            spread=np.array([0.01, 0.01]),
            returns_history=np.zeros((1, 2)),
            market_type="equity",
        )
        assert snap.n_assets == 2
        assert snap.market_type == "equity"
        assert snap.step == 0

    def test_frozen(self):
        snap = VenueSnapshot(
            timestamp=pd.Timestamp("2020-01-01"),
            step=0,
            assets=["A"],
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volume=np.array([0.0]),
            spread=np.array([0.01]),
            returns_history=np.zeros((1, 1)),
            market_type="equity",
        )
        with pytest.raises(AttributeError):
            snap.step = 1  # type: ignore[misc]

    def test_default_metadata(self):
        snap = VenueSnapshot(
            timestamp=pd.Timestamp("2020-01-01"),
            step=0,
            assets=["A"],
            prices=np.array([100.0]),
            returns=np.array([0.0]),
            volume=np.array([0.0]),
            spread=np.array([0.01]),
            returns_history=np.zeros((1, 1)),
            market_type="equity",
        )
        assert snap.metadata == {}


class TestEquityVenue:
    def test_implements_venue_protocol(self):
        venue = EquityVenue()
        assert isinstance(venue, Venue)

    def test_reset_returns_snapshot(self):
        venue = EquityVenue({"n_assets": 5, "n_steps": 100})
        snap = venue.reset(seed=42)
        assert isinstance(snap, VenueSnapshot)
        assert snap.n_assets == 5
        assert snap.step == 0
        assert snap.market_type == "equity"
        assert len(snap.prices) == 5
        assert len(snap.returns) == 5
        assert len(snap.volume) == 5
        assert len(snap.spread) == 5

    def test_step_advances_state(self):
        venue = EquityVenue({"n_assets": 3, "n_steps": 10})
        snap = venue.reset(seed=42)
        assert snap.step == 0

        orders = {"agent_0": np.array([0.1, -0.1, 0.0])}
        snap2 = venue.step(orders)
        assert snap2.step == 1
        assert not np.array_equal(snap.prices, snap2.prices)

    def test_deterministic_seeding(self):
        venue = EquityVenue({"n_assets": 5, "n_steps": 50})

        snap1 = venue.reset(seed=123)
        for _ in range(10):
            snap1 = venue.step({})

        snap2 = venue.reset(seed=123)
        for _ in range(10):
            snap2 = venue.step({})

        np.testing.assert_array_equal(snap1.prices, snap2.prices)
        np.testing.assert_array_equal(snap1.returns, snap2.returns)

    def test_different_seeds_diverge(self):
        venue = EquityVenue({"n_assets": 5, "n_steps": 50})

        venue.reset(seed=1)
        for _ in range(10):
            s1 = venue.step({})

        venue.reset(seed=2)
        for _ in range(10):
            s2 = venue.step({})

        assert not np.array_equal(s1.prices, s2.prices)

    def test_volume_from_orders(self):
        venue = EquityVenue({"n_assets": 3, "n_steps": 10})
        venue.reset(seed=42)

        orders = {
            "a1": np.array([0.5, -0.3, 0.0]),
            "a2": np.array([0.1, 0.2, -0.4]),
        }
        snap = venue.step(orders)
        expected_volume = np.abs(orders["a1"]) + np.abs(orders["a2"])
        np.testing.assert_array_almost_equal(snap.volume, expected_volume)

    def test_config_serializable(self):
        venue = EquityVenue({"n_assets": 5})
        cfg = venue.config
        assert cfg["type"] == "equity"
        assert cfg["params"]["n_assets"] == 5
        assert isinstance(cfg["params"]["seed"], int)

    def test_metadata_has_fee_schedule(self):
        venue = EquityVenue({"temporary_impact": 0.2, "permanent_impact": 0.05})
        snap = venue.reset(seed=42)
        assert "fee_schedule" in snap.metadata
        assert snap.metadata["fee_schedule"]["temporary_impact"] == 0.2
        assert snap.metadata["fee_schedule"]["permanent_impact"] == 0.05

    def test_returns_history_shape(self):
        venue = EquityVenue({"n_assets": 4, "n_steps": 100, "lookback_window": 20})
        venue.reset(seed=42)
        for _ in range(25):
            snap = venue.step({})
        # After 25 steps, history should be capped at lookback_window (20)
        assert snap.returns_history.shape[1] == 4
        assert snap.returns_history.shape[0] <= 20
