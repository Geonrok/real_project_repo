"""Tests for Stage6 Live Gate logic."""
from __future__ import annotations

import pandas as pd
import pytest

# Import functions from the script
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.stage6_live_gate import (
    classify_window_stress,
    get_market_window_status,
    apply_live_gate,
)


class TestClassifyWindowStress:
    """Tests for window stress classification."""

    def test_trades_zero_is_not_exercised(self):
        """trades=0 should result in NOT_EXERCISED."""
        df = pd.DataFrame(
            {
                "strategy_id": ["S1"],
                "market": ["M1"],
                "window_name": ["crisis"],
                "trades": [0],
            }
        )
        result = classify_window_stress(df, min_trades=10)
        assert result.iloc[0]["window_status"] == "NOT_EXERCISED"

    def test_trades_below_threshold_is_not_exercised(self):
        """trades < min_trades should result in NOT_EXERCISED."""
        df = pd.DataFrame(
            {
                "strategy_id": ["S1"],
                "market": ["M1"],
                "window_name": ["crisis"],
                "trades": [5],
            }
        )
        result = classify_window_stress(df, min_trades=10)
        assert result.iloc[0]["window_status"] == "NOT_EXERCISED"

    def test_trades_at_threshold_is_exercised(self):
        """trades >= min_trades should result in EXERCISED."""
        df = pd.DataFrame(
            {
                "strategy_id": ["S1"],
                "market": ["M1"],
                "window_name": ["crisis"],
                "trades": [10],
            }
        )
        result = classify_window_stress(df, min_trades=10)
        assert result.iloc[0]["window_status"] == "EXERCISED"

    def test_trades_above_threshold_is_exercised(self):
        """trades > min_trades should result in EXERCISED."""
        df = pd.DataFrame(
            {
                "strategy_id": ["S1"],
                "market": ["M1"],
                "window_name": ["crisis"],
                "trades": [100],
            }
        )
        result = classify_window_stress(df, min_trades=10)
        assert result.iloc[0]["window_status"] == "EXERCISED"


class TestGetMarketWindowStatus:
    """Tests for market-level window status aggregation."""

    def test_all_windows_exercised(self):
        """All windows EXERCISED -> market EXERCISED."""
        df = pd.DataFrame(
            {
                "strategy_id": ["S1", "S1", "S1"],
                "market": ["M1", "M1", "M1"],
                "window_name": ["crisis", "early", "late"],
                "trades": [100, 100, 100],
                "window_status": ["EXERCISED", "EXERCISED", "EXERCISED"],
            }
        )
        result = get_market_window_status(df)
        assert len(result) == 1
        assert result.iloc[0]["market_window_status"] == "EXERCISED"

    def test_no_windows_exercised(self):
        """No windows EXERCISED -> market NOT_EXERCISED."""
        df = pd.DataFrame(
            {
                "strategy_id": ["S1", "S1", "S1"],
                "market": ["M1", "M1", "M1"],
                "window_name": ["crisis", "early", "late"],
                "trades": [0, 0, 0],
                "window_status": ["NOT_EXERCISED", "NOT_EXERCISED", "NOT_EXERCISED"],
            }
        )
        result = get_market_window_status(df)
        assert len(result) == 1
        assert result.iloc[0]["market_window_status"] == "NOT_EXERCISED"

    def test_partial_windows_exercised(self):
        """Some windows EXERCISED -> market PARTIALLY_EXERCISED."""
        df = pd.DataFrame(
            {
                "strategy_id": ["S1", "S1", "S1"],
                "market": ["M1", "M1", "M1"],
                "window_name": ["crisis", "early", "late"],
                "trades": [100, 0, 0],
                "window_status": ["EXERCISED", "NOT_EXERCISED", "NOT_EXERCISED"],
            }
        )
        result = get_market_window_status(df)
        assert len(result) == 1
        assert result.iloc[0]["market_window_status"] == "PARTIALLY_EXERCISED"


class TestApplyLiveGate:
    """Tests for live gate application."""

    def _make_sensitivity(
        self, sharpe: float, mdd: float, trades: int = 100
    ) -> pd.DataFrame:
        """Helper to create sensitivity data."""
        return pd.DataFrame(
            {
                "strategy_id": ["S1"],
                "market": ["M1"],
                "fee_mult": [1.0],
                "slippage_mult": [1.0],
                "sharpe": [sharpe],
                "mdd": [mdd],
                "trades": [trades],
            }
        )

    def _make_window_status(self, status: str) -> pd.DataFrame:
        """Helper to create market window status."""
        return pd.DataFrame(
            {
                "strategy_id": ["S1"],
                "market": ["M1"],
                "exercised_windows": [3 if status == "EXERCISED" else 0],
                "total_windows": [3],
                "market_window_status": [status],
            }
        )

    def test_sharpe_negative_is_fail(self):
        """Negative sharpe should result in LIVE_FAIL."""
        sensitivity = self._make_sensitivity(sharpe=-0.5, mdd=-0.3)
        window_status = self._make_window_status("EXERCISED")

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        assert len(result) == 1
        assert result.iloc[0]["live_status"] == "LIVE_FAIL"
        assert result.iloc[0]["sharpe_pass"] == False
        assert "sharpe" in result.iloc[0]["fail_reasons"]

    def test_mdd_too_deep_is_fail(self):
        """MDD deeper than threshold should result in LIVE_FAIL."""
        sensitivity = self._make_sensitivity(sharpe=0.5, mdd=-0.9)
        window_status = self._make_window_status("EXERCISED")

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        assert len(result) == 1
        assert result.iloc[0]["live_status"] == "LIVE_FAIL"
        assert result.iloc[0]["mdd_pass"] == False
        assert "mdd" in result.iloc[0]["fail_reasons"]

    def test_window_not_exercised_is_inconclusive(self):
        """Window NOT_EXERCISED should result in LIVE_INCONCLUSIVE."""
        sensitivity = self._make_sensitivity(sharpe=0.5, mdd=-0.3)
        window_status = self._make_window_status("NOT_EXERCISED")

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        assert len(result) == 1
        assert result.iloc[0]["live_status"] == "LIVE_INCONCLUSIVE"
        assert "window_not_exercised" in result.iloc[0]["fail_reasons"]

    def test_all_conditions_pass_is_live_pass(self):
        """All conditions passing should result in LIVE_PASS."""
        sensitivity = self._make_sensitivity(sharpe=0.5, mdd=-0.3)
        window_status = self._make_window_status("EXERCISED")

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        assert len(result) == 1
        assert result.iloc[0]["live_status"] == "LIVE_PASS"
        assert result.iloc[0]["sharpe_pass"] == True
        assert result.iloc[0]["mdd_pass"] == True
        assert result.iloc[0]["window_pass"] == True
        assert result.iloc[0]["fail_reasons"] == ""

    def test_mdd_at_threshold_passes(self):
        """MDD exactly at threshold should pass."""
        sensitivity = self._make_sensitivity(sharpe=0.5, mdd=-0.60)
        window_status = self._make_window_status("EXERCISED")

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        assert result.iloc[0]["mdd_pass"] == True

    def test_sharpe_zero_fails_when_positive_required(self):
        """Sharpe=0 should fail when positive required."""
        sensitivity = self._make_sensitivity(sharpe=0.0, mdd=-0.3)
        window_status = self._make_window_status("EXERCISED")

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        assert result.iloc[0]["sharpe_pass"] == False
        assert result.iloc[0]["live_status"] == "LIVE_FAIL"

    def test_multiple_fail_reasons(self):
        """Multiple failures should all be listed."""
        sensitivity = self._make_sensitivity(sharpe=-0.5, mdd=-0.9)
        window_status = self._make_window_status("NOT_EXERCISED")

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        # Should be INCONCLUSIVE due to window, but sharpe and mdd also fail
        assert result.iloc[0]["live_status"] == "LIVE_INCONCLUSIVE"
        fail_reasons = result.iloc[0]["fail_reasons"]
        assert "sharpe" in fail_reasons
        assert "mdd" in fail_reasons
        assert "window" in fail_reasons


class TestMultipleStrategiesMarkets:
    """Tests for multiple strategies and markets."""

    def test_mixed_results(self):
        """Test mixed pass/fail/inconclusive across strategies."""
        sensitivity = pd.DataFrame(
            {
                "strategy_id": ["S1", "S2", "S3"],
                "market": ["M1", "M1", "M1"],
                "fee_mult": [1.0, 1.0, 1.0],
                "slippage_mult": [1.0, 1.0, 1.0],
                "sharpe": [0.5, -0.3, 0.2],  # S1: good, S2: bad sharpe, S3: good
                "mdd": [-0.3, -0.4, -0.8],  # S1: good, S2: good, S3: bad mdd
                "trades": [100, 100, 100],
            }
        )

        window_status = pd.DataFrame(
            {
                "strategy_id": ["S1", "S2", "S3"],
                "market": ["M1", "M1", "M1"],
                "exercised_windows": [3, 3, 3],
                "total_windows": [3, 3, 3],
                "market_window_status": ["EXERCISED", "EXERCISED", "EXERCISED"],
            }
        )

        result = apply_live_gate(
            sensitivity,
            window_status,
            require_sharpe_positive=True,
            max_mdd=-0.60,
        )

        assert len(result) == 3

        s1 = result[result["strategy_id"] == "S1"].iloc[0]
        s2 = result[result["strategy_id"] == "S2"].iloc[0]
        s3 = result[result["strategy_id"] == "S3"].iloc[0]

        assert s1["live_status"] == "LIVE_PASS"  # All good
        assert s2["live_status"] == "LIVE_FAIL"  # Bad sharpe
        assert s3["live_status"] == "LIVE_FAIL"  # Bad MDD
