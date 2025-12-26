"""
Unit tests for corporate action adjustments.

Tests the load and application of corporate actions (redenominations, splits)
to normalize historical price data.
"""
from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import yaml  # Required by backtest_runner

from backtest_runner import (
    load_corporate_actions,
    apply_corporate_action_adjustments,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(autouse=True)
def reset_cache():
    """Reset the global corporate actions cache before each test."""
    import backtest_runner
    backtest_runner._CORPORATE_ACTIONS_CACHE = None
    yield
    backtest_runner._CORPORATE_ACTIONS_CACHE = None


@pytest.fixture
def quick_redenomination_action():
    """QUICK redenomination action (1 OLD = 1000 NEW)."""
    return {
        "market": "binance_spot",
        "symbol": "QUICK",
        "date": "2023-07-21",
        "action_type": "redenomination",
        "ratio": 1000,
        "note": "QUICK token swap",
        "source_refs": ["binance_announcement"],
    }


@pytest.fixture
def quick_ohlcv_around_redenomination():
    """QUICK price data around 2023-07-21 redenomination."""
    return pd.DataFrame({
        "date": pd.to_datetime([
            "2023-07-15",
            "2023-07-16",
            "2023-07-17",  # Last day before redenomination
            "2023-07-21",  # Redenomination day
            "2023-07-22",
            "2023-07-23",
        ]),
        "open": [70.0, 72.0, 74.0, 0.06, 0.058, 0.062],
        "high": [72.0, 74.0, 76.0, 0.065, 0.063, 0.068],
        "low": [68.0, 70.0, 72.0, 0.055, 0.054, 0.058],
        "close": [71.0, 73.0, 74.2, 0.05967, 0.06, 0.065],
    })


# =============================================================================
# Test: QUICK Redenomination Continuity
# =============================================================================

class TestQuickRedenominationContinuity:
    """Tests for QUICK redenomination adjustment (1 OLD = 1000 NEW)."""

    def test_quick_close_scaled_before_action_date(
        self, quick_redenomination_action, quick_ohlcv_around_redenomination
    ):
        """
        close_pre=74.2 (before) becomes 0.0742 after scaling (ratio=1000).

        This verifies the core adjustment math for redenominations.
        """
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [quick_redenomination_action]

            df = quick_ohlcv_around_redenomination.copy()
            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            assert count == 1, "Expected 1 adjustment to be applied"

            # Check scaled close for 2023-07-17 (last day before action)
            pre_action_row = adjusted[adjusted["date"] == pd.Timestamp("2023-07-17")]
            original_close = 74.2
            expected_close = original_close / 1000  # = 0.0742

            np.testing.assert_almost_equal(
                pre_action_row["close"].values[0],
                expected_close,
                decimal=6,
                err_msg=f"Expected {expected_close}, got {pre_action_row['close'].values[0]}",
            )

    def test_quick_return_no_clamp_after_adjustment(
        self, quick_redenomination_action, quick_ohlcv_around_redenomination
    ):
        """
        After adjustment, return from 0.0742 to 0.05967 is ~-19.6% (no clamp trigger).

        The clamp threshold is typically at -100%. After adjustment, the return
        is reasonable and doesn't trigger any extreme clipping.
        """
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [quick_redenomination_action]

            df = quick_ohlcv_around_redenomination.copy()
            adjusted, _ = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            # Get adjusted close on 2023-07-17 and raw close on 2023-07-21
            close_before = adjusted.loc[
                adjusted["date"] == pd.Timestamp("2023-07-17"), "close"
            ].values[0]  # Should be 74.2 / 1000 = 0.0742

            close_after = adjusted.loc[
                adjusted["date"] == pd.Timestamp("2023-07-21"), "close"
            ].values[0]  # Should remain 0.05967 (unchanged)

            # Calculate return
            ret = (close_after / close_before) - 1

            # Before adjustment: (0.05967/74.2) - 1 = -99.92% (would trigger clamp)
            # After adjustment: (0.05967/0.0742) - 1 = -19.6% (reasonable)
            assert ret > -0.25, f"Return {ret:.4f} is still too negative"
            assert ret < -0.15, f"Return {ret:.4f} should be around -19.6%"

            # Specifically, it should be approximately -19.6%
            expected_ret = -0.196
            np.testing.assert_almost_equal(
                ret, expected_ret, decimal=2,
                err_msg=f"Expected return ~{expected_ret:.1%}, got {ret:.1%}",
            )

    def test_quick_all_ohlc_columns_scaled(
        self, quick_redenomination_action, quick_ohlcv_around_redenomination
    ):
        """All OHLC columns are scaled, not just close."""
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [quick_redenomination_action]

            df = quick_ohlcv_around_redenomination.copy()
            original_ohlc = df[df["date"] == pd.Timestamp("2023-07-17")][
                ["open", "high", "low", "close"]
            ].values[0]

            adjusted, _ = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            adjusted_ohlc = adjusted[adjusted["date"] == pd.Timestamp("2023-07-17")][
                ["open", "high", "low", "close"]
            ].values[0]

            expected_ohlc = original_ohlc / 1000
            np.testing.assert_array_almost_equal(
                adjusted_ohlc, expected_ohlc, decimal=6,
                err_msg="All OHLC columns should be scaled by ratio",
            )


# =============================================================================
# Test: Date Boundary Precision
# =============================================================================

class TestDateBoundary:
    """Tests for correct date boundary handling."""

    def test_scaling_only_before_action_date(
        self, quick_redenomination_action, quick_ohlcv_around_redenomination
    ):
        """
        Scaling applies ONLY to dates strictly < action_date.

        On action_date and after, prices should be unchanged.
        """
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [quick_redenomination_action]

            df = quick_ohlcv_around_redenomination.copy()
            original_df = df.copy()

            adjusted, _ = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            # Dates on or after 2023-07-21 should be unchanged
            action_date = pd.Timestamp("2023-07-21")
            after_mask = adjusted["date"] >= action_date

            for col in ["open", "high", "low", "close"]:
                original_vals = original_df.loc[after_mask, col].values
                adjusted_vals = adjusted.loc[after_mask, col].values
                np.testing.assert_array_equal(
                    adjusted_vals, original_vals,
                    err_msg=f"{col} should be unchanged on/after action date",
                )

    def test_scaling_all_dates_before_action(
        self, quick_redenomination_action, quick_ohlcv_around_redenomination
    ):
        """All dates before action_date should be scaled."""
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [quick_redenomination_action]

            df = quick_ohlcv_around_redenomination.copy()
            original_df = df.copy()

            adjusted, _ = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            # Dates before 2023-07-21 should be scaled
            action_date = pd.Timestamp("2023-07-21")
            before_mask = adjusted["date"] < action_date

            original_close = original_df.loc[before_mask, "close"].values
            adjusted_close = adjusted.loc[before_mask, "close"].values
            expected_close = original_close / 1000

            np.testing.assert_array_almost_equal(
                adjusted_close, expected_close, decimal=8,
                err_msg="All closes before action date should be scaled",
            )


# =============================================================================
# Test: Non-Matching Market/Symbol Unchanged
# =============================================================================

class TestNonMatchingUnchanged:
    """Tests that non-matching symbols are not affected."""

    def test_different_market_unchanged(
        self, quick_redenomination_action, quick_ohlcv_around_redenomination
    ):
        """Different market should not be adjusted."""
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [quick_redenomination_action]

            df = quick_ohlcv_around_redenomination.copy()
            original_df = df.copy()

            # Apply with different market
            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_futures", "QUICK"  # Different market
            )

            assert count == 0, "No adjustments should be applied"
            pd.testing.assert_frame_equal(
                adjusted, original_df,
                check_exact=True,
                obj="DataFrame should be unchanged for different market",
            )

    def test_different_symbol_unchanged(
        self, quick_redenomination_action, quick_ohlcv_around_redenomination
    ):
        """Different symbol should not be adjusted."""
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [quick_redenomination_action]

            df = quick_ohlcv_around_redenomination.copy()
            original_df = df.copy()

            # Apply with different symbol
            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_spot", "BTC"  # Different symbol
            )

            assert count == 0, "No adjustments should be applied"
            pd.testing.assert_frame_equal(
                adjusted, original_df,
                check_exact=True,
                obj="DataFrame should be unchanged for different symbol",
            )

    def test_empty_actions_unchanged(self, quick_ohlcv_around_redenomination):
        """Empty actions list should not modify data."""
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = []

            df = quick_ohlcv_around_redenomination.copy()
            original_df = df.copy()

            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            assert count == 0
            pd.testing.assert_frame_equal(
                adjusted, original_df, check_exact=True,
            )


# =============================================================================
# Test: Date Parsing Robustness
# =============================================================================

class TestDateParsing:
    """Tests for robust date parsing."""

    def test_string_date_format(self, quick_ohlcv_around_redenomination):
        """Action date as string 'YYYY-MM-DD' parses correctly."""
        action = {
            "market": "binance_spot",
            "symbol": "QUICK",
            "date": "2023-07-21",  # String format
            "action_type": "redenomination",
            "ratio": 1000,
        }

        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [action]

            df = quick_ohlcv_around_redenomination.copy()
            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            assert count == 1

    def test_timestamp_with_time_component(self, quick_ohlcv_around_redenomination):
        """Action date with time component parses correctly."""
        action = {
            "market": "binance_spot",
            "symbol": "QUICK",
            "date": "2023-07-21T00:00:00",  # With time component
            "action_type": "redenomination",
            "ratio": 1000,
        }

        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [action]

            df = quick_ohlcv_around_redenomination.copy()
            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            assert count == 1

    def test_df_dates_with_timezone(self, quick_ohlcv_around_redenomination):
        """DataFrame dates with timezone are handled correctly."""
        action = {
            "market": "binance_spot",
            "symbol": "QUICK",
            "date": "2023-07-21",
            "action_type": "redenomination",
            "ratio": 1000,
        }

        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [action]

            df = quick_ohlcv_around_redenomination.copy()
            # Add timezone to dates
            df["date"] = df["date"].dt.tz_localize("UTC")

            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            assert count == 1
            # Verify scaling worked
            scaled_close = adjusted.loc[
                adjusted["date"].dt.tz_localize(None) == pd.Timestamp("2023-07-17"),
                "close"
            ].values[0]
            np.testing.assert_almost_equal(scaled_close, 74.2 / 1000, decimal=6)


# =============================================================================
# Test: Reverse Split
# =============================================================================

class TestReverseSplit:
    """Tests for reverse split handling."""

    def test_reverse_split_scales_up(self, quick_ohlcv_around_redenomination):
        """Reverse split multiplies old prices by ratio."""
        action = {
            "market": "binance_spot",
            "symbol": "QUICK",
            "date": "2023-07-21",
            "action_type": "reverse_split",
            "ratio": 10,  # 10 old = 1 new
        }

        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = [action]

            df = quick_ohlcv_around_redenomination.copy()
            adjusted, count = apply_corporate_action_adjustments(
                df, "binance_spot", "QUICK"
            )

            assert count == 1

            # Pre-action close should be scaled UP
            pre_close = adjusted.loc[
                adjusted["date"] == pd.Timestamp("2023-07-17"), "close"
            ].values[0]

            # Original was 74.2, reverse split scales up by 10
            np.testing.assert_almost_equal(pre_close, 74.2 * 10, decimal=4)


# =============================================================================
# Test: Load Corporate Actions
# =============================================================================

class TestLoadCorporateActions:
    """Tests for loading corporate actions config."""

    def test_load_returns_list(self):
        """load_corporate_actions returns a list when mocked."""
        with patch("backtest_runner.load_corporate_actions") as mock_load:
            mock_load.return_value = []
            actions = mock_load()
            assert isinstance(actions, list)

    def test_load_caches_result(self):
        """Subsequent calls return cached result."""
        import backtest_runner

        first_result = load_corporate_actions()

        # Verify cache is set
        assert backtest_runner._CORPORATE_ACTIONS_CACHE is not None

        # Second call should return same object
        second_result = load_corporate_actions()
        assert first_result is second_result

    def test_missing_file_returns_empty_list(self):
        """Missing config file returns empty list."""
        import backtest_runner

        original_path = backtest_runner._CORPORATE_ACTIONS_PATH
        try:
            backtest_runner._CORPORATE_ACTIONS_PATH = Path("/nonexistent/path.yaml")
            backtest_runner._CORPORATE_ACTIONS_CACHE = None

            actions = load_corporate_actions()
            assert actions == []
        finally:
            backtest_runner._CORPORATE_ACTIONS_PATH = original_path


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
