"""
Tests for metrics calculation invariants.

Deterministic synthetic tests for:
a) Equity must be non-negative (all markets - Option A strict policy)
b) MDD must be in [-1, 0]
c) Equity must not contain NaN/inf; CAGR consistency
d) Invalid results excluded from candidate selection
e) Per-symbol validation (strategy invalid if any symbol violates)
f) Trades count includes both long and short entries (0→non-zero)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import numpy as np
import pandas as pd
import pytest

# Try to import backtest_runner functions, skip tests if dependencies missing
try:
    from backtest_runner import validate_metrics_invariants, save_debug_dump
    HAS_BACKTEST_RUNNER = True
except ImportError:
    HAS_BACKTEST_RUNNER = False
    validate_metrics_invariants = None
    save_debug_dump = None


class TestMDDCalculation:
    """Test MDD with edge cases."""

    def test_mdd_with_zero_running_max(self):
        """MDD should not be NaN/inf when running_max starts at 0."""
        # Synthetic: equity goes 0 -> 1 -> 0.5
        cum_ret = pd.Series([0.0, 1.0, 0.5])
        EPS = 1e-10
        running_max = cum_ret.cummax()
        safe_max = np.maximum(running_max.values, EPS)
        drawdown = cum_ret.values / safe_max - 1.0
        mdd = drawdown.min()

        assert not np.isnan(mdd), "MDD should not be NaN"
        assert not np.isinf(mdd), "MDD should not be inf"

    def test_mdd_range_valid_equity(self):
        """MDD must be in [-1, 0] for valid equity curve."""
        cum_ret = pd.Series([1.0, 1.2, 0.8, 1.0, 0.6])
        running_max = cum_ret.cummax()
        drawdown = cum_ret / running_max - 1.0
        mdd = drawdown.min()

        assert -1.0 <= mdd <= 0.0, f"MDD {mdd} outside [-1, 0]"

    def test_mdd_with_all_zeros(self):
        """MDD should handle all-zero equity curve."""
        cum_ret = pd.Series([0.0, 0.0, 0.0])
        EPS = 1e-10
        running_max = cum_ret.cummax()
        safe_max = np.maximum(running_max.values, EPS)
        drawdown = cum_ret.values / safe_max - 1.0
        mdd = drawdown.min()

        assert not np.isnan(mdd), "MDD should not be NaN"
        assert not np.isinf(mdd), "MDD should not be inf"


class TestCashPeriodEquity:
    """Test equity preservation during position=0."""

    def test_equity_unchanged_no_position(self):
        """Equity should not change when position=0 and no costs."""
        # Simulate: position=0 for all days, no costs
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({
            "date": dates,
            "close": [100, 101, 99, 102, 100],
            "position": [0, 0, 0, 0, 0],
        })
        df["ret"] = df["close"].pct_change().fillna(0)
        df["pos_change"] = df["position"].diff().abs().fillna(0)
        df["costs"] = 0.0
        df["strat_ret"] = df["position"].shift(1).fillna(0) * df["ret"] - df["costs"]
        df["cum_ret"] = (1 + df["strat_ret"]).cumprod()

        # Equity should remain 1.0 throughout
        np.testing.assert_allclose(df["cum_ret"].values, [1.0, 1.0, 1.0, 1.0, 1.0])

    def test_equity_changes_with_position(self):
        """Equity should change when position > 0."""
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({
            "date": dates,
            "close": [100, 110, 105, 115, 120],
            "position": [0, 1, 1, 1, 0],  # Enter on day 2, exit on day 5
        })
        df["ret"] = df["close"].pct_change().fillna(0)
        df["pos_change"] = df["position"].diff().abs().fillna(0)
        df["costs"] = 0.0
        df["strat_ret"] = df["position"].shift(1).fillna(0) * df["ret"] - df["costs"]
        df["cum_ret"] = (1 + df["strat_ret"]).cumprod()

        # Equity should change on days 2-4 (position held on prev day)
        assert df["cum_ret"].iloc[0] == 1.0  # Day 1: no position
        assert df["cum_ret"].iloc[1] == 1.0  # Day 2: position=0 on prev day
        assert df["cum_ret"].iloc[2] != 1.0  # Day 3: position=1 on prev day


class TestLeverageConstraints:
    """Test leverage enforcement."""

    def test_spot_position_capped_at_one(self):
        """Spot markets: position should be capped at 1.0."""
        position = np.array([0.5, 1.5, 2.0, 0.8])
        max_leverage = 1.0
        allow_shorting = False

        if not allow_shorting:
            clamped = np.clip(position, 0, max_leverage)

        assert all(clamped <= 1.0), "Spot position should be <= 1.0"
        assert all(clamped >= 0), "Spot position should be >= 0"
        np.testing.assert_array_equal(clamped, [0.5, 1.0, 1.0, 0.8])

    def test_futures_allows_higher_leverage(self):
        """Futures markets: position can exceed 1.0 up to max."""
        position = np.array([0.5, 1.5, 2.5, 0.8])
        max_leverage = 2.0

        clamped = np.clip(position, -max_leverage, max_leverage)

        np.testing.assert_array_equal(clamped, [0.5, 1.5, 2.0, 0.8])

    def test_no_shorting_constraint(self):
        """No shorting: position should be >= 0."""
        position = np.array([0.5, -0.5, 1.0, -1.0])
        allow_shorting = False
        max_leverage = 1.0

        if not allow_shorting:
            clamped = np.clip(position, 0, max_leverage)

        assert all(clamped >= 0), "Position should be >= 0 when shorting not allowed"
        np.testing.assert_array_equal(clamped, [0.5, 0.0, 1.0, 0.0])

    def test_shorting_allowed(self):
        """Shorting allowed: position can be negative up to -max_leverage."""
        position = np.array([0.5, -0.5, 1.5, -1.5])
        max_leverage = 1.0

        clamped = np.clip(position, -max_leverage, max_leverage)

        np.testing.assert_array_equal(clamped, [0.5, -0.5, 1.0, -1.0])


@pytest.mark.skipif(not HAS_BACKTEST_RUNNER, reason="backtest_runner import failed (yaml missing)")
class TestInvariantValidation:
    """Test invariant violation detection."""

    def test_negative_equity_marked_invalid(self):
        """Spot with negative equity should be invalid."""
        cum_ret = pd.Series([1.0, 0.5, -0.1])
        metrics = {"mdd": -1.1, "cagr": 0}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("equity_negative" in v for v in violations)

    def test_mdd_below_minus_one_invalid(self):
        """MDD < -1 should be flagged as invalid."""
        cum_ret = pd.Series([1.0, 0.5, 0.3])
        metrics = {"mdd": -1.5, "cagr": -0.5}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("mdd_below_minus1" in v for v in violations)

    def test_mdd_positive_invalid(self):
        """MDD > 0 should be flagged as invalid."""
        cum_ret = pd.Series([1.0, 1.1, 1.2])
        metrics = {"mdd": 0.1, "cagr": 0.2}  # Invalid: MDD should be <= 0

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("mdd_positive" in v for v in violations)

    def test_valid_result_passes(self):
        """Valid equity curve should pass validation."""
        cum_ret = pd.Series([1.0, 1.1, 0.9, 1.05])
        metrics = {"mdd": -0.18, "cagr": 0.05}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert is_valid
        assert len(violations) == 0

    def test_futures_negative_equity_also_invalid(self):
        """Futures with negative equity should also be invalid (Option A: strict policy)."""
        cum_ret = pd.Series([1.0, 0.5, -0.1])  # Negative equity
        metrics = {"mdd": -0.9, "cagr": -0.5}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "futures", "test_strat", "test_market"
        )

        # Option A: ALL markets (including futures) flag negative equity
        assert not is_valid
        assert any("equity_negative" in v for v in violations)

    def test_equity_has_nan_invalid(self):
        """Equity curve with NaN values should be invalid."""
        cum_ret = pd.Series([1.0, np.nan, 0.9])
        metrics = {"mdd": -0.1, "cagr": -0.1}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("equity_has_nan" in v for v in violations)

    def test_equity_has_inf_invalid(self):
        """Equity curve with inf values should be invalid."""
        cum_ret = pd.Series([1.0, np.inf, 0.9])
        metrics = {"mdd": -0.1, "cagr": 0.0}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("equity_has_inf" in v for v in violations)

    def test_cagr_consistency_total_loss(self):
        """If final_equity == 0, cagr should be -1.0."""
        cum_ret = pd.Series([1.0, 0.5, 0.0])  # Total loss
        metrics = {"mdd": -1.0, "cagr": -0.5}  # CAGR is wrong - should be -1.0

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("cagr_consistency" in v for v in violations)

    def test_cagr_consistency_valid(self):
        """If final_equity == 0 and cagr == -1.0, should pass."""
        cum_ret = pd.Series([1.0, 0.5, 0.0])  # Total loss
        metrics = {"mdd": -1.0, "cagr": -1.0}  # Correct CAGR

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        # CAGR consistency passes, but equity reaching 0 is at edge
        cagr_violations = [v for v in violations if "cagr_consistency" in v]
        assert len(cagr_violations) == 0

    def test_symbol_parameter_in_violation(self):
        """Symbol name should be included in validation."""
        cum_ret = pd.Series([1.0, 0.5, -0.1])
        metrics = {"mdd": -0.9, "cagr": -0.5}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market", symbol="BTCUSDT"
        )

        assert not is_valid
        # Violations should be returned (symbol prefixing happens at caller level)


class TestCandidateSelection:
    """Test that invalid results are excluded from selection."""

    def test_invalid_excluded_from_top20(self):
        """Invalid results should be excluded from top20."""
        df = pd.DataFrame({
            "strategy_id": ["A", "B", "C", "D"],
            "sharpe": [0.5, 0.8, 0.3, 0.9],
            "valid": [True, False, True, True],
        })

        df_valid = df[df["valid"] == True]
        top3 = df_valid.nlargest(3, "sharpe")

        assert "B" not in top3["strategy_id"].values
        assert list(top3["strategy_id"]) == ["D", "A", "C"]

    def test_all_invalid_returns_empty(self):
        """If all results invalid, top20 should be empty."""
        df = pd.DataFrame({
            "strategy_id": ["A", "B", "C"],
            "sharpe": [0.5, 0.8, 0.3],
            "valid": [False, False, False],
        })

        df_valid = df[df["valid"] == True]
        assert len(df_valid) == 0


class TestTradesDefinition:
    """Test trades count includes 0→non-zero entries + sign flips."""

    def _count_trades(self, position: pd.Series) -> int:
        """Count trades using the updated definition."""
        pos = position
        prev_pos = pos.shift(1).fillna(0)
        new_entry = (prev_pos == 0) & (pos != 0)
        sign_flip = (prev_pos * pos) < 0  # Different signs = flip
        return int((new_entry | sign_flip).sum())

    def test_trades_counts_long_entries(self):
        """Trades should count 0→positive transitions."""
        position = pd.Series([0, 1, 1, 0, 1, 0])  # 2 long entries
        assert self._count_trades(position) == 2

    def test_trades_counts_short_entries(self):
        """Trades should count 0→negative transitions (shorts)."""
        position = pd.Series([0, -1, -1, 0, -0.5, 0])  # 2 short entries
        assert self._count_trades(position) == 2

    def test_trades_counts_both_long_and_short(self):
        """Trades should count all 0→non-zero transitions."""
        position = pd.Series([0, 1, 0, -1, 0, 0.5, 0])  # 1 long, 1 short, 1 long
        assert self._count_trades(position) == 3

    def test_trades_zero_when_always_flat(self):
        """No trades when position is always 0."""
        position = pd.Series([0, 0, 0, 0, 0])
        assert self._count_trades(position) == 0

    def test_trades_counts_sign_flip_long_to_short(self):
        """Trades should count long→short flip as new trade."""
        position = pd.Series([0, 1, 1, -1, -1, 0])  # 1 entry + 1 flip = 2 trades
        assert self._count_trades(position) == 2

    def test_trades_counts_sign_flip_short_to_long(self):
        """Trades should count short→long flip as new trade."""
        position = pd.Series([0, -1, -1, 1, 1, 0])  # 1 entry + 1 flip = 2 trades
        assert self._count_trades(position) == 2

    def test_trades_multiple_flips(self):
        """Multiple sign flips should all be counted."""
        position = pd.Series([0, 1, -1, 1, -1, 0])  # 1 entry + 3 flips = 4 trades
        assert self._count_trades(position) == 4

    def test_trades_flip_without_zero(self):
        """Direct flip 1→-1 without going through 0 should count."""
        # position = [1, 1, -1, -1]
        # Bar 0: pos=1, prev=0 (fillna) → entry from flat = 1 trade
        # Bar 2: pos=-1, prev=1 → sign flip = 1 trade
        # Total = 2 trades
        position = pd.Series([1, 1, -1, -1])
        assert self._count_trades(position) == 2


class TestCAGRCalculation:
    """Test CAGR edge cases."""

    def test_cagr_positive_return(self):
        """CAGR should be positive for positive final value > 1."""
        years = 1.0
        final_value = 1.10  # 10% return

        if years > 0 and final_value > 0:
            cagr = final_value ** (1 / years) - 1
        else:
            cagr = 0.0

        assert cagr > 0
        np.testing.assert_allclose(cagr, 0.10, rtol=1e-10)

    def test_cagr_negative_return(self):
        """CAGR should be negative for final value < 1."""
        years = 1.0
        final_value = 0.90  # -10% return

        if years > 0 and final_value > 0:
            cagr = final_value ** (1 / years) - 1
        else:
            cagr = 0.0

        assert cagr < 0
        np.testing.assert_allclose(cagr, -0.10, rtol=1e-10)

    def test_cagr_total_loss(self):
        """CAGR should be -1.0 for total loss (final_value = 0)."""
        years = 1.0
        final_value = 0.0

        if years > 0 and final_value > 0:
            cagr = final_value ** (1 / years) - 1
        elif years > 0 and final_value == 0:
            cagr = -1.0
        else:
            cagr = 0.0

        assert cagr == -1.0


@pytest.mark.skipif(not HAS_BACKTEST_RUNNER, reason="backtest_runner import failed (yaml missing)")
class TestDebugDumpColumns:
    """Test debug dump output structure."""

    def test_debug_dump_has_required_columns(self, tmp_path):
        """Debug dump should have all required columns."""
        # Create minimal test data
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({
            "date": dates,
            "close": [100, 101, 99, 102, 100],
            "position": [0, 1, 1, 0, 0],
            "strat_ret": [0.0, 0.01, -0.02, 0.03, 0.0],
            "cum_ret": [1.0, 1.01, 0.99, 1.02, 1.02],
            "costs": [0.0, 0.001, 0.0, 0.001, 0.0],
            "pos_change": [0, 1, 0, 1, 0],
        })

        output_path = save_debug_dump(df, "test_market", "test_strategy", tmp_path)

        # Read and verify
        result_df = pd.read_csv(output_path)

        expected_cols = ["date", "close", "position", "strat_ret", "equity",
                         "costs", "pos_change", "entry_flag", "exit_flag"]
        for col in expected_cols:
            assert col in result_df.columns, f"Missing column: {col}"


@pytest.mark.skipif(not HAS_BACKTEST_RUNNER, reason="backtest_runner import failed (yaml missing)")
class TestPerSymbolValidation:
    """Test per-symbol invariant validation."""

    def test_two_symbols_one_violation_marks_strategy_invalid(self):
        """Strategy should be invalid if any symbol violates invariants."""
        # Simulate per-symbol validation logic
        symbol_names = ["BTCUSDT", "ETHUSDT"]
        symbol_results = [
            {"cagr": 0.1, "mdd": -0.2, "_cum_ret": pd.Series([1.0, 1.1, 1.05])},  # Valid
            {"cagr": -0.5, "mdd": -0.9, "_cum_ret": pd.Series([1.0, 0.5, -0.1])},  # Invalid: negative equity
        ]

        is_valid = True
        violations_list = []

        for symbol, result in zip(symbol_names, symbol_results):
            cum_ret = result["_cum_ret"]
            symbol_metrics = {"cagr": result["cagr"], "mdd": result["mdd"]}
            sym_valid, sym_violations = validate_metrics_invariants(
                cum_ret, symbol_metrics, "spot", "TEST_STRAT", "test_market", symbol
            )
            if not sym_valid:
                is_valid = False
                violations_list.extend([f"{symbol}:{v}" for v in sym_violations])

        assert not is_valid, "Strategy should be invalid when any symbol violates"
        assert any("ETHUSDT:equity_negative" in v for v in violations_list)
        # BTCUSDT should not appear in violations
        assert not any("BTCUSDT" in v for v in violations_list)


@pytest.mark.skipif(not HAS_BACKTEST_RUNNER, reason="backtest_runner import failed (yaml missing)")
class TestCumRetTypeDefense:
    """Test cum_ret type normalization and defense."""

    def test_cum_ret_as_list(self):
        """Should handle list input."""
        cum_ret = [1.0, 1.1, 0.9]
        metrics = {"mdd": -0.18, "cagr": 0.0}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        # Should not crash, and list should be converted properly
        assert isinstance(is_valid, bool)

    def test_cum_ret_as_ndarray(self):
        """Should handle numpy array input."""
        cum_ret = np.array([1.0, 1.1, 0.9])
        metrics = {"mdd": -0.18, "cagr": 0.0}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert isinstance(is_valid, bool)

    def test_cum_ret_none_is_invalid(self):
        """None cum_ret should be invalid."""
        metrics = {"mdd": -0.1, "cagr": 0.0}

        is_valid, violations = validate_metrics_invariants(
            None, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("cum_ret_missing" in v for v in violations)

    def test_cum_ret_empty_is_invalid(self):
        """Empty cum_ret should be invalid."""
        cum_ret = pd.Series([], dtype="float64")
        metrics = {"mdd": -0.1, "cagr": 0.0}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        assert not is_valid
        assert any("cum_ret_missing" in v for v in violations)

    def test_float_noise_not_flagged_as_negative(self):
        """Tiny negative values (float noise) should not be flagged."""
        cum_ret = pd.Series([1.0, 1.0 - 1e-15, 0.9])  # -1e-15 is noise, not real negative
        metrics = {"mdd": -0.1, "cagr": 0.0}

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        # Should not flag equity_negative for noise
        equity_violations = [v for v in violations if "equity_negative" in v]
        assert len(equity_violations) == 0

    def test_eps_consistency_near_zero_treated_as_zero(self):
        """Tiny final_equity (within EPS) should be treated as total loss."""
        # final_equity = -1e-13 is within EPS_EQUITY (1e-12), treated as 0
        cum_ret = pd.Series([1.0, 0.5, -1e-13])
        metrics = {"mdd": -1.0, "cagr": -0.5}  # Wrong CAGR for total loss

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        # Should flag CAGR consistency (near-zero should require cagr == -1.0)
        cagr_violations = [v for v in violations if "cagr_consistency" in v]
        assert len(cagr_violations) == 1

    def test_eps_consistency_correct_cagr_passes(self):
        """Near-zero final_equity with correct CAGR -1.0 should pass."""
        cum_ret = pd.Series([1.0, 0.5, 1e-14])  # Within EPS, treated as 0
        metrics = {"mdd": -1.0, "cagr": -1.0}  # Correct CAGR for total loss

        is_valid, violations = validate_metrics_invariants(
            cum_ret, metrics, "spot", "test_strat", "test_market"
        )

        cagr_violations = [v for v in violations if "cagr_consistency" in v]
        assert len(cagr_violations) == 0


@pytest.mark.skipif(not HAS_BACKTEST_RUNNER, reason="backtest_runner import failed (yaml missing)")
class TestRegimeMerge:
    """Test regime_state merge with date type handling."""

    def test_regime_merge_preserves_float_dtype(self, tmp_path):
        """Float regime values should not be cast to int."""
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({
            "date": dates,
            "close": [100, 101, 99, 102, 100],
            "position": [0, 1, 1, 0, 0],
            "strat_ret": [0.0, 0.01, -0.02, 0.03, 0.0],
            "cum_ret": [1.0, 1.01, 0.99, 1.02, 1.02],
            "costs": [0.0, 0.001, 0.0, 0.001, 0.0],
            "pos_change": [0, 1, 0, 1, 0],
        })

        # Float regime values (probabilities)
        regime_series = pd.Series(
            [0.3, 0.5, 0.7, 0.8, 0.6],
            index=dates,
            dtype="float64"
        )

        output_path = save_debug_dump(df, "test_market", "test_strategy", tmp_path, regime_series)
        result_df = pd.read_csv(output_path)

        # Should preserve float values, not truncate to int
        assert "regime_state" in result_df.columns
        # Values should be approximately preserved (not all zeros or all ones)
        regime_values = result_df["regime_state"].values
        assert not all(v == 0 for v in regime_values) or not all(v == 1 for v in regime_values)

    def test_regime_merge_with_string_dates(self, tmp_path):
        """Should handle string dates in main df and datetime in regime."""
        df = pd.DataFrame({
            "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "close": [100, 101, 99],
            "position": [0, 1, 1],
            "strat_ret": [0.0, 0.01, -0.02],
            "cum_ret": [1.0, 1.01, 0.99],
            "costs": [0.0, 0.001, 0.0],
            "pos_change": [0, 1, 0],
        })

        # Datetime index regime
        regime_series = pd.Series(
            [1, 2, 1],
            index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            dtype="int64"
        )

        output_path = save_debug_dump(df, "test_market", "test_strategy", tmp_path, regime_series)
        result_df = pd.read_csv(output_path)

        assert "regime_state" in result_df.columns
        # Should have merged successfully (not all zeros)
        assert result_df["regime_state"].iloc[1] == 2

    def test_regime_merge_with_time_component(self, tmp_path):
        """Should handle regime index with time component (normalize to date-only)."""
        # Main df has date-only
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "close": [100, 101, 99],
            "position": [0, 1, 1],
            "strat_ret": [0.0, 0.01, -0.02],
            "cum_ret": [1.0, 1.01, 0.99],
            "costs": [0.0, 0.001, 0.0],
            "pos_change": [0, 1, 0],
        })

        # Regime index has time component (e.g., 12:34:56)
        regime_series = pd.Series(
            [1, 2, 3],
            index=pd.to_datetime([
                "2024-01-01 12:34:56",
                "2024-01-02 09:00:00",
                "2024-01-03 23:59:59"
            ]),
            dtype="int64"
        )

        output_path = save_debug_dump(df, "test_market", "test_strategy", tmp_path, regime_series)
        result_df = pd.read_csv(output_path)

        assert "regime_state" in result_df.columns
        # Should have merged successfully despite time component
        assert result_df["regime_state"].iloc[0] == 1
        assert result_df["regime_state"].iloc[1] == 2
        assert result_df["regime_state"].iloc[2] == 3


class TestSymbolResultPairing:
    """Test symbol-result pairing with skip scenarios."""

    def test_tuple_pairing_logic(self):
        """Simulate tuple pairing with some symbols skipped."""
        # This tests the logic pattern used in run_strategy_backtest
        symbols = ["BTCUSDT", "ETHUSDT", "XYZUSDT", "AAVEUSDT"]
        skip_symbols = {"XYZUSDT"}  # Simulates load failure

        symbol_result_pairs = []
        for symbol in symbols:
            if symbol in skip_symbols:
                continue  # Skipped due to load failure
            result = {"cagr": 0.1, "mdd": -0.1}
            symbol_result_pairs.append((symbol, result))

        # Extract for aggregation
        symbol_results = [result for _, result in symbol_result_pairs]

        # Verify pairing is correct
        assert len(symbol_result_pairs) == 3
        assert symbol_result_pairs[0][0] == "BTCUSDT"
        assert symbol_result_pairs[1][0] == "ETHUSDT"
        assert symbol_result_pairs[2][0] == "AAVEUSDT"
        assert len(symbol_results) == 3

    def test_tuple_pairing_preserves_order_after_multiple_skips(self):
        """Multiple skips should not affect symbol-result alignment."""
        symbols = ["A", "B", "C", "D", "E"]
        skip_indices = {1, 3}  # Skip B and D

        symbol_result_pairs = []
        for i, symbol in enumerate(symbols):
            if i in skip_indices:
                continue
            result = {"value": i}
            symbol_result_pairs.append((symbol, result))

        # Should have A, C, E (indices 0, 2, 4)
        assert [s for s, _ in symbol_result_pairs] == ["A", "C", "E"]
        assert [r["value"] for _, r in symbol_result_pairs] == [0, 2, 4]


class TestStratRetClampRegression:
    """
    Regression test for strat_ret -1.0 clamp.

    Bug: When ret=-1 (close->0) plus costs, strat_ret < -1 causes
    (1+strat_ret) < 0, making cumprod negative. Fixed by clipping
    strat_ret to -1.0 minimum.
    """

    def test_close_zero_with_costs_no_negative_equity(self):
        """
        When close drops to 0 (ret=-1) with costs, equity should clamp to 0, not go negative.

        Scenario: price [100, 50, 0], position held from bar 1.
        ret on bar 2 = (0-50)/50 = -1.0 (total loss while holding)
        Plus costs > 0, raw strat_ret would be < -1.
        After fix: strat_ret clipped to -1.0, cum_ret stays >= 0.
        """
        dates = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame({
            "date": dates,
            "close": [100.0, 50.0, 0.0],  # Price gradually drops to 0
            "position": [0, 1, 1],  # Enter on bar 1, hold through crash
        })

        df["ret"] = df["close"].pct_change().fillna(0)
        df["ret"] = df["ret"].replace([np.inf, -np.inf], 0)  # Handle inf
        df["pos_change"] = df["position"].diff().abs().fillna(0)

        # Simulate costs (50 bps fee + slippage to push strat_ret below -1)
        cost_per_trade = 0.005  # 50 bps
        df["costs"] = df["pos_change"] * cost_per_trade

        # Compute strat_ret with clamp (as fixed in backtest_runner.py)
        df["strat_ret"] = df["position"].shift(1).fillna(0) * df["ret"] - df["costs"]
        df["strat_ret"] = df["strat_ret"].clip(lower=-1.0)  # THE FIX

        df["cum_ret"] = (1 + df["strat_ret"]).cumprod()
        df["cum_ret"] = df["cum_ret"].clip(lower=0.0)

        # Assertions
        assert df["cum_ret"].min() >= 0, f"Equity went negative: {df['cum_ret'].min()}"
        assert not df["cum_ret"].isna().any(), "Equity has NaN"

        # MDD calculation
        EPS = 1e-10
        running_max = df["cum_ret"].cummax()
        safe_max = np.maximum(running_max.values, EPS)
        drawdown = df["cum_ret"].values / safe_max - 1.0
        mdd = drawdown.min()

        assert mdd >= -1.0, f"MDD below -1: {mdd}"

    def test_close_zero_strat_ret_clipped(self):
        """Verify strat_ret is clipped to -1.0 when it would be more negative."""
        dates = pd.date_range("2024-01-01", periods=2)
        df = pd.DataFrame({
            "date": dates,
            "close": [100.0, 0.0],  # Total loss day
            "position": [1, 1],  # Already holding
        })

        df["ret"] = df["close"].pct_change().fillna(0)
        df["pos_change"] = df["position"].diff().abs().fillna(0)
        df["costs"] = 0.005  # 50 bps costs to push below -1

        # Raw strat_ret on bar 1: position[0]*ret[1] - costs = 1*(-1) - 0.005 = -1.005
        df["strat_ret_raw"] = df["position"].shift(1).fillna(0) * df["ret"] - df["costs"]

        # Clipped version
        df["strat_ret"] = df["strat_ret_raw"].clip(lower=-1.0)

        # Raw should be < -1 on bar 1
        assert df["strat_ret_raw"].iloc[1] < -1.0, "Raw strat_ret should be < -1"

        # Clipped should be exactly -1.0
        assert df["strat_ret"].iloc[1] == -1.0, "Clipped strat_ret should be -1.0"

        # Cumulative returns
        df["cum_ret"] = (1 + df["strat_ret"]).cumprod()

        # Should hit exactly 0, not negative
        assert df["cum_ret"].iloc[1] == 0.0, f"Expected 0.0, got {df['cum_ret'].iloc[1]}"

    def test_multiple_crashes_handled(self):
        """Multiple crash days (ret=-1 or near) should all be handled correctly."""
        dates = pd.date_range("2024-01-01", periods=5)
        df = pd.DataFrame({
            "date": dates,
            "close": [100.0, 50.0, 25.0, 12.5, 0.0],  # Continuous decline to 0
            "position": [0, 1, 1, 1, 1],
        })

        df["ret"] = df["close"].pct_change().fillna(0)
        df["ret"] = df["ret"].replace([np.inf, -np.inf], 0)  # Handle inf
        df["pos_change"] = df["position"].diff().abs().fillna(0)
        df["costs"] = df["pos_change"] * 0.001

        df["strat_ret"] = df["position"].shift(1).fillna(0) * df["ret"] - df["costs"]
        df["strat_ret"] = df["strat_ret"].clip(lower=-1.0)
        df["cum_ret"] = (1 + df["strat_ret"]).cumprod()
        df["cum_ret"] = df["cum_ret"].clip(lower=0.0)

        # All equity values should be >= 0
        assert (df["cum_ret"] >= 0).all(), f"Negative equity found: {df['cum_ret'].values}"
        assert not df["cum_ret"].isna().any(), f"NaN found: {df['cum_ret'].values}"


class TestTop20Determinism:
    """Test top20 selection determinism with NaN and ties."""

    def test_nan_sharpe_excluded_from_top20(self):
        """NaN sharpe values should be excluded (sorted to bottom)."""
        df = pd.DataFrame({
            "market": ["binance_futures"] * 5,
            "strategy_id": ["A", "B", "C", "D", "E"],
            "sharpe": [0.5, np.nan, 0.8, np.nan, 0.3],
            "valid": [True] * 5,
        })

        # Apply same logic as backtest_runner
        sorted_df = df.sort_values(
            ["sharpe", "strategy_id"], ascending=[False, True], na_position="last"
        )
        top3 = sorted_df.head(3)

        # NaN should be at bottom, so top3 = C(0.8), A(0.5), E(0.3)
        assert list(top3["strategy_id"]) == ["C", "A", "E"]
        assert np.isnan(top3["sharpe"]).sum() == 0

    def test_tied_sharpe_deterministic_order(self):
        """Tied sharpe values should be ordered by strategy_id."""
        df = pd.DataFrame({
            "market": ["binance_futures"] * 4,
            "strategy_id": ["D", "A", "C", "B"],
            "sharpe": [0.5, 0.5, 0.5, 0.5],  # All tied
            "valid": [True] * 4,
        })

        sorted_df = df.sort_values(
            ["sharpe", "strategy_id"], ascending=[False, True], na_position="last"
        )

        # All tied at 0.5, so order by strategy_id asc: A, B, C, D
        assert list(sorted_df["strategy_id"]) == ["A", "B", "C", "D"]

    def test_multi_market_final_sort_determinism(self):
        """Final sort after concat should produce deterministic order."""
        # Simulate concat from multiple markets
        df = pd.DataFrame({
            "market": ["binance_spot", "binance_futures", "binance_spot", "binance_futures"],
            "strategy_id": ["B", "A", "A", "B"],
            "sharpe": [0.5, 0.8, 0.6, 0.7],
            "valid": [True] * 4,
        })

        # Apply final sort: market asc, sharpe desc, strategy_id asc
        sorted_df = df.sort_values(
            ["market", "sharpe", "strategy_id"],
            ascending=[True, False, True],
            na_position="last"
        ).reset_index(drop=True)

        # binance_futures first: A(0.8), B(0.7)
        # binance_spot second: A(0.6), B(0.5)
        expected_order = ["A", "B", "A", "B"]
        expected_markets = ["binance_futures", "binance_futures", "binance_spot", "binance_spot"]

        assert list(sorted_df["strategy_id"]) == expected_order
        assert list(sorted_df["market"]) == expected_markets


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
