"""
Smoke tests for Stage1 backtest runner components.

Purpose: Verify indicators, regimes, and backtest core functions execute without error.
Uses dummy OHLCV data (no real data dependency).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from indicators import (
    kama, vidya, boll_mid, supertrend, frama, zlema, vma, tsmom,
    compute_trend_indicator, compute_momentum_indicator
)
from regimes import (
    compute_regime_btc_gt_ma50,
    compute_regime_eth_gt_ma50,
    compute_regime_btc_or_eth_gt_ma50,
    compute_regime_score_avg_gt_1,
    compute_regime_index_gt_ma50,
    compute_regime,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def dummy_ohlcv() -> pd.DataFrame:
    """Generate dummy OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Generate random walk for close prices
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    # Generate OHLC from close
    high = close * (1 + np.abs(np.random.randn(n)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n)) * 0.01)
    open_ = close * (1 + np.random.randn(n) * 0.005)
    volume = np.random.randint(1000, 10000, n).astype(float)

    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


@pytest.fixture
def dummy_btc_eth_prices() -> pd.DataFrame:
    """Generate dummy BTC/ETH prices for regime testing."""
    np.random.seed(42)
    n = 100

    dates = pd.date_range("2023-01-01", periods=n, freq="D")

    btc_returns = np.random.randn(n) * 0.03
    btc = 30000 * np.exp(np.cumsum(btc_returns))

    eth_returns = np.random.randn(n) * 0.04
    eth = 2000 * np.exp(np.cumsum(eth_returns))

    return pd.DataFrame({
        "date": dates,
        "BTC": btc,
        "ETH": eth,
    })


# =============================================================================
# Indicator Tests
# =============================================================================

class TestIndicators:
    """Smoke tests for technical indicators."""

    def test_kama_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """KAMA indicator executes without error."""
        result = kama(dummy_ohlcv["close"])
        assert len(result) == len(dummy_ohlcv)
        assert not result.isna().all()

    def test_vidya_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """VIDYA indicator executes without error."""
        result = vidya(dummy_ohlcv["close"])
        assert len(result) == len(dummy_ohlcv)
        assert not result.isna().all()

    def test_boll_mid_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """Bollinger Mid indicator executes without error."""
        result = boll_mid(dummy_ohlcv["close"])
        assert len(result) == len(dummy_ohlcv)
        assert not result.isna().all()

    def test_supertrend_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """Supertrend indicator executes without error."""
        result = supertrend(
            dummy_ohlcv["high"],
            dummy_ohlcv["low"],
            dummy_ohlcv["close"]
        )
        assert len(result) == len(dummy_ohlcv)
        assert not result.isna().all()

    def test_frama_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """FRAMA indicator executes without error."""
        result = frama(dummy_ohlcv["close"])
        assert len(result) == len(dummy_ohlcv)
        assert not result.isna().all()

    def test_zlema_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """ZLEMA indicator executes without error."""
        result = zlema(dummy_ohlcv["close"])
        assert len(result) == len(dummy_ohlcv)
        assert not result.isna().all()

    def test_vma_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """VMA indicator executes without error."""
        result = vma(dummy_ohlcv["close"])
        assert len(result) == len(dummy_ohlcv)
        assert not result.isna().all()

    def test_tsmom_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """TSMOM indicator executes without error."""
        result = tsmom(dummy_ohlcv["close"], lookback=30)
        assert len(result) == len(dummy_ohlcv)
        # First 30 values will be NaN due to lookback
        assert not result.iloc[30:].isna().all()

    def test_compute_trend_indicator_dispatcher(self, dummy_ohlcv: pd.DataFrame) -> None:
        """Trend indicator dispatcher works for all indicator types."""
        indicators = ["KAMA", "VIDYA", "BOLL_MID", "FRAMA", "ZLEMA", "VMA"]

        for ind_name in indicators:
            result = compute_trend_indicator(ind_name, dummy_ohlcv["close"])
            assert len(result) == len(dummy_ohlcv), f"{ind_name} failed"

    def test_compute_trend_indicator_supertrend(self, dummy_ohlcv: pd.DataFrame) -> None:
        """Supertrend dispatcher requires high/low."""
        result = compute_trend_indicator(
            "SUPERTREND",
            dummy_ohlcv["close"],
            dummy_ohlcv["high"],
            dummy_ohlcv["low"]
        )
        assert len(result) == len(dummy_ohlcv)

    def test_compute_momentum_indicator_dispatcher(self, dummy_ohlcv: pd.DataFrame) -> None:
        """Momentum indicator dispatcher works."""
        result = compute_momentum_indicator("TSMOM_30", dummy_ohlcv["close"])
        assert len(result) == len(dummy_ohlcv)


# =============================================================================
# Regime Tests
# =============================================================================

class TestRegimes:
    """Smoke tests for regime calculations."""

    def test_btc_gt_ma50_runs(self, dummy_btc_eth_prices: pd.DataFrame) -> None:
        """BTC > MA50 regime executes without error."""
        result = compute_regime_btc_gt_ma50(dummy_btc_eth_prices)
        assert len(result) == len(dummy_btc_eth_prices)
        assert set(result.unique()).issubset({0, 1})

    def test_eth_gt_ma50_runs(self, dummy_btc_eth_prices: pd.DataFrame) -> None:
        """ETH > MA50 regime executes without error."""
        result = compute_regime_eth_gt_ma50(dummy_btc_eth_prices)
        assert len(result) == len(dummy_btc_eth_prices)
        assert set(result.unique()).issubset({0, 1})

    def test_btc_or_eth_gt_ma50_runs(self, dummy_btc_eth_prices: pd.DataFrame) -> None:
        """BTC OR ETH > MA50 regime executes without error."""
        result = compute_regime_btc_or_eth_gt_ma50(dummy_btc_eth_prices)
        assert len(result) == len(dummy_btc_eth_prices)
        assert set(result.unique()).issubset({0, 1})

    def test_score_avg_gt_1_runs(self, dummy_btc_eth_prices: pd.DataFrame) -> None:
        """Score average > 1 regime executes without error."""
        result = compute_regime_score_avg_gt_1(dummy_btc_eth_prices)
        assert len(result) == len(dummy_btc_eth_prices)
        assert set(result.unique()).issubset({0, 1})

    def test_index_gt_ma50_runs(self, dummy_btc_eth_prices: pd.DataFrame) -> None:
        """Index > MA50 regime executes without error."""
        result = compute_regime_index_gt_ma50(dummy_btc_eth_prices)
        assert len(result) == len(dummy_btc_eth_prices)
        assert set(result.unique()).issubset({0, 1})

    def test_compute_regime_dispatcher(self, dummy_btc_eth_prices: pd.DataFrame) -> None:
        """Regime dispatcher works for all regime types."""
        regimes = [
            "BTC_GT_MA50",
            "ETH_GT_MA50",
            "BTC_OR_ETH_GT_MA50",
            "SCORE_AVG_GT_1",
            "INDEX_GT_MA50",
        ]

        for regime_name in regimes:
            result = compute_regime(regime_name, dummy_btc_eth_prices)
            assert len(result) == len(dummy_btc_eth_prices), f"{regime_name} failed"


# =============================================================================
# Backtest Core Tests
# =============================================================================

class TestBacktestCore:
    """Smoke tests for backtest core functions."""

    def test_backtest_symbol_runs(self, dummy_ohlcv: pd.DataFrame) -> None:
        """Symbol backtest executes without error."""
        from backtest_runner import backtest_symbol

        # Create simple signals DataFrame
        signals = pd.DataFrame({
            "date": dummy_ohlcv["date"],
            "close": dummy_ohlcv["close"],
            "position": [0] * 30 + [1] * 40 + [0] * 30,  # Simple position pattern
            "trend_ind": dummy_ohlcv["close"].rolling(20).mean(),
            "momentum_ind": dummy_ohlcv["close"].pct_change(30),
        })

        result = backtest_symbol(signals, fee_bps=10, slippage_bps=2)

        assert "cagr" in result
        assert "sharpe" in result
        assert "mdd" in result
        assert "win_rate" in result
        assert "turnover" in result
        assert "trades" in result

    def test_aggregate_symbol_results_runs(self) -> None:
        """Symbol aggregation executes without error."""
        from backtest_runner import aggregate_symbol_results

        results = [
            {"cagr": 0.10, "sharpe": 1.0, "mdd": -0.15, "win_rate": 0.55, "turnover": 2.0, "trades": 10},
            {"cagr": 0.15, "sharpe": 1.2, "mdd": -0.20, "win_rate": 0.50, "turnover": 2.5, "trades": 12},
        ]

        agg = aggregate_symbol_results(results)

        assert "cagr" in agg
        assert agg["mdd"] == -0.20  # Worst drawdown

    def test_results_to_dataframe_runs(self) -> None:
        """Results to DataFrame conversion works."""
        from backtest_runner import results_to_dataframe, BacktestResult

        results = [
            BacktestResult(
                market="test_market",
                strategy_id="KAMA_TSMOM_30_BTC_GT_MA50_GATE",
                trend="KAMA",
                momentum="TSMOM_30",
                regime="BTC_GT_MA50",
                regime_mode="GATE",
                cagr=0.12,
                sharpe=1.1,
                mdd=-0.18,
                win_rate=0.52,
                turnover=2.2,
                trades=15,
                start_date="2023-01-01",
                end_date="2023-12-31",
                days=365,
            )
        ]

        df = results_to_dataframe(results)

        assert len(df) == 1
        assert "market" in df.columns
        assert "cagr" in df.columns


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
