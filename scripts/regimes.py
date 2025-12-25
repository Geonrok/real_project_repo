#!/usr/bin/env python3
"""
Regime calculation module for Stage1 backtest.

Purpose: Generate regime time series based on BTC/ETH market conditions.

Regime Types:
- BTC_GT_MA50: BTC > MA50(BTC)
- ETH_GT_MA50: ETH > MA50(ETH)
- BTC_OR_ETH_GT_MA50: BTC > MA50(BTC) OR ETH > MA50(ETH)
- SCORE_AVG_GT_1: (BTC/MA50_BTC + ETH/MA50_ETH) / 2 > 1.0
- INDEX_GT_MA50: Index(0.5*ret_BTC + 0.5*ret_ETH) > MA50(Index)

Regime Source Modes:
- reference_market: Use binance_spot BTC/ETH for all markets (recommended)
- per_market: Use each market's own BTC/ETH
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def load_symbol_data(
    normalized_dir: Path,
    market: str,
    symbol: str
) -> pd.DataFrame | None:
    """
    Load normalized data for a specific symbol.

    Args:
        normalized_dir: Path to normalized_1d directory
        market: Market name (e.g., binance_spot)
        symbol: Symbol name (e.g., BTC)

    Returns:
        DataFrame with date, close columns or None if not found
    """
    # Try common symbol variations
    symbol_variants = [
        symbol,
        f"{symbol}USDT",
        f"{symbol}KRW",
        symbol.upper(),
        symbol.lower(),
    ]

    market_dir = normalized_dir / market
    if not market_dir.exists():
        return None

    for variant in symbol_variants:
        csv_path = market_dir / f"{variant}.csv"
        parquet_path = market_dir / f"{variant}.parquet"

        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "close"]].rename(columns={"close": symbol})
        elif parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            df["date"] = pd.to_datetime(df["date"])
            return df[["date", "close"]].rename(columns={"close": symbol})

    return None


def load_reference_prices(
    normalized_dir: Path,
    reference_market: str = "binance_spot"
) -> pd.DataFrame:
    """
    Load BTC and ETH prices from reference market.

    Args:
        normalized_dir: Path to normalized_1d directory
        reference_market: Market to use as reference

    Returns:
        DataFrame with date, BTC, ETH columns
    """
    btc_df = load_symbol_data(normalized_dir, reference_market, "BTC")
    eth_df = load_symbol_data(normalized_dir, reference_market, "ETH")

    if btc_df is None or eth_df is None:
        raise ValueError(f"BTC or ETH not found in {reference_market}")

    # Merge on date
    prices = btc_df.merge(eth_df, on="date", how="outer").sort_values("date")
    prices = prices.ffill().bfill()  # Forward/backward fill gaps

    return prices


def compute_regime_btc_gt_ma50(prices: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    """
    Regime: BTC > MA50(BTC)

    Returns:
        Series with 1 (regime on) or 0 (regime off)
    """
    btc_ma = _sma(prices["BTC"], ma_period)
    regime = (prices["BTC"] > btc_ma).astype(int)
    return pd.Series(regime.values, index=prices["date"], name="BTC_GT_MA50")


def compute_regime_eth_gt_ma50(prices: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    """
    Regime: ETH > MA50(ETH)

    Returns:
        Series with 1 (regime on) or 0 (regime off)
    """
    eth_ma = _sma(prices["ETH"], ma_period)
    regime = (prices["ETH"] > eth_ma).astype(int)
    return pd.Series(regime.values, index=prices["date"], name="ETH_GT_MA50")


def compute_regime_btc_or_eth_gt_ma50(prices: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    """
    Regime: BTC > MA50(BTC) OR ETH > MA50(ETH)

    Returns:
        Series with 1 (regime on) or 0 (regime off)
    """
    btc_ma = _sma(prices["BTC"], ma_period)
    eth_ma = _sma(prices["ETH"], ma_period)
    regime = ((prices["BTC"] > btc_ma) | (prices["ETH"] > eth_ma)).astype(int)
    return pd.Series(regime.values, index=prices["date"], name="BTC_OR_ETH_GT_MA50")


def compute_regime_score_avg_gt_1(prices: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    """
    Regime: (BTC/MA50_BTC + ETH/MA50_ETH) / 2 > 1.0

    This measures average relative strength of BTC and ETH vs their MAs.

    Returns:
        Series with 1 (regime on) or 0 (regime off)
    """
    btc_ma = _sma(prices["BTC"], ma_period)
    eth_ma = _sma(prices["ETH"], ma_period)

    btc_score = prices["BTC"] / btc_ma.replace(0, np.nan)
    eth_score = prices["ETH"] / eth_ma.replace(0, np.nan)

    avg_score = (btc_score + eth_score) / 2
    regime = (avg_score > 1.0).astype(int)

    return pd.Series(regime.values, index=prices["date"], name="SCORE_AVG_GT_1")


def compute_regime_score_avg_strength(prices: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    """
    Regime strength for SIZING mode: (BTC/MA50_BTC + ETH/MA50_ETH) / 2

    Returns value between 0 and 2+ (centered around 1.0).

    Returns:
        Series with continuous strength values
    """
    btc_ma = _sma(prices["BTC"], ma_period)
    eth_ma = _sma(prices["ETH"], ma_period)

    btc_score = prices["BTC"] / btc_ma.replace(0, np.nan)
    eth_score = prices["ETH"] / eth_ma.replace(0, np.nan)

    avg_score = (btc_score + eth_score) / 2
    return pd.Series(avg_score.values, index=prices["date"], name="SCORE_AVG_STRENGTH")


def compute_regime_index_gt_ma50(prices: pd.DataFrame, ma_period: int = 50) -> pd.Series:
    """
    Regime: Index > MA50(Index)
    Where Index = cumulative return of 0.5*BTC_ret + 0.5*ETH_ret

    Returns:
        Series with 1 (regime on) or 0 (regime off)
    """
    # Daily returns
    btc_ret = prices["BTC"].pct_change().fillna(0)
    eth_ret = prices["ETH"].pct_change().fillna(0)

    # Blended return
    blended_ret = 0.5 * btc_ret + 0.5 * eth_ret

    # Cumulative index (starting at 100)
    index = (1 + blended_ret).cumprod() * 100

    # Index vs MA50
    index_ma = _sma(index, ma_period)
    regime = (index > index_ma).astype(int)

    return pd.Series(regime.values, index=prices["date"], name="INDEX_GT_MA50")


def compute_regime(
    regime_name: str,
    prices: pd.DataFrame,
    ma_period: int = 50
) -> pd.Series:
    """
    Compute regime by name.

    Args:
        regime_name: One of the supported regime types
        prices: DataFrame with date, BTC, ETH columns
        ma_period: Moving average period

    Returns:
        Series with regime flag (0 or 1) indexed by date
    """
    if regime_name == "BTC_GT_MA50":
        return compute_regime_btc_gt_ma50(prices, ma_period)
    elif regime_name == "ETH_GT_MA50":
        return compute_regime_eth_gt_ma50(prices, ma_period)
    elif regime_name == "BTC_OR_ETH_GT_MA50":
        return compute_regime_btc_or_eth_gt_ma50(prices, ma_period)
    elif regime_name == "SCORE_AVG_GT_1":
        return compute_regime_score_avg_gt_1(prices, ma_period)
    elif regime_name == "INDEX_GT_MA50":
        return compute_regime_index_gt_ma50(prices, ma_period)
    else:
        raise ValueError(f"Unknown regime: {regime_name}")


def compute_regime_strength(
    regime_name: str,
    prices: pd.DataFrame,
    ma_period: int = 50
) -> pd.Series:
    """
    Compute regime strength for SIZING mode.

    For most regimes, this returns a normalized strength between 0 and 1.

    Args:
        regime_name: Regime type
        prices: DataFrame with date, BTC, ETH columns
        ma_period: Moving average period

    Returns:
        Series with continuous strength values (0 to 1 scale)
    """
    if regime_name == "SCORE_AVG_GT_1":
        # Use score directly, scale to 0-1 range
        raw_strength = compute_regime_score_avg_strength(prices, ma_period)
        # Clip to reasonable range and normalize
        strength = (raw_strength - 0.8).clip(0, 0.4) / 0.4  # 0.8-1.2 -> 0-1
        return pd.Series(strength.values, index=prices["date"], name="REGIME_STRENGTH")
    else:
        # For binary regimes, use the binary value as strength
        return compute_regime(regime_name, prices, ma_period).astype(float)


class RegimeCalculator:
    """
    Calculator for regime time series.

    Supports two modes:
    - reference_market: Use a single market's BTC/ETH for all calculations
    - per_market: Use each market's own BTC/ETH
    """

    def __init__(
        self,
        normalized_dir: Path,
        source_mode: Literal["reference_market", "per_market"] = "reference_market",
        reference_market: str = "binance_spot",
        ma_period: int = 50
    ):
        """
        Initialize regime calculator.

        Args:
            normalized_dir: Path to normalized data directory
            source_mode: "reference_market" or "per_market"
            reference_market: Market to use when source_mode is "reference_market"
            ma_period: Moving average period for regime calculations
        """
        self.normalized_dir = Path(normalized_dir)
        self.source_mode = source_mode
        self.reference_market = reference_market
        self.ma_period = ma_period

        # Cache for loaded prices
        self._price_cache: dict[str, pd.DataFrame] = {}

        # Load reference prices if using reference_market mode
        if source_mode == "reference_market":
            self._reference_prices = load_reference_prices(normalized_dir, reference_market)
        else:
            self._reference_prices = None

    def get_prices(self, market: str) -> pd.DataFrame:
        """Get BTC/ETH prices for a market."""
        if self.source_mode == "reference_market":
            return self._reference_prices

        if market not in self._price_cache:
            try:
                self._price_cache[market] = load_reference_prices(self.normalized_dir, market)
            except ValueError:
                # Fallback to reference market if market doesn't have BTC/ETH
                if self._reference_prices is None:
                    self._reference_prices = load_reference_prices(
                        self.normalized_dir, self.reference_market
                    )
                self._price_cache[market] = self._reference_prices

        return self._price_cache[market]

    def get_regime(self, regime_name: str, market: str) -> pd.Series:
        """
        Get regime time series for a market.

        Args:
            regime_name: Regime type
            market: Market name

        Returns:
            Series with regime flag indexed by date
        """
        prices = self.get_prices(market)
        return compute_regime(regime_name, prices, self.ma_period)

    def get_regime_strength(self, regime_name: str, market: str) -> pd.Series:
        """
        Get regime strength for SIZING mode.

        Args:
            regime_name: Regime type
            market: Market name

        Returns:
            Series with strength values indexed by date
        """
        prices = self.get_prices(market)
        return compute_regime_strength(regime_name, prices, self.ma_period)
