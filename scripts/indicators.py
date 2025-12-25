#!/usr/bin/env python3
"""
Technical indicators for Stage1 backtest.

Purpose: Provide trend and momentum indicators for strategy grid backtesting.

Indicators implemented:
- Trend: KAMA, VIDYA, BOLL_MID, SUPERTREND, FRAMA, ZLEMA, VMA
- Momentum: TSMOM_30 (30-day time-series momentum using log returns)

All functions take pandas Series and return pandas Series with same index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# =============================================================================
# Helper functions
# =============================================================================

def _ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, adjust=False).mean()


def _sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=1).mean()


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Calculate True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Average True Range."""
    tr = _true_range(high, low, close)
    return tr.rolling(window=period, min_periods=1).mean()


# =============================================================================
# Trend Indicators
# =============================================================================

def kama(
    close: pd.Series,
    period: int = 10,
    fast: int = 2,
    slow: int = 30
) -> pd.Series:
    """
    Kaufman Adaptive Moving Average (KAMA).

    Adapts to market volatility using efficiency ratio.

    Args:
        close: Close prices
        period: Efficiency ratio period
        fast: Fast EMA period for smoothing constant
        slow: Slow EMA period for smoothing constant

    Returns:
        KAMA series
    """
    # Efficiency Ratio
    change = (close - close.shift(period)).abs()
    volatility = close.diff().abs().rolling(window=period).sum()
    er = change / volatility.replace(0, np.nan)
    er = er.fillna(0)

    # Smoothing constants
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

    # Calculate KAMA
    kama_values = np.zeros(len(close))
    kama_values[:] = np.nan

    # Initialize with first valid close
    first_valid = close.first_valid_index()
    if first_valid is not None:
        start_idx = close.index.get_loc(first_valid)
        kama_values[start_idx] = close.iloc[start_idx]

        for i in range(start_idx + 1, len(close)):
            if not np.isnan(sc.iloc[i]):
                kama_values[i] = kama_values[i-1] + sc.iloc[i] * (close.iloc[i] - kama_values[i-1])
            else:
                kama_values[i] = kama_values[i-1]

    return pd.Series(kama_values, index=close.index, name="KAMA")


def vidya(
    close: pd.Series,
    period: int = 14,
    cmo_period: int = 9
) -> pd.Series:
    """
    Variable Index Dynamic Average (VIDYA).

    Uses Chande Momentum Oscillator for adaptive smoothing.

    Args:
        close: Close prices
        period: VIDYA period
        cmo_period: CMO calculation period

    Returns:
        VIDYA series
    """
    # Chande Momentum Oscillator (CMO)
    diff = close.diff()
    gains = diff.where(diff > 0, 0).rolling(window=cmo_period).sum()
    losses = (-diff).where(diff < 0, 0).rolling(window=cmo_period).sum()
    cmo = ((gains - losses) / (gains + losses).replace(0, np.nan)).abs()
    cmo = cmo.fillna(0)

    # Smoothing constant
    alpha = 2 / (period + 1)
    sc = alpha * cmo

    # Calculate VIDYA
    vidya_values = np.zeros(len(close))
    vidya_values[:] = np.nan

    first_valid = close.first_valid_index()
    if first_valid is not None:
        start_idx = close.index.get_loc(first_valid)
        vidya_values[start_idx] = close.iloc[start_idx]

        for i in range(start_idx + 1, len(close)):
            if not np.isnan(sc.iloc[i]):
                vidya_values[i] = sc.iloc[i] * close.iloc[i] + (1 - sc.iloc[i]) * vidya_values[i-1]
            else:
                vidya_values[i] = vidya_values[i-1]

    return pd.Series(vidya_values, index=close.index, name="VIDYA")


def boll_mid(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Bollinger Band Middle (Simple Moving Average).

    Args:
        close: Close prices
        period: SMA period

    Returns:
        Middle band (SMA) series
    """
    return _sma(close, period).rename("BOLL_MID")


def supertrend(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 10,
    multiplier: float = 3.0
) -> pd.Series:
    """
    Supertrend indicator.

    Trend-following indicator based on ATR.

    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        multiplier: ATR multiplier

    Returns:
        Supertrend series (acts as support/resistance)
    """
    atr = _atr(high, low, close, period)
    hl2 = (high + low) / 2

    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr

    supertrend_values = np.zeros(len(close))
    direction = np.zeros(len(close))

    supertrend_values[0] = upper_band.iloc[0]
    direction[0] = 1

    for i in range(1, len(close)):
        if close.iloc[i] > supertrend_values[i-1]:
            direction[i] = 1
        elif close.iloc[i] < supertrend_values[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        if direction[i] == 1:
            supertrend_values[i] = max(lower_band.iloc[i], supertrend_values[i-1]) if direction[i-1] == 1 else lower_band.iloc[i]
        else:
            supertrend_values[i] = min(upper_band.iloc[i], supertrend_values[i-1]) if direction[i-1] == -1 else upper_band.iloc[i]

    return pd.Series(supertrend_values, index=close.index, name="SUPERTREND")


def frama(close: pd.Series, period: int = 16) -> pd.Series:
    """
    Fractal Adaptive Moving Average (FRAMA).

    Uses fractal dimension for adaptive smoothing.

    Args:
        close: Close prices
        period: Base period (should be even)

    Returns:
        FRAMA series
    """
    half_period = period // 2

    # Calculate fractal dimension components
    n1 = (close.rolling(half_period).max() - close.rolling(half_period).min()) / half_period
    n2_high = close.shift(half_period).rolling(half_period).max()
    n2_low = close.shift(half_period).rolling(half_period).min()
    n2 = (n2_high - n2_low) / half_period
    n3 = (close.rolling(period).max() - close.rolling(period).min()) / period

    # Fractal dimension
    d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    d = d.replace([np.inf, -np.inf], np.nan).fillna(1)

    # Alpha (smoothing factor)
    alpha = np.exp(-4.6 * (d - 1))
    alpha = alpha.clip(0.01, 1)

    # Calculate FRAMA
    frama_values = np.zeros(len(close))
    frama_values[:] = np.nan

    first_valid = close.first_valid_index()
    if first_valid is not None:
        start_idx = close.index.get_loc(first_valid)
        frama_values[start_idx] = close.iloc[start_idx]

        for i in range(start_idx + 1, len(close)):
            if not np.isnan(alpha.iloc[i]):
                frama_values[i] = alpha.iloc[i] * close.iloc[i] + (1 - alpha.iloc[i]) * frama_values[i-1]
            else:
                frama_values[i] = frama_values[i-1]

    return pd.Series(frama_values, index=close.index, name="FRAMA")


def zlema(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Zero Lag Exponential Moving Average (ZLEMA).

    Reduces lag by de-lagging the price before EMA calculation.

    Args:
        close: Close prices
        period: EMA period

    Returns:
        ZLEMA series
    """
    lag = (period - 1) // 2
    delagged = 2 * close - close.shift(lag)
    return _ema(delagged, period).rename("ZLEMA")


def vma(close: pd.Series, period: int = 20) -> pd.Series:
    """
    Variable Moving Average (VMA).

    Adapts to volatility using standard deviation ratio.

    Args:
        close: Close prices
        period: Base period

    Returns:
        VMA series
    """
    # Volatility ratio
    std = close.rolling(window=period).std()
    std_smooth = std.rolling(window=period).mean()
    vol_ratio = (std / std_smooth.replace(0, np.nan)).fillna(1).clip(0.5, 2)

    # Adaptive alpha
    base_alpha = 2 / (period + 1)
    alpha = base_alpha * vol_ratio

    # Calculate VMA
    vma_values = np.zeros(len(close))
    vma_values[:] = np.nan

    first_valid = close.first_valid_index()
    if first_valid is not None:
        start_idx = close.index.get_loc(first_valid)
        vma_values[start_idx] = close.iloc[start_idx]

        for i in range(start_idx + 1, len(close)):
            if not np.isnan(alpha.iloc[i]):
                vma_values[i] = alpha.iloc[i] * close.iloc[i] + (1 - alpha.iloc[i]) * vma_values[i-1]
            else:
                vma_values[i] = vma_values[i-1]

    return pd.Series(vma_values, index=close.index, name="VMA")


# =============================================================================
# Momentum Indicators
# =============================================================================

def tsmom(
    close: pd.Series,
    lookback: int = 30,
    method: str = "log_return"
) -> pd.Series:
    """
    Time-Series Momentum (TSMOM).

    Measures momentum as the return over a lookback period.

    Args:
        close: Close prices
        lookback: Lookback period in days
        method: "log_return" or "simple_return"

    Returns:
        TSMOM series (positive = bullish, negative = bearish)

    Note:
        TSMOM_30 uses 30-day log returns by default.
        Log returns are preferred for their additive property across time.
    """
    if method == "log_return":
        # Log return over lookback period
        momentum = np.log(close / close.shift(lookback))
    else:
        # Simple return
        momentum = (close - close.shift(lookback)) / close.shift(lookback)

    return momentum.rename(f"TSMOM_{lookback}")


# =============================================================================
# Indicator dispatcher
# =============================================================================

def compute_trend_indicator(
    indicator_name: str,
    close: pd.Series,
    high: pd.Series | None = None,
    low: pd.Series | None = None,
    params: dict | None = None
) -> pd.Series:
    """
    Compute a trend indicator by name.

    Args:
        indicator_name: One of KAMA, VIDYA, BOLL_MID, SUPERTREND, FRAMA, ZLEMA, VMA
        close: Close prices
        high: High prices (required for SUPERTREND)
        low: Low prices (required for SUPERTREND)
        params: Optional parameter overrides

    Returns:
        Indicator series
    """
    params = params or {}

    if indicator_name == "KAMA":
        return kama(close, **{k: v for k, v in params.items() if k in ["period", "fast", "slow"]})
    elif indicator_name == "VIDYA":
        return vidya(close, **{k: v for k, v in params.items() if k in ["period", "cmo_period"]})
    elif indicator_name == "BOLL_MID":
        return boll_mid(close, **{k: v for k, v in params.items() if k in ["period"]})
    elif indicator_name == "SUPERTREND":
        if high is None or low is None:
            raise ValueError("SUPERTREND requires high and low prices")
        return supertrend(high, low, close, **{k: v for k, v in params.items() if k in ["period", "multiplier"]})
    elif indicator_name == "FRAMA":
        return frama(close, **{k: v for k, v in params.items() if k in ["period"]})
    elif indicator_name == "ZLEMA":
        return zlema(close, **{k: v for k, v in params.items() if k in ["period"]})
    elif indicator_name == "VMA":
        return vma(close, **{k: v for k, v in params.items() if k in ["period"]})
    else:
        raise ValueError(f"Unknown trend indicator: {indicator_name}")


def compute_momentum_indicator(
    indicator_name: str,
    close: pd.Series,
    params: dict | None = None
) -> pd.Series:
    """
    Compute a momentum indicator by name.

    Args:
        indicator_name: Currently only TSMOM_30
        close: Close prices
        params: Optional parameter overrides

    Returns:
        Indicator series
    """
    params = params or {}

    if indicator_name == "TSMOM_30":
        return tsmom(close, lookback=params.get("lookback", 30), method=params.get("method", "log_return"))
    else:
        raise ValueError(f"Unknown momentum indicator: {indicator_name}")
