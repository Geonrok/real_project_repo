#!/usr/bin/env python3
"""
Stage1 Strategy Grid Backtest Runner.

Purpose: Execute strategy grid backtests across 4 markets and generate comparison summary.

Usage:
    python scripts/backtest_runner.py \\
        --markets configs/markets.yaml \\
        --normalized-dir outputs/normalized_1d \\
        --grid configs/grid_stage1.yaml \\
        --out outputs/stage1 \\
        --regime-source reference_market \\
        --eval-mode both

Outputs:
    - stage1_summary.csv: All market × strategy combinations
    - stage1_top20.csv: Top 20 strategies per market
    - stage1_intersection_summary.csv: Performance on common date range
"""
from __future__ import annotations

import argparse
import sys
import warnings
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml

# Suppress pandas warnings for cleaner output
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

# Import local modules
sys.path.insert(0, str(Path(__file__).parent))
from indicators import compute_trend_indicator, compute_momentum_indicator
from regimes import RegimeCalculator


@dataclass
class BacktestResult:
    """Result of a single strategy backtest."""
    market: str
    strategy_id: str
    trend: str
    momentum: str
    regime: str
    regime_mode: str
    cagr: float
    sharpe: float
    mdd: float
    win_rate: float
    turnover: float
    trades: int
    start_date: str
    end_date: str
    days: int
    valid: bool = True  # False if any invariant violated
    violations: str = ""  # Comma-separated violation codes


def validate_metrics_invariants(
    cum_ret: pd.Series | np.ndarray | list | None,
    metrics: dict,
    market_type: str,
    strategy_id: str,
    market: str,
    symbol: str = ""
) -> tuple[bool, list[str]]:
    """
    Validate metrics invariants for a single symbol.

    Returns:
        (is_valid, list_of_violations)

    Invariants (checked in order):
    0) cum_ret must exist and be non-empty
    a) Equity must not contain NaN/inf
    b) Equity must be non-negative (all markets - Option A: strict policy)
    c) MDD must be in [-1, 0]
    d) CAGR consistency: if final_equity ≈ 0 (within EPS), cagr must be -1.0

    EPS Constants:
    - EPS_EQUITY (1e-12): Tolerance for float noise in equity checks
    - EPS_CAGR (1e-6): Tolerance for CAGR value comparison
    """
    EPS_EQUITY = 1e-12  # Tolerance for float noise in equity negativity/zero check
    EPS_CAGR = 1e-6     # Tolerance for CAGR value comparison

    violations = []
    is_valid = True
    location = f"{market}:{strategy_id}:{symbol}" if symbol else f"{market}:{strategy_id}"

    # (0) Type normalization and missing check
    if cum_ret is None:
        violations.append("cum_ret_missing:None")
        is_valid = False
        for v in violations:
            print(f"WARNING: [{location}] INVARIANT_VIOLATION: {v}")
        return is_valid, violations

    # Normalize to pd.Series with float64 dtype
    try:
        cum_ret = pd.Series(cum_ret, dtype="float64")
    except (ValueError, TypeError) as e:
        violations.append(f"cum_ret_invalid_type:{type(cum_ret).__name__}")
        is_valid = False
        for v in violations:
            print(f"WARNING: [{location}] INVARIANT_VIOLATION: {v}")
        return is_valid, violations

    if len(cum_ret) == 0:
        violations.append("cum_ret_missing:empty")
        is_valid = False
        for v in violations:
            print(f"WARNING: [{location}] INVARIANT_VIOLATION: {v}")
        return is_valid, violations

    # (a) NaN/inf check FIRST (before numeric comparisons)
    if cum_ret.isna().any():
        nan_count = int(cum_ret.isna().sum())
        violations.append(f"equity_has_nan:{nan_count}")
        is_valid = False
    if np.isinf(cum_ret).any():
        inf_count = int(np.isinf(cum_ret).sum())
        violations.append(f"equity_has_inf:{inf_count}")
        is_valid = False

    # (b) Equity >= 0 with EPS tolerance (applies to ALL markets - strict policy)
    if (cum_ret < -EPS_EQUITY).any():
        min_val = float(cum_ret.min())
        violations.append(f"equity_negative:{min_val:.6f}")
        is_valid = False

    # (c) MDD in [-1, 0]
    mdd = metrics.get("mdd", 0)
    if mdd < -1.0:
        violations.append(f"mdd_below_minus1:{mdd:.6f}")
        is_valid = False
    if mdd > 0:
        violations.append(f"mdd_positive:{mdd:.6f}")
        is_valid = False

    # (d) CAGR consistency: if final_equity ≈ 0 (within EPS_EQUITY), cagr must ≈ -1.0
    # Uses same EPS_EQUITY threshold for consistency with equity checks
    final_val = cum_ret.iloc[-1]
    cagr = metrics.get("cagr", 0)
    is_total_loss = abs(final_val) <= EPS_EQUITY  # Treat as 0 if within EPS
    if is_total_loss and abs(cagr - (-1.0)) > EPS_CAGR:
        violations.append(f"cagr_consistency:final={final_val:.6f},cagr={cagr:.6f}")
        is_valid = False

    # Print warnings
    for v in violations:
        print(f"WARNING: [{location}] INVARIANT_VIOLATION: {v}")

    return is_valid, violations


def save_invariant_violations(
    violations: list[dict],
    output_dir: Path
) -> None:
    """Append violations to outputs/<out>/invariant_violations.csv"""
    if not violations:
        return
    filepath = output_dir / "invariant_violations.csv"
    df = pd.DataFrame(violations)
    if filepath.exists():
        df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        df.to_csv(filepath, index=False)


def parse_debug_dump(arg: str) -> tuple[str, str] | None:
    """Parse 'market=X,strategy_id=Y' format."""
    if not arg:
        return None
    try:
        parts = dict(p.split("=") for p in arg.split(",") if "=" in p)
        market = parts.get("market", "")
        strategy_id = parts.get("strategy_id", "")
        if market and strategy_id:
            return (market, strategy_id)
    except ValueError:
        pass
    return None


def save_debug_dump(
    df: pd.DataFrame,
    market: str,
    strategy_id: str,
    output_dir: Path,
    regime_series: pd.Series | None = None
) -> str:
    """
    Save detailed debug equity dump.

    Output: outputs/<out>/debug/{market}_{strategy_id}_equity.csv

    Columns:
    - date, close, position
    - strat_ret, cum_ret (equity)
    - costs, pos_change
    - regime_state (if available)
    - entry_flag, exit_flag
    """
    debug_dir = output_dir / "debug"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Entry/exit flags
    pos = df["position"]
    entry_flag = ((pos > 0) & (pos.shift(1).fillna(0) == 0)).astype(int)
    exit_flag = ((pos == 0) & (pos.shift(1).fillna(0) > 0)).astype(int)

    debug_df = pd.DataFrame({
        "date": df["date"],
        "close": df["close"],
        "position": df["position"],
        "strat_ret": df["strat_ret"],
        "equity": df["cum_ret"],
        "costs": df["costs"],
        "pos_change": df["pos_change"],
        "entry_flag": entry_flag,
        "exit_flag": exit_flag,
    })

    if regime_series is not None:
        # Normalize date types and merge regime_state
        try:
            # Convert to datetime and normalize to date-only (strip time/tz)
            debug_dates = pd.to_datetime(debug_df["date"])
            # Remove timezone if present, then normalize to midnight
            if debug_dates.dt.tz is not None:
                debug_dates = debug_dates.dt.tz_localize(None)
            debug_df["date"] = debug_dates.dt.normalize()

            # Create regime DataFrame with normalized datetime index
            regime_dates = pd.to_datetime(regime_series.index)
            if regime_dates.tz is not None:
                regime_dates = regime_dates.tz_localize(None)
            regime_df = pd.DataFrame({
                "date": regime_dates.normalize(),
                "regime_state": regime_series.values
            })

            # Deduplicate dates (take last value if duplicates exist)
            regime_df = regime_df.groupby("date", as_index=False).last()

            # Merge on date (left join to preserve all debug_df rows)
            debug_df = debug_df.merge(regime_df, on="date", how="left")

            # Preserve dtype: only convert to int if original dtype is integer-like
            original_dtype = regime_series.dtype
            if pd.api.types.is_integer_dtype(original_dtype):
                debug_df["regime_state"] = debug_df["regime_state"].fillna(0).astype(int)
            else:
                debug_df["regime_state"] = debug_df["regime_state"].fillna(0.0)
        except Exception:
            debug_df["regime_state"] = 0
    else:
        debug_df["regime_state"] = 0

    filepath = debug_dir / f"{market}_{strategy_id}_equity.csv"
    debug_df.to_csv(filepath, index=False, float_format="%.8g")
    print(f"[DEBUG DUMP] Saved: {filepath}")
    return str(filepath)


def load_config(config_path: Path) -> dict:
    """Load YAML configuration file."""
    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_symbol_data(normalized_dir: Path, market: str, symbol: str) -> pd.DataFrame | None:
    """Load normalized OHLCV data for a symbol."""
    market_dir = normalized_dir / market
    csv_path = market_dir / f"{symbol}.csv"
    parquet_path = market_dir / f"{symbol}.parquet"

    if csv_path.exists():
        df = pd.read_csv(csv_path)
    elif parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    else:
        return None

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def list_symbols(normalized_dir: Path, market: str) -> list[str]:
    """List all symbols available for a market."""
    market_dir = normalized_dir / market
    if not market_dir.exists():
        return []

    symbols = []
    for f in market_dir.iterdir():
        if f.suffix in [".csv", ".parquet"] and not f.name.startswith("_"):
            symbols.append(f.stem)
    return sorted(symbols)


def compute_strategy_signals(
    df: pd.DataFrame,
    trend_name: str,
    momentum_name: str,
    regime_series: pd.Series,
    regime_mode: str,
    regime_strength: pd.Series | None,
    trend_params: dict,
    momentum_params: dict,
    vol_target_base: float,
    max_leverage: float = 1.0,
    allow_shorting: bool = False,
    allow_borrowing: bool = False
) -> pd.DataFrame:
    """
    Compute entry/exit signals for a strategy.

    Returns DataFrame with date, signal, position columns.
    """
    # Compute indicators
    trend_ind = compute_trend_indicator(
        trend_name,
        df["close"],
        df.get("high"),
        df.get("low"),
        trend_params.get(trend_name, {})
    )

    momentum_ind = compute_momentum_indicator(
        momentum_name,
        df["close"],
        momentum_params.get(momentum_name, {})
    )

    # Align regime to data dates
    df_with_regime = df.copy()
    df_with_regime["regime"] = df_with_regime["date"].map(
        lambda d: regime_series.get(d, 0) if hasattr(regime_series, "get") else 0
    )

    # If regime_series is a Series with DatetimeIndex
    if isinstance(regime_series.index, pd.DatetimeIndex):
        regime_dict = regime_series.to_dict()
        df_with_regime["regime"] = df_with_regime["date"].map(lambda d: regime_dict.get(d, 0))

    # Trend signal: price > trend indicator
    trend_up = df["close"] > trend_ind
    trend_down = df["close"] < trend_ind

    # Momentum signal: positive momentum
    momentum_positive = momentum_ind > 0
    momentum_negative = momentum_ind < 0

    # Regime signal
    regime_ok = df_with_regime["regime"] == 1
    regime_fail = df_with_regime["regime"] == 0

    # Entry: (TrendUp OR MomentumPositive) AND RegimeOK
    entry_signal = (trend_up | momentum_positive) & regime_ok

    # Exit: (TrendDown AND MomentumNegative) OR RegimeFail
    exit_signal = (trend_down & momentum_negative) | regime_fail

    # Build position series (0 or 1)
    position = np.zeros(len(df))
    for i in range(1, len(df)):
        if entry_signal.iloc[i] and position[i-1] == 0:
            position[i] = 1
        elif exit_signal.iloc[i] and position[i-1] == 1:
            position[i] = 0
        else:
            position[i] = position[i-1]

    # Apply regime mode for position sizing
    if regime_mode == "SIZING" and regime_strength is not None:
        # Map strength to dates
        if isinstance(regime_strength.index, pd.DatetimeIndex):
            strength_dict = regime_strength.to_dict()
            df_with_regime["strength"] = df_with_regime["date"].map(lambda d: strength_dict.get(d, 0))
        else:
            df_with_regime["strength"] = 0

        # Scale position by regime strength (0 to vol_target_base)
        position = position * df_with_regime["strength"].values.clip(0, 1)

    # Apply leverage constraints BEFORE returns are computed
    if not allow_shorting:
        position = np.clip(position, 0, max_leverage)
    else:
        position = np.clip(position, -max_leverage, max_leverage)

    # For spot: ensure no borrowing (position cannot exceed 1.0)
    if not allow_borrowing:
        position = np.minimum(position, 1.0)

    result = pd.DataFrame({
        "date": df["date"],
        "close": df["close"],
        "position": position,
        "trend_ind": trend_ind.values,
        "momentum_ind": momentum_ind.values,
    })

    return result


def backtest_symbol(
    signals: pd.DataFrame,
    fee_bps: float,
    slippage_bps: float
) -> dict:
    """
    Run backtest on signal DataFrame.

    Returns dict with performance metrics.
    """
    df = signals.copy()

    # Daily returns
    # Handle edge cases: NaN (first bar), inf (0->positive), -inf (negative price)
    df["ret"] = df["close"].pct_change().fillna(0)
    df["ret"] = df["ret"].replace([np.inf, -np.inf], 0)

    # Position change for cost calculation
    df["pos_change"] = df["position"].diff().abs().fillna(0)

    # Transaction costs (applied on position changes)
    cost_per_trade = (fee_bps + slippage_bps) / 10000
    df["costs"] = df["pos_change"] * cost_per_trade

    # Strategy returns
    # Clip strat_ret to -1.0 to prevent (1+strat_ret) < 0
    # This handles edge cases where ret=-1 (close->0) plus costs would cause negative equity
    df["strat_ret"] = df["position"].shift(1).fillna(0) * df["ret"] - df["costs"]
    df["strat_ret"] = df["strat_ret"].clip(lower=-1.0)

    # Cumulative returns (clip to 0 as safety net - should not be needed after strat_ret clip)
    df["cum_ret"] = (1 + df["strat_ret"]).cumprod()
    df["cum_ret"] = df["cum_ret"].clip(lower=0.0)

    # Performance metrics
    total_days = len(df)
    if total_days < 2:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "mdd": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "trades": 0,
        }

    # CAGR
    years = total_days / 252
    final_value = df["cum_ret"].iloc[-1]
    if years > 0 and final_value > 0:
        cagr = final_value ** (1 / years) - 1
    elif years > 0 and final_value == 0:
        cagr = -1.0  # Total loss
    else:
        cagr = 0.0
    # Note: negative final_value in spot is INVALID, handled by invariant check

    # Sharpe (daily, annualized)
    mean_ret = df["strat_ret"].mean()
    std_ret = df["strat_ret"].std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Max Drawdown - use eps to prevent division by zero
    EPS = 1e-10
    running_max = df["cum_ret"].cummax()
    safe_max = np.maximum(running_max.values, EPS)
    drawdown = df["cum_ret"].values / safe_max - 1.0
    mdd = float(drawdown.min())
    # DO NOT CLAMP - check invariant later and mark invalid if violated

    # Win Rate
    trading_days = df[df["position"].shift(1) > 0]
    if len(trading_days) > 0:
        win_rate = (trading_days["strat_ret"] > 0).mean()
    else:
        win_rate = 0.0

    # Turnover (average position changes per year)
    turnover = df["pos_change"].sum() / years if years > 0 else 0

    # Trade count: 0→non-zero entries + sign flips (long↔short)
    # Entry = (prev==0 & pos!=0) OR (prev*pos < 0, i.e., sign flip)
    pos = df["position"]
    prev_pos = pos.shift(1).fillna(0)
    new_entry = (prev_pos == 0) & (pos != 0)
    sign_flip = (prev_pos * pos) < 0  # Different signs = flip
    trades = (new_entry | sign_flip).sum()

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "turnover": turnover,
        "trades": int(trades),
        "_cum_ret": df["cum_ret"],  # For invariant validation
        "_signals_df": df,  # For debug dump
    }


def aggregate_symbol_results(
    symbol_results: list[dict],
    weights: list[float] | None = None
) -> dict:
    """
    Aggregate results across symbols using equal weights.

    Returns aggregate performance metrics.
    """
    if not symbol_results:
        return {
            "cagr": 0.0,
            "sharpe": 0.0,
            "mdd": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "trades": 0,
        }

    if weights is None:
        weights = [1.0 / len(symbol_results)] * len(symbol_results)

    agg = {
        "cagr": sum(r["cagr"] * w for r, w in zip(symbol_results, weights)),
        "sharpe": sum(r["sharpe"] * w for r, w in zip(symbol_results, weights)),
        "mdd": min(r["mdd"] for r in symbol_results),  # Worst drawdown
        "win_rate": sum(r["win_rate"] * w for r, w in zip(symbol_results, weights)),
        "turnover": sum(r["turnover"] * w for r, w in zip(symbol_results, weights)),
        "trades": sum(r["trades"] for r in symbol_results),
    }

    return agg


def run_strategy_backtest(
    market: str,
    normalized_dir: Path,
    regime_calc: RegimeCalculator,
    trend: str,
    momentum: str,
    regime_name: str,
    regime_mode: str,
    trend_params: dict,
    momentum_params: dict,
    vol_target_base: float,
    fee_bps: float,
    slippage_bps: float,
    date_range: tuple[str, str] | None = None,
    market_config: dict | None = None,
    debug_dump_target: tuple[str, str] | None = None,
    output_dir: Path | None = None
) -> BacktestResult | None:
    """
    Run backtest for a single strategy on a market.

    Returns BacktestResult or None if insufficient data.
    """
    symbols = list_symbols(normalized_dir, market)
    if not symbols:
        return None

    # Get regime series
    regime_series = regime_calc.get_regime(regime_name, market)
    regime_strength = regime_calc.get_regime_strength(regime_name, market) if regime_mode == "SIZING" else None

    # Get leverage constraints from market config
    if market_config:
        max_leverage = market_config.get("max_gross_leverage", 1.0)
        allow_shorting = market_config.get("allow_shorting", False)
        allow_borrowing = market_config.get("allow_borrowing", False)
        market_type = market_config.get("market_type", "spot")
    else:
        max_leverage = 1.0
        allow_shorting = False
        allow_borrowing = False
        market_type = "spot"

    strategy_id = f"{trend}_{momentum}_{regime_name}_{regime_mode}"

    # Use list of (symbol, result) tuples to guarantee matching
    # This prevents mismatch if continue/skip occurs in the loop
    symbol_result_pairs: list[tuple[str, dict]] = []
    all_dates = []

    for symbol in symbols:
        df = load_symbol_data(normalized_dir, market, symbol)
        if df is None or len(df) < 60:  # Require minimum 60 days
            continue

        # Apply date range filter if specified
        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if len(df) < 30:
                continue

        # Compute signals with leverage constraints
        try:
            signals = compute_strategy_signals(
                df, trend, momentum, regime_series, regime_mode,
                regime_strength, trend_params, momentum_params, vol_target_base,
                max_leverage=max_leverage,
                allow_shorting=allow_shorting,
                allow_borrowing=allow_borrowing
            )
        except Exception:
            continue

        # Run backtest
        result = backtest_symbol(signals, fee_bps, slippage_bps)
        # Append as tuple - symbol and result are always paired together
        symbol_result_pairs.append((symbol, result))
        all_dates.extend(df["date"].tolist())

        # Debug dump if this is the target
        if debug_dump_target and output_dir:
            target_market, target_strategy = debug_dump_target
            if market == target_market and strategy_id == target_strategy:
                if "_signals_df" in result:
                    save_debug_dump(
                        result["_signals_df"], market, strategy_id,
                        output_dir, regime_series
                    )

    if not symbol_result_pairs:
        return None

    # Extract results for aggregation
    symbol_results = [result for _, result in symbol_result_pairs]

    # Aggregate across symbols
    agg = aggregate_symbol_results(symbol_results)

    # Date range
    if all_dates:
        start_date = min(all_dates).strftime("%Y-%m-%d")
        end_date = max(all_dates).strftime("%Y-%m-%d")
        days = (max(all_dates) - min(all_dates)).days
    else:
        start_date = end_date = ""
        days = 0

    # Validate invariants PER-SYMBOL (strategy is invalid if ANY symbol violates)
    # Tuple approach guarantees symbol-result matching
    is_valid = True
    violations_list = []

    for symbol, result in symbol_result_pairs:
        if "_cum_ret" in result:
            cum_ret = result["_cum_ret"]
            symbol_metrics = {
                "cagr": result.get("cagr", 0),
                "mdd": result.get("mdd", 0),
            }
            sym_valid, sym_violations = validate_metrics_invariants(
                cum_ret, symbol_metrics, market_type, strategy_id, market, symbol
            )
            if not sym_valid:
                is_valid = False
                # Prefix violations with symbol name for clarity
                violations_list.extend([f"{symbol}:{v}" for v in sym_violations])

    return BacktestResult(
        market=market,
        strategy_id=strategy_id,
        trend=trend,
        momentum=momentum,
        regime=regime_name,
        regime_mode=regime_mode,
        cagr=agg["cagr"],
        sharpe=agg["sharpe"],
        mdd=agg["mdd"],
        win_rate=agg["win_rate"],
        turnover=agg["turnover"],
        trades=agg["trades"],
        start_date=start_date,
        end_date=end_date,
        days=days,
        valid=is_valid,
        violations=",".join(sorted(violations_list)),  # Sort for determinism
    )


def find_intersection_dates(
    normalized_dir: Path,
    markets: list[str]
) -> tuple[str, str] | None:
    """
    Find common date range across all markets.

    Returns (start_date, end_date) or None if no intersection.
    """
    date_ranges = []

    for market in markets:
        market_dir = normalized_dir / market
        if not market_dir.exists():
            continue

        market_dates = set()
        for f in market_dir.iterdir():
            if f.suffix == ".csv" and not f.name.startswith("_"):
                df = pd.read_csv(f, usecols=["date"])
                df["date"] = pd.to_datetime(df["date"])
                market_dates.update(df["date"].tolist())
                break  # Just need one file to get date range

        if market_dates:
            date_ranges.append((min(market_dates), max(market_dates)))

    if not date_ranges:
        return None

    # Find intersection
    start = max(dr[0] for dr in date_ranges)
    end = min(dr[1] for dr in date_ranges)

    if start >= end:
        return None

    return (start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))


def results_to_dataframe(results: list[BacktestResult]) -> pd.DataFrame:
    """Convert list of BacktestResult to DataFrame."""
    if not results:
        return pd.DataFrame()

    data = []
    for r in results:
        data.append({
            "market": r.market,
            "strategy_id": r.strategy_id,
            "trend": r.trend,
            "momentum": r.momentum,
            "regime": r.regime,
            "regime_mode": r.regime_mode,
            "cagr": r.cagr,
            "sharpe": r.sharpe,
            "mdd": r.mdd,
            "win_rate": r.win_rate,
            "turnover": r.turnover,
            "trades": r.trades,
            "start_date": r.start_date,
            "end_date": r.end_date,
            "days": r.days,
            "valid": r.valid,
            "violations": r.violations,
        })

    return pd.DataFrame(data)


def preload_market_data(
    normalized_dir: Path,
    market: str,
    sample_size: int | None = None
) -> dict[str, pd.DataFrame]:
    """
    Preload all symbol data for a market into memory.

    Args:
        normalized_dir: Path to normalized data
        market: Market name
        sample_size: If set, only load this many symbols (for quick testing)

    Returns:
        Dict mapping symbol name to DataFrame
    """
    symbols = list_symbols(normalized_dir, market)
    if sample_size and sample_size < len(symbols):
        # Prioritize BTC, ETH, then sample others
        priority = ["BTC", "ETH", "BTCUSDT", "ETHUSDT"]
        priority_symbols = [s for s in priority if s in symbols]
        other_symbols = [s for s in symbols if s not in priority]
        symbols = priority_symbols + other_symbols[:sample_size - len(priority_symbols)]

    data = {}
    for symbol in symbols:
        df = load_symbol_data(normalized_dir, market, symbol)
        if df is not None and len(df) >= 60:
            data[symbol] = df

    return data


def run_strategy_backtest_preloaded(
    market: str,
    market_data: dict[str, pd.DataFrame],
    regime_series: pd.Series,
    regime_strength: pd.Series | None,
    trend: str,
    momentum: str,
    regime_name: str,
    regime_mode: str,
    trend_params: dict,
    momentum_params: dict,
    vol_target_base: float,
    fee_bps: float,
    slippage_bps: float,
    date_range: tuple[str, str] | None = None,
    market_config: dict | None = None,
    debug_dump_target: tuple[str, str] | None = None,
    output_dir: Path | None = None
) -> BacktestResult | None:
    """
    Run backtest using preloaded data (faster than loading from disk each time).
    """
    # Get leverage constraints from market config
    if market_config:
        max_leverage = market_config.get("max_gross_leverage", 1.0)
        allow_shorting = market_config.get("allow_shorting", False)
        allow_borrowing = market_config.get("allow_borrowing", False)
        market_type = market_config.get("market_type", "spot")
    else:
        max_leverage = 1.0
        allow_shorting = False
        allow_borrowing = False
        market_type = "spot"

    strategy_id = f"{trend}_{momentum}_{regime_name}_{regime_mode}"

    # Use list of (symbol, result) tuples to guarantee matching
    # This prevents mismatch if continue/skip occurs in the loop
    symbol_result_pairs: list[tuple[str, dict]] = []
    all_dates = []

    for symbol, df in market_data.items():
        # Apply date range filter if specified
        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if len(df) < 30:
                continue

        # Compute signals with leverage constraints
        try:
            signals = compute_strategy_signals(
                df, trend, momentum, regime_series, regime_mode,
                regime_strength, trend_params, momentum_params, vol_target_base,
                max_leverage=max_leverage,
                allow_shorting=allow_shorting,
                allow_borrowing=allow_borrowing
            )
        except Exception:
            continue

        # Run backtest
        result = backtest_symbol(signals, fee_bps, slippage_bps)
        # Append as tuple - symbol and result are always paired together
        symbol_result_pairs.append((symbol, result))
        all_dates.extend(df["date"].tolist())

        # Debug dump if this is the target
        if debug_dump_target and output_dir:
            target_market, target_strategy = debug_dump_target
            if market == target_market and strategy_id == target_strategy:
                if "_signals_df" in result:
                    save_debug_dump(
                        result["_signals_df"], market, strategy_id,
                        output_dir, regime_series
                    )

    if not symbol_result_pairs:
        return None

    # Extract results for aggregation
    symbol_results = [result for _, result in symbol_result_pairs]

    # Aggregate across symbols
    agg = aggregate_symbol_results(symbol_results)

    # Date range
    if all_dates:
        start_date = min(all_dates).strftime("%Y-%m-%d")
        end_date = max(all_dates).strftime("%Y-%m-%d")
        days = (max(all_dates) - min(all_dates)).days
    else:
        start_date = end_date = ""
        days = 0

    # Validate invariants PER-SYMBOL (strategy is invalid if ANY symbol violates)
    # Tuple approach guarantees symbol-result matching
    is_valid = True
    violations_list = []

    for symbol, result in symbol_result_pairs:
        if "_cum_ret" in result:
            cum_ret = result["_cum_ret"]
            symbol_metrics = {
                "cagr": result.get("cagr", 0),
                "mdd": result.get("mdd", 0),
            }
            sym_valid, sym_violations = validate_metrics_invariants(
                cum_ret, symbol_metrics, market_type, strategy_id, market, symbol
            )
            if not sym_valid:
                is_valid = False
                violations_list.extend([f"{symbol}:{v}" for v in sym_violations])

    return BacktestResult(
        market=market,
        strategy_id=strategy_id,
        trend=trend,
        momentum=momentum,
        regime=regime_name,
        regime_mode=regime_mode,
        cagr=agg["cagr"],
        sharpe=agg["sharpe"],
        mdd=agg["mdd"],
        win_rate=agg["win_rate"],
        turnover=agg["turnover"],
        trades=agg["trades"],
        start_date=start_date,
        end_date=end_date,
        days=days,
        valid=is_valid,
        violations=",".join(sorted(violations_list)),  # Sort for determinism
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Stage1 Strategy Grid Backtest Runner"
    )
    parser.add_argument("--markets", type=Path, default=Path("configs/markets.yaml"))
    parser.add_argument("--normalized-dir", type=Path, default=Path("outputs/normalized_1d"))
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_stage1.yaml"))
    parser.add_argument("--out", type=Path, default=Path("outputs/stage1"))
    parser.add_argument("--regime-source", choices=["reference_market", "per_market"],
                        default="reference_market")
    parser.add_argument("--eval-mode", choices=["full", "intersection", "both"],
                        default="both")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Limit symbols per market for quick testing")
    parser.add_argument("--debug-dump", type=str, default=None,
                        help="Debug dump: 'market=X,strategy_id=Y' to save equity curve")
    parser.add_argument("--fail-on-invariant", action="store_true",
                        help="Exit non-zero if any invariant is violated (for CI)")
    args = parser.parse_args()

    # Parse debug dump target
    debug_dump_target = parse_debug_dump(args.debug_dump)

    print("[PURPOSE] Execute Stage1 grid backtest across 4 markets for strategy comparison.")
    print("-" * 70)

    # Load configurations
    markets_config = load_config(args.markets)
    grid_config = load_config(args.grid)

    # Get enabled markets
    enabled_markets = [
        name for name, cfg in markets_config.get("markets", {}).items()
        if cfg.get("enabled", True)
    ]

    print(f"Markets: {enabled_markets}")
    print(f"Regime source: {args.regime_source}")
    print(f"Eval mode: {args.eval_mode}")

    # Initialize regime calculator
    reference_market = grid_config.get("reference_market", "binance_spot")
    regime_calc = RegimeCalculator(
        args.normalized_dir,
        source_mode=args.regime_source,
        reference_market=reference_market,
        ma_period=grid_config.get("regime_params", {}).get("ma_period", 50)
    )

    # Generate strategy combinations
    trend_slots = grid_config.get("trend_slot", [])
    momentum_slots = grid_config.get("momentum_slot", [])
    regime_slots = grid_config.get("regime_slot", [])
    regime_modes = grid_config.get("regime_mode", [])

    combinations = list(product(trend_slots, momentum_slots, regime_slots, regime_modes))
    print(f"Strategy combinations: {len(combinations)}")

    # Get parameters
    trend_params = grid_config.get("trend_params", {})
    momentum_params = grid_config.get("momentum_params", {})
    vol_target_base = grid_config.get("vol_target_base", 0.30)

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    all_results = []
    intersection_results = []

    # Find intersection dates if needed
    intersection_dates = None
    if args.eval_mode in ["intersection", "both"]:
        intersection_dates = find_intersection_dates(args.normalized_dir, enabled_markets)
        if intersection_dates:
            print(f"Intersection date range: {intersection_dates[0]} to {intersection_dates[1]}")

    # Run backtests
    total_combos = len(combinations) * len(enabled_markets)
    combo_count = 0

    for market in enabled_markets:
        market_cfg = markets_config["markets"][market]
        fee_bps = market_cfg.get("fee_bps_roundtrip", 10)
        slippage_bps = market_cfg.get("slippage_bps_roundtrip", 2)

        # Preload all data for this market (major speedup)
        print(f"\n[{market}] Loading data...", flush=True)
        market_data = preload_market_data(args.normalized_dir, market, args.sample_size)
        print(f"[{market}] Loaded {len(market_data)} symbols. Processing {len(combinations)} strategies...", flush=True)

        for idx, (trend, momentum, regime, regime_mode) in enumerate(combinations, 1):
            combo_count += 1

            # Progress output every 10 combinations
            if idx % 10 == 0 or idx == len(combinations):
                print(f"  [{market}] {idx}/{len(combinations)} strategies...", flush=True)

            # Get regime series for this combination
            regime_series = regime_calc.get_regime(regime, market)
            regime_strength = regime_calc.get_regime_strength(regime, market) if regime_mode == "SIZING" else None

            # Full period backtest
            if args.eval_mode in ["full", "both"]:
                result = run_strategy_backtest_preloaded(
                    market, market_data, regime_series, regime_strength,
                    trend, momentum, regime, regime_mode,
                    trend_params, momentum_params, vol_target_base,
                    fee_bps, slippage_bps,
                    market_config=market_cfg,
                    debug_dump_target=debug_dump_target,
                    output_dir=args.out
                )
                if result:
                    all_results.append(result)

            # Intersection period backtest
            if args.eval_mode in ["intersection", "both"] and intersection_dates:
                result = run_strategy_backtest_preloaded(
                    market, market_data, regime_series, regime_strength,
                    trend, momentum, regime, regime_mode,
                    trend_params, momentum_params, vol_target_base,
                    fee_bps, slippage_bps,
                    date_range=intersection_dates,
                    market_config=market_cfg,
                    debug_dump_target=debug_dump_target,
                    output_dir=args.out
                )
                if result:
                    intersection_results.append(result)

        print(f"[{market}] Completed. Results: {sum(1 for r in all_results if r.market == market)}")

    # Save results
    print("\n" + "-" * 70)
    print("[STATUS] Saving results...")

    # Full results summary
    invariant_violation_count = 0

    if all_results:
        df_summary = results_to_dataframe(all_results)
        # Sort for determinism: market, strategy_id
        df_summary = df_summary.sort_values(["market", "strategy_id"]).reset_index(drop=True)
        df_summary.to_csv(args.out / "stage1_summary.csv", index=False, float_format="%.8g")
        print(f"  -> stage1_summary.csv ({len(df_summary)} rows)")

        # Count and save invariant violations
        invalid_results = [r for r in all_results if not r.valid]
        invariant_violation_count = len(invalid_results)
        if invalid_results:
            violations_data = [
                {"market": r.market, "strategy_id": r.strategy_id, "violations": r.violations}
                for r in invalid_results
            ]
            save_invariant_violations(violations_data, args.out)
            print(f"  -> invariant_violations.csv ({len(violations_data)} violations)")

        # Top 20 per market (ONLY valid results)
        top20_list = []
        for market in enabled_markets:
            market_df = df_summary[df_summary["market"] == market]
            # Filter to only valid results
            valid_market_df = market_df[market_df["valid"] == True]
            # Sort by sharpe desc, then strategy_id asc for tie-breaking determinism
            # na_position="last" ensures NaN sharpe goes to bottom (excluded from top20)
            sorted_df = valid_market_df.sort_values(
                ["sharpe", "strategy_id"], ascending=[False, True], na_position="last"
            )
            top20 = sorted_df.head(20)
            top20_list.append(top20)

        if top20_list:
            df_top20 = pd.concat(top20_list, ignore_index=True)
            # Final sort after concat for fully deterministic row order
            df_top20 = df_top20.sort_values(
                ["market", "sharpe", "strategy_id"],
                ascending=[True, False, True],
                na_position="last"
            ).reset_index(drop=True)
            df_top20.to_csv(args.out / "stage1_top20.csv", index=False, float_format="%.8g")
            print(f"  -> stage1_top20.csv ({len(df_top20)} rows, valid only)")

    # Intersection results
    if intersection_results:
        df_intersection = results_to_dataframe(intersection_results)
        # Sort for determinism
        df_intersection = df_intersection.sort_values(["market", "strategy_id"]).reset_index(drop=True)
        df_intersection.to_csv(args.out / "stage1_intersection_summary.csv", index=False, float_format="%.8g")
        print(f"  -> stage1_intersection_summary.csv ({len(df_intersection)} rows)")

    # Count valid/invalid
    valid_count = sum(1 for r in all_results if r.valid)
    invalid_count = sum(1 for r in all_results if not r.valid)

    print("\n" + "=" * 70)
    print("[DONE] Stage1 backtest completed.")
    print(f"  Total strategies tested: {combo_count}")
    print(f"  Full period results: {len(all_results)} (valid: {valid_count}, invalid: {invalid_count})")
    print(f"  Intersection results: {len(intersection_results)}")
    print("=" * 70)

    # Exit non-zero if --fail-on-invariant and violations found
    if args.fail_on_invariant and invariant_violation_count > 0:
        print(f"\n[ERROR] --fail-on-invariant: {invariant_violation_count} invariant violations found.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
