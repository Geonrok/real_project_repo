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
    vol_target_base: float
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
    df["ret"] = df["close"].pct_change().fillna(0)

    # Position change for cost calculation
    df["pos_change"] = df["position"].diff().abs().fillna(0)

    # Transaction costs (applied on position changes)
    cost_per_trade = (fee_bps + slippage_bps) / 10000
    df["costs"] = df["pos_change"] * cost_per_trade

    # Strategy returns
    df["strat_ret"] = df["position"].shift(1).fillna(0) * df["ret"] - df["costs"]

    # Cumulative returns
    df["cum_ret"] = (1 + df["strat_ret"]).cumprod()

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
    cagr = (final_value ** (1 / years) - 1) if years > 0 and final_value > 0 else 0

    # Sharpe (daily, annualized)
    mean_ret = df["strat_ret"].mean()
    std_ret = df["strat_ret"].std()
    sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0

    # Max Drawdown
    running_max = df["cum_ret"].cummax()
    drawdown = (df["cum_ret"] - running_max) / running_max
    mdd = drawdown.min()

    # Win Rate
    trading_days = df[df["position"].shift(1) > 0]
    if len(trading_days) > 0:
        win_rate = (trading_days["strat_ret"] > 0).mean()
    else:
        win_rate = 0.0

    # Turnover (average position changes per year)
    turnover = df["pos_change"].sum() / years if years > 0 else 0

    # Trade count (position going from 0 to >0)
    trades = ((df["position"] > 0) & (df["position"].shift(1) == 0)).sum()

    return {
        "cagr": cagr,
        "sharpe": sharpe,
        "mdd": mdd,
        "win_rate": win_rate,
        "turnover": turnover,
        "trades": int(trades),
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
    date_range: tuple[str, str] | None = None
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

    symbol_results = []
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

        # Compute signals
        try:
            signals = compute_strategy_signals(
                df, trend, momentum, regime_series, regime_mode,
                regime_strength, trend_params, momentum_params, vol_target_base
            )
        except Exception:
            continue

        # Run backtest
        result = backtest_symbol(signals, fee_bps, slippage_bps)
        symbol_results.append(result)
        all_dates.extend(df["date"].tolist())

    if not symbol_results:
        return None

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

    strategy_id = f"{trend}_{momentum}_{regime_name}_{regime_mode}"

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
    date_range: tuple[str, str] | None = None
) -> BacktestResult | None:
    """
    Run backtest using preloaded data (faster than loading from disk each time).
    """
    symbol_results = []
    all_dates = []

    for symbol, df in market_data.items():
        # Apply date range filter if specified
        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df["date"] >= start) & (df["date"] <= end)]
            if len(df) < 30:
                continue

        # Compute signals
        try:
            signals = compute_strategy_signals(
                df, trend, momentum, regime_series, regime_mode,
                regime_strength, trend_params, momentum_params, vol_target_base
            )
        except Exception:
            continue

        # Run backtest
        result = backtest_symbol(signals, fee_bps, slippage_bps)
        symbol_results.append(result)
        all_dates.extend(df["date"].tolist())

    if not symbol_results:
        return None

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

    strategy_id = f"{trend}_{momentum}_{regime_name}_{regime_mode}"

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
    args = parser.parse_args()

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
        print(f"\n[{market}] Loading data...")
        market_data = preload_market_data(args.normalized_dir, market, args.sample_size)
        print(f"[{market}] Loaded {len(market_data)} symbols. Processing {len(combinations)} strategies...")

        for trend, momentum, regime, regime_mode in combinations:
            combo_count += 1

            # Get regime series for this combination
            regime_series = regime_calc.get_regime(regime, market)
            regime_strength = regime_calc.get_regime_strength(regime, market) if regime_mode == "SIZING" else None

            # Full period backtest
            if args.eval_mode in ["full", "both"]:
                result = run_strategy_backtest_preloaded(
                    market, market_data, regime_series, regime_strength,
                    trend, momentum, regime, regime_mode,
                    trend_params, momentum_params, vol_target_base,
                    fee_bps, slippage_bps
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
                    date_range=intersection_dates
                )
                if result:
                    intersection_results.append(result)

        print(f"[{market}] Completed. Results: {sum(1 for r in all_results if r.market == market)}")

    # Save results
    print("\n" + "-" * 70)
    print("[STATUS] Saving results...")

    # Full results summary
    if all_results:
        df_summary = results_to_dataframe(all_results)
        df_summary.to_csv(args.out / "stage1_summary.csv", index=False)
        print(f"  -> stage1_summary.csv ({len(df_summary)} rows)")

        # Top 20 per market
        top20_list = []
        for market in enabled_markets:
            market_df = df_summary[df_summary["market"] == market]
            top20 = market_df.nlargest(20, "sharpe")
            top20_list.append(top20)

        if top20_list:
            df_top20 = pd.concat(top20_list, ignore_index=True)
            df_top20.to_csv(args.out / "stage1_top20.csv", index=False)
            print(f"  -> stage1_top20.csv ({len(df_top20)} rows)")

    # Intersection results
    if intersection_results:
        df_intersection = results_to_dataframe(intersection_results)
        df_intersection.to_csv(args.out / "stage1_intersection_summary.csv", index=False)
        print(f"  -> stage1_intersection_summary.csv ({len(df_intersection)} rows)")

    print("\n" + "=" * 70)
    print("[DONE] Stage1 backtest completed.")
    print(f"  Total strategies tested: {combo_count}")
    print(f"  Full period results: {len(all_results)}")
    print(f"  Intersection results: {len(intersection_results)}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
