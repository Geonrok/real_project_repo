#!/usr/bin/env python3
"""
Stage2 Window Stress Testing.

Tests robust candidates across different time windows (early/late/crisis)
to verify performance consistency across market regimes.

Usage:
    python scripts/stage2_window_stress.py \
        --stage1-dir outputs/stage1_full_v2 \
        --out outputs/stage2_full_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_runner import (
    RegimeCalculator,
    preload_market_data,
    run_strategy_backtest_preloaded,
)


def load_markets_config(markets_path: Path) -> dict:
    """Load markets configuration."""
    with open(markets_path) as f:
        return yaml.safe_load(f)


def load_grid_config(grid_path: Path) -> dict:
    """Load grid configuration with trend/momentum params."""
    with open(grid_path) as f:
        return yaml.safe_load(f)


def load_robust_candidates(stage1_dir: Path, candidates_csv: str) -> pd.DataFrame:
    """Load robust candidates from Stage1."""
    path = stage1_dir / candidates_csv
    if not path.exists():
        print(f"[ERROR] Candidates file not found: {path}")
        sys.exit(1)
    return pd.read_csv(path)


def load_stage1_summary(stage1_dir: Path) -> pd.DataFrame:
    """Load Stage1 summary to get strategy parameters."""
    path = stage1_dir / "stage1_summary.csv"
    if not path.exists():
        print(f"[ERROR] Stage1 summary not found: {path}")
        sys.exit(1)
    return pd.read_csv(path)


def get_strategy_params(summary_df: pd.DataFrame, strategy_id: str) -> dict | None:
    """Extract strategy parameters from summary."""
    rows = summary_df[summary_df["strategy_id"] == strategy_id]
    if len(rows) == 0:
        return None
    row = rows.iloc[0]
    return {
        "trend": row["trend"],
        "momentum": row["momentum"],
        "regime": row["regime"],
        "regime_mode": row["regime_mode"],
    }


def find_crisis_window(
    market_data: dict[str, pd.DataFrame],
    window_days: int = 180
) -> tuple[str, str] | None:
    """
    Find the worst-performing window based on BTC or first available symbol.

    Returns (start_date, end_date) of the crisis window.
    """
    # Try to find BTC first, otherwise use first symbol
    ref_symbol = None
    for symbol in ["BTC", "BTCUSDT", "BTC-KRW", "BTC_KRW"]:
        if symbol in market_data:
            ref_symbol = symbol
            break
    if ref_symbol is None and market_data:
        ref_symbol = list(market_data.keys())[0]

    if ref_symbol is None:
        return None

    df = market_data[ref_symbol].copy()
    if len(df) < window_days:
        return None

    df = df.sort_values("date").reset_index(drop=True)

    # Calculate rolling returns
    df["ret"] = df["close"].pct_change()
    df["rolling_ret"] = df["ret"].rolling(window_days).sum()

    # Find worst window (minimum rolling return)
    worst_idx = df["rolling_ret"].idxmin()
    if pd.isna(worst_idx):
        return None

    end_idx = int(worst_idx)
    start_idx = max(0, end_idx - window_days + 1)

    start_date = df.loc[start_idx, "date"].strftime("%Y-%m-%d")
    end_date = df.loc[end_idx, "date"].strftime("%Y-%m-%d")

    return start_date, end_date


def get_date_range_for_market(market_data: dict[str, pd.DataFrame]) -> tuple[str, str] | None:
    """Get the common date range across all symbols in a market."""
    if not market_data:
        return None

    min_dates = []
    max_dates = []

    for df in market_data.values():
        if len(df) > 0:
            min_dates.append(df["date"].min())
            max_dates.append(df["date"].max())

    if not min_dates:
        return None

    start = max(min_dates).strftime("%Y-%m-%d")
    end = min(max_dates).strftime("%Y-%m-%d")
    return start, end


def run_window_stress(
    stage1_dir: Path,
    candidates_csv: str,
    markets_config_path: Path,
    grid_config_path: Path,
    normalized_dir: Path,
    out_dir: Path,
    crisis_window_days: int,
    split: float,
) -> pd.DataFrame:
    """Run window stress analysis on robust candidates."""

    # Load data
    print(f"[INFO] Loading candidates from: {stage1_dir / candidates_csv}")
    candidates_df = load_robust_candidates(stage1_dir, candidates_csv)
    summary_df = load_stage1_summary(stage1_dir)
    markets_config = load_markets_config(markets_config_path)
    grid_config = load_grid_config(grid_config_path)

    # Get params from grid config
    trend_params = grid_config.get("trend_params", {})
    momentum_params = grid_config.get("momentum_params", {})
    vol_target_base = grid_config.get("vol_target_base", 0.2)

    strategy_ids = candidates_df["strategy_id"].tolist()
    print(f"[INFO] Found {len(strategy_ids)} robust candidates")

    markets = [
        m for m, cfg in markets_config.get("markets", {}).items()
        if cfg.get("enabled", True)
    ]
    print(f"[INFO] Markets: {markets}")

    results = []

    for market in markets:
        market_cfg = markets_config["markets"][market]
        fee_bps = market_cfg.get("fee_bps_roundtrip", 10)
        slippage_bps = market_cfg.get("slippage_bps_roundtrip", 2)

        print(f"\n[{market}] Loading data...")
        market_data = preload_market_data(normalized_dir, market)
        if not market_data:
            print(f"  [WARN] No data for {market}, skipping")
            continue

        # Get date range
        date_range = get_date_range_for_market(market_data)
        if date_range is None:
            print(f"  [WARN] Cannot determine date range for {market}")
            continue

        start_date, end_date = date_range
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        total_days = (end_dt - start_dt).days

        # Calculate split point
        split_days = int(total_days * split)
        mid_date = (start_dt + pd.Timedelta(days=split_days)).strftime("%Y-%m-%d")

        # Define windows
        windows = {
            "early": (start_date, mid_date),
            "late": (mid_date, end_date),
        }

        # Find crisis window
        crisis = find_crisis_window(market_data, crisis_window_days)
        if crisis:
            windows["crisis"] = crisis
            print(f"  Crisis window: {crisis[0]} to {crisis[1]}")

        # Initialize regime calculator
        regime_calc = RegimeCalculator(normalized_dir)

        for strategy_id in strategy_ids:
            params = get_strategy_params(summary_df, strategy_id)
            if params is None:
                continue

            regime_series = regime_calc.get_regime(params["regime"], market)
            regime_strength = (
                regime_calc.get_regime_strength(params["regime"], market)
                if params["regime_mode"] == "SIZING" else None
            )

            for window_name, (win_start, win_end) in windows.items():
                result = run_strategy_backtest_preloaded(
                    market=market,
                    market_data=market_data,
                    regime_series=regime_series,
                    regime_strength=regime_strength,
                    trend=params["trend"],
                    momentum=params["momentum"],
                    regime_name=params["regime"],
                    regime_mode=params["regime_mode"],
                    trend_params=trend_params.get(params["trend"], {}),
                    momentum_params=momentum_params.get(params["momentum"], {}),
                    vol_target_base=vol_target_base,
                    fee_bps=fee_bps,
                    slippage_bps=slippage_bps,
                    date_range=(win_start, win_end),
                    market_config=market_cfg,
                )

                if result:
                    results.append({
                        "market": market,
                        "strategy_id": strategy_id,
                        "window_name": window_name,
                        "window_start": win_start,
                        "window_end": win_end,
                        "sharpe": result.sharpe,
                        "cagr": result.cagr,
                        "mdd": result.mdd,
                        "trades": result.trades,
                        "valid": result.valid,
                        "violations": result.violations or "",
                    })

    # Create DataFrame and sort deterministically
    df = pd.DataFrame(results)
    df = df.sort_values(
        ["market", "strategy_id", "window_name"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage2 Window Stress Testing")
    parser.add_argument("--stage1-dir", type=Path, default=Path("outputs/stage1_full_v2"),
                        help="Stage1 output directory")
    parser.add_argument("--candidates-csv", default="candidates_robust_kof4_top20.csv",
                        help="Candidates CSV filename")
    parser.add_argument("--markets", type=Path, default=Path("configs/markets.yaml"),
                        help="Markets config path")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_stage1.yaml"),
                        help="Grid config path")
    parser.add_argument("--normalized-dir", type=Path, default=Path("outputs/normalized_1d"),
                        help="Normalized data directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Output directory")
    parser.add_argument("--crisis-window-days", type=int, default=180,
                        help="Crisis window size in days")
    parser.add_argument("--split", type=float, default=0.5,
                        help="Early/late split ratio")
    args = parser.parse_args()

    print("[PURPOSE] Stage2 Window Stress Testing")
    print("-" * 70)
    print(f"  Stage1 dir: {args.stage1_dir}")
    print(f"  Output dir: {args.out}")
    print(f"  Crisis window: {args.crisis_window_days} days")
    print(f"  Split: {args.split}")
    print("-" * 70)

    args.out.mkdir(parents=True, exist_ok=True)

    df = run_window_stress(
        stage1_dir=args.stage1_dir,
        candidates_csv=args.candidates_csv,
        markets_config_path=args.markets,
        grid_config_path=args.grid,
        normalized_dir=args.normalized_dir,
        out_dir=args.out,
        crisis_window_days=args.crisis_window_days,
        split=args.split,
    )

    out_path = args.out / "stage2_window_stress.csv"
    df.to_csv(out_path, index=False, float_format="%.8g")
    print(f"\n[SAVED] {out_path} ({len(df)} rows)")

    print("\n[SUMMARY]")
    print(f"  Total rows: {len(df)}")
    print(f"  Windows: {sorted(df['window_name'].unique())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
