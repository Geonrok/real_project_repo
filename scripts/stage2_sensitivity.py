#!/usr/bin/env python3
"""
Stage2 Cost Sensitivity Analysis.

Tests robust candidates across different fee/slippage multipliers
to verify performance stability under varying cost assumptions.

Usage:
    python scripts/stage2_sensitivity.py \
        --stage1-dir outputs/stage1_full_v2 \
        --out outputs/stage2_full_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.backtest_runner import (
    RegimeCalculator,
    preload_market_data,
    run_strategy_backtest_preloaded,
)


def load_grid_config(grid_path: Path) -> dict:
    """Load grid configuration with trend/momentum params."""
    with open(grid_path) as f:
        return yaml.safe_load(f)


def load_markets_config(markets_path: Path) -> dict:
    """Load markets configuration."""
    with open(markets_path) as f:
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


def run_sensitivity_analysis(
    stage1_dir: Path,
    candidates_csv: str,
    markets_config_path: Path,
    grid_config_path: Path,
    normalized_dir: Path,
    out_dir: Path,
    fee_mults: list[float],
) -> pd.DataFrame:
    """Run sensitivity analysis on robust candidates."""

    # Load data
    print(f"[INFO] Loading candidates from: {stage1_dir / candidates_csv}", flush=True)
    candidates_df = load_robust_candidates(stage1_dir, candidates_csv)
    summary_df = load_stage1_summary(stage1_dir)
    markets_config = load_markets_config(markets_config_path)
    grid_config = load_grid_config(grid_config_path)

    # Get params from grid config
    trend_params = grid_config.get("trend_params", {})
    momentum_params = grid_config.get("momentum_params", {})
    vol_target_base = grid_config.get("vol_target_base", 0.2)

    strategy_ids = candidates_df["strategy_id"].tolist()
    print(f"[INFO] Found {len(strategy_ids)} robust candidates", flush=True)

    # Get enabled markets
    markets = [
        m for m, cfg in markets_config.get("markets", {}).items()
        if cfg.get("enabled", True)
    ]
    print(f"[INFO] Markets: {markets}", flush=True)

    results = []
    total_combos = len(strategy_ids) * len(markets) * len(fee_mults)
    completed = 0

    for market in markets:
        market_cfg = markets_config["markets"][market]
        base_fee = market_cfg.get("fee_bps_roundtrip", 10)
        base_slippage = market_cfg.get("slippage_bps_roundtrip", 2)

        print(f"\n[{market}] Loading data...", flush=True)
        market_data = preload_market_data(normalized_dir, market)
        if not market_data:
            print(f"  [WARN] No data for {market}, skipping", flush=True)
            continue

        # Initialize regime calculator
        regime_calc = RegimeCalculator(normalized_dir)

        for strategy_id in strategy_ids:
            params = get_strategy_params(summary_df, strategy_id)
            if params is None:
                print(f"  [WARN] Strategy {strategy_id} not in summary, skipping")
                continue

            # Get regime series
            regime_series = regime_calc.get_regime(params["regime"], market)
            regime_strength = (
                regime_calc.get_regime_strength(params["regime"], market)
                if params["regime_mode"] == "SIZING" else None
            )

            for fee_mult in fee_mults:
                fee_bps = base_fee * fee_mult
                slippage_bps = base_slippage * fee_mult

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
                    market_config=market_cfg,
                )

                if result:
                    results.append({
                        "market": market,
                        "strategy_id": strategy_id,
                        "fee_mult": fee_mult,
                        "slippage_mult": fee_mult,
                        "fee_bps": fee_bps,
                        "slippage_bps": slippage_bps,
                        "sharpe": result.sharpe,
                        "cagr": result.cagr,
                        "mdd": result.mdd,
                        "trades": result.trades,
                        "turnover": result.turnover,
                        "win_rate": result.win_rate,
                        "valid": result.valid,
                        "violations": result.violations or "",
                        "zero_close_count": result.zero_close_count,
                        "ret_inf_count": result.ret_inf_count,
                        "strat_ret_clamp_count": result.strat_ret_clamp_count,
                        "cumret_clip_count": result.cumret_clip_count,
                    })

                completed += 1
                if completed % 10 == 0:
                    print(f"  Progress: {completed}/{total_combos}", flush=True)

    # Create DataFrame and sort deterministically
    df = pd.DataFrame(results)
    df = df.sort_values(
        ["market", "strategy_id", "fee_mult", "slippage_mult"],
        ascending=[True, True, True, True]
    ).reset_index(drop=True)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage2 Cost Sensitivity Analysis")
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
    parser.add_argument("--fee-mults", default="0.5,1.0,1.5,2.0",
                        help="Fee multipliers (comma-separated)")
    args = parser.parse_args()

    fee_mults = [float(x) for x in args.fee_mults.split(",")]

    print("[PURPOSE] Stage2 Cost Sensitivity Analysis", flush=True)
    print("-" * 70, flush=True)
    print(f"  Stage1 dir: {args.stage1_dir}", flush=True)
    print(f"  Output dir: {args.out}", flush=True)
    print(f"  Fee multipliers: {fee_mults}", flush=True)
    print("-" * 70, flush=True)

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Run analysis
    df = run_sensitivity_analysis(
        stage1_dir=args.stage1_dir,
        candidates_csv=args.candidates_csv,
        markets_config_path=args.markets,
        grid_config_path=args.grid,
        normalized_dir=args.normalized_dir,
        out_dir=args.out,
        fee_mults=fee_mults,
    )

    # Save results
    out_path = args.out / "stage2_sensitivity.csv"
    df.to_csv(out_path, index=False, float_format="%.8g")
    print(f"\n[SAVED] {out_path} ({len(df)} rows)")

    # Summary
    print("\n[SUMMARY]")
    print(f"  Total rows: {len(df)}")
    print(f"  Markets: {df['market'].nunique()}")
    print(f"  Strategies: {df['strategy_id'].nunique()}")
    print(f"  Fee multipliers: {sorted(df['fee_mult'].unique())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
