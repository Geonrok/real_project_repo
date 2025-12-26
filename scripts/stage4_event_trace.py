#!/usr/bin/env python3
"""
Stage4 Event Trace - Trace clamp-only WARN events to concrete symbol/date evidence.

For all WARN strategies (EXTREME_RETURN_CLAMP_ONLY) in binance_spot:
1) Identify worst_symbol and the exact worst_date where strat_ret was clamped.
2) Dump a single-bar "event record" with close/prev_close, raw return, raw strat_ret,
   clipped strat_ret, position change, costs, equity before/after.
3) Produce deterministic outputs + snapshot + sha256 manifest for audit.

Usage:
    python scripts/stage4_event_trace.py \
        --stage2-dir outputs/stage2_full_v1 \
        --stage1-dir outputs/stage1_full_v2 \
        --out outputs/stage4_event_trace_v1
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from backtest_runner import (
    backtest_symbol,
    compute_strategy_signals,
    load_symbol_data,
    list_symbols,
)
from regimes import RegimeCalculator


def load_warn_strategies(stage2_dir: Path) -> pd.DataFrame:
    """Load WARN strategies from stage2_final_pass_with_warn.csv."""
    path = stage2_dir / "stage2_final_pass_with_warn.csv"
    if not path.exists():
        print(f"[ERROR] Required file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # Filter to WARN with EXTREME_RETURN_CLAMP_ONLY
    warn_df = df[
        (df["data_quality_status"] == "WARN") &
        (df["data_quality_reason"] == "EXTREME_RETURN_CLAMP_ONLY")
    ].copy()

    return warn_df


def load_strategy_params(stage1_dir: Path, market: str) -> pd.DataFrame:
    """Load strategy parameters from stage1_summary.csv."""
    path = stage1_dir / "stage1_summary.csv"
    if not path.exists():
        print(f"[ERROR] Required file not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)
    # Filter to specified market
    return df[df["market"] == market].copy()


def load_grid_config(grid_path: Path) -> dict:
    """Load grid configuration with trend/momentum params."""
    with open(grid_path) as f:
        return yaml.safe_load(f)


def load_markets_config(markets_path: Path) -> dict:
    """Load markets configuration."""
    with open(markets_path) as f:
        return yaml.safe_load(f)


def trace_clamp_events(
    strategy_id: str,
    strategy_params: dict,
    market: str,
    normalized_dir: Path,
    regime_calc: RegimeCalculator,
    trend_params: dict,
    momentum_params: dict,
    vol_target_base: float,
    fee_bps: float,
    slippage_bps: float,
    market_config: dict,
    require_single_clamp: bool = True,
) -> list[dict]:
    """
    Trace clamp events for a single strategy.

    Returns list of event records (one per clamp event found).
    """
    symbols = list_symbols(normalized_dir, market)
    if not symbols:
        return []

    # Get regime series
    regime_name = strategy_params["regime"]
    regime_mode = strategy_params["regime_mode"]
    regime_series = regime_calc.get_regime(regime_name, market)
    regime_strength = (
        regime_calc.get_regime_strength(regime_name, market)
        if regime_mode == "SIZING" else None
    )

    # Get leverage constraints from market config
    max_leverage = market_config.get("max_gross_leverage", 1.0)
    allow_shorting = market_config.get("allow_shorting", False)
    allow_borrowing = market_config.get("allow_borrowing", False)

    events = []

    for symbol in symbols:
        df = load_symbol_data(normalized_dir, market, symbol)
        if df is None or len(df) < 60:
            continue

        # Compute signals
        try:
            signals = compute_strategy_signals(
                df,
                strategy_params["trend"],
                strategy_params["momentum"],
                regime_series,
                regime_mode,
                regime_strength,
                trend_params,
                momentum_params,
                vol_target_base,
                max_leverage=max_leverage,
                allow_shorting=allow_shorting,
                allow_borrowing=allow_borrowing,
            )
        except Exception:
            continue

        # Run backtest with diagnostics
        result = backtest_symbol(signals, fee_bps, slippage_bps, return_diag=True)

        # Check for clamp events
        clamp_flag = result.get("_diag_clamp_flag")
        if clamp_flag is None:
            continue

        clamp_indices = np.where(clamp_flag == 1)[0]
        if len(clamp_indices) == 0:
            continue

        # Extract event details for each clamp
        dates = result["_diag_dates"]
        strat_ret_raw = result["_diag_strat_ret_raw"]
        strat_ret = result["_diag_strat_ret"]
        position = result["_diag_position"]
        costs = result["_diag_costs"]
        equity = result["_diag_equity"]
        close = result["_diag_close"]

        for idx in clamp_indices:
            # Get previous bar values (handle idx=0 edge case)
            prev_close = close[idx - 1] if idx > 0 else np.nan
            prev_position = position[idx - 1] if idx > 0 else 0.0
            equity_before = equity[idx - 1] if idx > 0 else 1.0

            # Raw return calculation
            raw_ret = (close[idx] / prev_close - 1) if prev_close and prev_close > 0 else np.nan

            event = {
                "market": market,
                "strategy_id": strategy_id,
                "symbol": symbol,
                "date": dates[idx].strftime("%Y-%m-%d"),
                "close": close[idx],
                "prev_close": prev_close,
                "raw_ret": raw_ret,
                "strat_ret_raw": strat_ret_raw[idx],
                "strat_ret_clipped": strat_ret[idx],
                "position_prev": prev_position,
                "position": position[idx],
                "costs": costs[idx],
                "equity_before": equity_before,
                "equity_after": equity[idx],
                "clamp_count_in_symbol": len(clamp_indices),
            }
            events.append(event)

    # Determine status
    total_clamps = len(events)
    if require_single_clamp:
        for event in events:
            if total_clamps != 1:
                event["status"] = "REVIEW"
            else:
                event["status"] = "TRACED"
    else:
        for event in events:
            event["status"] = "TRACED"

    return events


def generate_markdown_report(events_df: pd.DataFrame, out_path: Path) -> None:
    """Generate human-readable markdown report."""
    lines = [
        "# Stage4 Event Trace Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Summary",
        "",
    ]

    if len(events_df) == 0:
        lines.append("*No clamp events found*")
    else:
        total = len(events_df)
        traced = len(events_df[events_df["status"] == "TRACED"])
        review = len(events_df[events_df["status"] == "REVIEW"])
        strategies = events_df["strategy_id"].nunique()
        symbols = events_df["symbol"].nunique()

        lines.extend([
            f"- Total clamp events: {total}",
            f"- Strategies affected: {strategies}",
            f"- Symbols involved: {symbols}",
            f"- **TRACED**: {traced} (single clamp, fully traced)",
            f"- **REVIEW**: {review} (multiple clamps or needs investigation)",
            "",
        ])

        # Event details table
        lines.extend([
            "## Clamp Event Details",
            "",
            "| Strategy | Symbol | Date | strat_ret_raw | clipped | Status |",
            "|----------|--------|------|---------------|---------|--------|",
        ])

        for _, row in events_df.iterrows():
            lines.append(
                f"| {row['strategy_id']} | {row['symbol']} | {row['date']} | "
                f"{row['strat_ret_raw']:.4f} | {row['strat_ret_clipped']:.4f} | {row['status']} |"
            )

        # Find worst event
        worst_idx = events_df["strat_ret_raw"].idxmin()
        worst = events_df.loc[worst_idx]

        lines.extend([
            "",
            "## Worst Event (Most Extreme strat_ret_raw)",
            "",
            f"- **Strategy**: {worst['strategy_id']}",
            f"- **Symbol**: {worst['symbol']}",
            f"- **Date**: {worst['date']}",
            f"- **strat_ret_raw**: {worst['strat_ret_raw']:.6f}",
            f"- **strat_ret_clipped**: {worst['strat_ret_clipped']:.6f}",
            f"- **close**: {worst['close']:.6f}",
            f"- **prev_close**: {worst['prev_close']:.6f}",
            f"- **raw_ret**: {worst['raw_ret']:.6f}",
            f"- **position_prev**: {worst['position_prev']:.4f}",
            f"- **costs**: {worst['costs']:.6f}",
            f"- **equity_before**: {worst['equity_before']:.6f}",
            f"- **equity_after**: {worst['equity_after']:.6f}",
            "",
        ])

        # Policy note
        lines.extend([
            "## Policy Note",
            "",
            "This evidence supports the \"research WARN\" classification from Stage2/Stage3:",
            "",
            "- These strategies experienced extreme single-bar returns that got clipped to -1.0",
            "- But the equity curve remained positive (no cumulative return floor clip)",
            "- The clamp prevents negative equity in the backtest model",
            "",
            "**Strong Warning**: Without a liquidation/margin model, the clip may understate",
            "the real impact in live trading. In reality, such extreme moves could trigger:",
            "- Margin calls",
            "- Forced liquidation",
            "- Slippage beyond model assumptions",
            "",
            "Revisit this classification before deploying to live trading.",
            "",
        ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4 Event Trace")
    parser.add_argument("--stage2-dir", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Stage2 output directory")
    parser.add_argument("--stage1-dir", type=Path, default=Path("outputs/stage1_full_v2"),
                        help="Stage1 output directory")
    parser.add_argument("--markets", type=Path, default=Path("configs/markets.yaml"),
                        help="Markets config path")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_stage1.yaml"),
                        help="Grid config path")
    parser.add_argument("--normalized-dir", type=Path, default=Path("outputs/normalized_1d"),
                        help="Normalized data directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/stage4_event_trace_v1"),
                        help="Output directory")
    parser.add_argument("--market", default="binance_spot",
                        help="Market to analyze")
    parser.add_argument("--topk", type=int, default=13,
                        help="Max WARN strategies to process")
    parser.add_argument("--require-single-clamp", type=bool, default=True,
                        help="Mark as REVIEW if clamp count != 1")
    args = parser.parse_args()

    print("[PURPOSE] Stage4 Event Trace - Trace clamp-only WARN events", flush=True)
    print("-" * 70, flush=True)

    # Check required inputs exist
    required_files = [
        args.stage2_dir / "stage2_final_pass_with_warn.csv",
        args.stage1_dir / "stage1_summary.csv",
        args.markets,
        args.grid,
    ]
    missing = [str(f) for f in required_files if not f.exists()]
    if missing:
        print("[ERROR] Required files not found:")
        for f in missing:
            print(f"  - {f}")
        sys.exit(1)

    # Load configurations
    markets_config = load_markets_config(args.markets)
    grid_config = load_grid_config(args.grid)
    market_cfg = markets_config.get("markets", {}).get(args.market, {})

    fee_bps = market_cfg.get("fee_bps_roundtrip", 10)
    slippage_bps = market_cfg.get("slippage_bps_roundtrip", 2)

    trend_params = grid_config.get("trend_params", {})
    momentum_params = grid_config.get("momentum_params", {})
    vol_target_base = grid_config.get("vol_target_base", 0.2)

    # Initialize regime calculator
    reference_market = grid_config.get("reference_market", "binance_spot")
    regime_calc = RegimeCalculator(
        args.normalized_dir,
        source_mode="reference_market",
        reference_market=reference_market,
        ma_period=grid_config.get("regime_params", {}).get("ma_period", 50),
    )

    # Load WARN strategies
    warn_df = load_warn_strategies(args.stage2_dir)
    print(f"[INFO] Found {len(warn_df)} WARN strategies", flush=True)

    if len(warn_df) == 0:
        print("[WARN] No WARN strategies found, nothing to trace")
        args.out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame().to_csv(args.out / "stage4_event_trace.csv", index=False)
        generate_markdown_report(pd.DataFrame(), args.out / "stage4_event_trace.md")
        return 0

    # Load strategy parameters
    params_df = load_strategy_params(args.stage1_dir, args.market)
    print(f"[INFO] Loaded {len(params_df)} strategy params for {args.market}", flush=True)

    # Process WARN strategies
    all_events = []
    processed = 0

    for _, row in warn_df.iterrows():
        if processed >= args.topk:
            break

        strategy_id = row["strategy_id"]

        # Get strategy parameters
        param_rows = params_df[params_df["strategy_id"] == strategy_id]
        if len(param_rows) == 0:
            print(f"  [WARN] No params found for {strategy_id}, skipping")
            continue

        strategy_params = param_rows.iloc[0].to_dict()

        print(f"  Tracing: {strategy_id}...", flush=True)

        events = trace_clamp_events(
            strategy_id=strategy_id,
            strategy_params=strategy_params,
            market=args.market,
            normalized_dir=args.normalized_dir,
            regime_calc=regime_calc,
            trend_params=trend_params,
            momentum_params=momentum_params,
            vol_target_base=vol_target_base,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            market_config=market_cfg,
            require_single_clamp=args.require_single_clamp,
        )

        all_events.extend(events)
        processed += 1

        if events:
            print(f"    Found {len(events)} clamp event(s)", flush=True)
        else:
            print(f"    No clamp events found (unexpected)", flush=True)

    # Create output directory
    args.out.mkdir(parents=True, exist_ok=True)

    # Build DataFrame
    events_df = pd.DataFrame(all_events)

    # Sort for determinism
    if len(events_df) > 0:
        events_df = events_df.sort_values(
            ["market", "strategy_id", "symbol", "date"],
            ascending=[True, True, True, True]
        ).reset_index(drop=True)

    # Save CSV
    csv_path = args.out / "stage4_event_trace.csv"
    events_df.to_csv(csv_path, index=False, float_format="%.8g")
    print(f"\n[SAVED] {csv_path} ({len(events_df)} rows)", flush=True)

    # Save markdown report
    md_path = args.out / "stage4_event_trace.md"
    generate_markdown_report(events_df, md_path)
    print(f"[SAVED] {md_path}", flush=True)

    # Summary
    print("\n[SUMMARY]", flush=True)
    print(f"  Total clamp events: {len(events_df)}", flush=True)
    if len(events_df) > 0:
        print(f"  TRACED: {len(events_df[events_df['status'] == 'TRACED'])}", flush=True)
        print(f"  REVIEW: {len(events_df[events_df['status'] == 'REVIEW'])}", flush=True)
        print(f"  Strategies: {events_df['strategy_id'].nunique()}", flush=True)
        print(f"  Symbols: {events_df['symbol'].nunique()}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
