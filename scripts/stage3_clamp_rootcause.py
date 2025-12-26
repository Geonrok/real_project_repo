#!/usr/bin/env python3
"""
Stage3 Clamp Root-Cause Analysis.

Analyzes strategies that failed Stage2 data_quality due to clamp events,
classifying root causes as DATA_ZERO_CLOSE, DATA_RET_INF, EXTREME_RETURN_CLAMP_ONLY,
or EQUITY_FLOOR_CLIP.

Usage:
    python scripts/stage3_clamp_rootcause.py \
        --stage2-dir outputs/stage2_full_v1 \
        --stage1-dir outputs/stage1_full_v2 \
        --out outputs/stage3_rootcause_v1
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
    list_symbols,
)


def load_data_quality(stage2_dir: Path) -> pd.DataFrame:
    """Load Stage2 data quality results."""
    path = stage2_dir / "stage2_data_quality.csv"
    if not path.exists():
        print(f"[ERROR] Data quality file not found: {path}", flush=True)
        sys.exit(1)
    return pd.read_csv(path)


def load_stage1_summary(stage1_dir: Path) -> pd.DataFrame:
    """Load Stage1 summary for strategy parameters."""
    path = stage1_dir / "stage1_summary.csv"
    if not path.exists():
        print(f"[ERROR] Stage1 summary not found: {path}", flush=True)
        sys.exit(1)
    return pd.read_csv(path)


def load_grid_config(grid_path: Path) -> dict:
    """Load grid configuration."""
    with open(grid_path) as f:
        return yaml.safe_load(f)


def load_markets_config(markets_path: Path) -> dict:
    """Load markets configuration."""
    with open(markets_path) as f:
        return yaml.safe_load(f)


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


def _safe_int(val) -> int:
    """Convert value to int, treating None/NaN as 0."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def classify_root_cause(row: pd.Series) -> str:
    """
    Classify root cause based on diagnostic counters.

    Priority order:
    1. DATA_ZERO_CLOSE - close price was zero/negative (data issue)
    2. DATA_RET_INF - infinite return (0→positive transition, data issue)
    3. EXTREME_RETURN_CLAMP_ONLY - strat_ret clipped but equity survived
    4. EQUITY_FLOOR_CLIP - cumulative return went negative
    5. UNKNOWN - none of the above
    """
    zero_close = _safe_int(row.get("zero_close_count", 0))
    ret_inf = _safe_int(row.get("ret_inf_count", 0))
    clamp = _safe_int(row.get("strat_ret_clamp_count", 0))
    cumret_clip = _safe_int(row.get("cumret_clip_count", 0))

    if zero_close > 0:
        return "DATA_ZERO_CLOSE"
    elif ret_inf > 0:
        return "DATA_RET_INF"
    elif clamp > 0 and cumret_clip == 0:
        return "EXTREME_RETURN_CLAMP_ONLY"
    elif cumret_clip > 0:
        return "EQUITY_FLOOR_CLIP"
    else:
        return "UNKNOWN"


def find_worst_symbols_for_strategy(
    market: str,
    strategy_id: str,
    params: dict,
    market_data: dict[str, pd.DataFrame],
    regime_calc: RegimeCalculator,
    grid_config: dict,
    market_config: dict,
    topk: int = 10,
) -> tuple[list[str], list[str]]:
    """
    Find symbols with clamp events for a given strategy.

    Returns (worst_symbols, worst_dates) lists.
    """
    trend_params = grid_config.get("trend_params", {}).get(params["trend"], {})
    momentum_params = grid_config.get("momentum_params", {}).get(params["momentum"], {})
    vol_target_base = grid_config.get("vol_target_base", 0.2)

    fee_bps = market_config.get("fee_bps_roundtrip", 10)
    slippage_bps = market_config.get("slippage_bps_roundtrip", 2)

    regime_series = regime_calc.get_regime(params["regime"], market)
    regime_strength = (
        regime_calc.get_regime_strength(params["regime"], market)
        if params["regime_mode"] == "SIZING" else None
    )

    # Track symbols with issues
    symbol_clamp_counts = {}
    symbol_worst_dates = {}

    for symbol, df in market_data.items():
        if len(df) < 30:
            continue

        df = df.copy()
        df = df.sort_values("date").reset_index(drop=True)

        # Check for zero/negative close
        zero_close_mask = (df["close"] <= 0) | df["close"].isna()
        zero_close_count = int(zero_close_mask.sum())

        # Calculate returns
        df["ret"] = df["close"].pct_change().fillna(0)
        ret_inf_count = int(np.isinf(df["ret"]).sum())
        df["ret"] = df["ret"].replace([np.inf, -np.inf], 0)

        # Simple position and return calculation (simplified for worst-symbol detection)
        df["position"] = 1.0  # Assume always in position for detection
        cost_per_trade = (fee_bps + slippage_bps) / 10000
        df["strat_ret"] = df["position"].shift(1).fillna(0) * df["ret"] - cost_per_trade * 0.01

        # Count clamp events
        strat_ret_clamp_mask = df["strat_ret"] < -1.0
        clamp_count = int(strat_ret_clamp_mask.sum())

        if clamp_count > 0 or zero_close_count > 0 or ret_inf_count > 0:
            total_issues = clamp_count + zero_close_count + ret_inf_count
            symbol_clamp_counts[symbol] = total_issues

            # Get dates of clamp events
            if clamp_count > 0:
                clamp_dates = df.loc[strat_ret_clamp_mask, "date"].tolist()
                symbol_worst_dates[symbol] = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in clamp_dates[:topk]]
            elif zero_close_count > 0:
                zero_dates = df.loc[zero_close_mask, "date"].tolist()
                symbol_worst_dates[symbol] = [d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d) for d in zero_dates[:topk]]

    # Sort by issue count and get top symbols
    sorted_symbols = sorted(symbol_clamp_counts.items(), key=lambda x: -x[1])
    worst_symbols = [s for s, _ in sorted_symbols[:topk]]

    # Collect dates from worst symbols
    all_dates = []
    for sym in worst_symbols[:3]:  # Top 3 symbols' dates
        if sym in symbol_worst_dates:
            all_dates.extend(symbol_worst_dates[sym])

    return worst_symbols, all_dates[:topk]


def run_rootcause_analysis(
    stage2_dir: Path,
    stage1_dir: Path,
    out_dir: Path,
    market_filter: str,
    only_failed: bool,
    topk_symbols: int,
    topk_dates: int,
    debug_dump_top1: bool,
    markets_config_path: Path,
    grid_config_path: Path,
    normalized_dir: Path,
) -> pd.DataFrame:
    """Run root cause analysis on clamp events."""

    print(f"[INFO] Loading data quality from: {stage2_dir}", flush=True)
    quality_df = load_data_quality(stage2_dir)
    summary_df = load_stage1_summary(stage1_dir)
    grid_config = load_grid_config(grid_config_path)
    markets_config = load_markets_config(markets_config_path)

    # Filter to market and rows with issues
    if market_filter:
        quality_df = quality_df[quality_df["market"] == market_filter]

    if only_failed:
        # Filter to rows with any diagnostic issue
        issue_mask = (
            (quality_df["zero_close_count"] > 0) |
            (quality_df["ret_inf_count"] > 0) |
            (quality_df["strat_ret_clamp_count"] > 0) |
            (quality_df["cumret_clip_count"] > 0)
        )
        quality_df = quality_df[issue_mask]

    print(f"[INFO] Found {len(quality_df)} rows with issues", flush=True)

    if len(quality_df) == 0:
        print("[WARN] No issues found, creating empty output", flush=True)
        return pd.DataFrame(columns=[
            "market", "strategy_id", "root_cause",
            "zero_close_count", "ret_inf_count", "strat_ret_clamp_count", "cumret_clip_count",
            "worst_symbols", "worst_dates", "notes"
        ])

    results = []

    # Group by market for data loading efficiency
    for market in quality_df["market"].unique():
        market_rows = quality_df[quality_df["market"] == market]
        print(f"\n[{market}] Processing {len(market_rows)} strategies with issues...", flush=True)

        market_config = markets_config.get("markets", {}).get(market, {})

        # Load market data
        market_data = preload_market_data(normalized_dir, market)
        if not market_data:
            print(f"  [WARN] No data for {market}, skipping", flush=True)
            continue

        regime_calc = RegimeCalculator(normalized_dir)

        for _, row in market_rows.iterrows():
            strategy_id = row["strategy_id"]
            params = get_strategy_params(summary_df, strategy_id)

            if params is None:
                print(f"  [WARN] Strategy {strategy_id} not in summary, skipping", flush=True)
                continue

            root_cause = classify_root_cause(row)

            # Find worst symbols and dates
            worst_symbols, worst_dates = find_worst_symbols_for_strategy(
                market=market,
                strategy_id=strategy_id,
                params=params,
                market_data=market_data,
                regime_calc=regime_calc,
                grid_config=grid_config,
                market_config=market_config,
                topk=topk_symbols,
            )

            # Generate notes based on root cause
            notes = ""
            if root_cause == "DATA_ZERO_CLOSE":
                notes = "FAIL: Data contains zero/negative close prices"
            elif root_cause == "DATA_RET_INF":
                notes = "FAIL: Data contains infinite returns (0→positive)"
            elif root_cause == "EXTREME_RETURN_CLAMP_ONLY":
                notes = "WARN: Extreme return clipped, but equity survived"
            elif root_cause == "EQUITY_FLOOR_CLIP":
                notes = "FAIL: Equity went negative, clipped to zero"
            else:
                notes = "UNKNOWN: Needs investigation"

            results.append({
                "market": market,
                "strategy_id": strategy_id,
                "root_cause": root_cause,
                "zero_close_count": int(row.get("zero_close_count", 0) or 0),
                "ret_inf_count": int(row.get("ret_inf_count", 0) or 0),
                "strat_ret_clamp_count": int(row.get("strat_ret_clamp_count", 0) or 0),
                "cumret_clip_count": int(row.get("cumret_clip_count", 0) or 0),
                "worst_symbols": ";".join(worst_symbols[:topk_symbols]) if worst_symbols else "",
                "worst_dates": ";".join(worst_dates[:topk_dates]) if worst_dates else "",
                "notes": notes,
            })

            print(f"  {strategy_id}: {root_cause} (worst: {worst_symbols[:3] if worst_symbols else 'N/A'})", flush=True)

    # Create DataFrame and sort deterministically
    df = pd.DataFrame(results)
    if len(df) > 0:
        df = df.sort_values(
            ["market", "root_cause", "strategy_id"],
            ascending=[True, True, True],
            na_position="last"
        ).reset_index(drop=True)

    return df


def generate_markdown_report(df: pd.DataFrame, out_path: Path):
    """Generate human-readable markdown report."""
    lines = [
        "# Stage3 Clamp Root-Cause Report",
        "",
        f"Total strategies analyzed: {len(df)}",
        "",
        "## Root Cause Distribution",
        "",
        "| Root Cause | Count | Recommendation |",
        "|------------|-------|----------------|",
    ]

    recommendations = {
        "DATA_ZERO_CLOSE": "FAIL - exclude symbol",
        "DATA_RET_INF": "FAIL - exclude symbol",
        "EXTREME_RETURN_CLAMP_ONLY": "WARN - review",
        "EQUITY_FLOOR_CLIP": "FAIL - investigate",
        "UNKNOWN": "INVESTIGATE",
    }

    if len(df) > 0:
        cause_counts = df["root_cause"].value_counts()
        for cause, count in cause_counts.items():
            rec = recommendations.get(cause, "INVESTIGATE")
            lines.append(f"| {cause} | {count} | {rec} |")
    else:
        lines.append("| (none) | 0 | N/A |")

    lines.extend([
        "",
        "## Top Affected Symbols",
        "",
    ])

    if len(df) > 0 and "worst_symbols" in df.columns:
        # Collect all worst symbols across strategies
        all_symbols = []
        for syms in df["worst_symbols"].dropna():
            if syms:
                all_symbols.extend(syms.split(";"))

        if all_symbols:
            from collections import Counter
            symbol_counts = Counter(all_symbols)
            lines.append("| Symbol | Occurrence |")
            lines.append("|--------|------------|")
            for sym, count in symbol_counts.most_common(10):
                lines.append(f"| {sym} | {count} |")
        else:
            lines.append("*No worst symbols identified*")
    else:
        lines.append("*No data available*")

    lines.extend([
        "",
        "## Top Affected Dates",
        "",
    ])

    if len(df) > 0 and "worst_dates" in df.columns:
        all_dates = []
        for dates in df["worst_dates"].dropna():
            if dates:
                all_dates.extend(dates.split(";"))

        if all_dates:
            from collections import Counter
            date_counts = Counter(all_dates)
            lines.append("| Date | Occurrence |")
            lines.append("|------|------------|")
            for date, count in date_counts.most_common(10):
                lines.append(f"| {date} | {count} |")
        else:
            lines.append("*No worst dates identified*")
    else:
        lines.append("*No data available*")

    lines.extend([
        "",
        "## Strategy Details",
        "",
        "| Strategy | Root Cause | Clamp Count | Notes |",
        "|----------|------------|-------------|-------|",
    ])

    if len(df) > 0:
        for _, row in df.iterrows():
            strat = row["strategy_id"][:40] + "..." if len(row["strategy_id"]) > 40 else row["strategy_id"]
            lines.append(f"| {strat} | {row['root_cause']} | {row['strat_ret_clamp_count']} | {row['notes'][:30]}... |")

    lines.extend([
        "",
        "## Policy Recommendation",
        "",
        "Based on the analysis:",
        "",
    ])

    if len(df) > 0:
        cause_counts = df["root_cause"].value_counts()

        if "EXTREME_RETURN_CLAMP_ONLY" in cause_counts:
            lines.append(f"- **EXTREME_RETURN_CLAMP_ONLY** ({cause_counts.get('EXTREME_RETURN_CLAMP_ONLY', 0)} strategies): "
                        "These experienced extreme single-bar returns but equity remained positive. "
                        "Consider WARN status - may be acceptable in high-volatility environments.")

        if "DATA_ZERO_CLOSE" in cause_counts or "DATA_RET_INF" in cause_counts:
            lines.append(f"- **DATA issues** ({cause_counts.get('DATA_ZERO_CLOSE', 0) + cause_counts.get('DATA_RET_INF', 0)} strategies): "
                        "These are caused by bad data (zero close, infinite returns). "
                        "Recommend using `--exclude-bad-symbols` option in future Stage4 runs.")
    else:
        lines.append("*No issues to recommend*")

    lines.extend([
        "",
        "## Future: --exclude-bad-symbols Option",
        "",
        "To re-evaluate candidates excluding problematic symbols, a future Stage4 could implement:",
        "",
        "```",
        "python scripts/stage4_reevaluate.py \\",
        "    --exclude-bad-symbols outputs/stage3_rootcause_v1/bad_symbols.txt",
        "```",
        "",
        "This would exclude symbols with `zero_close_count > 0` or `ret_inf_count > 0` from backtest.",
        "",
    ])

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage3 Clamp Root-Cause Analysis")
    parser.add_argument("--stage2-dir", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Stage2 output directory")
    parser.add_argument("--stage1-dir", type=Path, default=Path("outputs/stage1_full_v2"),
                        help="Stage1 output directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/stage3_rootcause_v1"),
                        help="Output directory")
    parser.add_argument("--market-filter", default="binance_spot",
                        help="Market to filter (empty for all)")
    parser.add_argument("--only-failed", type=bool, default=True,
                        help="Only analyze strategies with issues")
    parser.add_argument("--topk-symbols", type=int, default=10,
                        help="Top K worst symbols to report")
    parser.add_argument("--topk-dates", type=int, default=10,
                        help="Top K worst dates to report")
    parser.add_argument("--debug-dump-top1", type=bool, default=True,
                        help="Dump debug info for worst case")
    parser.add_argument("--markets", type=Path, default=Path("configs/markets.yaml"),
                        help="Markets config path")
    parser.add_argument("--grid", type=Path, default=Path("configs/grid_stage1.yaml"),
                        help="Grid config path")
    parser.add_argument("--normalized-dir", type=Path, default=Path("outputs/normalized_1d"),
                        help="Normalized data directory")
    args = parser.parse_args()

    print("[PURPOSE] Stage3 Clamp Root-Cause Analysis", flush=True)
    print("-" * 70, flush=True)
    print(f"  Stage2 dir: {args.stage2_dir}", flush=True)
    print(f"  Stage1 dir: {args.stage1_dir}", flush=True)
    print(f"  Output dir: {args.out}", flush=True)
    print(f"  Market filter: {args.market_filter}", flush=True)
    print("-" * 70, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    df = run_rootcause_analysis(
        stage2_dir=args.stage2_dir,
        stage1_dir=args.stage1_dir,
        out_dir=args.out,
        market_filter=args.market_filter,
        only_failed=args.only_failed,
        topk_symbols=args.topk_symbols,
        topk_dates=args.topk_dates,
        debug_dump_top1=args.debug_dump_top1,
        markets_config_path=args.markets,
        grid_config_path=args.grid,
        normalized_dir=args.normalized_dir,
    )

    # Save CSV
    csv_path = args.out / "stage3_clamp_rootcause.csv"
    df.to_csv(csv_path, index=False, float_format="%.8g")
    print(f"\n[SAVED] {csv_path} ({len(df)} rows)", flush=True)

    # Save Markdown
    md_path = args.out / "stage3_clamp_rootcause.md"
    generate_markdown_report(df, md_path)
    print(f"[SAVED] {md_path}", flush=True)

    # Summary
    print("\n[SUMMARY]", flush=True)
    print(f"  Total strategies analyzed: {len(df)}", flush=True)
    if len(df) > 0:
        print(f"  Root cause distribution:", flush=True)
        for cause, count in df["root_cause"].value_counts().items():
            print(f"    {cause}: {count}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
