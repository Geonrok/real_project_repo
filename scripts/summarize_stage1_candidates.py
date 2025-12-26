#!/usr/bin/env python3
"""
Summarize Stage1 backtest results and extract robust candidates.

Purpose: Identify strategies that perform well across multiple markets.

Usage:
    python scripts/summarize_stage1_candidates.py \
        --summary-csv outputs/stage1_full/stage1_summary.csv \
        --out-dir outputs/stage1_full \
        --topn 20 \
        --robust-k 3

Outputs (3 required):
    - candidates_top20_by_market.csv: Top N per market
    - candidates_top20_intersection.csv: Top N from intersection summary
    - candidates_robust_kof4_top20.csv: Strategies in top N of K+ markets
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_summary(path: Path, require_sharpe: bool = True) -> pd.DataFrame:
    """Load summary CSV and validate columns."""
    if not path.exists():
        print(f"[ERROR] File not found: {path}")
        sys.exit(1)

    df = pd.read_csv(path)

    # Check for sharpe column
    if require_sharpe and "sharpe" not in df.columns:
        print(f"[ERROR] Required column 'sharpe' not found in {path}")
        print(f"  Available columns: {list(df.columns)}")
        sys.exit(1)

    # Check for market column
    if "market" not in df.columns:
        print(f"[ERROR] Required column 'market' not found in {path}")
        sys.exit(1)

    # Filter valid only if column exists
    if "valid" in df.columns:
        original_len = len(df)
        df = df[df["valid"] == True].copy()
        filtered = original_len - len(df)
        if filtered > 0:
            print(f"  [INFO] Filtered {filtered} invalid rows (valid != True)")
    else:
        print(f"  [WARN] 'valid' column not found, using all rows")

    return df


def generate_top20_by_market(df: pd.DataFrame, topn: int) -> pd.DataFrame:
    """
    Generate Top N strategies per market.

    Sorting: market asc, sharpe desc, strategy_id asc
    Columns: market, rank, strategy_id, sharpe, cagr, mdd, trades, valid
    """
    results = []

    for market in sorted(df["market"].unique()):
        market_df = df[df["market"] == market].copy()

        # Sort by sharpe desc, then strategy_id asc for tie-break
        market_df = market_df.sort_values(
            ["sharpe", "strategy_id"],
            ascending=[False, True]
        )

        # Take top N
        top_n = market_df.head(topn).copy()
        top_n["rank"] = range(1, len(top_n) + 1)
        results.append(top_n)

    if not results:
        return pd.DataFrame()

    combined = pd.concat(results, ignore_index=True)

    # Select and order columns
    base_cols = ["market", "rank", "strategy_id", "sharpe"]
    optional_cols = ["cagr", "mdd", "trades", "valid"]

    cols = base_cols + [c for c in optional_cols if c in combined.columns]
    return combined[cols]


def generate_top20_intersection(
    summary_df: pd.DataFrame,
    intersection_path: Path,
    topn: int
) -> pd.DataFrame | None:
    """
    Generate Top N from intersection summary.

    If intersection file exists, use it. Otherwise return None.
    Sorting: sharpe desc, strategy_id asc (add market asc if present)
    """
    if not intersection_path.exists():
        print(f"  [WARN] Intersection file not found: {intersection_path}")
        return None

    df = load_summary(intersection_path, require_sharpe=True)

    if len(df) == 0:
        print(f"  [WARN] Intersection file is empty after filtering")
        return None

    # Sort: if market column exists, market asc then sharpe desc, strategy_id asc
    if "market" in df.columns:
        df = df.sort_values(
            ["market", "sharpe", "strategy_id"],
            ascending=[True, False, True]
        )
    else:
        df = df.sort_values(
            ["sharpe", "strategy_id"],
            ascending=[False, True]
        )

    # Take top N overall
    top_n = df.head(topn).copy()
    top_n["rank"] = range(1, len(top_n) + 1)

    # Select columns
    base_cols = ["rank", "strategy_id", "sharpe"]
    if "market" in top_n.columns:
        base_cols = ["market"] + base_cols
    optional_cols = ["cagr", "mdd", "trades", "valid"]

    cols = base_cols + [c for c in optional_cols if c in top_n.columns]
    return top_n[cols]


def generate_robust_candidates(df: pd.DataFrame, topn: int, robust_k: int) -> pd.DataFrame:
    """
    Find strategies in Top N of K+ markets.

    Columns: strategy_id, count_markets, markets_list, sharpe_mean,
             sharpe_min, sharpe_max, cagr_mean, mdd_worst, trades_mean
    Sorting: count_markets desc, sharpe_mean desc, strategy_id asc
    """
    markets = sorted(df["market"].unique())

    # Build top N set per market
    strategy_info: dict[str, dict] = {}

    for market in markets:
        market_df = df[df["market"] == market].copy()
        market_df = market_df.sort_values(
            ["sharpe", "strategy_id"],
            ascending=[False, True]
        )
        top_n = market_df.head(topn)

        for _, row in top_n.iterrows():
            strat = row["strategy_id"]
            if strat not in strategy_info:
                strategy_info[strat] = {
                    "markets": [],
                    "sharpes": [],
                    "cagrs": [],
                    "mdds": [],
                    "trades": [],
                }
            strategy_info[strat]["markets"].append(market)
            strategy_info[strat]["sharpes"].append(row["sharpe"])
            if "cagr" in row:
                strategy_info[strat]["cagrs"].append(row["cagr"])
            if "mdd" in row:
                strategy_info[strat]["mdds"].append(row["mdd"])
            if "trades" in row:
                strategy_info[strat]["trades"].append(row["trades"])

    # Filter by robust_k
    robust = []
    for strat, info in strategy_info.items():
        if len(info["markets"]) >= robust_k:
            row = {
                "strategy_id": strat,
                "count_markets": len(info["markets"]),
                "markets_list": ",".join(info["markets"]),
                "sharpe_mean": sum(info["sharpes"]) / len(info["sharpes"]),
                "sharpe_min": min(info["sharpes"]),
                "sharpe_max": max(info["sharpes"]),
            }
            if info["cagrs"]:
                row["cagr_mean"] = sum(info["cagrs"]) / len(info["cagrs"])
            if info["mdds"]:
                row["mdd_worst"] = min(info["mdds"])
            if info["trades"]:
                row["trades_mean"] = sum(info["trades"]) / len(info["trades"])
            robust.append(row)

    if not robust:
        return pd.DataFrame()

    robust_df = pd.DataFrame(robust)

    # Sort: count_markets desc, sharpe_mean desc, strategy_id asc
    robust_df = robust_df.sort_values(
        ["count_markets", "sharpe_mean", "strategy_id"],
        ascending=[False, False, True]
    )

    return robust_df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize Stage1 results and extract candidates"
    )
    parser.add_argument("--summary-csv", type=Path, required=True,
                        help="Path to stage1_summary.csv")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Output directory for candidate CSVs")
    parser.add_argument("--topn", type=int, default=20,
                        help="Top N per market (default: 20)")
    parser.add_argument("--robust-k", type=int, default=3,
                        help="Minimum markets for robust candidate (default: 3)")
    args = parser.parse_args()

    print("[PURPOSE] Extract robust strategy candidates from Stage1 results.")
    print("-" * 70)

    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load summary data
    print(f"\nLoading summary from: {args.summary_csv}")
    summary_df = load_summary(args.summary_csv)
    print(f"  Loaded {len(summary_df)} valid rows")
    print(f"  Markets: {sorted(summary_df['market'].unique())}")

    # Infer intersection path
    intersection_path = args.summary_csv.parent / "stage1_intersection_summary.csv"

    # 1) Generate Top N by market
    print(f"\n[1/3] Generating Top {args.topn} by market...")
    by_market_df = generate_top20_by_market(summary_df, args.topn)
    by_market_path = args.out_dir / "candidates_top20_by_market.csv"
    by_market_df.to_csv(by_market_path, index=False)
    print(f"  Saved {len(by_market_df)} rows to: {by_market_path}")

    # 2) Generate Top N from intersection
    print(f"\n[2/3] Generating Top {args.topn} from intersection...")
    intersection_df = generate_top20_intersection(summary_df, intersection_path, args.topn)
    intersection_out_path = args.out_dir / "candidates_top20_intersection.csv"
    if intersection_df is not None and len(intersection_df) > 0:
        intersection_df.to_csv(intersection_out_path, index=False)
        print(f"  Saved {len(intersection_df)} rows to: {intersection_out_path}")
    else:
        # Create empty file with header
        pd.DataFrame(columns=["rank", "strategy_id", "sharpe"]).to_csv(
            intersection_out_path, index=False
        )
        print(f"  [WARN] No intersection data, created empty file: {intersection_out_path}")

    # 3) Generate robust candidates
    print(f"\n[3/3] Finding robust candidates (top{args.topn} in {args.robust_k}+ markets)...")
    robust_df = generate_robust_candidates(summary_df, args.topn, args.robust_k)
    robust_path = args.out_dir / "candidates_robust_kof4_top20.csv"
    robust_df.to_csv(robust_path, index=False)
    print(f"  Saved {len(robust_df)} rows to: {robust_path}")

    # Summary
    print("\n" + "=" * 70)
    print("[STATUS] Candidate extraction completed successfully.")
    print("-" * 70)
    print(f"  candidates_top20_by_market.csv    : {len(by_market_df)} rows")
    if intersection_df is not None:
        print(f"  candidates_top20_intersection.csv : {len(intersection_df)} rows")
    else:
        print(f"  candidates_top20_intersection.csv : 0 rows (no data)")
    print(f"  candidates_robust_kof4_top20.csv  : {len(robust_df)} rows")
    print("=" * 70)

    # Print top 5 from each
    print("\n[TOP 5 strategy_id per output]")

    print("\n  By Market (first 5):")
    for _, row in by_market_df.head(5).iterrows():
        print(f"    {row['market']}: {row['strategy_id']} (sharpe={row['sharpe']:.4f})")

    if intersection_df is not None and len(intersection_df) > 0:
        print("\n  Intersection (top 5):")
        for _, row in intersection_df.head(5).iterrows():
            market_str = f"{row['market']}: " if "market" in row else ""
            print(f"    {market_str}{row['strategy_id']} (sharpe={row['sharpe']:.4f})")

    if len(robust_df) > 0:
        print("\n  Robust (top 5):")
        for _, row in robust_df.head(5).iterrows():
            print(f"    {row['strategy_id']} (markets={row['count_markets']}, sharpe_mean={row['sharpe_mean']:.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
