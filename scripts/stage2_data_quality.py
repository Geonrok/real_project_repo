#!/usr/bin/env python3
"""
Stage2 Data Quality Diagnostics.

Aggregates diagnostic counters (clamp, inf, zero_close) from sensitivity results
and provides per-strategy quality scores.

Usage:
    python scripts/stage2_data_quality.py \
        --stage1-dir outputs/stage1_full_v2 \
        --stage2-dir outputs/stage2_full_v1 \
        --out outputs/stage2_full_v1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def load_sensitivity_results(stage2_dir: Path) -> pd.DataFrame | None:
    """Load sensitivity results which contain diagnostic counters."""
    path = stage2_dir / "stage2_sensitivity.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def aggregate_data_quality(sensitivity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate data quality metrics per market/strategy.

    Uses fee_mult=1.0 (base case) for quality assessment.
    """
    # Filter to base case only
    base_df = sensitivity_df[sensitivity_df["fee_mult"] == 1.0].copy()

    if len(base_df) == 0:
        # Fallback: use all data
        base_df = sensitivity_df.copy()

    # Define diagnostic columns
    diag_cols = [
        "zero_close_count",
        "ret_inf_count",
        "strat_ret_clamp_count",
        "cumret_clip_count",
    ]

    # Fill missing diagnostic columns with 0
    for col in diag_cols:
        if col not in base_df.columns:
            base_df[col] = 0

    # Aggregate per market/strategy
    agg_data = []
    for (market, strategy_id), group in base_df.groupby(["market", "strategy_id"]):
        row = {
            "market": market,
            "strategy_id": strategy_id,
            "zero_close_count": int(group["zero_close_count"].sum()),
            "ret_inf_count": int(group["ret_inf_count"].sum()),
            "strat_ret_clamp_count": int(group["strat_ret_clamp_count"].sum()),
            "cumret_clip_count": int(group["cumret_clip_count"].sum()),
        }

        # Calculate quality score (1.0 = perfect, lower = worse)
        total_issues = (
            row["zero_close_count"] +
            row["ret_inf_count"] +
            row["strat_ret_clamp_count"] +
            row["cumret_clip_count"]
        )
        row["total_issues"] = total_issues
        row["quality_score"] = 1.0 if total_issues == 0 else max(0.0, 1.0 - total_issues / 100)

        agg_data.append(row)

    df = pd.DataFrame(agg_data)

    # Sort deterministically
    df = df.sort_values(
        ["market", "strategy_id"],
        ascending=[True, True]
    ).reset_index(drop=True)

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage2 Data Quality Diagnostics")
    parser.add_argument("--stage1-dir", type=Path, default=Path("outputs/stage1_full_v2"),
                        help="Stage1 output directory")
    parser.add_argument("--stage2-dir", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Stage2 output directory (for sensitivity results)")
    parser.add_argument("--out", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Output directory")
    args = parser.parse_args()

    print("[PURPOSE] Stage2 Data Quality Diagnostics")
    print("-" * 70)
    print(f"  Stage2 dir: {args.stage2_dir}")
    print(f"  Output dir: {args.out}")
    print("-" * 70)

    args.out.mkdir(parents=True, exist_ok=True)

    # Load sensitivity results
    sensitivity_df = load_sensitivity_results(args.stage2_dir)
    if sensitivity_df is None:
        print("[ERROR] Sensitivity results not found. Run stage2_sensitivity.py first.")
        return 1

    print(f"[INFO] Loaded {len(sensitivity_df)} sensitivity rows")

    # Aggregate data quality
    df = aggregate_data_quality(sensitivity_df)

    out_path = args.out / "stage2_data_quality.csv"
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path} ({len(df)} rows)")

    # Summary
    print("\n[SUMMARY]")
    print(f"  Total rows: {len(df)}")
    print(f"  Strategies with issues: {(df['total_issues'] > 0).sum()}")
    print(f"  Perfect quality (score=1.0): {(df['quality_score'] == 1.0).sum()}")

    # Show worst offenders
    if (df["total_issues"] > 0).any():
        print("\n[TOP ISSUES]")
        worst = df[df["total_issues"] > 0].nlargest(5, "total_issues")
        for _, row in worst.iterrows():
            print(f"  {row['market']}/{row['strategy_id']}: {row['total_issues']} issues")

    return 0


if __name__ == "__main__":
    sys.exit(main())
