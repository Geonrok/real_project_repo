#!/usr/bin/env python3
"""
Stage6 Live Gate Analysis.

Applies strict live-readiness criteria beyond data quality:
- Window stress must be exercised (trades > 0)
- Sharpe must be positive
- MDD must be within acceptable bounds

Usage:
    python scripts/stage6_live_gate.py \
        --stage2-dir outputs/stage2_full_v3_ca \
        --out outputs/stage6_live_gate_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# =============================================================================
# LIVE GATE CONFIGURATION (Conservative defaults)
# =============================================================================
MIN_TRADES_PER_WINDOW = 10  # Minimum trades required per window to be "exercised"
REQUIRE_SHARPE_POSITIVE = True  # Sharpe must be > 0 for LIVE_PASS
MAX_MDD = -0.60  # MDD must be >= this value (less negative) for LIVE_PASS

# Research mode relaxes constraints
RESEARCH_MIN_TRADES_PER_WINDOW = 3


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def classify_window_stress(
    window_stress_df: pd.DataFrame,
    min_trades: int,
) -> pd.DataFrame:
    """
    Classify window stress results.

    Returns DataFrame with columns:
    - strategy_id, market, window_name
    - trades
    - window_status: EXERCISED, NOT_EXERCISED
    """
    df = window_stress_df.copy()

    # Classify each window
    df["window_status"] = df["trades"].apply(
        lambda t: "EXERCISED" if t >= min_trades else "NOT_EXERCISED"
    )

    return df[["strategy_id", "market", "window_name", "trades", "window_status"]]


def get_market_window_status(
    window_classified: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate window status per (strategy_id, market).

    Returns DataFrame with:
    - strategy_id, market
    - exercised_windows: count of EXERCISED windows
    - total_windows: total windows
    - market_window_status: EXERCISED, PARTIALLY_EXERCISED, NOT_EXERCISED
    """
    agg = (
        window_classified.groupby(["strategy_id", "market"])
        .agg(
            exercised_windows=("window_status", lambda x: (x == "EXERCISED").sum()),
            total_windows=("window_status", "count"),
        )
        .reset_index()
    )

    def classify_market_windows(row):
        if row["exercised_windows"] == row["total_windows"]:
            return "EXERCISED"
        elif row["exercised_windows"] == 0:
            return "NOT_EXERCISED"
        else:
            return "PARTIALLY_EXERCISED"

    agg["market_window_status"] = agg.apply(classify_market_windows, axis=1)

    return agg


def apply_live_gate(
    sensitivity_df: pd.DataFrame,
    market_window_status: pd.DataFrame,
    require_sharpe_positive: bool = True,
    max_mdd: float = -0.60,
) -> pd.DataFrame:
    """
    Apply live gate rules to sensitivity data at fee_mult=1, slippage_mult=1.

    Returns DataFrame with:
    - strategy_id, market
    - sharpe, mdd, trades
    - sharpe_pass, mdd_pass, window_pass
    - live_status: LIVE_PASS, LIVE_FAIL, LIVE_INCONCLUSIVE
    - fail_reasons: list of reasons if FAIL
    """
    # Filter to baseline (fee_mult=1, slippage_mult=1)
    baseline = sensitivity_df[
        (sensitivity_df["fee_mult"] == 1.0) & (sensitivity_df["slippage_mult"] == 1.0)
    ].copy()

    # Merge with window status
    merged = baseline.merge(
        market_window_status[["strategy_id", "market", "market_window_status"]],
        on=["strategy_id", "market"],
        how="left",
    )

    # Fill missing window status (shouldn't happen but be safe)
    merged["market_window_status"] = merged["market_window_status"].fillna(
        "NOT_EXERCISED"
    )

    # Apply gate rules
    results = []
    for _, row in merged.iterrows():
        fail_reasons = []

        # Sharpe check
        if require_sharpe_positive:
            sharpe_pass = row["sharpe"] > 0
            if not sharpe_pass:
                fail_reasons.append(f"sharpe={row['sharpe']:.3f}<=0")
        else:
            sharpe_pass = True

        # MDD check
        mdd_pass = row["mdd"] >= max_mdd
        if not mdd_pass:
            fail_reasons.append(f"mdd={row['mdd']:.3f}<{max_mdd}")

        # Window check
        window_status = row["market_window_status"]
        if window_status == "EXERCISED":
            window_pass = True
            window_inconclusive = False
        elif window_status == "PARTIALLY_EXERCISED":
            window_pass = False
            window_inconclusive = True
            fail_reasons.append("window_partially_exercised")
        else:  # NOT_EXERCISED
            window_pass = False
            window_inconclusive = True
            fail_reasons.append("window_not_exercised")

        # Final classification
        if window_inconclusive:
            live_status = "LIVE_INCONCLUSIVE"
        elif sharpe_pass and mdd_pass and window_pass:
            live_status = "LIVE_PASS"
        else:
            live_status = "LIVE_FAIL"

        results.append(
            {
                "strategy_id": row["strategy_id"],
                "market": row["market"],
                "sharpe": row["sharpe"],
                "mdd": row["mdd"],
                "trades": row["trades"],
                "sharpe_pass": sharpe_pass,
                "mdd_pass": mdd_pass,
                "window_status": window_status,
                "window_pass": window_pass,
                "live_status": live_status,
                "fail_reasons": "; ".join(fail_reasons) if fail_reasons else "",
            }
        )

    return pd.DataFrame(results)


def generate_report_md(
    live_gate_df: pd.DataFrame,
    config: dict,
    out_path: Path,
) -> None:
    """Generate markdown report."""
    lines = []
    lines.append("# Stage6 Live Gate Report")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- **MIN_TRADES_PER_WINDOW**: {config['min_trades_per_window']}")
    lines.append(f"- **REQUIRE_SHARPE_POSITIVE**: {config['require_sharpe_positive']}")
    lines.append(f"- **MAX_MDD**: {config['max_mdd']}")
    lines.append(f"- **Research Mode**: {config.get('research_mode', False)}")
    lines.append("")

    # Overall summary
    status_counts = live_gate_df["live_status"].value_counts()
    lines.append("## Overall Summary")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in ["LIVE_PASS", "LIVE_FAIL", "LIVE_INCONCLUSIVE"]:
        count = status_counts.get(status, 0)
        lines.append(f"| {status} | {count} |")
    lines.append("")

    # Per-market summary
    lines.append("## Per-Market Summary")
    lines.append("")
    markets = sorted(live_gate_df["market"].unique())
    lines.append("| Market | LIVE_PASS | LIVE_FAIL | LIVE_INCONCLUSIVE | Recommendation |")
    lines.append("|--------|-----------|-----------|-------------------|----------------|")

    market_recommendations = {}
    for m in markets:
        mdata = live_gate_df[live_gate_df["market"] == m]
        pass_count = (mdata["live_status"] == "LIVE_PASS").sum()
        fail_count = (mdata["live_status"] == "LIVE_FAIL").sum()
        inc_count = (mdata["live_status"] == "LIVE_INCONCLUSIVE").sum()

        # Recommendation logic
        if pass_count == 0 and fail_count == len(mdata):
            rec = "EXCLUDE MARKET"
        elif pass_count == 0 and inc_count > 0:
            rec = "NEEDS MORE DATA"
        elif pass_count > 0:
            rec = f"PARTIAL ({pass_count}/{len(mdata)})"
        else:
            rec = "REVIEW NEEDED"

        market_recommendations[m] = rec
        lines.append(f"| {m} | {pass_count} | {fail_count} | {inc_count} | {rec} |")
    lines.append("")

    # Market exclusion warnings
    excluded = [m for m, r in market_recommendations.items() if r == "EXCLUDE MARKET"]
    if excluded:
        lines.append("## Market Exclusion Warnings")
        lines.append("")
        lines.append("The following markets have **0 strategies passing live gate**:")
        lines.append("")
        for m in excluded:
            mdata = live_gate_df[live_gate_df["market"] == m]
            avg_sharpe = mdata["sharpe"].mean()
            avg_mdd = mdata["mdd"].mean()
            lines.append(f"- **{m}**: avg_sharpe={avg_sharpe:.3f}, avg_mdd={avg_mdd:.3f}")
            lines.append(f"  - Recommendation: Exclude from live trading or redesign strategy parameters")
        lines.append("")

    # LIVE_PASS strategies
    live_pass = live_gate_df[live_gate_df["live_status"] == "LIVE_PASS"]
    lines.append("## LIVE_PASS Strategies")
    lines.append("")
    if len(live_pass) > 0:
        lines.append(f"**{len(live_pass)} strategy-market combinations passed all live gates.**")
        lines.append("")
        lines.append("| Strategy ID | Market | Sharpe | MDD | Trades |")
        lines.append("|-------------|--------|--------|-----|--------|")
        for _, row in live_pass.sort_values(["market", "sharpe"], ascending=[True, False]).iterrows():
            lines.append(
                f"| {row['strategy_id']} | {row['market']} | {row['sharpe']:.3f} | {row['mdd']:.3f} | {int(row['trades'])} |"
            )
    else:
        lines.append("**No strategies passed all live gates.**")
        lines.append("")
        lines.append("This indicates:")
        lines.append("- Current strategy parameters may not be suitable for live trading")
        lines.append("- Consider relaxing gate thresholds (research mode)")
        lines.append("- Or redesign strategies with better risk/return characteristics")
    lines.append("")

    # LIVE_FAIL breakdown
    live_fail = live_gate_df[live_gate_df["live_status"] == "LIVE_FAIL"]
    if len(live_fail) > 0:
        lines.append("## LIVE_FAIL Analysis")
        lines.append("")

        # Count failure reasons
        all_reasons = []
        for reasons in live_fail["fail_reasons"]:
            if reasons:
                all_reasons.extend(reasons.split("; "))

        from collections import Counter

        reason_counts = Counter(all_reasons)
        lines.append("| Failure Reason | Count |")
        lines.append("|----------------|-------|")
        for reason, count in reason_counts.most_common():
            lines.append(f"| {reason} | {count} |")
        lines.append("")

    # LIVE_INCONCLUSIVE
    live_inc = live_gate_df[live_gate_df["live_status"] == "LIVE_INCONCLUSIVE"]
    if len(live_inc) > 0:
        lines.append("## LIVE_INCONCLUSIVE Analysis")
        lines.append("")
        lines.append(f"**{len(live_inc)} strategy-market combinations have insufficient window stress data.**")
        lines.append("")
        lines.append("These cannot be classified as PASS or FAIL because window stress testing")
        lines.append("was not exercised (trades=0 in stress windows).")
        lines.append("")
        lines.append("Options:")
        lines.append("- Extend backtest period to include more stress events")
        lines.append("- Use research mode (--research-mode) with relaxed thresholds")
        lines.append("- Accept as research-only candidates")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage6 Live Gate Analysis")
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("outputs/stage2_full_v3_ca"),
        help="Stage2 results directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/stage6_live_gate_v1"),
        help="Output directory",
    )
    parser.add_argument(
        "--research-mode",
        action="store_true",
        help="Use relaxed thresholds for research",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=None,
        help="Override MIN_TRADES_PER_WINDOW",
    )
    parser.add_argument(
        "--max-mdd",
        type=float,
        default=None,
        help="Override MAX_MDD threshold",
    )
    args = parser.parse_args()

    stage2_dir = args.stage2_dir
    out_dir = args.out

    # Check for alternate path if primary doesn't exist
    if not stage2_dir.exists():
        alt_path = Path("docs/snapshots/stage2_full_v3_ca")
        if alt_path.exists():
            print(f"[INFO] Primary path not found, using: {alt_path}")
            stage2_dir = alt_path
        else:
            print(f"[ERROR] Stage2 directory not found: {stage2_dir}")
            return 1

    # Required files
    sensitivity_file = stage2_dir / "stage2_sensitivity.csv"
    window_stress_file = stage2_dir / "stage2_window_stress.csv"

    if not sensitivity_file.exists():
        print(f"[ERROR] Missing: {sensitivity_file}")
        return 1
    if not window_stress_file.exists():
        print(f"[ERROR] Missing: {window_stress_file}")
        return 1

    # Determine thresholds
    if args.research_mode:
        min_trades = args.min_trades or RESEARCH_MIN_TRADES_PER_WINDOW
        max_mdd = args.max_mdd or -0.80  # More lenient in research mode
    else:
        min_trades = args.min_trades or MIN_TRADES_PER_WINDOW
        max_mdd = args.max_mdd or MAX_MDD

    config = {
        "min_trades_per_window": min_trades,
        "require_sharpe_positive": REQUIRE_SHARPE_POSITIVE,
        "max_mdd": max_mdd,
        "research_mode": args.research_mode,
    }

    print("[PURPOSE] Stage6 Live Gate Analysis")
    print("-" * 70)
    print(f"  Stage2 dir: {stage2_dir}")
    print(f"  Output dir: {out_dir}")
    print(f"  MIN_TRADES_PER_WINDOW: {min_trades}")
    print(f"  REQUIRE_SHARPE_POSITIVE: {REQUIRE_SHARPE_POSITIVE}")
    print(f"  MAX_MDD: {max_mdd}")
    print(f"  Research mode: {args.research_mode}")
    print("-" * 70)

    # Load data
    sensitivity_df = pd.read_csv(sensitivity_file)
    window_stress_df = pd.read_csv(window_stress_file)

    print(f"[INFO] Loaded {len(sensitivity_df)} sensitivity rows")
    print(f"[INFO] Loaded {len(window_stress_df)} window stress rows")

    # Classify window stress
    window_classified = classify_window_stress(window_stress_df, min_trades)
    market_window_status = get_market_window_status(window_classified)

    # Apply live gate
    live_gate_df = apply_live_gate(
        sensitivity_df,
        market_window_status,
        require_sharpe_positive=REQUIRE_SHARPE_POSITIVE,
        max_mdd=max_mdd,
    )

    # Sort deterministically
    live_gate_df = live_gate_df.sort_values(
        ["market", "strategy_id"]
    ).reset_index(drop=True)

    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV
    csv_path = out_dir / "stage6_live_gate.csv"
    live_gate_df.to_csv(csv_path, index=False)
    print(f"[SAVED] {csv_path} ({len(live_gate_df)} rows)")

    # Save MD report
    md_path = out_dir / "stage6_live_gate.md"
    generate_report_md(live_gate_df, config, md_path)
    print(f"[SAVED] {md_path}")

    # Save window classification detail
    window_detail_path = out_dir / "stage6_window_detail.csv"
    window_classified.to_csv(window_detail_path, index=False)
    print(f"[SAVED] {window_detail_path}")

    # Save metadata
    metadata = {
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "config": config,
        "source_files": {
            "sensitivity": str(sensitivity_file),
            "window_stress": str(window_stress_file),
        },
        "counts": {
            "total_combinations": len(live_gate_df),
            "live_pass": int((live_gate_df["live_status"] == "LIVE_PASS").sum()),
            "live_fail": int((live_gate_df["live_status"] == "LIVE_FAIL").sum()),
            "live_inconclusive": int(
                (live_gate_df["live_status"] == "LIVE_INCONCLUSIVE").sum()
            ),
        },
    }

    metadata_path = out_dir / "run_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {metadata_path}")

    # Save manifest
    manifest_path = out_dir / "sha256_manifest.txt"
    files_to_hash = [csv_path, md_path, window_detail_path]
    with open(manifest_path, "w", encoding="utf-8") as f:
        for path in sorted(files_to_hash):
            if path.exists():
                hash_val = sha256_file(path)
                f.write(f"{hash_val}  {path.name}\n")
    print(f"[SAVED] {manifest_path}")

    # Summary
    print("\n[SUMMARY]")
    print(f"  Total strategy-market combinations: {len(live_gate_df)}")
    print(f"  LIVE_PASS: {metadata['counts']['live_pass']}")
    print(f"  LIVE_FAIL: {metadata['counts']['live_fail']}")
    print(f"  LIVE_INCONCLUSIVE: {metadata['counts']['live_inconclusive']}")

    # Per-market
    print("\n[PER-MARKET]")
    for m in sorted(live_gate_df["market"].unique()):
        mdata = live_gate_df[live_gate_df["market"] == m]
        pass_c = (mdata["live_status"] == "LIVE_PASS").sum()
        fail_c = (mdata["live_status"] == "LIVE_FAIL").sum()
        inc_c = (mdata["live_status"] == "LIVE_INCONCLUSIVE").sum()
        print(f"  {m}: PASS={pass_c}, FAIL={fail_c}, INCONCLUSIVE={inc_c}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
