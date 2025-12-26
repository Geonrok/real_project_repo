#!/usr/bin/env python3
"""
Stage2 Gate Report Generator.

Evaluates robust candidates against pass/fail criteria and generates
a summary report with final candidate list.

Usage:
    python scripts/stage2_gate_report.py \
        --stage2-dir outputs/stage2_full_v1 \
        --out outputs/stage2_full_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load CSV if exists, else return None."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def evaluate_sensitivity(sensitivity_df: pd.DataFrame) -> dict:
    """
    Evaluate cost sensitivity.

    PASS if sharpe at 2x cost >= 50% of sharpe at 1x cost.
    """
    if sensitivity_df is None or len(sensitivity_df) == 0:
        return {"passed": [], "failed": [], "skipped": True}

    base = sensitivity_df[sensitivity_df["fee_mult"] == 1.0]
    high = sensitivity_df[sensitivity_df["fee_mult"] == 2.0]

    passed = []
    failed = []

    for strategy_id in sensitivity_df["strategy_id"].unique():
        base_rows = base[base["strategy_id"] == strategy_id]
        high_rows = high[high["strategy_id"] == strategy_id]

        if len(base_rows) == 0 or len(high_rows) == 0:
            continue

        # Check per market
        markets_passed = 0
        markets_total = 0

        for market in base_rows["market"].unique():
            base_sharpe = base_rows[base_rows["market"] == market]["sharpe"].values
            high_sharpe = high_rows[high_rows["market"] == market]["sharpe"].values

            if len(base_sharpe) == 0 or len(high_sharpe) == 0:
                continue

            markets_total += 1
            # Pass if high_sharpe >= 0.5 * base_sharpe (allowing for negative sharpes)
            if base_sharpe[0] <= 0 or high_sharpe[0] >= 0.5 * base_sharpe[0]:
                markets_passed += 1

        if markets_total > 0 and markets_passed >= markets_total / 2:
            passed.append(strategy_id)
        else:
            failed.append({"strategy_id": strategy_id, "reason": "sensitivity_fail"})

    return {"passed": passed, "failed": failed, "skipped": False}


def evaluate_window_stress(stress_df: pd.DataFrame) -> dict:
    """
    Evaluate window stress.

    PASS if sharpe >= 0 in at least 2 of 3 windows.
    """
    if stress_df is None or len(stress_df) == 0:
        return {"passed": [], "failed": [], "skipped": True}

    passed = []
    failed = []

    for strategy_id in stress_df["strategy_id"].unique():
        strat_rows = stress_df[stress_df["strategy_id"] == strategy_id]

        # Count windows with positive sharpe per market
        markets_passed = 0
        markets_total = 0

        for market in strat_rows["market"].unique():
            market_rows = strat_rows[strat_rows["market"] == market]
            positive_windows = (market_rows["sharpe"] >= 0).sum()
            total_windows = len(market_rows)

            markets_total += 1
            if total_windows >= 2 and positive_windows >= 2:
                markets_passed += 1

        if markets_total > 0 and markets_passed >= markets_total / 2:
            passed.append(strategy_id)
        else:
            failed.append({"strategy_id": strategy_id, "reason": "stress_fail"})

    return {"passed": passed, "failed": failed, "skipped": False}


def evaluate_data_quality(quality_df: pd.DataFrame) -> dict:
    """
    Evaluate data quality.

    PASS if total_issues == 0 for all markets.
    """
    if quality_df is None or len(quality_df) == 0:
        return {"passed": [], "failed": [], "skipped": True}

    passed = []
    failed = []

    for strategy_id in quality_df["strategy_id"].unique():
        strat_rows = quality_df[quality_df["strategy_id"] == strategy_id]
        total_issues = strat_rows["total_issues"].sum()

        if total_issues == 0:
            passed.append(strategy_id)
        else:
            failed.append({
                "strategy_id": strategy_id,
                "reason": "quality_fail",
                "issues": int(total_issues)
            })

    return {"passed": passed, "failed": failed, "skipped": False}


def generate_gate_report(stage2_dir: Path, out_dir: Path) -> dict:
    """Generate comprehensive gate report."""

    # Load all results
    sensitivity_df = load_csv_safe(stage2_dir / "stage2_sensitivity.csv")
    stress_df = load_csv_safe(stage2_dir / "stage2_window_stress.csv")
    quality_df = load_csv_safe(stage2_dir / "stage2_data_quality.csv")

    # Evaluate each criterion
    sens_result = evaluate_sensitivity(sensitivity_df)
    stress_result = evaluate_window_stress(stress_df)
    quality_result = evaluate_data_quality(quality_df)

    # Find strategies that pass all criteria
    all_strategies = set()
    if sensitivity_df is not None:
        all_strategies.update(sensitivity_df["strategy_id"].unique())

    sens_passed = set(sens_result["passed"])
    stress_passed = set(stress_result["passed"])
    quality_passed = set(quality_result["passed"])

    # Final candidates must pass all non-skipped criteria
    final_candidates = all_strategies.copy()
    if not sens_result["skipped"]:
        final_candidates &= sens_passed
    if not stress_result["skipped"]:
        final_candidates &= stress_passed
    if not quality_result["skipped"]:
        final_candidates &= quality_passed

    final_candidates = sorted(final_candidates)

    # Build report
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(all_strategies),
        "final_candidates_count": len(final_candidates),
        "final_candidates": final_candidates,
        "criteria": {
            "sensitivity": {
                "skipped": sens_result["skipped"],
                "passed_count": len(sens_result["passed"]),
                "failed_count": len(sens_result["failed"]),
            },
            "window_stress": {
                "skipped": stress_result["skipped"],
                "passed_count": len(stress_result["passed"]),
                "failed_count": len(stress_result["failed"]),
            },
            "data_quality": {
                "skipped": quality_result["skipped"],
                "passed_count": len(quality_result["passed"]),
                "failed_count": len(quality_result["failed"]),
            },
        },
        "failures": {
            "sensitivity": sens_result["failed"][:10],  # Top 10
            "window_stress": stress_result["failed"][:10],
            "data_quality": quality_result["failed"][:10],
        },
    }

    return report


def write_markdown_report(report: dict, out_path: Path):
    """Write human-readable markdown report."""
    lines = [
        "# Stage2 Gate Report",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Summary",
        "",
        f"- Total candidates evaluated: {report['total_candidates']}",
        f"- **Final candidates passed: {report['final_candidates_count']}**",
        "",
        "## Criteria Results",
        "",
        "| Criterion | Passed | Failed | Skipped |",
        "|-----------|--------|--------|---------|",
    ]

    for name, data in report["criteria"].items():
        skipped = "Yes" if data["skipped"] else "No"
        lines.append(f"| {name} | {data['passed_count']} | {data['failed_count']} | {skipped} |")

    lines.extend([
        "",
        "## Final Candidates",
        "",
    ])

    if report["final_candidates"]:
        for i, strat in enumerate(report["final_candidates"], 1):
            lines.append(f"{i}. {strat}")
    else:
        lines.append("*No candidates passed all criteria*")

    lines.extend([
        "",
        "## Top Failures",
        "",
    ])

    for criterion, failures in report["failures"].items():
        if failures:
            lines.append(f"### {criterion}")
            for f in failures[:5]:
                if isinstance(f, dict):
                    lines.append(f"- {f['strategy_id']}: {f.get('reason', 'unknown')}")
                else:
                    lines.append(f"- {f}")
            lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage2 Gate Report Generator")
    parser.add_argument("--stage2-dir", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Stage2 output directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Output directory")
    args = parser.parse_args()

    print("[PURPOSE] Stage2 Gate Report Generation")
    print("-" * 70)

    args.out.mkdir(parents=True, exist_ok=True)

    report = generate_gate_report(args.stage2_dir, args.out)

    # Save JSON
    json_path = args.out / "stage2_gate_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] {json_path}")

    # Save Markdown
    md_path = args.out / "stage2_gate_report.md"
    write_markdown_report(report, md_path)
    print(f"[SAVED] {md_path}")

    # Summary
    print("\n[SUMMARY]")
    print(f"  Total candidates: {report['total_candidates']}")
    print(f"  Final passed: {report['final_candidates_count']}")
    print(f"  Final candidates: {report['final_candidates'][:5]}{'...' if len(report['final_candidates']) > 5 else ''}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
