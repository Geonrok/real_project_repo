#!/usr/bin/env python3
"""
Stage2 Gate Report Generator.

Evaluates robust candidates against pass/fail/warn criteria and generates
a summary report with final candidate lists.

WARN channel (Stage2.1):
- EXTREME_RETURN_CLAMP_ONLY: strat_ret_clamp_count > 0 but no data defects
  (zero_close_count=0, ret_inf_count=0, cumret_clip_count=0)
- Treated as acceptable for research candidate selection; revisit for live gating.

Usage:
    python scripts/stage2_gate_report.py \
        --stage2-dir outputs/stage2_full_v1 \
        --out outputs/stage2_full_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


# Column name mapping for data_quality.csv
DATA_QUALITY_COLUMNS = {
    "zero_close": "zero_close_count",
    "ret_inf": "ret_inf_count",
    "strat_ret_clamp": "strat_ret_clamp_count",
    "cumret_clip": "cumret_clip_count",
}


def load_csv_safe(path: Path) -> pd.DataFrame | None:
    """Load CSV if exists, else return None."""
    if not path.exists():
        return None
    return pd.read_csv(path)


def _safe_int(val) -> int:
    """Convert value to int, treating None/NaN as 0."""
    import numpy as np
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return 0
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0


def classify_data_quality_status(row: pd.Series) -> tuple[str, str]:
    """
    Classify data quality status for a single market×strategy row.

    Returns (status, reason):
    - ("FAIL", "DATA_ZERO_CLOSE") if zero_close_count > 0
    - ("FAIL", "DATA_RET_INF") if ret_inf_count > 0
    - ("FAIL", "EQUITY_FLOOR_CLIP") if cumret_clip_count > 0
    - ("WARN", "EXTREME_RETURN_CLAMP_ONLY") if strat_ret_clamp_count > 0 and no defects
    - ("PASS", "") if all counts are 0
    """
    zero_close = _safe_int(row.get(DATA_QUALITY_COLUMNS["zero_close"], 0))
    ret_inf = _safe_int(row.get(DATA_QUALITY_COLUMNS["ret_inf"], 0))
    clamp = _safe_int(row.get(DATA_QUALITY_COLUMNS["strat_ret_clamp"], 0))
    cumret_clip = _safe_int(row.get(DATA_QUALITY_COLUMNS["cumret_clip"], 0))

    # FAIL conditions (true data defects / equity floor)
    if zero_close > 0:
        return ("FAIL", "DATA_ZERO_CLOSE")
    if ret_inf > 0:
        return ("FAIL", "DATA_RET_INF")
    if cumret_clip > 0:
        return ("FAIL", "EQUITY_FLOOR_CLIP")

    # WARN condition (clamp-only, equity survived)
    if clamp > 0:
        return ("WARN", "EXTREME_RETURN_CLAMP_ONLY")

    return ("PASS", "")


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

    for strategy_id in sorted(sensitivity_df["strategy_id"].unique()):
        base_rows = base[base["strategy_id"] == strategy_id]
        high_rows = high[high["strategy_id"] == strategy_id]

        if len(base_rows) == 0 or len(high_rows) == 0:
            continue

        # Check per market
        markets_passed = 0
        markets_total = 0

        for market in sorted(base_rows["market"].unique()):
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

    # Sort for determinism
    passed = sorted(passed)
    failed = sorted(failed, key=lambda x: x["strategy_id"])

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

    for strategy_id in sorted(stress_df["strategy_id"].unique()):
        strat_rows = stress_df[stress_df["strategy_id"] == strategy_id]

        # Count windows with positive sharpe per market
        markets_passed = 0
        markets_total = 0

        for market in sorted(strat_rows["market"].unique()):
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

    # Sort for determinism
    passed = sorted(passed)
    failed = sorted(failed, key=lambda x: x["strategy_id"])

    return {"passed": passed, "failed": failed, "skipped": False}


def evaluate_data_quality(quality_df: pd.DataFrame) -> dict:
    """
    Evaluate data quality with tri-state PASS/WARN/FAIL.

    - PASS: no issues
    - WARN: EXTREME_RETURN_CLAMP_ONLY (clamp but no data defects)
    - FAIL: DATA_ZERO_CLOSE / DATA_RET_INF / EQUITY_FLOOR_CLIP
    """
    if quality_df is None or len(quality_df) == 0:
        return {"passed": [], "warned": [], "failed": [], "skipped": True}

    # Check required columns exist
    required_cols = list(DATA_QUALITY_COLUMNS.values())
    missing = [c for c in required_cols if c not in quality_df.columns]
    if missing:
        print(f"[ERROR] Missing columns in data_quality.csv: {missing}", flush=True)
        print(f"[INFO] Available columns: {quality_df.columns.tolist()}", flush=True)
        sys.exit(1)

    # Classify each row
    quality_df = quality_df.copy()
    quality_df[["_status", "_reason"]] = quality_df.apply(
        classify_data_quality_status, axis=1, result_type="expand"
    )

    # Aggregate per strategy (worst status across markets)
    # Priority: FAIL > WARN > PASS
    status_priority = {"FAIL": 0, "WARN": 1, "PASS": 2}

    passed = []
    warned = []
    failed = []

    for strategy_id in sorted(quality_df["strategy_id"].unique()):
        strat_rows = quality_df[quality_df["strategy_id"] == strategy_id]
        statuses = strat_rows["_status"].tolist()
        reasons = strat_rows["_reason"].tolist()

        # Find worst status
        worst_idx = min(range(len(statuses)), key=lambda i: status_priority[statuses[i]])
        worst_status = statuses[worst_idx]
        worst_reason = reasons[worst_idx]

        # Get counts for the worst market
        worst_row = strat_rows.iloc[worst_idx]
        counts = {
            "zero_close_count": _safe_int(worst_row.get(DATA_QUALITY_COLUMNS["zero_close"], 0)),
            "ret_inf_count": _safe_int(worst_row.get(DATA_QUALITY_COLUMNS["ret_inf"], 0)),
            "strat_ret_clamp_count": _safe_int(worst_row.get(DATA_QUALITY_COLUMNS["strat_ret_clamp"], 0)),
            "cumret_clip_count": _safe_int(worst_row.get(DATA_QUALITY_COLUMNS["cumret_clip"], 0)),
        }

        if worst_status == "PASS":
            passed.append(strategy_id)
        elif worst_status == "WARN":
            warned.append({
                "strategy_id": strategy_id,
                "market": worst_row["market"],
                "reason": worst_reason,
                "counts": counts,
            })
        else:  # FAIL
            failed.append({
                "strategy_id": strategy_id,
                "market": worst_row["market"],
                "reason": worst_reason,
                "counts": counts,
            })

    # Sort for determinism
    passed = sorted(passed)
    warned = sorted(warned, key=lambda x: (x["market"], x["strategy_id"]))
    failed = sorted(failed, key=lambda x: (x["market"], x["strategy_id"]))

    return {"passed": passed, "warned": warned, "failed": failed, "skipped": False}


def generate_final_candidates_csv(
    all_strategies: set,
    sens_passed: set,
    stress_passed: set,
    quality_result: dict,
    sens_skipped: bool,
    stress_skipped: bool,
    quality_skipped: bool,
    out_dir: Path,
) -> tuple[int, int]:
    """
    Generate two final candidate CSVs:
    1. stage2_final_pass.csv - strict PASS only
    2. stage2_final_pass_with_warn.csv - PASS + WARN

    Returns (strict_count, with_warn_count).
    """
    quality_passed = set(quality_result.get("passed", []))
    quality_warned = {w["strategy_id"]: w for w in quality_result.get("warned", [])}
    quality_warned_set = set(quality_warned.keys())

    # Strict PASS: must pass all criteria
    strict_pass = all_strategies.copy()
    if not sens_skipped:
        strict_pass &= sens_passed
    if not stress_skipped:
        strict_pass &= stress_passed
    if not quality_skipped:
        strict_pass &= quality_passed

    # PASS + WARN: can have WARN in data_quality
    pass_with_warn = all_strategies.copy()
    if not sens_skipped:
        pass_with_warn &= sens_passed
    if not stress_skipped:
        pass_with_warn &= stress_passed
    if not quality_skipped:
        pass_with_warn &= (quality_passed | quality_warned_set)

    # Build DataFrames
    strict_rows = [{"strategy_id": s} for s in sorted(strict_pass)]
    strict_df = pd.DataFrame(strict_rows)

    warn_rows = []
    for s in sorted(pass_with_warn):
        if s in quality_warned:
            warn_rows.append({
                "strategy_id": s,
                "data_quality_status": "WARN",
                "data_quality_reason": quality_warned[s]["reason"],
            })
        else:
            warn_rows.append({
                "strategy_id": s,
                "data_quality_status": "PASS",
                "data_quality_reason": "",
            })
    warn_df = pd.DataFrame(warn_rows)

    # Sort for determinism (strategy_id asc)
    if len(strict_df) > 0:
        strict_df = strict_df.sort_values("strategy_id").reset_index(drop=True)
    if len(warn_df) > 0:
        warn_df = warn_df.sort_values("strategy_id").reset_index(drop=True)

    # Save
    strict_path = out_dir / "stage2_final_pass.csv"
    warn_path = out_dir / "stage2_final_pass_with_warn.csv"

    strict_df.to_csv(strict_path, index=False)
    warn_df.to_csv(warn_path, index=False, float_format="%.8g")

    return len(strict_df), len(warn_df)


def generate_gate_report(stage2_dir: Path, out_dir: Path) -> dict:
    """Generate comprehensive gate report with WARN channel."""

    # Load all results
    sensitivity_df = load_csv_safe(stage2_dir / "stage2_sensitivity.csv")
    stress_df = load_csv_safe(stage2_dir / "stage2_window_stress.csv")
    quality_df = load_csv_safe(stage2_dir / "stage2_data_quality.csv")

    # Evaluate each criterion
    sens_result = evaluate_sensitivity(sensitivity_df)
    stress_result = evaluate_window_stress(stress_df)
    quality_result = evaluate_data_quality(quality_df)

    # Get all strategies
    all_strategies = set()
    if sensitivity_df is not None:
        all_strategies.update(sensitivity_df["strategy_id"].unique())

    sens_passed = set(sens_result["passed"])
    stress_passed = set(stress_result["passed"])
    quality_passed = set(quality_result["passed"])
    quality_warned_set = set(w["strategy_id"] for w in quality_result.get("warned", []))

    # Generate final candidate CSVs
    strict_count, warn_count = generate_final_candidates_csv(
        all_strategies=all_strategies,
        sens_passed=sens_passed,
        stress_passed=stress_passed,
        quality_result=quality_result,
        sens_skipped=sens_result["skipped"],
        stress_skipped=stress_result["skipped"],
        quality_skipped=quality_result["skipped"],
        out_dir=out_dir,
    )

    # Final candidates (strict PASS)
    final_candidates = all_strategies.copy()
    if not sens_result["skipped"]:
        final_candidates &= sens_passed
    if not stress_result["skipped"]:
        final_candidates &= stress_passed
    if not quality_result["skipped"]:
        final_candidates &= quality_passed

    final_candidates = sorted(final_candidates)

    # Final candidates with warn
    final_with_warn = all_strategies.copy()
    if not sens_result["skipped"]:
        final_with_warn &= sens_passed
    if not stress_result["skipped"]:
        final_with_warn &= stress_passed
    if not quality_result["skipped"]:
        final_with_warn &= (quality_passed | quality_warned_set)

    final_with_warn = sorted(final_with_warn)

    # Build report
    report = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "total_candidates": len(all_strategies),
        "final_candidates_count": len(final_candidates),
        "final_candidates": final_candidates,
        "final_with_warn_count": len(final_with_warn),
        "final_with_warn": final_with_warn,
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
                "warned_count": len(quality_result.get("warned", [])),
                "failed_count": len(quality_result["failed"]),
            },
        },
        "warnings": {
            "data_quality": quality_result.get("warned", []),
        },
        "failures": {
            "sensitivity": sens_result["failed"][:10],
            "window_stress": stress_result["failed"][:10],
            "data_quality": quality_result["failed"][:10],
        },
    }

    return report


def write_markdown_report(report: dict, out_path: Path):
    """Write human-readable markdown report with WARN section."""
    lines = [
        "# Stage2 Gate Report",
        "",
        f"Generated: {report['generated_utc']}",
        "",
        "## Summary",
        "",
        f"- Total candidates evaluated: {report['total_candidates']}",
        f"- **Final candidates (strict PASS): {report['final_candidates_count']}**",
        f"- **Final candidates (PASS + WARN): {report['final_with_warn_count']}**",
        "",
        "## Criteria Results",
        "",
        "| Criterion | Passed | Warned | Failed | Skipped |",
        "|-----------|--------|--------|--------|---------|",
    ]

    for name, data in report["criteria"].items():
        skipped = "Yes" if data["skipped"] else "No"
        warned = data.get("warned_count", "-")
        lines.append(f"| {name} | {data['passed_count']} | {warned} | {data['failed_count']} | {skipped} |")

    lines.extend([
        "",
        "## Final Candidates (Strict PASS)",
        "",
    ])

    if report["final_candidates"]:
        for i, strat in enumerate(report["final_candidates"], 1):
            lines.append(f"{i}. {strat}")
    else:
        lines.append("*No candidates passed all criteria strictly*")

    lines.extend([
        "",
        "## Final Candidates (PASS + WARN)",
        "",
    ])

    if report["final_with_warn"]:
        for i, strat in enumerate(report["final_with_warn"], 1):
            marker = " ⚠️" if strat not in report["final_candidates"] else ""
            lines.append(f"{i}. {strat}{marker}")
    else:
        lines.append("*No candidates passed*")

    # WARN section
    warnings = report.get("warnings", {})
    if warnings.get("data_quality"):
        lines.extend([
            "",
            "## Warnings (Data Quality)",
            "",
            "| Strategy | Market | Reason | Clamp Count |",
            "|----------|--------|--------|-------------|",
        ])
        for w in warnings["data_quality"]:
            clamp = w.get("counts", {}).get("strat_ret_clamp_count", 0)
            lines.append(f"| {w['strategy_id']} | {w['market']} | {w['reason']} | {clamp} |")

        lines.extend([
            "",
            "### WARN Policy Note",
            "",
            "- **EXTREME_RETURN_CLAMP_ONLY**: Strategies experienced extreme single-bar returns",
            "  that got clipped to -1, but equity curve remained positive (no `cumret_clip`).",
            "- Treated as WARN (not FAIL) for research candidate selection based on Stage3 analysis.",
            "- **Revisit this policy for live trading gates.**",
            "",
        ])

    # Failures section
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
                    reason = f.get("reason", "unknown")
                    lines.append(f"- {f['strategy_id']}: {reason}")
                else:
                    lines.append(f"- {f}")
            lines.append("")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage2 Gate Report Generator")
    parser.add_argument("--stage2-dir", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Stage2 output directory")
    parser.add_argument("--out", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Output directory")
    args = parser.parse_args()

    print("[PURPOSE] Stage2 Gate Report Generation (with WARN channel)", flush=True)
    print("-" * 70, flush=True)

    args.out.mkdir(parents=True, exist_ok=True)

    report = generate_gate_report(args.stage2_dir, args.out)

    # Save JSON
    json_path = args.out / "stage2_gate_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[SAVED] {json_path}", flush=True)

    # Save Markdown
    md_path = args.out / "stage2_gate_report.md"
    write_markdown_report(report, md_path)
    print(f"[SAVED] {md_path}", flush=True)

    # Summary
    print("\n[SUMMARY]", flush=True)
    print(f"  Total candidates: {report['total_candidates']}", flush=True)
    print(f"  Final (strict PASS): {report['final_candidates_count']}", flush=True)
    print(f"  Final (PASS + WARN): {report['final_with_warn_count']}", flush=True)
    print(f"  Data quality: PASS={report['criteria']['data_quality']['passed_count']}, "
          f"WARN={report['criteria']['data_quality']['warned_count']}, "
          f"FAIL={report['criteria']['data_quality']['failed_count']}", flush=True)

    # List files created
    print("\n[FILES]", flush=True)
    print(f"  {args.out / 'stage2_gate_report.json'}", flush=True)
    print(f"  {args.out / 'stage2_gate_report.md'}", flush=True)
    print(f"  {args.out / 'stage2_final_pass.csv'}", flush=True)
    print(f"  {args.out / 'stage2_final_pass_with_warn.csv'}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
