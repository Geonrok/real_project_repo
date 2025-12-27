#!/usr/bin/env python3
"""
Stage6 Finalize - Generate v4 final candidate packs.

Creates final artifacts in docs/final/ and docs/snapshots/ based on Stage6 live gate results.
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage6 Finalize")
    parser.add_argument(
        "--stage6-dir",
        type=Path,
        default=Path("outputs/stage6_live_gate_v1"),
        help="Stage6 results directory",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/final"),
        help="Output directory for final CSVs",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("docs/snapshots/final_candidates_v4_live_gate"),
        help="Snapshot directory for metadata",
    )
    args = parser.parse_args()

    stage6_dir = args.stage6_dir
    out_dir = args.out_dir
    snapshot_dir = args.snapshot_dir

    # Check input
    live_gate_file = stage6_dir / "stage6_live_gate.csv"
    if not live_gate_file.exists():
        print(f"[ERROR] Live gate file not found: {live_gate_file}")
        return 1

    # Load data
    live_gate_df = pd.read_csv(live_gate_file)

    # Create directories
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Extract LIVE_PASS only
    live_pass = live_gate_df[live_gate_df["live_status"] == "LIVE_PASS"].copy()

    # Get unique strategies that passed in at least one market
    if len(live_pass) > 0:
        pass_strategies = live_pass["strategy_id"].unique().tolist()
    else:
        pass_strategies = []

    # Counts by status
    status_counts = live_gate_df["live_status"].value_counts().to_dict()
    n_pass = status_counts.get("LIVE_PASS", 0)
    n_fail = status_counts.get("LIVE_FAIL", 0)
    n_inc = status_counts.get("LIVE_INCONCLUSIVE", 0)

    # Per-market summary
    market_summary = (
        live_gate_df.groupby("market")["live_status"]
        .value_counts()
        .unstack(fill_value=0)
        .to_dict("index")
    )

    # Write final candidates CSV (LIVE_PASS only)
    final_csv = out_dir / "final_candidates_v4_live_gate.csv"
    if len(live_pass) > 0:
        live_pass.to_csv(final_csv, index=False)
        print(f"[INFO] Wrote {len(live_pass)} LIVE_PASS rows to {final_csv}")
    else:
        # Write empty file with header
        pd.DataFrame(columns=live_gate_df.columns).to_csv(final_csv, index=False)
        print(f"[INFO] Wrote 0 LIVE_PASS rows to {final_csv} (empty)")

    # Write markdown report
    final_md = out_dir / "final_candidates_v4_live_gate.md"
    lines = []
    lines.append("# Final Candidates v4 (Live Gate Applied)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("This version applies strict live-readiness criteria:")
    lines.append("")
    lines.append("- **Sharpe > 0** required")
    lines.append("- **MDD >= -0.60** required")
    lines.append("- **Window stress exercised** (trades >= 10 per window)")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| LIVE_PASS | {n_pass} |")
    lines.append(f"| LIVE_FAIL | {n_fail} |")
    lines.append(f"| LIVE_INCONCLUSIVE | {n_inc} |")
    lines.append("")

    lines.append("## Per-Market Results")
    lines.append("")
    lines.append("| Market | LIVE_PASS | LIVE_FAIL | LIVE_INCONCLUSIVE |")
    lines.append("|--------|-----------|-----------|-------------------|")
    for m in sorted(market_summary.keys()):
        s = market_summary[m]
        lines.append(
            f"| {m} | {s.get('LIVE_PASS', 0)} | {s.get('LIVE_FAIL', 0)} | {s.get('LIVE_INCONCLUSIVE', 0)} |"
        )
    lines.append("")

    if len(live_pass) > 0:
        lines.append("## LIVE_PASS Strategies")
        lines.append("")
        lines.append("| Strategy ID | Market | Sharpe | MDD | Trades |")
        lines.append("|-------------|--------|--------|-----|--------|")
        for _, row in live_pass.sort_values(["market", "sharpe"], ascending=[True, False]).iterrows():
            lines.append(
                f"| {row['strategy_id']} | {row['market']} | {row['sharpe']:.3f} | {row['mdd']:.3f} | {int(row['trades'])} |"
            )
        lines.append("")
    else:
        lines.append("## Result")
        lines.append("")
        lines.append("**No strategies passed all live gate criteria.**")
        lines.append("")
        lines.append("### Key Issues")
        lines.append("")
        lines.append("1. **Window Stress Not Exercised**: 3/4 markets have trades=0 in stress windows")
        lines.append("2. **binance_spot**: All 18 strategies fail due to negative Sharpe and deep MDD")
        lines.append("")
        lines.append("### Recommendations")
        lines.append("")
        lines.append("1. Extend backtest data to include more stress periods")
        lines.append("2. Review strategy parameters for binance_spot market")
        lines.append("3. Consider market-specific strategies rather than universal application")
        lines.append("4. Reduce position sizing to limit MDD")
        lines.append("")

    lines.append("## Comparison with v3_ca")
    lines.append("")
    lines.append("| Version | Criteria | PASS Count | Note |")
    lines.append("|---------|----------|------------|------|")
    lines.append("| v3_ca | Data Quality Only | 18 | All strategies have clean data |")
    lines.append(f"| v4 | Live Gate (Sharpe/MDD/Window) | {n_pass} | Strict live-readiness |")
    lines.append("")
    lines.append("**Conclusion**: v3_ca 'PASS' means data is clean, not that strategies are profitable or safe.")
    lines.append("v4 applies real trading constraints and finds no strategies currently suitable for live trading.")
    lines.append("")

    with open(final_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[INFO] Wrote {final_md}")

    # Write metadata
    metadata = {
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source_files": {
            "stage6_live_gate": str(live_gate_file),
        },
        "counts": {
            "live_pass": n_pass,
            "live_fail": n_fail,
            "live_inconclusive": n_inc,
            "unique_pass_strategies": len(pass_strategies),
        },
        "market_summary": market_summary,
    }

    metadata_file = snapshot_dir / "run_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote {metadata_file}")

    # Write manifest
    manifest_file = snapshot_dir / "sha256_manifest.txt"
    files_to_hash = [final_csv, final_md]
    with open(manifest_file, "w", encoding="utf-8") as f:
        for path in sorted(files_to_hash):
            if path.exists():
                hash_val = sha256_file(path)
                f.write(f"{hash_val}  {path.name}\n")
    print(f"[INFO] Wrote {manifest_file}")

    # Write README
    readme_file = snapshot_dir / "README.md"
    readme_lines = [
        "# Final Candidates v4 (Live Gate)",
        "",
        f"Generated: {metadata['timestamp_utc']}",
        "",
        "## Summary",
        "",
        f"- **LIVE_PASS**: {n_pass}",
        f"- **LIVE_FAIL**: {n_fail}",
        f"- **LIVE_INCONCLUSIVE**: {n_inc}",
        "",
        "## Criteria",
        "",
        "- Sharpe > 0",
        "- MDD >= -0.60",
        "- Window stress exercised (trades >= 10 per window)",
        "",
        "## Files",
        "",
        "- `run_metadata.json`: Generation metadata",
        "- `sha256_manifest.txt`: SHA256 hashes",
        "",
        "## Conclusion",
        "",
    ]
    if n_pass > 0:
        readme_lines.append(f"{n_pass} strategy-market combinations are live-eligible.")
    else:
        readme_lines.append("No strategies currently meet live-trading criteria.")
        readme_lines.append("Review `docs/final/final_candidates_v4_live_gate.md` for recommendations.")

    with open(readme_file, "w", encoding="utf-8") as f:
        f.write("\n".join(readme_lines))
    print(f"[INFO] Wrote {readme_file}")

    print(f"\n[SUCCESS] v4 artifacts generated")
    print(f"  LIVE_PASS: {n_pass}")
    print(f"  LIVE_FAIL: {n_fail}")
    print(f"  LIVE_INCONCLUSIVE: {n_inc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
