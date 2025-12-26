#!/usr/bin/env python3
"""
Finalize Candidates Pack Generator.

Purpose: Generate final candidate CSVs and snapshots from Stage2 results.

Usage:
    python scripts/finalize_candidates_pack.py \
        --stage2-dir docs/snapshots/stage2_full_v1 \
        --out-dir docs/final \
        --snapshot-dir docs/snapshots/final_candidates_v1
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
    parser = argparse.ArgumentParser(description="Generate final candidate packs")
    parser.add_argument(
        "--stage2-dir",
        type=Path,
        default=Path("docs/snapshots/stage2_full_v1"),
        help="Stage2 results directory",
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
        default=Path("docs/snapshots/final_candidates_v1"),
        help="Snapshot directory for metadata",
    )
    args = parser.parse_args()

    # Validate inputs
    stage2_dir = args.stage2_dir
    out_dir = args.out_dir
    snapshot_dir = args.snapshot_dir

    strict_pass_file = stage2_dir / "stage2_final_pass.csv"
    pass_with_warn_file = stage2_dir / "stage2_final_pass_with_warn.csv"

    if not strict_pass_file.exists():
        print(f"[ERROR] Strict pass file not found: {strict_pass_file}")
        return 1
    if not pass_with_warn_file.exists():
        print(f"[ERROR] Pass with warn file not found: {pass_with_warn_file}")
        return 1

    # Create directories
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Read source CSVs
    strict_df = pd.read_csv(strict_pass_file)
    pass_warn_df = pd.read_csv(pass_with_warn_file)

    # Sort by strategy_id ascending
    strict_df = strict_df.sort_values("strategy_id").reset_index(drop=True)
    pass_warn_df = pass_warn_df.sort_values("strategy_id").reset_index(drop=True)

    # Write final CSVs
    final_strict = out_dir / "final_candidates_v1.csv"
    final_research = out_dir / "research_candidates_v1.csv"

    strict_df.to_csv(final_strict, index=False)
    pass_warn_df.to_csv(final_research, index=False)

    print(f"[INFO] Wrote {len(strict_df)} strict pass strategies to {final_strict}")
    print(f"[INFO] Wrote {len(pass_warn_df)} research candidates to {final_research}")

    # Count by status
    n_pass = len(pass_warn_df[pass_warn_df.get("data_quality_status", "") == "PASS"])
    n_warn = len(pass_warn_df[pass_warn_df.get("data_quality_status", "") == "WARN"])
    print(f"[INFO] Research breakdown: {n_pass} PASS + {n_warn} WARN = {len(pass_warn_df)}")

    # Write FINAL_CANDIDATES.md
    final_md = out_dir / "FINAL_CANDIDATES.md"
    with open(final_md, "w", encoding="utf-8") as f:
        f.write("# Final Candidates\n\n")
        f.write("## Strict Pass (Live-Eligible)\n\n")
        f.write(f"Total: **{len(strict_df)} strategies**\n\n")
        f.write("| Strategy ID |\n")
        f.write("|-------------|\n")
        for _, row in strict_df.iterrows():
            f.write(f"| {row['strategy_id']} |\n")
        f.write("\n")

        f.write("## Research Candidates (Pass + Warn)\n\n")
        f.write(f"Total: **{len(pass_warn_df)} strategies** ({n_pass} PASS + {n_warn} WARN)\n\n")
        f.write("| Strategy ID | Status | Reason |\n")
        f.write("|-------------|--------|--------|\n")
        for _, row in pass_warn_df.iterrows():
            status = row.get("data_quality_status", "")
            reason = row.get("data_quality_reason", "")
            f.write(f"| {row['strategy_id']} | {status} | {reason} |\n")
        f.write("\n")

        f.write("## Policy\n\n")
        f.write("- **PASS**: Live-eligible. These strategies passed all data quality gates.\n")
        f.write("- **WARN**: Research-only. These strategies triggered extreme return clamps but remained viable.\n\n")
        f.write("## Stage4 Evidence\n\n")
        f.write("All 13 WARN strategies trace to a single event:\n\n")
        f.write("- **Symbol**: QUICK\n")
        f.write("- **Date**: 2023-07-21\n")
        f.write("- **Event**: Token redenomination (1 OLD = 1000 NEW)\n")
        f.write("- **Price**: 74.20 -> 0.05967 (-99.92%)\n\n")
        f.write("This was not a market crash but a token migration. The extreme return clamp ")
        f.write("prevented equity from going negative, which is correct model behavior. ")
        f.write("However, in live trading, such events could trigger margin calls or forced liquidation.\n\n")
        f.write("See `docs/snapshots/stage4_event_trace_v1/` for full event trace details.\n")

    print(f"[INFO] Wrote {final_md}")

    # Write run_metadata.json
    metadata = {
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "source_files": {
            "strict_pass": str(strict_pass_file),
            "pass_with_warn": str(pass_with_warn_file),
        },
        "output_files": {
            "final_candidates": str(final_strict),
            "research_candidates": str(final_research),
            "readme": str(final_md),
        },
        "counts": {
            "strict_pass": len(strict_df),
            "research_total": len(pass_warn_df),
            "research_pass": n_pass,
            "research_warn": n_warn,
        },
    }

    metadata_file = snapshot_dir / "run_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[INFO] Wrote {metadata_file}")

    # Write sha256_manifest.txt
    manifest_file = snapshot_dir / "sha256_manifest.txt"
    files_to_hash = [final_strict, final_research, final_md]

    with open(manifest_file, "w", encoding="utf-8") as f:
        for path in sorted(files_to_hash):
            if path.exists():
                hash_val = sha256_file(path)
                f.write(f"{hash_val}  {path.name}\n")
    print(f"[INFO] Wrote {manifest_file}")

    # Write README.md
    readme_file = snapshot_dir / "README.md"
    with open(readme_file, "w", encoding="utf-8") as f:
        f.write("# Final Candidates Snapshot (v1)\n\n")
        f.write(f"Generated: {metadata['timestamp_utc']}\n\n")
        f.write("## Contents\n\n")
        f.write("- `run_metadata.json`: Generation metadata\n")
        f.write("- `sha256_manifest.txt`: SHA256 hashes for verification\n\n")
        f.write("## Summary\n\n")
        f.write(f"- **Strict Pass**: {len(strict_df)} strategies (live-eligible)\n")
        f.write(f"- **Research Candidates**: {len(pass_warn_df)} strategies ({n_pass} PASS + {n_warn} WARN)\n\n")
        f.write("## Source\n\n")
        f.write(f"- Stage2 results from: `{stage2_dir}`\n")
        f.write(f"- Git commit: `{metadata['git_commit'][:8]}...`\n")
    print(f"[INFO] Wrote {readme_file}")

    print("\n[SUCCESS] Final candidate pack generated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
