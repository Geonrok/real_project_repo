#!/usr/bin/env python3
"""
Stage4 Finalization - Generate metadata and SHA256 manifest.

Usage:
    python scripts/stage4_finalize.py --stage4-dir outputs/stage4_event_trace_v1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def get_python_version() -> str:
    """Get Python version string."""
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_metadata(stage4_dir: Path) -> dict:
    """Generate run metadata."""
    # Count results
    trace_csv = stage4_dir / "stage4_event_trace.csv"

    trace_rows = 0
    traced_count = 0
    review_count = 0
    strategies_count = 0
    symbols_count = 0

    if trace_csv.exists():
        import pandas as pd
        df = pd.read_csv(trace_csv)
        trace_rows = len(df)
        if len(df) > 0:
            traced_count = len(df[df["status"] == "TRACED"]) if "status" in df.columns else 0
            review_count = len(df[df["status"] == "REVIEW"]) if "status" in df.columns else 0
            strategies_count = df["strategy_id"].nunique() if "strategy_id" in df.columns else 0
            symbols_count = df["symbol"].nunique() if "symbol" in df.columns else 0

    return {
        "stage": "stage4_event_trace_v1",
        "purpose": "Trace clamp-only WARN events to concrete symbol/date evidence",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "command": "scripts\\run_stage4_trace.cmd",
        "environment": {
            "python": get_python_version(),
            "platform": "Windows"
        },
        "results_summary": {
            "total_events": trace_rows,
            "traced_count": traced_count,
            "review_count": review_count,
            "strategies_count": strategies_count,
            "symbols_count": symbols_count,
        }
    }


def generate_manifest(stage4_dir: Path) -> list[tuple[str, str]]:
    """Generate SHA256 manifest for all output files."""
    files = sorted(stage4_dir.glob("*"))
    manifest = []

    for f in files:
        if f.is_file() and f.suffix in [".csv", ".json", ".md", ".txt"]:
            sha = compute_sha256(f)
            manifest.append((sha, f.name))

    return manifest


def copy_to_snapshots(stage4_dir: Path):
    """Copy metadata files to docs/snapshots."""
    snapshots_dir = Path("docs/snapshots/stage4_event_trace_v1")
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Copy metadata files
    for fname in ["run_metadata.json", "sha256_manifest.txt"]:
        src = stage4_dir / fname
        if src.exists():
            shutil.copy2(src, snapshots_dir / fname)

    # Create README
    readme_path = snapshots_dir / "README.md"
    readme_content = """# Stage4 Event Trace V1 Snapshot

This snapshot provides reproducibility evidence for Stage4 clamp event tracing.

## What This Contains

- `run_metadata.json`: Execution context (git commit, command, environment, results summary)
- `sha256_manifest.txt`: SHA256 hashes of all output CSV/JSON/MD files

## Purpose

Stage4 traces WARN strategies (EXTREME_RETURN_CLAMP_ONLY) from Stage2/Stage3 to:
- Identify the exact symbol and date where each clamp event occurred
- Extract bar-level details: close, prev_close, raw return, strat_ret before/after clip
- Provide concrete evidence for the "research WARN" classification

## Reproduction

```powershell
cd E:\\repos\\real_project_repo
scripts\\run_stage4_trace.cmd

# Compare SHA256
.\\.venv\\Scripts\\python.exe -c "
import hashlib
from pathlib import Path

for f in sorted(Path('outputs/stage4_event_trace_v1').glob('*')):
    if f.is_file() and f.suffix in ['.csv', '.json', '.md']:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f'{sha}  {f.name}')
"
```

## Key Results

See `run_metadata.json` for:
- Total clamp events traced
- TRACED vs REVIEW counts
- Strategies and symbols involved
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage4 Finalization")
    parser.add_argument("--stage4-dir", type=Path, default=Path("outputs/stage4_event_trace_v1"),
                        help="Stage4 output directory")
    args = parser.parse_args()

    print("[PURPOSE] Stage4 Finalization")
    print("-" * 70)

    # Generate metadata
    metadata = generate_metadata(args.stage4_dir)
    metadata_path = args.stage4_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {metadata_path}")

    # Generate manifest
    manifest = generate_manifest(args.stage4_dir)
    manifest_path = args.stage4_dir / "sha256_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"# SHA256 Manifest for Stage4_event_trace_v1\n")
        f.write(f"# Generated: {metadata['created_utc']}\n")
        f.write(f"# Git commit: {metadata['git_commit']}\n")
        f.write("#\n")
        for sha, name in manifest:
            f.write(f"{sha}  {name}\n")
    print(f"[SAVED] {manifest_path}")

    # Copy to snapshots
    copy_to_snapshots(args.stage4_dir)
    print(f"[COPIED] docs/snapshots/stage4_event_trace_v1/")

    print("\n[SUMMARY]")
    print(f"  Files in manifest: {len(manifest)}")
    results = metadata["results_summary"]
    print(f"  Total events: {results['total_events']}")
    print(f"  TRACED: {results['traced_count']}")
    print(f"  REVIEW: {results['review_count']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
