#!/usr/bin/env python3
"""
Stage3 Finalization - Generate metadata and SHA256 manifest.

Usage:
    python scripts/stage3_finalize.py --stage3-dir outputs/stage3_rootcause_v1
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

import pandas as pd


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


def generate_metadata(stage3_dir: Path) -> dict:
    """Generate run metadata."""
    # Count results
    rootcause_csv = stage3_dir / "stage3_clamp_rootcause.csv"

    rootcause_rows = 0
    root_cause_dist = {}

    if rootcause_csv.exists():
        df = pd.read_csv(rootcause_csv)
        rootcause_rows = len(df)
        if len(df) > 0 and "root_cause" in df.columns:
            root_cause_dist = df["root_cause"].value_counts().to_dict()

    return {
        "stage": "stage3_rootcause_v1",
        "purpose": "Clamp root-cause analysis for Stage2 failed strategies",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "command": "scripts\\run_stage3_all.cmd",
        "environment": {
            "python": get_python_version(),
            "platform": "Windows"
        },
        "inputs": {
            "stage2_dir": "outputs/stage2_full_v1",
            "stage1_dir": "outputs/stage1_full_v2",
        },
        "results_summary": {
            "rootcause_rows": rootcause_rows,
            "root_cause_distribution": root_cause_dist,
        }
    }


def generate_manifest(stage3_dir: Path) -> list[tuple[str, str]]:
    """Generate SHA256 manifest for all output files."""
    files = sorted(stage3_dir.glob("*"))
    manifest = []

    for f in files:
        if f.is_file() and f.suffix in [".csv", ".json", ".md"]:
            sha = compute_sha256(f)
            manifest.append((sha, f.name))

    return manifest


def copy_to_snapshots(stage3_dir: Path, metadata: dict):
    """Copy metadata files to docs/snapshots."""
    snapshots_dir = Path("docs/snapshots/stage3_rootcause_v1")
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for fname in ["run_metadata.json", "sha256_manifest.txt"]:
        src = stage3_dir / fname
        if src.exists():
            shutil.copy2(src, snapshots_dir / fname)

    # Create README
    readme_path = snapshots_dir / "README.md"

    # Get summary info
    rootcause_rows = metadata["results_summary"]["rootcause_rows"]
    root_cause_dist = metadata["results_summary"]["root_cause_distribution"]

    dist_str = ", ".join(f"{k}: {v}" for k, v in root_cause_dist.items()) if root_cause_dist else "N/A"

    readme_content = f"""# Stage3 Root-Cause Analysis Snapshot

## Summary

- **Strategies analyzed**: {rootcause_rows}
- **Market**: binance_spot
- **Root cause distribution**: {dist_str}
- **Recommendation**: EXTREME_RETURN_CLAMP_ONLY → WARN (review), DATA_* → FAIL (exclude symbol)

## What This Contains

- `run_metadata.json`: Execution context (git commit, command, environment, results summary)
- `sha256_manifest.txt`: SHA256 hashes of all output CSV/JSON/MD files

## Reproduction

```powershell
cd E:\\repos\\real_project_repo
scripts\\run_stage3_all.cmd

# Compare SHA256
.\\.venv\\Scripts\\python.exe -c "
import hashlib
from pathlib import Path

for f in sorted(Path('outputs/stage3_rootcause_v1').glob('*')):
    if f.is_file() and f.suffix in ['.csv', '.json', '.md']:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f'{{sha}}  {{f.name}}')
"
```

## Manifest Location

- `docs/snapshots/stage3_rootcause_v1/sha256_manifest.txt`
"""
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage3 Finalization")
    parser.add_argument("--stage3-dir", type=Path, default=Path("outputs/stage3_rootcause_v1"),
                        help="Stage3 output directory")
    args = parser.parse_args()

    print("[PURPOSE] Stage3 Finalization", flush=True)
    print("-" * 70, flush=True)

    # Generate metadata
    metadata = generate_metadata(args.stage3_dir)
    metadata_path = args.stage3_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {metadata_path}", flush=True)

    # Generate manifest
    manifest = generate_manifest(args.stage3_dir)
    manifest_path = args.stage3_dir / "sha256_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"# SHA256 Manifest for Stage3_rootcause_v1\n")
        f.write(f"# Generated: {metadata['created_utc']}\n")
        f.write(f"# Git commit: {metadata['git_commit']}\n")
        f.write("#\n")
        for sha, name in manifest:
            f.write(f"{sha}  {name}\n")
    print(f"[SAVED] {manifest_path}", flush=True)

    # Copy to snapshots
    copy_to_snapshots(args.stage3_dir, metadata)
    print(f"[COPIED] docs/snapshots/stage3_rootcause_v1/", flush=True)

    print("\n[SUMMARY]", flush=True)
    print(f"  Files in manifest: {len(manifest)}", flush=True)
    print(f"  Strategies analyzed: {metadata['results_summary']['rootcause_rows']}", flush=True)
    print(f"  Root cause distribution: {metadata['results_summary']['root_cause_distribution']}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
