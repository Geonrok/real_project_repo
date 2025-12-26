#!/usr/bin/env python3
"""
Stage2 Finalization - Generate metadata and SHA256 manifest.

Usage:
    python scripts/stage2_finalize.py --stage2-dir outputs/stage2_full_v1
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


def generate_metadata(stage2_dir: Path) -> dict:
    """Generate run metadata."""
    # Count results
    sensitivity = stage2_dir / "stage2_sensitivity.csv"
    stress = stage2_dir / "stage2_window_stress.csv"
    quality = stage2_dir / "stage2_data_quality.csv"
    gate = stage2_dir / "stage2_gate_report.json"

    sens_rows = sum(1 for _ in open(sensitivity)) - 1 if sensitivity.exists() else 0
    stress_rows = sum(1 for _ in open(stress)) - 1 if stress.exists() else 0
    quality_rows = sum(1 for _ in open(quality)) - 1 if quality.exists() else 0

    final_candidates = 0
    if gate.exists():
        with open(gate) as f:
            gate_data = json.load(f)
            final_candidates = gate_data.get("final_candidates_count", 0)

    return {
        "stage": "stage2_full_v1",
        "purpose": "Robust candidate verification (sensitivity, window stress, data quality)",
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": get_git_commit(),
        "command": "scripts\\run_stage2_all.cmd",
        "environment": {
            "python": get_python_version(),
            "platform": "Windows"
        },
        "results_summary": {
            "sensitivity_rows": sens_rows,
            "stress_rows": stress_rows,
            "quality_rows": quality_rows,
            "final_candidates": final_candidates,
        }
    }


def generate_manifest(stage2_dir: Path) -> list[tuple[str, str]]:
    """Generate SHA256 manifest for all output files."""
    files = sorted(stage2_dir.glob("*"))
    manifest = []

    for f in files:
        if f.is_file() and f.suffix in [".csv", ".json", ".md"]:
            sha = compute_sha256(f)
            manifest.append((sha, f.name))

    return manifest


def copy_to_snapshots(stage2_dir: Path):
    """Copy metadata files to docs/snapshots."""
    snapshots_dir = Path("docs/snapshots/stage2_full_v1")
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    # Copy files
    for fname in ["run_metadata.json", "sha256_manifest.txt"]:
        src = stage2_dir / fname
        if src.exists():
            shutil.copy2(src, snapshots_dir / fname)

    # Create README
    readme_path = snapshots_dir / "README.md"
    readme_content = """# Stage2 Full V1 Snapshot

This snapshot provides reproducibility evidence for Stage2 verification results.

## What This Contains

- `run_metadata.json`: Execution context (git commit, command, environment, results summary)
- `sha256_manifest.txt`: SHA256 hashes of all output CSV/JSON/MD files

## Verification

After running Stage2, verify your local outputs match this snapshot:

```powershell
cd E:\\repos\\real_project_repo
scripts\\run_stage2_all.cmd

# Compare SHA256
..\\.venv\\Scripts\\python.exe -c "
import hashlib
from pathlib import Path

for f in sorted(Path('outputs/stage2_full_v1').glob('*')):
    if f.is_file() and f.suffix in ['.csv', '.json', '.md']:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f'{sha}  {f.name}')
"
```

## Reproduction Command

```powershell
cd E:\\repos\\real_project_repo
scripts\\run_stage2_all.cmd
```
"""
    with open(readme_path, "w") as f:
        f.write(readme_content)


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage2 Finalization")
    parser.add_argument("--stage2-dir", type=Path, default=Path("outputs/stage2_full_v1"),
                        help="Stage2 output directory")
    args = parser.parse_args()

    print("[PURPOSE] Stage2 Finalization")
    print("-" * 70)

    # Generate metadata
    metadata = generate_metadata(args.stage2_dir)
    metadata_path = args.stage2_dir / "run_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[SAVED] {metadata_path}")

    # Generate manifest
    manifest = generate_manifest(args.stage2_dir)
    manifest_path = args.stage2_dir / "sha256_manifest.txt"
    with open(manifest_path, "w") as f:
        f.write(f"# SHA256 Manifest for Stage2_full_v1\n")
        f.write(f"# Generated: {metadata['created_utc']}\n")
        f.write(f"# Git commit: {metadata['git_commit']}\n")
        f.write("#\n")
        for sha, name in manifest:
            f.write(f"{sha}  {name}\n")
    print(f"[SAVED] {manifest_path}")

    # Copy to snapshots
    copy_to_snapshots(args.stage2_dir)
    print(f"[COPIED] docs/snapshots/stage2_full_v1/")

    print("\n[SUMMARY]")
    print(f"  Files in manifest: {len(manifest)}")
    print(f"  Final candidates: {metadata['results_summary']['final_candidates']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
