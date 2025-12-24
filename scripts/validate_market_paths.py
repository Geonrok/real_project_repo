#!/usr/bin/env python3
"""
Validate market data paths from markets.yaml configuration.

Usage:
    python scripts/validate_market_paths.py --markets configs/markets.yaml

Exit codes:
    0 - All enabled markets have valid paths with data files
    2 - One or more enabled markets have missing/empty paths
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def load_markets_config(config_path: Path) -> dict:
    """Load and parse markets.yaml configuration."""
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_market(name: str, cfg: dict) -> tuple[bool, dict]:
    """
    Validate a single market configuration.

    Returns:
        (is_valid, details) where details contains validation info
    """
    root = Path(cfg.get("root", ""))
    file_glob = cfg.get("file_glob", "*.csv")
    enabled = cfg.get("enabled", True)

    details = {
        "name": name,
        "enabled": enabled,
        "root": str(root),
        "root_exists": False,
        "file_count": 0,
        "sample_files": [],
        "status": "unknown",
    }

    if not enabled:
        details["status"] = "disabled (config)"
        return True, details

    if not root.exists():
        details["status"] = "INVALID - root path missing"
        return False, details

    details["root_exists"] = True

    # Count matching files
    matching_files = sorted(root.glob(file_glob))
    details["file_count"] = len(matching_files)
    details["sample_files"] = [f.name for f in matching_files[:5]]

    if len(matching_files) == 0:
        details["status"] = "INVALID - no matching files"
        return False, details

    details["status"] = "OK"
    return True, details


def print_validation_report(results: list[dict]) -> None:
    """Print formatted validation report."""
    print("=" * 70)
    print("MARKET PATH VALIDATION REPORT")
    print("=" * 70)

    for r in results:
        print(f"\n[{r['name']}]")
        print(f"  Status:     {r['status']}")
        print(f"  Enabled:    {r['enabled']}")
        print(f"  Root:       {r['root']}")
        print(f"  Root exists: {r['root_exists']}")
        print(f"  File count: {r['file_count']}")
        if r["sample_files"]:
            print("  Sample files:")
            for fname in r["sample_files"]:
                print(f"    - {fname}")

    print("\n" + "=" * 70)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate market data paths from configuration"
    )
    parser.add_argument(
        "--markets",
        type=Path,
        default=Path("configs/markets.yaml"),
        help="Path to markets.yaml config file",
    )
    args = parser.parse_args()

    config = load_markets_config(args.markets)
    markets = config.get("markets", {})

    if not markets:
        print("[ERROR] No markets defined in configuration")
        return 1

    results = []
    all_valid = True

    for name, cfg in markets.items():
        is_valid, details = validate_market(name, cfg)
        results.append(details)
        if not is_valid:
            all_valid = False

    print_validation_report(results)

    # Summary
    valid_count = sum(1 for r in results if r["status"] == "OK")
    disabled_count = sum(1 for r in results if "disabled" in r["status"])
    invalid_count = sum(1 for r in results if "INVALID" in r["status"])

    print(f"Summary: {valid_count} OK, {disabled_count} disabled, {invalid_count} invalid")

    if not all_valid:
        print("\n[WARN] One or more enabled markets have invalid paths")
        return 2

    print("\n[OK] All enabled markets validated successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
