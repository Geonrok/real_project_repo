#!/usr/bin/env python3
"""
Inspect sample files from a market to understand schema and data format.

Usage:
    python scripts/inspect_market_sample.py --markets configs/markets.yaml --market upbit_krw
    python scripts/inspect_market_sample.py --markets configs/markets.yaml --all

Outputs:
    - Column names and data types
    - Date range (if date column detected)
    - Row count
    - Sample rows (head/tail)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml


def load_markets_config(config_path: Path) -> dict:
    """Load and parse markets.yaml configuration."""
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def inspect_sample_file(file_path: Path, market_name: str) -> None:
    """Inspect a single CSV file and print schema info."""
    print(f"\n{'='*70}")
    print(f"FILE: {file_path.name}")
    print(f"MARKET: {market_name}")
    print("=" * 70)

    try:
        df = pd.read_csv(file_path, nrows=1000)  # Read sample
    except Exception as e:
        print(f"[ERROR] Failed to read file: {e}")
        return

    print(f"\nRows (sampled): {len(df)}")
    print(f"Columns ({len(df.columns)}): {list(df.columns)}")

    print("\nColumn Types:")
    for col in df.columns:
        dtype = df[col].dtype
        sample_val = df[col].iloc[0] if len(df) > 0 else "N/A"
        print(f"  {col:20} {str(dtype):15} sample: {sample_val}")

    # Detect date columns
    date_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
    if date_cols:
        print("\nDate Range Detection:")
        for col in date_cols:
            try:
                dates = pd.to_datetime(df[col], errors="coerce")
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    print(f"  {col}: {valid_dates.min()} to {valid_dates.max()}")
            except Exception:
                print(f"  {col}: unable to parse as datetime")

    print("\nHead (3 rows):")
    print(df.head(3).to_string(index=False))

    print("\nTail (3 rows):")
    print(df.tail(3).to_string(index=False))


def inspect_market(name: str, cfg: dict) -> None:
    """Inspect sample files from a market."""
    root = Path(cfg.get("root", ""))
    file_glob = cfg.get("file_glob", "*.csv")
    enabled = cfg.get("enabled", True)

    print(f"\n{'#'*70}")
    print(f"# MARKET: {name}")
    print(f"# Enabled: {enabled}")
    print(f"# Root: {root}")
    print("#" * 70)

    if not enabled:
        print("[SKIP] Market is disabled in config")
        return

    if not root.exists():
        print(f"[ERROR] Root path does not exist: {root}")
        return

    matching_files = sorted(root.glob(file_glob))
    if not matching_files:
        print(f"[ERROR] No files matching {file_glob}")
        return

    print(f"Found {len(matching_files)} files matching {file_glob}")

    # Inspect first 2 sample files
    for sample_file in matching_files[:2]:
        inspect_sample_file(sample_file, name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Inspect sample files from market data directories"
    )
    parser.add_argument(
        "--markets",
        type=Path,
        default=Path("configs/markets.yaml"),
        help="Path to markets.yaml config file",
    )
    parser.add_argument(
        "--market",
        type=str,
        help="Specific market to inspect (e.g., upbit_krw)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Inspect all enabled markets",
    )
    args = parser.parse_args()

    if not args.market and not args.all:
        print("[ERROR] Must specify --market NAME or --all")
        return 1

    config = load_markets_config(args.markets)
    markets = config.get("markets", {})

    if args.market:
        if args.market not in markets:
            print(f"[ERROR] Market '{args.market}' not found in config")
            print(f"Available markets: {list(markets.keys())}")
            return 1
        inspect_market(args.market, markets[args.market])
    else:
        for name, cfg in markets.items():
            inspect_market(name, cfg)

    return 0


if __name__ == "__main__":
    sys.exit(main())
