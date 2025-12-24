#!/usr/bin/env python3
"""
Normalize multi-market OHLCV data to a common schema.

This is a scaffolding file - full implementation is TODO.

Usage:
    python scripts/normalize_data.py --markets configs/markets.yaml --output data/normalized/

Target Schema (from markets.yaml normalized_schema):
    - date (YYYY-MM-DD)
    - symbol (extracted from filename)
    - open, high, low, close, volume

Normalization Steps:
    1. Load raw CSV files from each market
    2. Extract symbol from filename (e.g., "BTC.csv" -> "BTC")
    3. Parse date column to consistent format
    4. Validate OHLCV columns exist
    5. Handle timezone normalization (if needed)
    6. Output to common format
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


def normalize_market(name: str, cfg: dict, output_dir: Path) -> int:
    """
    Normalize all files from a single market.

    Returns:
        Number of files processed
    """
    root = Path(cfg.get("root", ""))
    file_glob = cfg.get("file_glob", "*.csv")
    enabled = cfg.get("enabled", True)

    if not enabled:
        print(f"[{name}] Skipped (disabled)")
        return 0

    if not root.exists():
        print(f"[{name}] ERROR: Root path missing: {root}")
        return 0

    matching_files = sorted(root.glob(file_glob))
    if not matching_files:
        print(f"[{name}] ERROR: No matching files")
        return 0

    market_output = output_dir / name
    market_output.mkdir(parents=True, exist_ok=True)

    processed = 0
    for fpath in matching_files:
        try:
            symbol = fpath.stem  # filename without extension
            df = normalize_file(fpath, symbol, cfg)
            if df is not None:
                out_path = market_output / f"{symbol}.csv"
                df.to_csv(out_path, index=False)
                processed += 1
        except Exception as e:
            print(f"[{name}] WARN: Failed to process {fpath.name}: {e}")

    print(f"[{name}] Processed {processed}/{len(matching_files)} files")
    return processed


def normalize_file(fpath: Path, symbol: str, cfg: dict) -> pd.DataFrame | None:
    """
    Normalize a single CSV file to common schema.

    TODO: Implement full normalization logic
    """
    # Basic implementation - read and add symbol column
    df = pd.read_csv(fpath)

    # Add symbol column
    df["symbol"] = symbol

    # Ensure required columns exist
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  WARN: {fpath.name} missing columns: {missing}")
        return None

    # Normalize date format
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # Reorder columns to match schema
    output_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
    df = df[output_cols]

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Normalize multi-market OHLCV data to common schema"
    )
    parser.add_argument(
        "--markets",
        type=Path,
        default=Path("configs/markets.yaml"),
        help="Path to markets.yaml config file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/normalized"),
        help="Output directory for normalized data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing files",
    )
    args = parser.parse_args()

    config = load_markets_config(args.markets)
    markets = config.get("markets", {})

    if not markets:
        print("[ERROR] No markets defined in configuration")
        return 1

    if args.dry_run:
        print("[DRY-RUN] Would normalize the following markets:")
        for name, cfg in markets.items():
            enabled = cfg.get("enabled", True)
            root = cfg.get("root", "N/A")
            status = "enabled" if enabled else "disabled"
            print(f"  {name}: {status} ({root})")
        return 0

    print(f"Output directory: {args.output}")
    args.output.mkdir(parents=True, exist_ok=True)

    total_processed = 0
    for name, cfg in markets.items():
        total_processed += normalize_market(name, cfg, args.output)

    print(f"\n[DONE] Total files processed: {total_processed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
