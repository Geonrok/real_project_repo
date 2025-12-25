#!/usr/bin/env python3
"""
Normalize multi-market OHLCV data to a common schema.

Usage:
    python scripts/normalize_data.py --markets configs/markets.yaml --out-dir outputs/normalized_1d

Features:
    - Column auto-mapping (case-insensitive, aliases supported)
    - Date normalization to YYYY-MM-DD format
    - Duplicate date removal (keeps last)
    - Diagnostics output for failures/warnings
    - Parquet output (CSV fallback if pyarrow unavailable)

Output Schema:
    date, symbol, open, high, low, close, volume
"""
from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Column name mappings (lowercase -> canonical)
COLUMN_ALIASES: dict[str, list[str]] = {
    "date": ["date", "datetime", "timestamp", "time", "dt", "trade_date"],
    "open": ["open", "open_price", "o", "opening"],
    "high": ["high", "high_price", "h", "highest"],
    "low": ["low", "low_price", "l", "lowest"],
    "close": ["close", "close_price", "c", "closing", "last"],
    "volume": ["volume", "vol", "v", "qty", "quantity", "trade_volume"],
}

# Try to import pyarrow for parquet support
try:
    import pyarrow  # noqa: F401
    PARQUET_AVAILABLE = True
except ImportError:
    PARQUET_AVAILABLE = False


@dataclass
class DiagnosticEntry:
    """Single diagnostic record for a file processing result."""
    market: str
    symbol: str
    file_path: str
    status: str  # "success", "warning", "error"
    message: str
    rows_in: int = 0
    rows_out: int = 0
    date_min: str = ""
    date_max: str = ""


@dataclass
class MarketStats:
    """Statistics for a single market."""
    name: str
    total_files: int = 0
    success_count: int = 0
    warning_count: int = 0
    error_count: int = 0
    has_btc: bool = False
    has_eth: bool = False
    date_min: str = ""
    date_max: str = ""


@dataclass
class NormalizationContext:
    """Context object holding state during normalization."""
    output_dir: Path
    diagnostics: list[DiagnosticEntry] = field(default_factory=list)
    market_stats: dict[str, MarketStats] = field(default_factory=dict)
    use_parquet: bool = True


def load_markets_config(config_path: Path) -> dict:
    """Load and parse markets.yaml configuration."""
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        sys.exit(1)

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def find_column(df_columns: list[str], canonical: str) -> str | None:
    """
    Find a column in the dataframe that matches the canonical name.
    Returns the actual column name or None if not found.
    """
    aliases = COLUMN_ALIASES.get(canonical, [canonical])
    df_cols_lower = {c.lower(): c for c in df_columns}

    for alias in aliases:
        if alias.lower() in df_cols_lower:
            return df_cols_lower[alias.lower()]
    return None


def map_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Map dataframe columns to canonical schema.
    Returns (mapped_df, warnings).
    """
    warnings = []
    column_mapping = {}
    required = ["date", "open", "high", "low", "close", "volume"]

    for canonical in required:
        actual = find_column(list(df.columns), canonical)
        if actual:
            column_mapping[actual] = canonical
        else:
            warnings.append(f"Missing column: {canonical}")

    if warnings:
        return df, warnings

    # Rename columns to canonical names
    df = df.rename(columns=column_mapping)
    return df, warnings


def normalize_date(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Normalize date column to YYYY-MM-DD format.
    Returns (normalized_df, warnings).
    """
    warnings = []

    if "date" not in df.columns:
        return df, ["No date column found"]

    try:
        # Parse dates with flexible format detection
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

        # Check for unparseable dates
        null_count = df["date"].isna().sum()
        if null_count > 0:
            warnings.append(f"{null_count} unparseable dates dropped")
            df = df.dropna(subset=["date"])

        # Convert to YYYY-MM-DD string format
        df["date"] = df["date"].dt.strftime("%Y-%m-%d")

        # Sort by date and remove duplicates (keep last)
        df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")

    except Exception as e:
        warnings.append(f"Date normalization error: {e}")

    return df, warnings


def normalize_file(
    fpath: Path,
    symbol: str,
    market_name: str,
    ctx: NormalizationContext
) -> pd.DataFrame | None:
    """
    Normalize a single CSV file to common schema.
    Returns normalized DataFrame or None on failure.
    """
    diag = DiagnosticEntry(
        market=market_name,
        symbol=symbol,
        file_path=str(fpath),
        status="success",
        message="OK"
    )

    try:
        # Read CSV
        df = pd.read_csv(fpath)
        diag.rows_in = len(df)

        if df.empty:
            diag.status = "warning"
            diag.message = "Empty file"
            ctx.diagnostics.append(diag)
            return None

        # Map columns
        df, col_warnings = map_columns(df)
        if col_warnings:
            diag.status = "error"
            diag.message = "; ".join(col_warnings)
            ctx.diagnostics.append(diag)
            return None

        # Normalize dates
        df, date_warnings = normalize_date(df)
        if date_warnings:
            if "error" in " ".join(date_warnings).lower():
                diag.status = "error"
            else:
                diag.status = "warning"
            diag.message = "; ".join(date_warnings)

        if df.empty:
            diag.status = "error"
            diag.message = "No valid rows after date normalization"
            ctx.diagnostics.append(diag)
            return None

        # Add symbol column
        df["symbol"] = symbol

        # Ensure numeric columns are numeric
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Drop rows with NaN in critical columns
        df = df.dropna(subset=["open", "high", "low", "close"])

        # Fill volume NaN with 0
        df["volume"] = df["volume"].fillna(0)

        # Reorder columns
        output_cols = ["date", "symbol", "open", "high", "low", "close", "volume"]
        df = df[output_cols]

        # Update diagnostics
        diag.rows_out = len(df)
        if len(df) > 0:
            diag.date_min = df["date"].min()
            diag.date_max = df["date"].max()

        ctx.diagnostics.append(diag)
        return df

    except Exception as e:
        diag.status = "error"
        diag.message = str(e)
        ctx.diagnostics.append(diag)
        return None


def normalize_market(name: str, cfg: dict, ctx: NormalizationContext) -> MarketStats:
    """
    Normalize all files from a single market.
    Returns market statistics.
    """
    stats = MarketStats(name=name)
    ctx.market_stats[name] = stats

    root = Path(cfg.get("root", ""))
    file_glob = cfg.get("file_glob", "*.csv")
    enabled = cfg.get("enabled", True)

    if not enabled:
        print(f"[{name}] Skipped (disabled)")
        return stats

    if not root.exists():
        print(f"[{name}] ERROR: Root path missing: {root}")
        return stats

    matching_files = sorted(root.glob(file_glob))
    stats.total_files = len(matching_files)

    if not matching_files:
        print(f"[{name}] ERROR: No matching files")
        return stats

    market_output = ctx.output_dir / name
    market_output.mkdir(parents=True, exist_ok=True)

    all_date_min = []
    all_date_max = []

    for fpath in matching_files:
        symbol = fpath.stem  # filename without extension
        if not symbol:
            symbol = "UNKNOWN"

        df = normalize_file(fpath, symbol, name, ctx)

        if df is not None and len(df) > 0:
            # Output file
            if ctx.use_parquet and PARQUET_AVAILABLE:
                out_path = market_output / f"{symbol}.parquet"
                df.to_parquet(out_path, index=False)
            else:
                out_path = market_output / f"{symbol}.csv"
                df.to_csv(out_path, index=False)

            stats.success_count += 1

            # Track BTC/ETH
            if symbol.upper() in ("BTC", "BTCUSDT", "BTCKRW"):
                stats.has_btc = True
            if symbol.upper() in ("ETH", "ETHUSDT", "ETHKRW"):
                stats.has_eth = True

            # Track date range
            all_date_min.append(df["date"].min())
            all_date_max.append(df["date"].max())

    # Calculate overall date range
    if all_date_min:
        stats.date_min = min(all_date_min)
        stats.date_max = max(all_date_max)

    # Count warnings/errors
    for diag in ctx.diagnostics:
        if diag.market == name:
            if diag.status == "warning":
                stats.warning_count += 1
            elif diag.status == "error":
                stats.error_count += 1

    print(f"[{name}] Processed {stats.success_count}/{stats.total_files} files "
          f"(warn={stats.warning_count}, err={stats.error_count})")

    return stats


def write_diagnostics(ctx: NormalizationContext) -> None:
    """Write diagnostics CSV file."""
    diag_path = ctx.output_dir / "_diagnostics.csv"

    with open(diag_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "market", "symbol", "file_path", "status", "message",
            "rows_in", "rows_out", "date_min", "date_max"
        ])
        writer.writeheader()
        for diag in ctx.diagnostics:
            writer.writerow({
                "market": diag.market,
                "symbol": diag.symbol,
                "file_path": diag.file_path,
                "status": diag.status,
                "message": diag.message,
                "rows_in": diag.rows_in,
                "rows_out": diag.rows_out,
                "date_min": diag.date_min,
                "date_max": diag.date_max,
            })

    print(f"\nDiagnostics written to: {diag_path}")


def print_verification_report(ctx: NormalizationContext) -> None:
    """Print verification report summary."""
    print("\n" + "=" * 70)
    print("NORMALIZATION VERIFICATION REPORT")
    print("=" * 70)

    total_success = 0
    total_files = 0

    for name, stats in ctx.market_stats.items():
        if stats.total_files == 0:
            continue

        total_files += stats.total_files
        total_success += stats.success_count

        print(f"\n[{name}]")
        print(f"  Files: {stats.success_count}/{stats.total_files} success")
        print(f"  Warnings: {stats.warning_count}, Errors: {stats.error_count}")
        print(f"  BTC: {'YES' if stats.has_btc else 'NO'}, "
              f"ETH: {'YES' if stats.has_eth else 'NO'}")
        if stats.date_min:
            print(f"  Date range: {stats.date_min} to {stats.date_max}")

    print("\n" + "-" * 70)
    print(f"TOTAL: {total_success}/{total_files} files normalized successfully")
    print(f"Output format: {'Parquet' if ctx.use_parquet and PARQUET_AVAILABLE else 'CSV'}")
    print("=" * 70)


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
        "--out-dir",
        type=Path,
        default=Path("outputs/normalized_1d"),
        help="Output directory for normalized data",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "csv", "auto"],
        default="auto",
        help="Output format (auto uses parquet if available)",
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
        print(f"\nOutput format: {'Parquet' if PARQUET_AVAILABLE else 'CSV (pyarrow not installed)'}")
        return 0

    # Determine output format
    use_parquet = args.format == "parquet" or (args.format == "auto" and PARQUET_AVAILABLE)
    if args.format == "parquet" and not PARQUET_AVAILABLE:
        print("[WARN] Parquet requested but pyarrow not installed, falling back to CSV")
        use_parquet = False

    # Create context
    ctx = NormalizationContext(
        output_dir=args.out_dir,
        use_parquet=use_parquet,
    )

    print(f"Output directory: {args.out_dir}")
    print(f"Output format: {'Parquet' if use_parquet else 'CSV'}")
    print("-" * 70)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Process each market
    for name, cfg in markets.items():
        normalize_market(name, cfg, ctx)

    # Write diagnostics
    write_diagnostics(ctx)

    # Print verification report
    print_verification_report(ctx)

    return 0


if __name__ == "__main__":
    sys.exit(main())
