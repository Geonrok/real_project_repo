"""
examples/generate_phase25_sweep.py - Generate Phase 2.5 min_notional_cash sweep

This script reruns Phase 1 from real OHLCV data, then runs the Phase 2.5 sweep.
"""

import sys
import os
from typing import List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from phase1_anchor_engine import Phase1Config
from adapters.run_phase1_from_loader import run_phase1_from_loader
from adapters.run_phase2_from_loader import (
    Phase2Config,
    MIN_NOTIONAL_SWEEP_GRID,
    run_min_notional_sweep,
)


def _parse_grid(value: Optional[str]) -> Optional[List[float]]:
    if value is None or value.strip() == "":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [float(p) for p in parts]


def _resolve_venue_path(venue_path: Optional[str], base_data: Optional[str], venue: Optional[str]) -> str:
    if venue_path:
        return venue_path
    if base_data and venue:
        return os.path.join(base_data, venue)
    raise ValueError("Provide --venue-path or both --base-data and --venue.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate phase25_min_notional_sweep.csv from real data")
    parser.add_argument("--venue-path", default=None, help="Path to a venue data directory")
    parser.add_argument("--base-data", default=None, help="Base data directory containing venue subfolders")
    parser.add_argument("--venue", default=None, help="Venue subfolder name under --base-data")
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "outputs"))
    parser.add_argument("--grid", default=None, help="Comma-separated grid, e.g. 0,50,100,200,500")
    parser.add_argument("--min-history-days", type=int, default=60)

    args = parser.parse_args()

    venue_path = _resolve_venue_path(args.venue_path, args.base_data, args.venue)
    if not os.path.exists(venue_path):
        raise FileNotFoundError(f"Venue path not found: {venue_path}")

    phase1_config = Phase1Config(min_history_days=args.min_history_days)
    phase1_result = run_phase1_from_loader(venue_path, config=phase1_config, min_days=args.min_history_days)
    if phase1_result is None:
        raise RuntimeError(f"Phase 1 failed for venue path: {venue_path}")

    grid = _parse_grid(args.grid) or MIN_NOTIONAL_SWEEP_GRID
    base_config = Phase2Config(phase1_config=phase1_config, save_outputs=False)

    run_min_notional_sweep(
        phase1_result=phase1_result,
        grid=grid,
        base_config=base_config,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
