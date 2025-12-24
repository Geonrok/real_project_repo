"""
examples/analyze_phase26_recommendation.py - Phase 2.6 Default Selection & Pareto Report

Inputs:
- outputs/phase25_min_notional_sweep.csv (Phase 2.5 sweep output)
- or --sweep-csv path override

Outputs:
- outputs/phase26_recommendation.json
- outputs/phase26_pareto.csv
- outputs/phase26_tradeoff.png (optional)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from glob import glob
from typing import List, Optional

import pandas as pd

from adapters.run_phase2_from_loader import save_phase26_reports


def _parse_grid(value: Optional[str]) -> Optional[List[float]]:
    if value is None or value.strip() == "":
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return [float(p) for p in parts]


def _resolve_sweep_path(
    output_dir: str,
    sweep_csv: Optional[str],
    project_root: str,
) -> str:
    if sweep_csv:
        if os.path.exists(sweep_csv):
            return sweep_csv
        raise FileNotFoundError(f"Missing sweep file: {sweep_csv}")

    default_path = os.path.join(output_dir, "phase25_min_notional_sweep.csv")
    if os.path.exists(default_path):
        return default_path

    matches = glob(os.path.join(project_root, "**", "phase25_min_notional_sweep.csv"), recursive=True)
    if len(matches) == 1:
        return matches[0]
    if len(matches) == 0:
        raise FileNotFoundError(
            "Missing sweep file. Looked for:\n"
            f"- {default_path}\n"
            "To generate one, run one of:\n"
            "  python examples/generate_phase25_sweep.py --venue-path <DATA_DIR> --output-dir outputs\n"
            "  python examples/generate_phase25_sweep.py --base-data <BASE_DIR> --venue binance_futures --output-dir outputs"
        )
    raise FileNotFoundError(
        "Multiple sweep files found. Pass --sweep-csv to select one:\n"
        + "\n".join(f"- {path}" for path in matches)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2.6 min_notional_cash recommendation")
    parser.add_argument("--grid", default=None, help="Comma-separated grid, e.g. 0,50,100,200,500")
    parser.add_argument("--objective", choices=["max_nav", "min_cost"], default="max_nav")
    parser.add_argument("--l1-max", type=float, default=0.8)
    parser.add_argument("--l1-mean", type=float, default=0.2)
    parser.add_argument("--require-positive-nav", action="store_true", default=True)
    parser.add_argument("--output-dir", default=os.path.join(os.getcwd(), "outputs"))
    parser.add_argument("--sweep-csv", default=None, help="Path to phase25_min_notional_sweep.csv")
    parser.add_argument("--save-plot", action="store_true", default=False)

    args = parser.parse_args()

    output_dir = args.output_dir
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sweep_path = _resolve_sweep_path(output_dir, args.sweep_csv, project_root)

    sweep_df = pd.read_csv(sweep_path)
    grid = _parse_grid(args.grid)
    if grid is not None:
        sweep_df = sweep_df[sweep_df["min_notional_cash"].isin(grid)].reset_index(drop=True)

    constraints = {
        "require_positive_nav": args.require_positive_nav,
        "guardrail_l1_distance_max": args.l1_max,
        "guardrail_l1_distance_mean": args.l1_mean,
    }

    result = save_phase26_reports(
        sweep_df=sweep_df,
        output_dir=output_dir,
        constraints=constraints,
        objective=args.objective,
        selected_grid=grid,
        save_plot=args.save_plot,
    )

    selected = result["recommendation"]["selected_min_notional_cash"]
    print(f"Phase 2.6 recommendation saved. selected_min_notional_cash={selected}")


if __name__ == "__main__":
    main()
