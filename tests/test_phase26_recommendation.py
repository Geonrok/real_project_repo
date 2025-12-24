"""
tests/test_phase26_recommendation.py - Phase 2.6 Default Selection & Pareto Report tests
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd

from adapters.run_phase2_from_loader import (
    recommend_min_notional_from_sweep,
    build_phase26_pareto_df,
    save_phase26_reports,
)


def _make_sweep_df() -> pd.DataFrame:
    return pd.DataFrame({
        "min_notional_cash": [0, 50, 100, 200],
        "final_nav_phase2": [1000.0, 999.5, 990.0, 980.0],
        "total_extra_cost": [10.0, 9.5, 9.0, 8.0],
        "filter_ratio": [0.4, 0.2, 0.3, 0.1],
        "l1_distance_mean": [0.19, 0.10, 0.25, 0.15],
        "l1_distance_max": [0.7, 0.7, 0.9, 0.7],
    })


def test_recommendation_tie_breaker():
    sweep_df = _make_sweep_df()
    rec = recommend_min_notional_from_sweep(sweep_df)

    # 0 and 50 are within 0.1% of max NAV, so tie-breaker picks lower l1_distance_mean.
    assert rec["selected_min_notional_cash"] == 50.0


def test_pareto_feasibility_flags():
    sweep_df = _make_sweep_df()
    pareto = build_phase26_pareto_df(sweep_df)

    feasible = pareto.set_index("min_notional_cash")["is_feasible"].to_dict()
    assert feasible[0] is True
    assert feasible[50] is True
    assert feasible[100] is False  # l1_distance_mean and l1_distance_max exceed guardrails
    assert feasible[200] is True

    ranks = pareto.set_index("min_notional_cash")["rank"]
    assert np.isnan(ranks[100])


def test_save_phase26_reports_updates_manifest(tmp_path):
    sweep_df = _make_sweep_df()
    manifest_path = tmp_path / "real_run_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"schema_version": "2.5.0"}, f)

    result = save_phase26_reports(
        sweep_df=sweep_df,
        output_dir=str(tmp_path),
        constraints={"guardrail_l1_distance_max": 0.8, "guardrail_l1_distance_mean": 0.2},
        objective="max_nav",
        selected_grid=[0, 50, 100, 200],
        save_plot=False,
    )

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert manifest["schema_version"] == "2.6.0"
    assert "recommendation" in manifest
    assert manifest["recommendation"]["min_notional_cash"] == 50.0
    assert manifest["recommendation"]["objective"] == "max_nav"
    assert manifest["recommendation"]["selected_from_grid"] == [0, 50, 100, 200]

    assert result["recommendation"]["selected_min_notional_cash"] == 50.0
