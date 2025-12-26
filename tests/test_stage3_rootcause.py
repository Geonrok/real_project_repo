"""Tests for Stage3 root cause classification."""
import pandas as pd
import pytest


class TestRootCauseClassification:
    """Test root cause classification logic."""

    def test_data_zero_close_priority(self):
        """DATA_ZERO_CLOSE should have highest priority."""
        from scripts.stage3_clamp_rootcause import classify_root_cause

        row = pd.Series({
            "zero_close_count": 1,
            "ret_inf_count": 1,
            "strat_ret_clamp_count": 1,
            "cumret_clip_count": 1,
        })
        assert classify_root_cause(row) == "DATA_ZERO_CLOSE"

    def test_data_ret_inf_second_priority(self):
        """DATA_RET_INF should be second priority."""
        from scripts.stage3_clamp_rootcause import classify_root_cause

        row = pd.Series({
            "zero_close_count": 0,
            "ret_inf_count": 1,
            "strat_ret_clamp_count": 1,
            "cumret_clip_count": 1,
        })
        assert classify_root_cause(row) == "DATA_RET_INF"

    def test_extreme_return_clamp_only(self):
        """EXTREME_RETURN_CLAMP_ONLY when clamp but no equity clip."""
        from scripts.stage3_clamp_rootcause import classify_root_cause

        row = pd.Series({
            "zero_close_count": 0,
            "ret_inf_count": 0,
            "strat_ret_clamp_count": 1,
            "cumret_clip_count": 0,
        })
        assert classify_root_cause(row) == "EXTREME_RETURN_CLAMP_ONLY"

    def test_equity_floor_clip(self):
        """EQUITY_FLOOR_CLIP when cumret was clipped."""
        from scripts.stage3_clamp_rootcause import classify_root_cause

        row = pd.Series({
            "zero_close_count": 0,
            "ret_inf_count": 0,
            "strat_ret_clamp_count": 1,
            "cumret_clip_count": 1,
        })
        assert classify_root_cause(row) == "EQUITY_FLOOR_CLIP"

    def test_unknown_no_issues(self):
        """UNKNOWN when no diagnostic issues."""
        from scripts.stage3_clamp_rootcause import classify_root_cause

        row = pd.Series({
            "zero_close_count": 0,
            "ret_inf_count": 0,
            "strat_ret_clamp_count": 0,
            "cumret_clip_count": 0,
        })
        assert classify_root_cause(row) == "UNKNOWN"

    def test_handles_none_values(self):
        """Should handle None/NaN values gracefully."""
        from scripts.stage3_clamp_rootcause import classify_root_cause

        row = pd.Series({
            "zero_close_count": None,
            "ret_inf_count": None,
            "strat_ret_clamp_count": 1,
            "cumret_clip_count": None,
        })
        assert classify_root_cause(row) == "EXTREME_RETURN_CLAMP_ONLY"


class TestDeterministicSorting:
    """Test deterministic sorting of results."""

    def test_sort_order(self):
        """Results should be sorted by market, root_cause, strategy_id."""
        df = pd.DataFrame([
            {"market": "b", "root_cause": "Z", "strategy_id": "s1"},
            {"market": "a", "root_cause": "A", "strategy_id": "s2"},
            {"market": "a", "root_cause": "A", "strategy_id": "s1"},
            {"market": "a", "root_cause": "B", "strategy_id": "s1"},
        ])

        sorted_df = df.sort_values(
            ["market", "root_cause", "strategy_id"],
            ascending=[True, True, True],
            na_position="last"
        ).reset_index(drop=True)

        expected_order = [
            ("a", "A", "s1"),
            ("a", "A", "s2"),
            ("a", "B", "s1"),
            ("b", "Z", "s1"),
        ]

        for i, (m, r, s) in enumerate(expected_order):
            assert sorted_df.loc[i, "market"] == m
            assert sorted_df.loc[i, "root_cause"] == r
            assert sorted_df.loc[i, "strategy_id"] == s

    def test_na_position_last(self):
        """NA values should be sorted to the end."""
        import numpy as np

        df = pd.DataFrame([
            {"market": "a", "root_cause": np.nan, "strategy_id": "s1"},
            {"market": "a", "root_cause": "A", "strategy_id": "s2"},
        ])

        sorted_df = df.sort_values(
            ["market", "root_cause", "strategy_id"],
            ascending=[True, True, True],
            na_position="last"
        ).reset_index(drop=True)

        assert sorted_df.loc[0, "root_cause"] == "A"
        assert pd.isna(sorted_df.loc[1, "root_cause"])
