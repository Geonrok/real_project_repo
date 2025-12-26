"""
Tests for Stage2 Gate Report WARN channel functionality.

Purpose: Verify the tri-state PASS/WARN/FAIL classification for data quality.
Tests the classify_data_quality_status, evaluate_data_quality, and
generate_final_candidates_csv functions.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from stage2_gate_report import (
    classify_data_quality_status,
    evaluate_data_quality,
    generate_final_candidates_csv,
    _safe_int,
    DATA_QUALITY_COLUMNS,
)


# =============================================================================
# Helper Function Tests
# =============================================================================

class TestSafeInt:
    """Tests for _safe_int helper function."""

    def test_int_value(self) -> None:
        """Integer values pass through."""
        assert _safe_int(5) == 5
        assert _safe_int(0) == 0

    def test_float_value(self) -> None:
        """Float values are converted to int."""
        assert _safe_int(5.0) == 5
        assert _safe_int(5.7) == 5

    def test_none_returns_zero(self) -> None:
        """None returns 0."""
        assert _safe_int(None) == 0

    def test_nan_returns_zero(self) -> None:
        """NaN returns 0."""
        import numpy as np
        assert _safe_int(float("nan")) == 0
        assert _safe_int(np.nan) == 0

    def test_string_returns_zero(self) -> None:
        """Invalid string returns 0."""
        assert _safe_int("invalid") == 0


# =============================================================================
# Classification Tests
# =============================================================================

class TestClassifyDataQualityStatus:
    """Tests for classify_data_quality_status function."""

    def test_all_zeros_returns_pass(self) -> None:
        """All counts at 0 returns PASS."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 0,
            DATA_QUALITY_COLUMNS["ret_inf"]: 0,
            DATA_QUALITY_COLUMNS["strat_ret_clamp"]: 0,
            DATA_QUALITY_COLUMNS["cumret_clip"]: 0,
        })
        status, reason = classify_data_quality_status(row)
        assert status == "PASS"
        assert reason == ""

    def test_zero_close_returns_fail(self) -> None:
        """zero_close_count > 0 returns FAIL."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 1,
            DATA_QUALITY_COLUMNS["ret_inf"]: 0,
            DATA_QUALITY_COLUMNS["strat_ret_clamp"]: 0,
            DATA_QUALITY_COLUMNS["cumret_clip"]: 0,
        })
        status, reason = classify_data_quality_status(row)
        assert status == "FAIL"
        assert reason == "DATA_ZERO_CLOSE"

    def test_ret_inf_returns_fail(self) -> None:
        """ret_inf_count > 0 returns FAIL."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 0,
            DATA_QUALITY_COLUMNS["ret_inf"]: 1,
            DATA_QUALITY_COLUMNS["strat_ret_clamp"]: 0,
            DATA_QUALITY_COLUMNS["cumret_clip"]: 0,
        })
        status, reason = classify_data_quality_status(row)
        assert status == "FAIL"
        assert reason == "DATA_RET_INF"

    def test_cumret_clip_returns_fail(self) -> None:
        """cumret_clip_count > 0 returns FAIL."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 0,
            DATA_QUALITY_COLUMNS["ret_inf"]: 0,
            DATA_QUALITY_COLUMNS["strat_ret_clamp"]: 0,
            DATA_QUALITY_COLUMNS["cumret_clip"]: 1,
        })
        status, reason = classify_data_quality_status(row)
        assert status == "FAIL"
        assert reason == "EQUITY_FLOOR_CLIP"

    def test_strat_ret_clamp_only_returns_warn(self) -> None:
        """strat_ret_clamp_count > 0 with no other issues returns WARN."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 0,
            DATA_QUALITY_COLUMNS["ret_inf"]: 0,
            DATA_QUALITY_COLUMNS["strat_ret_clamp"]: 3,  # Any positive count
            DATA_QUALITY_COLUMNS["cumret_clip"]: 0,
        })
        status, reason = classify_data_quality_status(row)
        assert status == "WARN"
        assert reason == "EXTREME_RETURN_CLAMP_ONLY"

    def test_fail_priority_over_warn(self) -> None:
        """FAIL conditions take priority over WARN."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 1,  # FAIL condition
            DATA_QUALITY_COLUMNS["ret_inf"]: 0,
            DATA_QUALITY_COLUMNS["strat_ret_clamp"]: 5,  # Would be WARN
            DATA_QUALITY_COLUMNS["cumret_clip"]: 0,
        })
        status, reason = classify_data_quality_status(row)
        assert status == "FAIL"
        assert reason == "DATA_ZERO_CLOSE"

    def test_fail_priority_order(self) -> None:
        """FAIL conditions check in priority order: zero_close > ret_inf > cumret_clip."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 1,
            DATA_QUALITY_COLUMNS["ret_inf"]: 1,
            DATA_QUALITY_COLUMNS["strat_ret_clamp"]: 1,
            DATA_QUALITY_COLUMNS["cumret_clip"]: 1,
        })
        status, reason = classify_data_quality_status(row)
        assert status == "FAIL"
        assert reason == "DATA_ZERO_CLOSE"

    def test_handles_missing_columns(self) -> None:
        """Missing columns treated as 0."""
        row = pd.Series({
            DATA_QUALITY_COLUMNS["zero_close"]: 0,
            # Missing other columns
        })
        status, reason = classify_data_quality_status(row)
        assert status == "PASS"
        assert reason == ""


# =============================================================================
# Evaluate Data Quality Tests
# =============================================================================

class TestEvaluateDataQuality:
    """Tests for evaluate_data_quality function."""

    def test_empty_df_returns_skipped(self) -> None:
        """Empty DataFrame returns skipped=True."""
        result = evaluate_data_quality(pd.DataFrame())
        assert result["skipped"] is True
        assert result["passed"] == []
        assert result["warned"] == []
        assert result["failed"] == []

    def test_none_returns_skipped(self) -> None:
        """None input returns skipped=True."""
        result = evaluate_data_quality(None)
        assert result["skipped"] is True

    def test_all_pass_strategies(self) -> None:
        """All-pass strategies correctly identified."""
        df = pd.DataFrame([
            {"market": "m1", "strategy_id": "s1", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},
            {"market": "m1", "strategy_id": "s2", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},
        ])
        result = evaluate_data_quality(df)
        assert result["skipped"] is False
        assert set(result["passed"]) == {"s1", "s2"}
        assert result["warned"] == []
        assert result["failed"] == []

    def test_warn_strategies(self) -> None:
        """WARN strategies correctly identified with details."""
        df = pd.DataFrame([
            {"market": "m1", "strategy_id": "s1", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 1, "cumret_clip_count": 0},
        ])
        result = evaluate_data_quality(df)
        assert result["passed"] == []
        assert len(result["warned"]) == 1
        warn = result["warned"][0]
        assert warn["strategy_id"] == "s1"
        assert warn["reason"] == "EXTREME_RETURN_CLAMP_ONLY"
        assert result["failed"] == []

    def test_fail_strategies(self) -> None:
        """FAIL strategies correctly identified with details."""
        df = pd.DataFrame([
            {"market": "m1", "strategy_id": "s1", "zero_close_count": 1, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},
        ])
        result = evaluate_data_quality(df)
        assert result["passed"] == []
        assert result["warned"] == []
        assert len(result["failed"]) == 1
        fail = result["failed"][0]
        assert fail["strategy_id"] == "s1"
        assert fail["reason"] == "DATA_ZERO_CLOSE"

    def test_mixed_pass_warn_fail(self) -> None:
        """Mixed PASS/WARN/FAIL correctly categorized."""
        df = pd.DataFrame([
            {"market": "m1", "strategy_id": "s1", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},  # PASS
            {"market": "m1", "strategy_id": "s2", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 2, "cumret_clip_count": 0},  # WARN
            {"market": "m1", "strategy_id": "s3", "zero_close_count": 0, "ret_inf_count": 1, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},  # FAIL
        ])
        result = evaluate_data_quality(df)
        assert result["passed"] == ["s1"]
        assert len(result["warned"]) == 1
        assert result["warned"][0]["strategy_id"] == "s2"
        assert len(result["failed"]) == 1
        assert result["failed"][0]["strategy_id"] == "s3"

    def test_multi_market_worst_status_aggregation(self) -> None:
        """Multi-market strategies use worst status across markets."""
        df = pd.DataFrame([
            {"market": "m1", "strategy_id": "s1", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},  # PASS
            {"market": "m2", "strategy_id": "s1", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 1, "cumret_clip_count": 0},  # WARN
        ])
        result = evaluate_data_quality(df)
        # Strategy s1 should be WARN (worst of PASS and WARN)
        assert result["passed"] == []
        assert len(result["warned"]) == 1
        assert result["warned"][0]["strategy_id"] == "s1"

    def test_fail_takes_priority_over_warn_multi_market(self) -> None:
        """FAIL takes priority over WARN across markets."""
        df = pd.DataFrame([
            {"market": "m1", "strategy_id": "s1", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 5, "cumret_clip_count": 0},  # WARN
            {"market": "m2", "strategy_id": "s1", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 1},  # FAIL
        ])
        result = evaluate_data_quality(df)
        # Strategy s1 should be FAIL (worst of WARN and FAIL)
        assert result["passed"] == []
        assert result["warned"] == []
        assert len(result["failed"]) == 1
        assert result["failed"][0]["strategy_id"] == "s1"
        assert result["failed"][0]["reason"] == "EQUITY_FLOOR_CLIP"

    def test_deterministic_sorting(self) -> None:
        """Output is deterministically sorted."""
        df = pd.DataFrame([
            {"market": "m1", "strategy_id": "z_strat", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},
            {"market": "m1", "strategy_id": "a_strat", "zero_close_count": 0, "ret_inf_count": 0, "strat_ret_clamp_count": 0, "cumret_clip_count": 0},
        ])
        result = evaluate_data_quality(df)
        assert result["passed"] == ["a_strat", "z_strat"]


# =============================================================================
# Final Candidates CSV Tests
# =============================================================================

class TestGenerateFinalCandidatesCsv:
    """Tests for generate_final_candidates_csv function."""

    def test_strict_pass_excludes_warn(self, tmp_path: Path) -> None:
        """Strict PASS excludes WARN strategies."""
        quality_result = {
            "passed": ["s1"],
            "warned": [{"strategy_id": "s2", "reason": "EXTREME_RETURN_CLAMP_ONLY"}],
            "failed": [],
        }
        strict_count, warn_count = generate_final_candidates_csv(
            all_strategies={"s1", "s2"},
            sens_passed={"s1", "s2"},
            stress_passed={"s1", "s2"},
            quality_result=quality_result,
            sens_skipped=False,
            stress_skipped=False,
            quality_skipped=False,
            out_dir=tmp_path,
        )
        assert strict_count == 1
        assert warn_count == 2

        # Verify strict CSV
        strict_df = pd.read_csv(tmp_path / "stage2_final_pass.csv")
        assert len(strict_df) == 1
        assert strict_df.iloc[0]["strategy_id"] == "s1"

        # Verify with-warn CSV
        warn_df = pd.read_csv(tmp_path / "stage2_final_pass_with_warn.csv")
        assert len(warn_df) == 2
        assert set(warn_df["strategy_id"]) == {"s1", "s2"}

    def test_warn_csv_has_status_column(self, tmp_path: Path) -> None:
        """WARN CSV includes data_quality_status column."""
        quality_result = {
            "passed": ["s1"],
            "warned": [{"strategy_id": "s2", "reason": "EXTREME_RETURN_CLAMP_ONLY"}],
            "failed": [],
        }
        generate_final_candidates_csv(
            all_strategies={"s1", "s2"},
            sens_passed={"s1", "s2"},
            stress_passed={"s1", "s2"},
            quality_result=quality_result,
            sens_skipped=False,
            stress_skipped=False,
            quality_skipped=False,
            out_dir=tmp_path,
        )

        warn_df = pd.read_csv(tmp_path / "stage2_final_pass_with_warn.csv")
        assert "data_quality_status" in warn_df.columns
        assert "data_quality_reason" in warn_df.columns

        s1_row = warn_df[warn_df["strategy_id"] == "s1"].iloc[0]
        assert s1_row["data_quality_status"] == "PASS"

        s2_row = warn_df[warn_df["strategy_id"] == "s2"].iloc[0]
        assert s2_row["data_quality_status"] == "WARN"
        assert s2_row["data_quality_reason"] == "EXTREME_RETURN_CLAMP_ONLY"

    def test_sensitivity_filter_applied(self, tmp_path: Path) -> None:
        """Sensitivity filter excludes non-passing strategies."""
        quality_result = {
            "passed": ["s1", "s2"],
            "warned": [],
            "failed": [],
        }
        strict_count, warn_count = generate_final_candidates_csv(
            all_strategies={"s1", "s2"},
            sens_passed={"s1"},  # s2 fails sensitivity
            stress_passed={"s1", "s2"},
            quality_result=quality_result,
            sens_skipped=False,
            stress_skipped=False,
            quality_skipped=False,
            out_dir=tmp_path,
        )
        assert strict_count == 1
        assert warn_count == 1

    def test_skipped_criteria_not_filtered(self, tmp_path: Path) -> None:
        """Skipped criteria don't filter strategies."""
        quality_result = {
            "passed": [],
            "warned": [],
            "failed": [{"strategy_id": "s1", "reason": "DATA_ZERO_CLOSE"}],
        }
        strict_count, warn_count = generate_final_candidates_csv(
            all_strategies={"s1"},
            sens_passed=set(),  # Would fail if not skipped
            stress_passed=set(),  # Would fail if not skipped
            quality_result=quality_result,
            sens_skipped=True,  # Skip sensitivity
            stress_skipped=True,  # Skip stress
            quality_skipped=True,  # Skip quality
            out_dir=tmp_path,
        )
        # All criteria skipped, so s1 should pass
        assert strict_count == 1
        assert warn_count == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
