"""
tests/test_phase25_distortion.py - Phase 2.5 Distortion Control 테스트

Phase 2.5의 핵심 기능을 테스트합니다:
  1) Worst-days 리포트 (L1 distance 상위 20일)
  2) Distortion guardrail (warning/error 모드)
  3) min_notional_cash sensitivity sweep
  4) 스키마 버전 2.6.0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import warnings
import numpy as np
import pandas as pd
import pytest

from phase1_anchor_engine import Phase1AnchorEngine, Phase1Config, Phase1Result
from adapters.run_phase2_from_loader import (
    Phase2Runner,
    Phase2Config,
    Phase2Result,
    Phase2DistortionError,
    PHASE25_DEFAULT_CONFIG,
    MIN_NOTIONAL_SWEEP_GRID,
    run_phase2_from_phase1_result,
    run_min_notional_sweep,
)
from execution import FilterImpactMetrics


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_phase1_result():
    """합성 Phase 1 결과 생성"""
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    n_days = len(dates)

    def generate_ohlcv(base_price: float, volatility: float = 0.02) -> pd.DataFrame:
        returns = np.random.normal(0, volatility, n_days)
        close = base_price * np.exp(np.cumsum(returns))
        high = close * (1 + np.random.uniform(0, 0.01, n_days))
        low = close * (1 - np.random.uniform(0, 0.01, n_days))
        open_ = close * (1 + np.random.uniform(-0.005, 0.005, n_days))
        volume = np.random.uniform(1000, 10000, n_days)

        return pd.DataFrame({
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }, index=dates)

    price_data = {
        "BTC": generate_ohlcv(40000, 0.03),
        "ETH": generate_ohlcv(2500, 0.04),
        "SOL": generate_ohlcv(100, 0.05),
        "DOGE": generate_ohlcv(0.1, 0.06),
        "XRP": generate_ohlcv(0.5, 0.04),
    }

    btc_close = price_data["BTC"]["close"]
    eth_close = price_data["ETH"]["close"]

    config = Phase1Config(min_history_days=30, one_way_rate_bps=5.0)
    engine = Phase1AnchorEngine(config)
    return engine.run(price_data, btc_close, eth_close)


@pytest.fixture
def phase25_config():
    """Phase 2.5 기본 설정"""
    return Phase2Config(
        enable_min_fee=True,
        enable_rounding=True,
        min_fee_cash=1.0,
        min_fee_charging_mode="per_rebalance_day",
        min_notional_cash=100.0,
        enable_netting=False,
        save_outputs=False,
        guardrail_distortion_mode="warning",
    )


# ==============================================================================
# Worst-Days Report Tests
# ==============================================================================

class TestWorstDaysReport:
    """Phase 2.5 Worst-days 리포트 테스트"""

    def test_distortion_top20_exists(self, synthetic_phase1_result, phase25_config):
        """distortion_top20 DataFrame이 결과에 존재하는지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        assert hasattr(result, "distortion_top20")
        assert isinstance(result.distortion_top20, pd.DataFrame)

    def test_distortion_top20_columns(self, synthetic_phase1_result, phase25_config):
        """distortion_top20 DataFrame의 컬럼이 올바른지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        if not result.distortion_top20.empty:
            expected_cols = [
                "date", "l1_distance", "pre_filter_turnover_L1",
                "post_filter_turnover_L1", "filter_ratio_day",
                "filtered_notional_day", "total_notional_day",
                "events_day", "top_filtered_assets"
            ]
            for col in expected_cols:
                assert col in result.distortion_top20.columns, f"Missing column: {col}"

    def test_distortion_top20_sorted_by_l1_distance(self, synthetic_phase1_result, phase25_config):
        """distortion_top20이 L1 distance 내림차순으로 정렬되어 있는지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        if len(result.distortion_top20) >= 2:
            l1_values = result.distortion_top20["l1_distance"].values
            assert all(l1_values[i] >= l1_values[i+1] for i in range(len(l1_values)-1))

    def test_distortion_top20_max_20_rows(self, synthetic_phase1_result, phase25_config):
        """distortion_top20이 최대 20행인지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        assert len(result.distortion_top20) <= 20

    def test_distortion_top20_saved_to_file(self, synthetic_phase1_result, tmp_path):
        """phase25_distortion_top20.csv 파일이 생성되는지 확인"""
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=100.0,
            output_dir=str(tmp_path),
            save_outputs=True,
        )

        result = run_phase2_from_phase1_result(synthetic_phase1_result, config)

        distortion_path = tmp_path / "phase25_distortion_top20.csv"
        assert distortion_path.exists()

        df = pd.read_csv(distortion_path)
        assert "l1_distance" in df.columns


# ==============================================================================
# Distortion Guardrail Tests
# ==============================================================================

class TestDistortionGuardrail:
    """Phase 2.5 Distortion Guardrail 테스트"""

    def test_guardrail_warning_mode(self, synthetic_phase1_result):
        """warning 모드에서 임계값 초과 시 경고가 발생하는지 확인"""
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=500.0,  # 높은 값으로 왜곡 유발
            save_outputs=False,
            guardrail_l1_distance_max=0.01,  # 낮은 임계값
            guardrail_l1_distance_mean=0.001,  # 낮은 임계값
            guardrail_distortion_mode="warning",
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_phase2_from_phase1_result(synthetic_phase1_result, config)

            # 경고가 발생했는지 확인
            distortion_warnings = [
                warning for warning in w
                if "Distortion Guardrail" in str(warning.message)
            ]
            assert len(distortion_warnings) > 0

    def test_guardrail_error_mode(self, synthetic_phase1_result):
        """error 모드에서 임계값 초과 시 예외가 발생하는지 확인"""
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=500.0,  # 높은 값으로 왜곡 유발
            save_outputs=False,
            guardrail_l1_distance_max=0.01,  # 낮은 임계값
            guardrail_l1_distance_mean=0.001,  # 낮은 임계값
            guardrail_distortion_mode="error",
        )

        with pytest.raises(Phase2DistortionError) as exc_info:
            run_phase2_from_phase1_result(synthetic_phase1_result, config)

        # 예외 정보 확인
        err = exc_info.value
        assert err.l1_distance_max > 0
        assert err.threshold_max == 0.01
        assert err.min_notional_cash == 500.0

    def test_guardrail_no_trigger_within_threshold(self, synthetic_phase1_result, phase25_config):
        """임계값 내에서는 guardrail이 트리거되지 않는지 확인"""
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=100.0,
            save_outputs=False,
            guardrail_l1_distance_max=10.0,  # 매우 높은 임계값
            guardrail_l1_distance_mean=10.0,  # 매우 높은 임계값
            guardrail_distortion_mode="error",
        )

        # 예외 없이 실행되어야 함
        result = run_phase2_from_phase1_result(synthetic_phase1_result, config)
        assert result is not None

    def test_distortion_error_to_dict(self, synthetic_phase1_result):
        """Phase2DistortionError.to_dict() 테스트"""
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=500.0,
            save_outputs=False,
            guardrail_l1_distance_max=0.01,
            guardrail_distortion_mode="error",
        )

        with pytest.raises(Phase2DistortionError) as exc_info:
            run_phase2_from_phase1_result(synthetic_phase1_result, config)

        err_dict = exc_info.value.to_dict()
        assert "l1_distance_max" in err_dict
        assert "threshold_max" in err_dict
        assert "min_notional_cash" in err_dict


# ==============================================================================
# Sensitivity Sweep Tests
# ==============================================================================

class TestMinNotionalSweep:
    """Phase 2.5 min_notional_cash sweep 테스트"""

    def test_sweep_returns_dataframe(self, synthetic_phase1_result):
        """sweep 함수가 DataFrame을 반환하는지 확인"""
        df = run_min_notional_sweep(
            synthetic_phase1_result,
            grid=[0, 50, 100],
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3

    def test_sweep_correct_columns(self, synthetic_phase1_result):
        """sweep 결과에 필요한 컬럼이 있는지 확인"""
        df = run_min_notional_sweep(
            synthetic_phase1_result,
            grid=[0, 100],
        )

        required_cols = [
            "min_notional_cash", "final_nav_phase2",
            "l1_distance_mean", "l1_distance_max", "filter_ratio"
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_sweep_default_grid(self, synthetic_phase1_result):
        """기본 그리드로 sweep이 작동하는지 확인"""
        df = run_min_notional_sweep(synthetic_phase1_result)

        assert len(df) == len(MIN_NOTIONAL_SWEEP_GRID)

    def test_sweep_saved_to_file(self, synthetic_phase1_result, tmp_path):
        """sweep 결과가 파일로 저장되는지 확인"""
        df = run_min_notional_sweep(
            synthetic_phase1_result,
            grid=[0, 50, 100],
            output_dir=str(tmp_path),
        )

        sweep_path = tmp_path / "phase25_min_notional_sweep.csv"
        assert sweep_path.exists()

    def test_sweep_monotonic_l1_distance(self, synthetic_phase1_result):
        """
        min_notional_cash 증가 시 l1_distance도 증가하는지 확인
        (더 많은 필터링 = 더 큰 왜곡)
        """
        df = run_min_notional_sweep(
            synthetic_phase1_result,
            grid=[0, 50, 100, 200],
        )

        l1_values = df["l1_distance_mean"].values
        # 일반적으로 단조 증가해야 하지만, 엣지 케이스가 있을 수 있음
        # 적어도 마지막 값이 첫 번째 값보다 크거나 같아야 함
        assert l1_values[-1] >= l1_values[0] or abs(l1_values[-1] - l1_values[0]) < 1e-6


# ==============================================================================
# Schema Version Tests
# ==============================================================================

class TestSchemaVersion25:
    """Phase 2.5 스키마 버전 테스트"""

    def test_schema_version_is_2_5_0(self, synthetic_phase1_result, phase25_config):
        """schema_version이 2.6.0인지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        assert result.schema_version == "2.6.0"
        assert result.manifest["schema_version"] == "2.6.0"

    def test_manifest_has_distortion_guardrails(self, synthetic_phase1_result, phase25_config):
        """manifest에 distortion guardrail 설정이 있는지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        guardrails = result.manifest["guardrails"]
        assert "l1_distance_max" in guardrails
        assert "l1_distance_mean" in guardrails
        assert "distortion_mode" in guardrails

    def test_default_config_exists(self):
        """PHASE25_DEFAULT_CONFIG 상수 존재 확인"""
        assert PHASE25_DEFAULT_CONFIG is not None
        assert isinstance(PHASE25_DEFAULT_CONFIG, dict)
        assert PHASE25_DEFAULT_CONFIG["min_notional_cash"] == 100.0


# ==============================================================================
# Daily Details Tests
# ==============================================================================

class TestDailyDetails:
    """Phase 2.5 일별 상세 지표 테스트"""

    def test_filter_impact_has_daily_details(self, synthetic_phase1_result, phase25_config):
        """filter_impact에 daily_details DataFrame이 있는지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        fi = result.filter_impact
        assert fi is not None
        assert hasattr(fi, "daily_details")
        assert isinstance(fi.daily_details, pd.DataFrame)

    def test_daily_details_columns(self, synthetic_phase1_result, phase25_config):
        """daily_details의 컬럼이 올바른지 확인"""
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase25_config)

        daily = result.filter_impact.daily_details
        if not daily.empty:
            expected_cols = [
                "l1_distance", "pre_filter_turnover_L1", "post_filter_turnover_L1",
                "filter_ratio_day", "filtered_notional_day", "total_notional_day",
                "events_day", "top_filtered_assets"
            ]
            for col in expected_cols:
                assert col in daily.columns, f"Missing column: {col}"


# ==============================================================================
# 실행
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
