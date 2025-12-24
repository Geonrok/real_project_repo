"""
tests/test_phase2_sensitivity_smoke.py - Phase 2.1 Sensitivity 분석 Smoke 테스트

작은 synthetic fixture로 핵심 함수를 테스트합니다:
  (a) 결과 shape/컬럼 존재
  (b) P2 <= P1 불변량
  (c) cost 합산 불변량
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from phase1_anchor_engine import Phase1AnchorEngine, Phase1Config, Phase1Result
from examples.analyze_phase2_sensitivity import (
    run_sensitivity_single,
    run_sensitivity_grid,
    analyze_recommendation,
    SensitivityRow,
    DEFAULT_CHARGING_MODES,
    DEFAULT_EPS_TRADE_GRID,
)


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_price_data():
    """합성 가격 데이터 생성"""
    np.random.seed(42)
    n_days = 100
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")

    def generate_ohlcv(base_price, volatility=0.02):
        returns = np.random.normal(0, volatility, n_days)
        close = base_price * np.exp(np.cumsum(returns))
        high = close * (1 + np.random.uniform(0, 0.01, n_days))
        low = close * (1 - np.random.uniform(0, 0.01, n_days))
        open_ = close * (1 + np.random.uniform(-0.005, 0.005, n_days))
        volume = np.random.uniform(1000, 10000, n_days)
        return pd.DataFrame({
            "open": open_, "high": high, "low": low,
            "close": close, "volume": volume
        }, index=dates)

    return {
        "BTC": generate_ohlcv(40000, 0.03),
        "ETH": generate_ohlcv(2500, 0.04),
        "SOL": generate_ohlcv(100, 0.05),
    }


@pytest.fixture
def phase1_result(synthetic_price_data):
    """Phase 1 결과 생성"""
    config = Phase1Config(
        min_history_days=30,
        one_way_rate_bps=5.0,
        initial_nav=10000.0,
    )
    engine = Phase1AnchorEngine(config)

    btc_close = synthetic_price_data["BTC"]["close"]
    eth_close = synthetic_price_data["ETH"]["close"]

    return engine.run(synthetic_price_data, btc_close, eth_close)


# ==============================================================================
# Tests
# ==============================================================================

class TestSensitivitySingle:
    """run_sensitivity_single 테스트"""

    def test_returns_sensitivity_row(self, phase1_result):
        """SensitivityRow 반환 확인"""
        row = run_sensitivity_single(
            phase1_result,
            charging_mode="per_asset_day",
            eps_trade=1e-12,
        )

        assert isinstance(row, SensitivityRow)
        assert row.mode == "per_asset_day"
        assert row.eps_trade == 1e-12

    def test_p2_leq_p1_invariant(self, phase1_result):
        """P2 NAV <= P1 NAV 불변량"""
        for mode in DEFAULT_CHARGING_MODES:
            row = run_sensitivity_single(
                phase1_result,
                charging_mode=mode,
                eps_trade=1e-12,
            )

            assert row.final_nav_p2 <= row.final_nav_p1, \
                f"P2 NAV should be <= P1 NAV for mode={mode}"

    def test_cost_sum_invariant(self, phase1_result):
        """cost_cash_phase2 = cost_cash_phase1 + extra_cost 불변량"""
        for mode in DEFAULT_CHARGING_MODES:
            row = run_sensitivity_single(
                phase1_result,
                charging_mode=mode,
                eps_trade=1e-12,
            )

            expected_p2 = row.total_cost_p1 + row.total_extra_cost
            assert np.isclose(row.total_cost_p2, expected_p2, rtol=1e-6), \
                f"Cost sum mismatch for mode={mode}"

    def test_event_count_non_negative(self, phase1_result):
        """이벤트 카운트 비음수"""
        row = run_sensitivity_single(
            phase1_result,
            charging_mode="per_asset_day",
            eps_trade=1e-12,
        )

        assert row.min_fee_event_count >= 0
        assert row.days_with_min_fee >= 0
        assert row.avg_events_per_day >= 0

    def test_eps_trade_affects_event_count(self, phase1_result):
        """eps_trade가 이벤트 카운트에 영향"""
        row_strict = run_sensitivity_single(
            phase1_result,
            charging_mode="per_asset_day",
            eps_trade=0.0,  # 모든 거래 인식
        )

        row_loose = run_sensitivity_single(
            phase1_result,
            charging_mode="per_asset_day",
            eps_trade=1e-6,  # 소규모 거래 무시
        )

        # 더 느슨한 eps_trade는 더 적은 이벤트
        assert row_loose.min_fee_event_count <= row_strict.min_fee_event_count


class TestSensitivityGrid:
    """run_sensitivity_grid 테스트"""

    def test_returns_dataframe(self, phase1_result):
        """DataFrame 반환 확인"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=["per_asset_day"],
            eps_trade_grid=[1e-12],
        )

        assert isinstance(df, pd.DataFrame)

    def test_correct_shape(self, phase1_result):
        """결과 shape 확인"""
        modes = ["per_asset_day", "per_rebalance_day"]
        eps_grid = [1e-12, 1e-8]

        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=modes,
            eps_trade_grid=eps_grid,
        )

        expected_rows = len(modes) * len(eps_grid)
        assert len(df) == expected_rows

    def test_required_columns_exist(self, phase1_result):
        """필수 컬럼 존재 확인"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=["per_asset_day"],
            eps_trade_grid=[1e-12],
        )

        required_columns = [
            "mode", "eps_trade",
            "final_nav_p1", "final_nav_p2",
            "total_cost_p1", "total_extra_cost", "total_cost_p2",
            "min_fee_cost", "rounding_cost",
            "min_fee_event_count", "days_with_min_fee", "avg_events_per_day",
        ]

        for col in required_columns:
            assert col in df.columns, f"Missing column: {col}"

    def test_all_modes_represented(self, phase1_result):
        """모든 모드가 결과에 포함"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=DEFAULT_CHARGING_MODES,
            eps_trade_grid=[1e-12],
        )

        result_modes = set(df["mode"].unique())
        expected_modes = set(DEFAULT_CHARGING_MODES)

        assert result_modes == expected_modes

    def test_p2_leq_p1_all_rows(self, phase1_result):
        """모든 행에서 P2 <= P1"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=DEFAULT_CHARGING_MODES,
            eps_trade_grid=[0.0, 1e-12, 1e-8],
        )

        assert (df["final_nav_p2"] <= df["final_nav_p1"]).all(), \
            "P2 NAV should be <= P1 NAV for all rows"

    def test_cost_sum_all_rows(self, phase1_result):
        """모든 행에서 비용 합산 일치"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=DEFAULT_CHARGING_MODES,
            eps_trade_grid=[1e-12],
        )

        expected = df["total_cost_p1"] + df["total_extra_cost"]
        match = np.isclose(df["total_cost_p2"], expected, rtol=1e-6)

        assert match.all(), "Cost sum should match for all rows"


class TestAnalyzeRecommendation:
    """analyze_recommendation 테스트"""

    def test_returns_dict(self, phase1_result):
        """Dict 반환 확인"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=DEFAULT_CHARGING_MODES,
            eps_trade_grid=[1e-12],
        )

        result = analyze_recommendation(df)

        assert isinstance(result, dict)
        assert "recommendations" in result
        assert "invariants_verified" in result
        assert "total_combinations" in result

    def test_invariants_verified(self, phase1_result):
        """불변량 검증 통과"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=DEFAULT_CHARGING_MODES,
            eps_trade_grid=DEFAULT_EPS_TRADE_GRID,
        )

        result = analyze_recommendation(df)

        assert result["invariants_verified"] is True

    def test_recommendations_have_required_fields(self, phase1_result):
        """권장값에 필수 필드 존재"""
        df = run_sensitivity_grid(
            phase1_result,
            charging_modes=DEFAULT_CHARGING_MODES,
            eps_trade_grid=[1e-12],
        )

        result = analyze_recommendation(df)

        for rec in result["recommendations"]:
            assert "rank" in rec
            assert "mode" in rec
            assert "eps_trade" in rec
            assert "rationale" in rec
            assert "nav_p2" in rec
            assert "min_fee_cost" in rec


class TestEventCountDiagnostics:
    """min_fee_event_count 진단 테스트"""

    def test_per_rebalance_day_fewer_events(self, phase1_result):
        """per_rebalance_day는 per_asset_day보다 이벤트 적음"""
        row_asset = run_sensitivity_single(
            phase1_result,
            charging_mode="per_asset_day",
            eps_trade=1e-12,
        )

        row_rebal = run_sensitivity_single(
            phase1_result,
            charging_mode="per_rebalance_day",
            eps_trade=1e-12,
        )

        # per_rebalance_day는 하루 최대 1회이므로 이벤트 수가 적어야 함
        assert row_rebal.min_fee_event_count <= row_asset.min_fee_event_count

    def test_days_with_min_fee_leq_total_days(self, phase1_result):
        """min_fee가 있는 날 <= 전체 거래일"""
        total_days = len(phase1_result.timeseries)

        row = run_sensitivity_single(
            phase1_result,
            charging_mode="per_asset_day",
            eps_trade=1e-12,
        )

        assert row.days_with_min_fee <= total_days

    def test_avg_events_consistency(self, phase1_result):
        """평균 이벤트 수 일관성"""
        row = run_sensitivity_single(
            phase1_result,
            charging_mode="per_asset_day",
            eps_trade=1e-12,
        )

        total_days = len(phase1_result.timeseries)
        expected_avg = row.min_fee_event_count / total_days if total_days > 0 else 0

        assert np.isclose(row.avg_events_per_day, expected_avg, rtol=1e-6)


# ==============================================================================
# 실행
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
