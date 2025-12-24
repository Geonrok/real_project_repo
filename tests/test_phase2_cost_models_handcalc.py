"""
tests/test_phase2_cost_models_handcalc.py - Phase 2 비용 모델 수동 검증 테스트 (Phase 2.1)

이 테스트는 비용 모델의 정확성을 수동 계산과 비교하여 검증합니다.

Phase 2.1 추가:
- charging_mode 3종 (per_asset_day, per_asset_side_day, per_rebalance_day) 검증
- 2자산+CASH 사례에서 buy/sell 발생 시나리오별 기대값 비교
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from costs.base import CostModelResult
from costs.proportional import ProportionalCostModel, ProportionalCostConfig
from costs.min_fee import MinFeeCostModel, MinFeeCostConfig
from costs.rounding import RoundingCostModel, RoundingCostConfig
from costs.combined import CombinedCostModel


# ==============================================================================
# 테스트 픽스처
# ==============================================================================

@pytest.fixture
def sample_data():
    """기본 테스트 데이터"""
    dates = pd.date_range("2024-01-01", periods=5, freq="D")

    nav = pd.Series([10000, 10100, 10050, 10200, 10150], index=dates, dtype=float)
    turnover_L1 = pd.Series([0.0, 0.1, 0.0, 0.05, 0.02], index=dates, dtype=float)

    # delta_w: 2개 자산
    delta_w = pd.DataFrame({
        "BTC": [0.0, 0.05, 0.0, 0.03, 0.01],
        "ETH": [0.0, 0.05, 0.0, 0.02, 0.01],
    }, index=dates)

    return nav, turnover_L1, delta_w


@pytest.fixture
def zero_turnover_data():
    """턴오버가 0인 데이터"""
    dates = pd.date_range("2024-01-01", periods=3, freq="D")

    nav = pd.Series([10000, 10100, 10200], index=dates, dtype=float)
    turnover_L1 = pd.Series([0.0, 0.0, 0.0], index=dates, dtype=float)
    delta_w = pd.DataFrame({
        "BTC": [0.0, 0.0, 0.0],
        "ETH": [0.0, 0.0, 0.0],
    }, index=dates)

    return nav, turnover_L1, delta_w


@pytest.fixture
def two_asset_plus_cash_data():
    """
    2자산 + CASH 테스트 데이터 (Phase 2.1 charging_mode 검증용)

    시나리오:
    - Day 0: 초기 상태, 거래 없음
    - Day 1: BTC buy만 발생 (Δw > 0)
    - Day 2: ETH sell만 발생 (Δw < 0)
    - Day 3: BTC buy + ETH sell 발생 (양쪽)
    - Day 4: 거래 없음
    """
    dates = pd.date_range("2024-01-01", periods=5, freq="D")

    nav = pd.Series([10000, 10100, 10050, 10200, 10150], index=dates, dtype=float)
    turnover_L1 = pd.Series([0.0, 0.05, 0.03, 0.08, 0.0], index=dates, dtype=float)

    # delta_w 설계:
    # Day 0: 모두 0
    # Day 1: BTC +0.05, ETH 0 (buy only)
    # Day 2: BTC 0, ETH -0.03 (sell only)
    # Day 3: BTC +0.04, ETH -0.04 (both buy and sell)
    # Day 4: 모두 0
    delta_w = pd.DataFrame({
        "BTC": [0.0, 0.05, 0.0, 0.04, 0.0],
        "ETH": [0.0, 0.0, -0.03, -0.04, 0.0],
        "CASH": [0.0, -0.05, 0.03, 0.0, 0.0],  # CASH는 제외되어야 함
    }, index=dates)

    return nav, turnover_L1, delta_w


# ==============================================================================
# ProportionalCostModel 테스트
# ==============================================================================

class TestProportionalCostModel:
    """비례 비용 모델 테스트"""

    def test_basic_calculation(self, sample_data):
        """기본 비용 계산 검증"""
        nav, turnover_L1, delta_w = sample_data

        config = ProportionalCostConfig(one_way_rate_bps=5.0)
        model = ProportionalCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        # 수동 계산
        # cost_ratio = turnover * 0.0005 (5bps)
        expected_ratio = pd.Series([0, 0.00005, 0, 0.000025, 0.00001],
                                   index=nav.index)

        np.testing.assert_allclose(
            result.cost_ratio.values,
            expected_ratio.values,
            rtol=1e-10
        )

        # cost_cash = nav * cost_ratio
        expected_cash = nav * expected_ratio
        np.testing.assert_allclose(
            result.cost_cash.values,
            expected_cash.values,
            rtol=1e-10
        )

    def test_zero_turnover(self, zero_turnover_data):
        """턴오버 0일 때 비용 0"""
        nav, turnover_L1, delta_w = zero_turnover_data

        model = ProportionalCostModel()
        result = model.compute(nav, turnover_L1, delta_w)

        assert result.total_cost_cash() == 0.0
        assert result.total_cost_ratio() == 0.0


# ==============================================================================
# MinFeeCostModel 테스트 - 기본
# ==============================================================================

class TestMinFeeCostModelBasic:
    """최소 수수료 모델 기본 테스트"""

    def test_turnover_zero_no_fee(self, zero_turnover_data):
        """CASE: turnover=0인 날에는 min_fee=0이어야 함"""
        nav, turnover_L1, delta_w = zero_turnover_data

        config = MinFeeCostConfig(min_fee_cash=1.0, charging_mode="per_asset_day")
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        # 모든 날에 turnover=0이므로 비용 0
        assert result.total_cost_cash() == 0.0
        assert (result.cost_cash == 0).all()

    def test_turnover_positive_fee_applied(self, sample_data):
        """CASE: turnover>0이면 min_fee 적용 (per_asset_day)"""
        nav, turnover_L1, delta_w = sample_data

        config = MinFeeCostConfig(min_fee_cash=1.0, charging_mode="per_asset_day")
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        # per_asset_day: 거래 발생한 자산 개수 × min_fee
        # day 0: delta_w = [0, 0] -> 0 trades -> cost = 0
        # day 1: delta_w = [0.05, 0.05] -> 2 trades -> cost = 2
        # day 2: delta_w = [0, 0] -> 0 trades -> cost = 0
        # day 3: delta_w = [0.03, 0.02] -> 2 trades -> cost = 2
        # day 4: delta_w = [0.01, 0.01] -> 2 trades -> cost = 2

        expected_cost = pd.Series([0, 2, 0, 2, 2], index=nav.index, dtype=float)
        np.testing.assert_array_equal(result.cost_cash.values, expected_cost.values)


# ==============================================================================
# MinFeeCostModel 테스트 - charging_mode 3종 (Phase 2.1)
# ==============================================================================

class TestMinFeeCostModelChargingModes:
    """
    최소 수수료 모델 charging_mode 테스트 (Phase 2.1)

    테스트 데이터 (two_asset_plus_cash_data):
    - Day 0: 거래 없음
    - Day 1: BTC buy만 (Δw=+0.05)
    - Day 2: ETH sell만 (Δw=-0.03)
    - Day 3: BTC buy + ETH sell (양쪽)
    - Day 4: 거래 없음

    CASH 컬럼은 제외되어야 함
    """

    def test_per_asset_day_mode(self, two_asset_plus_cash_data):
        """
        per_asset_day 모드:
        - 자산별로 |Δw| > eps면 min_fee 1회

        기대값:
        - Day 0: 0 (거래 없음)
        - Day 1: 1 (BTC만 거래)
        - Day 2: 1 (ETH만 거래)
        - Day 3: 2 (BTC, ETH 모두 거래)
        - Day 4: 0 (거래 없음)
        """
        nav, turnover_L1, delta_w = two_asset_plus_cash_data

        config = MinFeeCostConfig(
            min_fee_cash=1.0,
            charging_mode="per_asset_day",
            eps_trade=1e-12
        )
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        expected = pd.Series([0, 1, 1, 2, 0], index=nav.index, dtype=float)
        np.testing.assert_array_equal(
            result.cost_cash.values,
            expected.values,
            err_msg="per_asset_day mode failed"
        )

    def test_per_asset_side_day_mode(self, two_asset_plus_cash_data):
        """
        per_asset_side_day 모드:
        - 자산별/사이드별로 거래 발생 시 min_fee 부과
        - buy: Δw > eps
        - sell: Δw < -eps

        기대값:
        - Day 0: 0
        - Day 1: 1 (BTC buy 1건)
        - Day 2: 1 (ETH sell 1건)
        - Day 3: 2 (BTC buy 1건 + ETH sell 1건)
        - Day 4: 0
        """
        nav, turnover_L1, delta_w = two_asset_plus_cash_data

        config = MinFeeCostConfig(
            min_fee_cash=1.0,
            charging_mode="per_asset_side_day",
            eps_trade=1e-12
        )
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        expected = pd.Series([0, 1, 1, 2, 0], index=nav.index, dtype=float)
        np.testing.assert_array_equal(
            result.cost_cash.values,
            expected.values,
            err_msg="per_asset_side_day mode failed"
        )

    def test_per_asset_side_day_with_both_sides(self):
        """
        per_asset_side_day: 같은 자산에서 buy+sell 모두 발생하는 경우
        (실제로는 드물지만 테스트용)
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        nav = pd.Series([10000, 10100, 10050], index=dates, dtype=float)
        turnover_L1 = pd.Series([0.0, 0.1, 0.0], index=dates, dtype=float)

        # 극단적 케이스: 하루에 BTC를 사고 ETH를 팔고
        delta_w = pd.DataFrame({
            "BTC": [0.0, 0.05, 0.0],  # buy
            "ETH": [0.0, -0.03, 0.0],  # sell
        }, index=dates)

        config = MinFeeCostConfig(
            min_fee_cash=1.0,
            charging_mode="per_asset_side_day"
        )
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        # Day 1: BTC buy(1) + ETH sell(1) = 2
        expected = pd.Series([0, 2, 0], index=nav.index, dtype=float)
        np.testing.assert_array_equal(result.cost_cash.values, expected.values)

    def test_per_rebalance_day_mode(self, two_asset_plus_cash_data):
        """
        per_rebalance_day 모드:
        - 포트폴리오에서 거래가 1건이라도 있으면 min_fee 1회만

        기대값:
        - Day 0: 0
        - Day 1: 1 (거래 있음)
        - Day 2: 1 (거래 있음)
        - Day 3: 1 (거래 있음)
        - Day 4: 0
        """
        nav, turnover_L1, delta_w = two_asset_plus_cash_data

        config = MinFeeCostConfig(
            min_fee_cash=1.0,
            charging_mode="per_rebalance_day",
            eps_trade=1e-12
        )
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        expected = pd.Series([0, 1, 1, 1, 0], index=nav.index, dtype=float)
        np.testing.assert_array_equal(
            result.cost_cash.values,
            expected.values,
            err_msg="per_rebalance_day mode failed"
        )

    def test_cash_excluded(self, two_asset_plus_cash_data):
        """CASH 컬럼이 비용 계산에서 제외되는지 확인"""
        nav, turnover_L1, delta_w = two_asset_plus_cash_data

        # CASH에만 거래가 있는 날은 없어야 함
        # Day 1: BTC +0.05, CASH -0.05 -> BTC만 count
        config = MinFeeCostConfig(
            min_fee_cash=1.0,
            charging_mode="per_asset_day"
        )
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        # CASH가 포함되었다면 Day 1에 cost=2가 됨
        # CASH 제외 시 Day 1에 cost=1
        assert result.cost_cash.iloc[1] == 1.0, "CASH should be excluded"

    def test_eps_trade_threshold(self):
        """eps_trade 임계값 테스트"""
        dates = pd.date_range("2024-01-01", periods=3, freq="D")
        nav = pd.Series([10000, 10100, 10050], index=dates, dtype=float)
        turnover_L1 = pd.Series([0.0, 0.001, 0.0], index=dates, dtype=float)

        # 매우 작은 거래량
        delta_w = pd.DataFrame({
            "BTC": [0.0, 1e-10, 0.0],  # eps_trade보다 작음
            "ETH": [0.0, 0.001, 0.0],  # eps_trade보다 큼
        }, index=dates)

        config = MinFeeCostConfig(
            min_fee_cash=1.0,
            charging_mode="per_asset_day",
            eps_trade=1e-8  # BTC는 이보다 작으므로 거래 없음 처리
        )
        model = MinFeeCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        # Day 1: ETH만 거래로 인식 (BTC는 eps_trade 미만)
        expected = pd.Series([0, 1, 0], index=nav.index, dtype=float)
        np.testing.assert_array_equal(result.cost_cash.values, expected.values)


class TestMinFeeCostModelValidation:
    """MinFeeCostModel 설정 검증 테스트"""

    def test_invalid_charging_mode_raises(self):
        """잘못된 charging_mode는 오류 발생"""
        with pytest.raises(ValueError, match="Invalid charging_mode"):
            MinFeeCostModel(MinFeeCostConfig(charging_mode="invalid_mode"))

    def test_negative_eps_trade_raises(self):
        """음수 eps_trade는 오류 발생"""
        with pytest.raises(ValueError, match="eps_trade must be non-negative"):
            MinFeeCostModel(MinFeeCostConfig(eps_trade=-1.0))

    def test_negative_min_fee_raises(self):
        """음수 min_fee는 오류 발생"""
        with pytest.raises(ValueError, match="min_fee_cash must be non-negative"):
            MinFeeCostModel(MinFeeCostConfig(min_fee_cash=-1.0))


# ==============================================================================
# RoundingCostModel 테스트
# ==============================================================================

class TestRoundingCostModel:
    """라운딩 비용 모델 테스트"""

    def test_basic_rounding_cost(self, sample_data):
        """기본 라운딩 비용 계산"""
        nav, turnover_L1, delta_w = sample_data

        config = RoundingCostConfig(avg_rounding_pct=0.0001)  # 1bps
        model = RoundingCostModel(config)
        result = model.compute(nav, turnover_L1, delta_w)

        # cost_ratio = turnover * 0.0001
        expected_ratio = turnover_L1 * 0.0001
        np.testing.assert_allclose(
            result.cost_ratio.values,
            expected_ratio.values,
            rtol=1e-10
        )

    def test_zero_turnover_no_cost(self, zero_turnover_data):
        """턴오버 0일 때 라운딩 비용 0"""
        nav, turnover_L1, delta_w = zero_turnover_data

        model = RoundingCostModel()
        result = model.compute(nav, turnover_L1, delta_w)

        assert result.total_cost_cash() == 0.0


# ==============================================================================
# CombinedCostModel 테스트
# ==============================================================================

class TestCombinedCostModel:
    """합성 비용 모델 테스트"""

    def test_combined_sum_correct(self, sample_data):
        """CASE: combined 모델 합산이 정확"""
        nav, turnover_L1, delta_w = sample_data

        # 개별 모델 생성
        proportional = ProportionalCostModel(ProportionalCostConfig(one_way_rate_bps=5.0))
        min_fee = MinFeeCostModel(MinFeeCostConfig(min_fee_cash=1.0, charging_mode="per_asset_day"))
        rounding = RoundingCostModel(RoundingCostConfig(avg_rounding_pct=0.0001))

        # 개별 계산
        r1 = proportional.compute(nav, turnover_L1, delta_w)
        r2 = min_fee.compute(nav, turnover_L1, delta_w)
        r3 = rounding.compute(nav, turnover_L1, delta_w)

        expected_total_cash = r1.cost_cash + r2.cost_cash + r3.cost_cash
        expected_total_ratio = r1.cost_ratio + r2.cost_ratio + r3.cost_ratio

        # 합성 모델
        combined = CombinedCostModel([proportional, min_fee, rounding])
        result = combined.compute(nav, turnover_L1, delta_w)

        # 검증
        np.testing.assert_allclose(
            result.total_cost_cash.values,
            expected_total_cash.values,
            rtol=1e-10
        )
        np.testing.assert_allclose(
            result.total_cost_ratio.values,
            expected_total_ratio.values,
            rtol=1e-10
        )

        # 개별 결과 확인
        assert len(result.individual_results) == 3

    def test_empty_models_raises_error(self):
        """빈 모델 리스트는 오류"""
        with pytest.raises(ValueError):
            CombinedCostModel([])

    def test_breakdown_by_model(self, sample_data):
        """모델별 비용 분해"""
        nav, turnover_L1, delta_w = sample_data

        proportional = ProportionalCostModel()
        min_fee = MinFeeCostModel(MinFeeCostConfig(min_fee_cash=2.0))

        combined = CombinedCostModel([proportional, min_fee])
        result = combined.compute(nav, turnover_L1, delta_w)

        breakdown = result.breakdown_by_model()
        assert "proportional" in breakdown
        assert "min_fee" in breakdown


# ==============================================================================
# 엣지 케이스 테스트
# ==============================================================================

class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_single_day(self):
        """단일 날짜 데이터"""
        dates = pd.date_range("2024-01-01", periods=1, freq="D")
        nav = pd.Series([10000], index=dates)
        turnover = pd.Series([0.1], index=dates)
        delta_w = pd.DataFrame({"BTC": [0.1]}, index=dates)

        model = ProportionalCostModel()
        result = model.compute(nav, turnover, delta_w)

        assert len(result.cost_cash) == 1
        assert result.cost_cash.iloc[0] == nav.iloc[0] * turnover.iloc[0] * 0.0005

    def test_very_small_turnover(self, sample_data):
        """매우 작은 턴오버"""
        nav, _, delta_w = sample_data
        turnover = pd.Series([1e-15, 1e-15, 1e-15, 1e-15, 1e-15], index=nav.index)

        # delta_w도 매우 작게 조정
        delta_w_small = delta_w * 1e-15

        model = MinFeeCostModel(MinFeeCostConfig(
            min_fee_cash=1.0,
            charging_mode="per_asset_day",
            eps_trade=1e-10  # 임계값
        ))
        result = model.compute(nav, turnover, delta_w_small)

        # 임계값보다 작으므로 거래 없음으로 처리
        assert result.total_cost_cash() == 0.0


# ==============================================================================
# 메인 실행
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
