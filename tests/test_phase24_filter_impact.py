"""
tests/test_phase24_filter_impact.py - Phase 2.4 필터 영향 테스트

Phase 2.4의 핵심 기능을 테스트합니다:
  1) trade_notional 기준 필터가 정확히 작동하는지 (handcalc)
  2) 불변량: Σw=1, P2 NAV <= P1 NAV, cost_sum 일관성
  3) FilterImpactMetrics 계산 정확성
  4) w_exec_filtered 생성 확인
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from phase1_anchor_engine import Phase1AnchorEngine, Phase1Config, Phase1Result
from adapters.run_phase2_from_loader import (
    Phase2Runner,
    Phase2Config,
    Phase2Result,
    PHASE24_DEFAULT_CONFIG,
    run_phase2_from_phase1_result,
)
from execution import TradeEventBuilder, TradeEventConfig, FilterImpactMetrics


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture
def synthetic_phase1_result():
    """
    합성 Phase 1 결과 생성 (테스트용)
    """
    np.random.seed(42)

    # 100일 데이터
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

    # 5개 심볼 생성
    price_data = {
        "BTC": generate_ohlcv(40000, 0.03),
        "ETH": generate_ohlcv(2500, 0.04),
        "SOL": generate_ohlcv(100, 0.05),
        "DOGE": generate_ohlcv(0.1, 0.06),
        "XRP": generate_ohlcv(0.5, 0.04),
    }

    btc_close = price_data["BTC"]["close"]
    eth_close = price_data["ETH"]["close"]

    config = Phase1Config(
        min_history_days=30,
        one_way_rate_bps=5.0,
    )
    engine = Phase1AnchorEngine(config)
    return engine.run(price_data, btc_close, eth_close)


@pytest.fixture
def phase24_config():
    """
    Phase 2.4 기본 설정
    """
    return Phase2Config(
        enable_min_fee=True,
        enable_rounding=True,
        min_fee_cash=1.0,
        min_fee_charging_mode="per_rebalance_day",
        min_notional_cash=100.0,
        enable_netting=False,
        save_outputs=False,
    )


# ==============================================================================
# Handcalc Tests: trade_notional 기준 필터 정확성
# ==============================================================================

class TestTradeNotionalFilterHandcalc:
    """trade_notional 기준 필터의 정확성을 수작업 계산으로 검증"""

    def test_filter_clips_delta_w_correctly(self):
        """
        Handcalc 1: trade_notional < min_notional이면 delta_w가 0으로 클립되는지 확인

        설정:
          - NAV = 10,000 USDT
          - delta_w = [0.01, 0.005, 0.02] for [BTC, ETH, SOL]
          - min_notional_cash = 100 USDT

        계산:
          - BTC: 10,000 * 0.01 = 100 >= 100 → 유지
          - ETH: 10,000 * 0.005 = 50 < 100 → 클립
          - SOL: 10,000 * 0.02 = 200 >= 100 → 유지
        """
        dates = pd.date_range("2024-01-01", periods=3, freq="D")

        # 간단한 delta_w
        delta_w = pd.DataFrame({
            "BTC": [0.01, 0.0, 0.0],
            "ETH": [0.005, 0.0, 0.0],
            "SOL": [0.02, 0.0, 0.0],
            "CASH": [0.0, 0.0, 0.0],
        }, index=dates)

        nav = pd.Series([10000.0, 10000.0, 10000.0], index=dates)

        # w_exec_raw (시작 상태)
        w_exec_raw = pd.DataFrame({
            "BTC": [0.3, 0.31, 0.31],
            "ETH": [0.2, 0.205, 0.205],
            "SOL": [0.1, 0.12, 0.12],
            "CASH": [0.4, 0.365, 0.365],
        }, index=dates)

        config = TradeEventConfig(
            eps_trade=1e-12,
            min_notional_cash=100.0,
            enable_netting=False,
        )
        builder = TradeEventBuilder(config)
        result = builder.build(delta_w, nav, w_exec_raw=w_exec_raw)

        # 첫째 날 검증
        filtered_delta = result.delta_w_filtered

        # BTC: 0.01 유지 (100 USDT >= 100)
        assert abs(filtered_delta.loc[dates[0], "BTC"] - 0.01) < 1e-10

        # ETH: 0으로 클립 (50 USDT < 100)
        assert abs(filtered_delta.loc[dates[0], "ETH"]) < 1e-10

        # SOL: 0.02 유지 (200 USDT >= 100)
        assert abs(filtered_delta.loc[dates[0], "SOL"] - 0.02) < 1e-10

    def test_filter_boundary_case(self):
        """
        Handcalc 2: 경계값 테스트 (정확히 min_notional인 경우)

        설정:
          - NAV = 10,000 USDT
          - delta_w = 0.01 (BTC)
          - min_notional_cash = 100 USDT (정확히 경계)

        계산:
          - BTC: 10,000 * 0.01 = 100 == 100 → 유지 (>= 조건)
        """
        dates = pd.date_range("2024-01-01", periods=1, freq="D")

        delta_w = pd.DataFrame({
            "BTC": [0.01],
            "CASH": [0.0],
        }, index=dates)

        nav = pd.Series([10000.0], index=dates)

        w_exec_raw = pd.DataFrame({
            "BTC": [0.5],
            "CASH": [0.5],
        }, index=dates)

        config = TradeEventConfig(
            eps_trade=1e-12,
            min_notional_cash=100.0,
        )
        builder = TradeEventBuilder(config)
        result = builder.build(delta_w, nav, w_exec_raw=w_exec_raw)

        # 정확히 100 USDT → 유지
        assert abs(result.delta_w_filtered.loc[dates[0], "BTC"] - 0.01) < 1e-10

        # 경계 바로 아래 (100.0 + epsilon)
        config2 = TradeEventConfig(
            eps_trade=1e-12,
            min_notional_cash=100.01,  # 바로 위
        )
        builder2 = TradeEventBuilder(config2)
        result2 = builder2.build(delta_w, nav, w_exec_raw=w_exec_raw)

        # 100 < 100.01 → 클립
        assert abs(result2.delta_w_filtered.loc[dates[0], "BTC"]) < 1e-10


# ==============================================================================
# Invariant Tests: 불변량 검증
# ==============================================================================

class TestInvariants:
    """Phase 2.4 불변량 테스트"""

    def test_w_exec_sum_equals_one(self, synthetic_phase1_result, phase24_config):
        """
        불변량 1: w_exec_filtered의 행합은 항상 1
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        if result.w_exec_filtered is not None and not result.w_exec_filtered.empty:
            w_sum = result.w_exec_filtered.sum(axis=1)
            np.testing.assert_allclose(w_sum, 1.0, rtol=1e-6)

    def test_phase2_nav_leq_phase1_nav(self, synthetic_phase1_result, phase24_config):
        """
        불변량 2: Phase 2 NAV <= Phase 1 NAV (모든 시점)
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        ts = result.timeseries
        nav_p1 = ts["nav_post_open_phase1"]
        nav_p2 = ts["nav_post_open_phase2"]

        # 모든 시점에서 P2 <= P1
        assert (nav_p2 <= nav_p1 + 1e-6).all()

    def test_cost_sum_consistency(self, synthetic_phase1_result, phase24_config):
        """
        불변량 3: 비용 합계 일관성

        total_cost_phase2 = total_cost_phase1 + sum(extra_costs)
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        ts = result.timeseries

        total_p1 = ts["cost_cash_phase1"].sum()
        total_extra = ts["extra_cost_cash"].sum()
        total_p2 = ts["cost_cash_phase2"].sum()

        np.testing.assert_allclose(total_p2, total_p1 + total_extra, rtol=1e-10)

    def test_cost_breakdown_consistency(self, synthetic_phase1_result, phase24_config):
        """
        불변량 4: 비용 분해 합 = extra_cost 합
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        breakdown = result.summary["cost_breakdown_by_model"]
        total_extra = result.summary["total_extra_cost_cash"]

        sum_breakdown = sum(breakdown.values())
        np.testing.assert_allclose(sum_breakdown, total_extra, rtol=1e-10)


# ==============================================================================
# FilterImpactMetrics Tests
# ==============================================================================

class TestFilterImpactMetrics:
    """FilterImpactMetrics 계산 정확성 테스트"""

    def test_filter_impact_exists(self, synthetic_phase1_result, phase24_config):
        """
        filter_impact이 결과에 존재하는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        assert result.filter_impact is not None
        assert isinstance(result.filter_impact, FilterImpactMetrics)

    def test_filter_impact_in_summary(self, synthetic_phase1_result, phase24_config):
        """
        filter_impact이 summary에 포함되는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        assert "filter_impact" in result.summary
        fi = result.summary["filter_impact"]

        assert "pre_filter_turnover_L1" in fi
        assert "post_filter_turnover_L1" in fi
        assert "l1_distance_mean" in fi
        assert "l1_distance_max" in fi
        assert "filter_ratio" in fi

    def test_filter_impact_turnover_reduction(self, synthetic_phase1_result, phase24_config):
        """
        필터링 후 turnover가 감소하는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        fi = result.filter_impact

        # post <= pre (필터링하면 turnover 감소 또는 동일)
        assert fi.post_filter_turnover_L1 <= fi.pre_filter_turnover_L1 + 1e-10

    def test_filter_impact_l1_distance_nonnegative(self, synthetic_phase1_result, phase24_config):
        """
        L1 distance는 0 이상
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        fi = result.filter_impact

        assert fi.l1_distance_mean >= 0
        assert fi.l1_distance_max >= 0
        assert fi.l1_distance_max >= fi.l1_distance_mean

    def test_filter_ratio_range(self, synthetic_phase1_result, phase24_config):
        """
        filter_ratio는 [0, 1] 범위
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        fi = result.filter_impact

        assert 0.0 <= fi.filter_ratio <= 1.0

    def test_high_min_notional_increases_filter_ratio(self, synthetic_phase1_result):
        """
        높은 min_notional_cash는 filter_ratio를 증가시킴
        """
        # 낮은 임계값
        config_low = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=10.0,
            save_outputs=False,
        )
        result_low = run_phase2_from_phase1_result(synthetic_phase1_result, config_low)

        # 높은 임계값
        config_high = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=1000.0,
            save_outputs=False,
        )
        result_high = run_phase2_from_phase1_result(synthetic_phase1_result, config_high)

        # 높은 임계값 → 더 많은 필터링
        fi_low = result_low.filter_impact
        fi_high = result_high.filter_impact

        assert fi_high.filter_ratio >= fi_low.filter_ratio


# ==============================================================================
# w_exec_filtered Tests
# ==============================================================================

class TestWExecFiltered:
    """w_exec_filtered 생성 및 정확성 테스트"""

    def test_w_exec_filtered_exists(self, synthetic_phase1_result, phase24_config):
        """
        w_exec_filtered이 결과에 존재하는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        assert result.w_exec_filtered is not None
        assert isinstance(result.w_exec_filtered, pd.DataFrame)

    def test_w_exec_filtered_has_cash(self, synthetic_phase1_result, phase24_config):
        """
        w_exec_filtered에 CASH 컬럼이 존재하는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        if not result.w_exec_filtered.empty:
            assert "CASH" in result.w_exec_filtered.columns

    def test_w_exec_filtered_same_index(self, synthetic_phase1_result, phase24_config):
        """
        w_exec_filtered의 인덱스가 w_exec와 동일한지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        if not result.w_exec_filtered.empty:
            w_exec_orig = result.phase1_result.w_exec
            assert len(result.w_exec_filtered) == len(w_exec_orig)


# ==============================================================================
# PHASE24_DEFAULT_CONFIG Tests
# ==============================================================================

class TestPhase24DefaultConfig:
    """Phase 2.4 기본 설정 테스트"""

    def test_default_config_exists(self):
        """
        PHASE24_DEFAULT_CONFIG 상수 존재 확인
        """
        assert PHASE24_DEFAULT_CONFIG is not None
        assert isinstance(PHASE24_DEFAULT_CONFIG, dict)

    def test_default_config_values(self):
        """
        Phase 2.4 기본값이 올바른지 확인
        """
        assert PHASE24_DEFAULT_CONFIG["charging_mode"] == "per_rebalance_day"
        assert PHASE24_DEFAULT_CONFIG["eps_trade"] == 1e-12
        assert PHASE24_DEFAULT_CONFIG["min_notional_cash"] == 100.0
        assert PHASE24_DEFAULT_CONFIG["enable_netting"] == False
        assert PHASE24_DEFAULT_CONFIG["min_fee_cash"] == 1.0
        assert PHASE24_DEFAULT_CONFIG["quote_ccy"] == "USDT"

    def test_filter_impact_in_manifest(self, synthetic_phase1_result, phase24_config):
        """
        filter_impact_summary가 manifest에 포함되는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, phase24_config)

        assert "filter_impact_summary" in result.manifest
        fi = result.manifest["filter_impact_summary"]

        assert "pre_filter_turnover_L1" in fi
        assert "post_filter_turnover_L1" in fi


# ==============================================================================
# Output File Tests
# ==============================================================================

class TestPhase24OutputFiles:
    """Phase 2.4 출력 파일 테스트"""

    def test_filter_impact_json_saved(self, synthetic_phase1_result, tmp_path):
        """
        phase24_filter_impact_summary.json 파일이 생성되는지 확인
        """
        import json

        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=100.0,
            output_dir=str(tmp_path),
            save_outputs=True,
        )

        result = run_phase2_from_phase1_result(synthetic_phase1_result, config)

        filter_impact_path = tmp_path / "phase24_filter_impact_summary.json"
        assert filter_impact_path.exists()

        with open(filter_impact_path, "r") as f:
            fi = json.load(f)

        assert "pre_filter_turnover_L1" in fi
        assert "l1_distance_mean" in fi

    def test_w_exec_filtered_csv_saved(self, synthetic_phase1_result, tmp_path):
        """
        real_w_exec_filtered.csv 파일이 생성되는지 확인
        """
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            min_notional_cash=100.0,
            output_dir=str(tmp_path),
            save_outputs=True,
        )

        result = run_phase2_from_phase1_result(synthetic_phase1_result, config)

        w_exec_filtered_path = tmp_path / "real_w_exec_filtered.csv"
        assert w_exec_filtered_path.exists()

        df = pd.read_csv(w_exec_filtered_path, index_col=0)
        assert len(df) > 0


# ==============================================================================
# Netting Tests (Phase 2.4)
# ==============================================================================

class TestNettingPhase24:
    """Phase 2.4 Netting 기능 테스트"""

    def test_netting_reduces_events(self, synthetic_phase1_result):
        """
        netting 활성화 시 이벤트 수가 감소하는지 확인
        """
        # Netting 비활성화
        config_no_netting = Phase2Config(
            enable_min_fee=True,
            enable_rounding=False,
            min_fee_charging_mode="per_asset_day",
            min_notional_cash=0.0,
            enable_netting=False,
            save_outputs=False,
            guardrail_nav_collapse=False,
        )
        result_no_netting = run_phase2_from_phase1_result(
            synthetic_phase1_result, config_no_netting
        )

        # Netting 활성화
        config_with_netting = Phase2Config(
            enable_min_fee=True,
            enable_rounding=False,
            min_fee_charging_mode="per_asset_day",
            min_notional_cash=0.0,
            enable_netting=True,
            save_outputs=False,
            guardrail_nav_collapse=False,
        )
        result_with_netting = run_phase2_from_phase1_result(
            synthetic_phase1_result, config_with_netting
        )

        # Netting으로 이벤트 수가 감소하거나 동일해야 함
        events_no_netting = result_no_netting.event_stats.get("min_fee_event_count", 0)
        events_with_netting = result_with_netting.event_stats.get("min_fee_event_count", 0)

        assert events_with_netting <= events_no_netting


# ==============================================================================
# 실행
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
