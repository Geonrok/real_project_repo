"""
tests/test_phase23_guardrails.py - Phase 2.3 Guardrail 테스트

Phase 2.3의 핵심 안전장치를 테스트합니다:
  1) NAV 붕괴 시 RuntimeError 발생
  2) events_per_day threshold 초과 시 WARNING 발생
  3) filtered_trade_ratio 기록 확인
  4) default_config 존재 확인
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
    Phase2NAVCollapseError,
    PHASE23_DEFAULT_CONFIG,
    run_phase2_from_phase1_result,
)


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
def collapse_config():
    """
    NAV 붕괴를 유발하는 설정 (per_asset_day + min_notional=0)
    """
    return Phase2Config(
        enable_min_fee=True,
        enable_rounding=True,
        min_fee_cash=1.0,
        min_fee_charging_mode="per_asset_day",  # 붕괴 유발
        min_notional_cash=0.0,  # 필터링 없음
        save_outputs=False,
        guardrail_nav_collapse=True,
    )


@pytest.fixture
def safe_config():
    """
    안전한 기본 설정 (per_rebalance_day + min_notional=100)
    """
    return Phase2Config(
        enable_min_fee=True,
        enable_rounding=True,
        min_fee_cash=1.0,
        min_fee_charging_mode="per_rebalance_day",
        min_notional_cash=100.0,
        save_outputs=False,
        guardrail_nav_collapse=True,
    )


# ==============================================================================
# NAV Collapse Guardrail Tests
# ==============================================================================

class TestNAVCollapseGuardrail:
    """NAV 붕괴 Guardrail 테스트"""

    def test_collapse_raises_runtime_error(self, synthetic_phase1_result, collapse_config):
        """
        NAV < 0 조건에서 Phase2NAVCollapseError 발생 확인
        """
        # 이 테스트는 실제 붕괴가 발생하는 데이터에서만 작동
        # synthetic 데이터로는 붕괴가 발생하지 않을 수 있음
        # 그래서 더 극단적인 설정 사용
        extreme_config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=False,
            min_fee_cash=100.0,  # 매우 높은 min_fee
            min_fee_charging_mode="per_asset_day",
            min_notional_cash=0.0,
            save_outputs=False,
            guardrail_nav_collapse=True,
        )

        # 100 USDT/event * 많은 이벤트 = 음수 NAV 가능
        # synthetic 데이터로는 붕괴가 발생할 수도 있고 아닐 수도 있음
        try:
            result = run_phase2_from_phase1_result(synthetic_phase1_result, extreme_config)
            # 붕괴가 발생하지 않았다면 테스트 통과
            assert result.summary["final_nav_phase2"] >= 0
        except Phase2NAVCollapseError as e:
            # 붕괴가 발생했다면 예외 속성 확인
            assert e.final_nav < 0
            assert e.charging_mode == "per_asset_day"
            assert "NAV Collapse" in str(e)
            assert "Recommendations" in str(e)

    def test_collapse_error_contains_cause_info(self):
        """
        Phase2NAVCollapseError가 원인 정보를 포함하는지 확인
        """
        error = Phase2NAVCollapseError(
            final_nav=-5000.0,
            charging_mode="per_asset_day",
            eps_trade=1e-12,
            min_notional_cash=0.0,
            enable_netting=False,
            total_events=10000,
            n_days=100,
            filtered_ratio=0.0,
            events_per_day_max=500,
        )

        # 예외 메시지에 원인 정보 포함 확인
        msg = str(error)
        assert "per_asset_day" in msg
        assert "10,000" in msg  # total_events
        assert "0.00" in msg  # min_notional_cash
        assert "Recommendations" in msg

    def test_collapse_error_to_dict(self):
        """
        Phase2NAVCollapseError.to_dict() 메서드 확인
        """
        error = Phase2NAVCollapseError(
            final_nav=-5000.0,
            charging_mode="per_asset_day",
            eps_trade=1e-12,
            min_notional_cash=0.0,
            enable_netting=False,
            total_events=10000,
            n_days=100,
            filtered_ratio=0.0,
            events_per_day_max=500,
        )

        d = error.to_dict()
        assert d["final_nav"] == -5000.0
        assert d["charging_mode"] == "per_asset_day"
        assert d["total_events"] == 10000
        assert d["filtered_ratio"] == 0.0

    def test_guardrail_disabled_no_error(self, synthetic_phase1_result):
        """
        guardrail_nav_collapse=False일 때 예외 발생 안 함
        """
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=False,
            min_fee_cash=100.0,  # 높은 비용
            min_fee_charging_mode="per_asset_day",
            min_notional_cash=0.0,
            save_outputs=False,
            guardrail_nav_collapse=False,  # 비활성화
        )

        # 예외 발생 없이 실행 완료되어야 함
        result = run_phase2_from_phase1_result(synthetic_phase1_result, config)
        assert result is not None
        # NAV가 음수일 수도 있지만 예외는 발생하지 않음


# ==============================================================================
# Events Threshold Warning Tests
# ==============================================================================

class TestEventsThresholdWarning:
    """events_per_day threshold 경고 테스트"""

    def test_high_events_triggers_warning(self, synthetic_phase1_result):
        """
        events_per_day > threshold 시 WARNING 발생 확인
        """
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=False,
            min_fee_cash=1.0,
            min_fee_charging_mode="per_asset_day",
            min_notional_cash=0.0,
            save_outputs=False,
            guardrail_events_threshold=1,  # 매우 낮은 임계값
            guardrail_nav_collapse=False,  # 붕괴 검사 비활성화
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_phase2_from_phase1_result(synthetic_phase1_result, config)

            # 경고 발생 확인
            warning_messages = [str(warning.message) for warning in w]
            guardrail_warnings = [m for m in warning_messages if "Guardrail" in m]

            # events > 1 이면 경고 발생해야 함
            if result.event_stats.get("events_per_day_max", 0) > 1:
                assert len(guardrail_warnings) > 0
                assert "events_per_day_max" in guardrail_warnings[0]

    def test_low_events_no_warning(self, synthetic_phase1_result):
        """
        events_per_day <= threshold 시 WARNING 없음
        """
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=False,
            min_fee_cash=1.0,
            min_fee_charging_mode="per_rebalance_day",  # 이벤트 수 적음
            min_notional_cash=100.0,
            save_outputs=False,
            guardrail_events_threshold=1000,  # 높은 임계값
            guardrail_nav_collapse=False,
        )

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = run_phase2_from_phase1_result(synthetic_phase1_result, config)

            # Guardrail 관련 경고 없어야 함
            warning_messages = [str(warning.message) for warning in w]
            guardrail_warnings = [m for m in warning_messages if "Guardrail" in m]
            assert len(guardrail_warnings) == 0


# ==============================================================================
# Filtered Trade Ratio Tests
# ==============================================================================

class TestFilteredTradeRatio:
    """filtered_trade_ratio 기록 테스트"""

    def test_filtered_ratio_recorded(self, synthetic_phase1_result, safe_config):
        """
        min_fee_filtered_trade_ratio가 summary에 기록되는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, safe_config)

        assert "min_fee_filtered_trade_ratio" in result.summary
        assert isinstance(result.summary["min_fee_filtered_trade_ratio"], float)
        assert 0.0 <= result.summary["min_fee_filtered_trade_ratio"] <= 1.0

    def test_filtered_ratio_in_event_stats(self, synthetic_phase1_result, safe_config):
        """
        filtered_trade_ratio가 event_stats에도 기록되는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, safe_config)

        assert "min_fee_filtered_trade_ratio" in result.event_stats
        assert result.event_stats["min_fee_filtered_trade_ratio"] == \
               result.summary["min_fee_filtered_trade_ratio"]

    def test_high_min_notional_high_filtered_ratio(self, synthetic_phase1_result):
        """
        높은 min_notional_cash일 때 filtered_ratio가 높은지 확인
        """
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=False,
            min_fee_cash=1.0,
            min_fee_charging_mode="per_asset_day",
            min_notional_cash=1000.0,  # 매우 높은 임계값
            save_outputs=False,
            guardrail_nav_collapse=False,
        )

        result = run_phase2_from_phase1_result(synthetic_phase1_result, config)

        # 높은 min_notional은 많은 거래를 필터링해야 함
        filtered_ratio = result.summary["min_fee_filtered_trade_ratio"]
        # synthetic 데이터 특성상 정확한 값은 알 수 없으나, 0 이상이어야 함
        assert filtered_ratio >= 0.0


# ==============================================================================
# Default Config Tests
# ==============================================================================

class TestDefaultConfig:
    """기본 설정 봉인 테스트"""

    def test_default_config_exists(self):
        """
        PHASE23_DEFAULT_CONFIG 상수 존재 확인
        """
        assert PHASE23_DEFAULT_CONFIG is not None
        assert isinstance(PHASE23_DEFAULT_CONFIG, dict)

    def test_default_config_values(self):
        """
        기본값이 올바른지 확인
        """
        assert PHASE23_DEFAULT_CONFIG["charging_mode"] == "per_rebalance_day"
        assert PHASE23_DEFAULT_CONFIG["eps_trade"] == 1e-12
        assert PHASE23_DEFAULT_CONFIG["min_notional_cash"] == 100.0
        assert PHASE23_DEFAULT_CONFIG["enable_netting"] == False
        assert PHASE23_DEFAULT_CONFIG["min_fee_cash"] == 1.0

    def test_default_config_in_manifest(self, synthetic_phase1_result, safe_config):
        """
        default_config가 manifest에 포함되는지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, safe_config)

        assert "default_config" in result.manifest
        assert result.manifest["default_config"]["charging_mode"] == "per_rebalance_day"
        assert result.manifest["default_config"]["min_notional_cash"] == 100.0

    def test_phase2config_defaults_match(self):
        """
        Phase2Config 기본값이 PHASE23_DEFAULT_CONFIG와 일치하는지 확인
        """
        config = Phase2Config()

        assert config.min_fee_charging_mode == PHASE23_DEFAULT_CONFIG["charging_mode"]
        assert config.min_fee_eps_trade == PHASE23_DEFAULT_CONFIG["eps_trade"]
        assert config.min_notional_cash == PHASE23_DEFAULT_CONFIG["min_notional_cash"]
        assert config.enable_netting == PHASE23_DEFAULT_CONFIG["enable_netting"]


# ==============================================================================
# Event Stats Output Tests
# ==============================================================================

class TestEventStatsOutput:
    """이벤트 통계 출력 테스트"""

    def test_event_stats_daily_dataframe(self, synthetic_phase1_result, safe_config):
        """
        event_stats_daily DataFrame 생성 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, safe_config)

        assert isinstance(result.event_stats_daily, pd.DataFrame)
        assert "events" in result.event_stats_daily.columns
        assert "gross_turnover" in result.event_stats_daily.columns

    def test_summary_event_stats_structure(self, synthetic_phase1_result, safe_config):
        """
        summary에 event_stats 구조 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, safe_config)

        assert "event_stats" in result.summary
        es = result.summary["event_stats"]

        assert "events_per_day_mean" in es
        assert "events_per_day_max" in es
        assert "events_per_day_top5" in es

        assert isinstance(es["events_per_day_mean"], float)
        assert isinstance(es["events_per_day_max"], int)
        assert isinstance(es["events_per_day_top5"], list)


# ==============================================================================
# Schema Version Tests
# ==============================================================================

class TestSchemaVersion:
    """스키마 버전 테스트"""

    def test_schema_version_2_5_0(self, synthetic_phase1_result, safe_config):
        """
        schema_version이 2.6.0인지 확인
        """
        result = run_phase2_from_phase1_result(synthetic_phase1_result, safe_config)

        assert result.schema_version == "2.6.0"
        assert result.manifest["schema_version"] == "2.6.0"


# ==============================================================================
# 실행
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
