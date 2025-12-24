"""
tests/test_phase2_integration_smoke.py - Phase 2 통합 스모크 테스트

synthetic fixture를 이용해 Phase 2 실행 후 산출물 스키마/컬럼 존재를 확인합니다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
import pytest

from phase1_anchor_engine import Phase1AnchorEngine, Phase1Config, Phase1Result
from adapters.run_phase2_from_loader import (
    Phase2Runner,
    Phase2Config,
    Phase2Result,
    run_phase2_from_phase1_result,
)
from costs import (
    ProportionalCostModel,
    MinFeeCostModel,
    RoundingCostModel,
    CombinedCostModel,
)


# ==============================================================================
# Synthetic Fixture
# ==============================================================================

@pytest.fixture
def synthetic_price_data():
    """
    합성 가격 데이터 생성

    Returns:
        (price_data, btc_close, eth_close)
    """
    np.random.seed(42)

    # 100일 데이터
    dates = pd.date_range("2024-01-01", periods=100, freq="D")
    n_days = len(dates)

    def generate_ohlcv(base_price: float, volatility: float = 0.02) -> pd.DataFrame:
        """OHLCV 데이터 생성"""
        returns = np.random.normal(0, volatility, n_days)
        close = base_price * np.exp(np.cumsum(returns))

        # High/Low: close 기준 ±1%
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

    return price_data, btc_close, eth_close


@pytest.fixture
def phase1_result(synthetic_price_data):
    """Phase 1 결과 생성"""
    price_data, btc_close, eth_close = synthetic_price_data

    config = Phase1Config(
        min_history_days=30,
        one_way_rate_bps=5.0,
    )
    engine = Phase1AnchorEngine(config)
    return engine.run(price_data, btc_close, eth_close)


@pytest.fixture
def phase2_result(phase1_result):
    """Phase 2 결과 생성"""
    config = Phase2Config(
        enable_min_fee=True,
        enable_rounding=True,
        min_fee_cash=1.0,
        rounding_pct=0.0001,
        save_outputs=False,  # 테스트에서는 저장 안 함
    )
    return run_phase2_from_phase1_result(phase1_result, config)


# ==============================================================================
# Phase 1 스모크 테스트
# ==============================================================================

class TestPhase1Smoke:
    """Phase 1 기본 동작 테스트"""

    def test_phase1_result_structure(self, phase1_result: Phase1Result):
        """Phase 1 결과 구조 확인"""
        assert phase1_result is not None
        assert phase1_result.schema_version == "1.5.0"

        # 타임시리즈 확인
        ts = phase1_result.timeseries
        assert len(ts) > 0

        # 필수 컬럼
        required_cols = [
            "nav_pre_open",
            "nav_post_open",
            "nav_post_open_phase1",
            "turnover_L1",
            "cost_ratio_phase1",
            "cost_cash_phase1",
            "gross_leverage",
            "regime_mult",
            "daily_return_gross",
            "daily_return_net",
        ]
        for col in required_cols:
            assert col in ts.columns, f"Missing column: {col}"

    def test_phase1_nav_positive(self, phase1_result: Phase1Result):
        """NAV가 항상 양수"""
        nav = phase1_result.timeseries["nav_post_open"]
        assert (nav > 0).all()

    def test_phase1_weights_sum_valid(self, phase1_result: Phase1Result):
        """비중 합이 레버리지 제한 이내"""
        w_exec = phase1_result.w_exec
        gross = w_exec.abs().sum(axis=1)

        # 초기화 기간 이후에는 레버리지 제한 확인
        assert (gross <= 1.0 + 1e-6).all()

    def test_phase1_summary_keys(self, phase1_result: Phase1Result):
        """Summary에 필요한 키 존재"""
        summary = phase1_result.summary
        required_keys = [
            "initial_nav",
            "final_nav_phase1",
            "total_return",
            "cagr",
            "ann_volatility",
            "sharpe_ratio",
            "max_drawdown",
            "total_turnover_L1",
            "total_cost_cash_phase1",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"


# ==============================================================================
# Phase 2 스모크 테스트
# ==============================================================================

class TestPhase2Smoke:
    """Phase 2 기본 동작 테스트"""

    def test_phase2_result_structure(self, phase2_result: Phase2Result):
        """Phase 2 결과 구조 확인"""
        assert phase2_result is not None
        assert phase2_result.schema_version == "2.6.0"

        # Phase 1 결과 포함
        assert phase2_result.phase1_result is not None

    def test_phase2_timeseries_columns(self, phase2_result: Phase2Result):
        """Phase 2 타임시리즈 컬럼 확인"""
        ts = phase2_result.timeseries

        # Phase 1 컬럼
        assert "nav_post_open_phase1" in ts.columns
        assert "cost_cash_phase1" in ts.columns

        # Phase 2 추가 컬럼
        required_phase2_cols = [
            "nav_post_open_phase2",
            "extra_cost_cash",
            "extra_cost_ratio",
            "cost_cash_phase2",
            "cost_ratio_phase2",
            "daily_return_phase2",
        ]
        for col in required_phase2_cols:
            assert col in ts.columns, f"Missing Phase 2 column: {col}"

    def test_phase2_nav_less_than_phase1(self, phase2_result: Phase2Result):
        """Phase 2 NAV가 Phase 1 이하 (추가 비용으로 인해)"""
        ts = phase2_result.timeseries
        nav_p1 = ts["nav_post_open_phase1"]
        nav_p2 = ts["nav_post_open_phase2"]

        # 마지막 값 기준
        assert nav_p2.iloc[-1] <= nav_p1.iloc[-1]

    def test_phase2_extra_cost_nonnegative(self, phase2_result: Phase2Result):
        """추가 비용은 항상 0 이상"""
        extra_cost = phase2_result.timeseries["extra_cost_cash"]
        assert (extra_cost >= 0).all()

    def test_phase2_summary_keys(self, phase2_result: Phase2Result):
        """Summary에 필요한 키 존재"""
        summary = phase2_result.summary
        required_keys = [
            "final_nav_phase1",
            "final_nav_phase2",
            "total_cost_cash_phase1",
            "total_extra_cost_cash",
            "total_cost_cash_phase2",
            "cost_breakdown_by_model",
            "nav_diff_phase1_vs_phase2",
        ]
        for key in required_keys:
            assert key in summary, f"Missing key: {key}"

    def test_phase2_cost_breakdown(self, phase2_result: Phase2Result):
        """비용 분해 확인"""
        breakdown = phase2_result.summary["cost_breakdown_by_model"]

        # min_fee와 rounding이 활성화되어 있으므로
        assert "min_fee" in breakdown
        assert "rounding" in breakdown

        # 비용이 0 이상
        for model, cost in breakdown.items():
            assert cost >= 0, f"{model} has negative cost"

    def test_phase2_manifest_structure(self, phase2_result: Phase2Result):
        """Manifest 구조 확인"""
        manifest = phase2_result.manifest
        assert manifest["schema_version"] == "2.6.0"
        assert "cost_models" in manifest
        assert "cost_model_params" in manifest
        assert "phase1_config" in manifest

        # 비용 모델 목록
        assert "proportional" in manifest["cost_models"]  # Phase 1 기본
        assert "min_fee" in manifest["cost_models"]
        assert "rounding" in manifest["cost_models"]


# ==============================================================================
# 비용 일관성 테스트
# ==============================================================================

class TestCostConsistency:
    """비용 계산 일관성 테스트"""

    def test_total_cost_phase2_equals_sum(self, phase2_result: Phase2Result):
        """Phase 2 총비용 = Phase 1 비용 + 추가 비용"""
        ts = phase2_result.timeseries

        total_p1 = ts["cost_cash_phase1"].sum()
        total_extra = ts["extra_cost_cash"].sum()
        total_p2 = ts["cost_cash_phase2"].sum()

        np.testing.assert_allclose(total_p2, total_p1 + total_extra, rtol=1e-10)

    def test_breakdown_sums_to_extra(self, phase2_result: Phase2Result):
        """개별 비용 합 = 추가 비용 합"""
        breakdown = phase2_result.summary["cost_breakdown_by_model"]
        total_extra = phase2_result.summary["total_extra_cost_cash"]

        sum_breakdown = sum(breakdown.values())
        np.testing.assert_allclose(sum_breakdown, total_extra, rtol=1e-10)


# ==============================================================================
# 출력 저장 테스트
# ==============================================================================

class TestOutputSaving:
    """출력 파일 저장 테스트"""

    def test_save_outputs(self, phase1_result, tmp_path):
        """출력 파일 저장 테스트"""
        config = Phase2Config(
            enable_min_fee=True,
            enable_rounding=True,
            output_dir=str(tmp_path),
            save_outputs=True,
        )

        result = run_phase2_from_phase1_result(phase1_result, config)

        # 파일 존재 확인
        assert (tmp_path / "real_run_manifest.json").exists()
        assert (tmp_path / "real_summary.json").exists()
        assert (tmp_path / "real_timeseries.csv").exists()
        assert (tmp_path / "real_w_exec.csv").exists()

        # Manifest 읽기 테스트
        with open(tmp_path / "real_run_manifest.json", "r") as f:
            manifest = json.load(f)
        assert manifest["schema_version"] == "2.6.0"

        # Summary 읽기 테스트
        with open(tmp_path / "real_summary.json", "r") as f:
            summary = json.load(f)
        assert "final_nav_phase2" in summary

        # Timeseries 읽기 테스트
        ts = pd.read_csv(tmp_path / "real_timeseries.csv", index_col=0)
        assert "nav_post_open_phase2" in ts.columns


# ==============================================================================
# 메인 실행
# ==============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
