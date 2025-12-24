"""
examples/analyze_phase2_sensitivity.py - Phase 2.2 Sensitivity 분석

charging_mode, eps_trade, min_notional_cash, netting 조합에 따른
비용 민감도를 분석하여 최적의 기본값을 선택하는 데 도움을 줍니다.

Phase 2.2 변경사항:
- --min-notional: 최소 거래 금액 임계값
- --netting: 동일 자산 buy/sell 상쇄 활성화

출력: outputs/phase2_sensitivity_summary.csv

기본값 선택은 sensitivity 결과로 결정한다:
  - per_rebalance_day: 가장 보수적 (min_fee 1회/일)
  - per_asset_day + min_notional: 중간 (소액 거래 필터링)
  - per_asset_side_day: 가장 상세 (자산별 + buy/sell 구분)

Phase 2.1에서 per_asset_day 모드가 붕괴한 이유:
  - 모든 미세 Δw를 거래로 간주하여 이벤트 136,000+ 발생
  - Phase 2.2에서 min_notional_cash로 소액 거래 필터링하여 해결
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from itertools import product

import numpy as np
import pandas as pd

from phase1_anchor_engine import Phase1Config, Phase1Result
from adapters.run_phase2_from_loader import (
    Phase2Config,
    Phase2Result,
    run_phase2_from_loader,
    run_phase2_from_phase1_result,
)
from adapters.run_phase1_from_loader import run_phase1_from_loader


# ==============================================================================
# 설정
# ==============================================================================

BASE_DATA = r"E:\OneDrive\고형석\코인\data"
OUTPUT_DIR = r"C:\Users\고형석\outputs"

# 기본 분석 그리드
DEFAULT_CHARGING_MODES = [
    "per_asset_day",
    "per_asset_side_day",
    "per_rebalance_day",
]

DEFAULT_EPS_TRADE_GRID = [
    0.0,
    1e-12,
    1e-8,
    1e-6,
]

# Phase 2.2: min_notional 그리드
DEFAULT_MIN_NOTIONAL_GRID = [
    0.0,      # 비활성 (기존 동작)
    1.0,      # 1 USDT
    10.0,     # 10 USDT
    100.0,    # 100 USDT
]


@dataclass
class SensitivityRow:
    """Sensitivity 분석 결과 행 (Phase 2.2)"""
    mode: str
    eps_trade: float
    min_notional_cash: float
    enable_netting: bool
    final_nav_p1: float
    final_nav_p2: float
    total_cost_p1: float
    total_extra_cost: float
    total_cost_p2: float
    min_fee_cost: float
    rounding_cost: float
    min_fee_event_count: int
    days_with_min_fee: int
    avg_events_per_day: float
    # Phase 2.2 신규
    ignored_small_trades: int
    ignored_by_netting: int
    total_ignored: int


def run_sensitivity_single(
    phase1_result: Phase1Result,
    charging_mode: str,
    eps_trade: float,
    min_notional_cash: float = 0.0,
    enable_netting: bool = False,
    min_fee_cash: float = 1.0,
    rounding_pct: float = 0.0001,
    quote_ccy: str = "USDT",
) -> SensitivityRow:
    """
    단일 파라미터 조합으로 Phase 2 실행

    Args:
        phase1_result: Phase 1 결과 (Anchor)
        charging_mode: min_fee 부과 모드
        eps_trade: 거래 인식 임계값
        min_notional_cash: 최소 거래 금액 임계값 (Phase 2.2)
        enable_netting: 동일 자산 상쇄 (Phase 2.2)
        min_fee_cash: 최소 수수료 금액
        rounding_pct: 반올림 비용 비율
        quote_ccy: 기준 통화

    Returns:
        SensitivityRow
    """
    config = Phase2Config(
        enable_proportional=True,
        enable_min_fee=True,
        enable_rounding=True,
        min_fee_cash=min_fee_cash,
        min_fee_charging_mode=charging_mode,
        min_fee_eps_trade=eps_trade,
        min_notional_cash=min_notional_cash,
        enable_netting=enable_netting,
        quote_ccy=quote_ccy,
        rounding_pct=rounding_pct,
        save_outputs=False,
    )

    p2_result = run_phase2_from_phase1_result(phase1_result, config)
    summary = p2_result.summary
    breakdown = summary.get("cost_breakdown_by_model", {})

    # min_fee 진단 정보 추출
    min_fee_meta = {}
    if "min_fee" in p2_result.cost_breakdown:
        min_fee_meta = p2_result.cost_breakdown["min_fee"].metadata

    return SensitivityRow(
        mode=charging_mode,
        eps_trade=eps_trade,
        min_notional_cash=min_notional_cash,
        enable_netting=enable_netting,
        final_nav_p1=summary["final_nav_phase1"],
        final_nav_p2=summary["final_nav_phase2"],
        total_cost_p1=summary["total_cost_cash_phase1"],
        total_extra_cost=summary["total_extra_cost_cash"],
        total_cost_p2=summary["total_cost_cash_phase2"],
        min_fee_cost=breakdown.get("min_fee", 0.0),
        rounding_cost=breakdown.get("rounding", 0.0),
        min_fee_event_count=min_fee_meta.get("min_fee_event_count", 0),
        days_with_min_fee=min_fee_meta.get("days_with_min_fee", 0),
        avg_events_per_day=min_fee_meta.get("avg_events_per_day", 0.0),
        # Phase 2.2 신규
        ignored_small_trades=min_fee_meta.get("min_fee_ignored_small_trades", 0),
        ignored_by_netting=min_fee_meta.get("min_fee_ignored_by_netting", 0),
        total_ignored=min_fee_meta.get("min_fee_total_ignored", 0),
    )


def run_sensitivity_grid(
    phase1_result: Phase1Result,
    charging_modes: List[str] = None,
    eps_trade_grid: List[float] = None,
    min_notional_grid: List[float] = None,
    netting_options: List[bool] = None,
    min_fee_cash: float = 1.0,
    rounding_pct: float = 0.0001,
    quote_ccy: str = "USDT",
) -> pd.DataFrame:
    """
    전체 그리드에 대해 sensitivity 분석 실행

    Args:
        phase1_result: Phase 1 결과
        charging_modes: 분석할 charging_mode 목록
        eps_trade_grid: 분석할 eps_trade 목록
        min_notional_grid: 분석할 min_notional_cash 목록 (Phase 2.2)
        netting_options: netting 활성화 여부 목록 (Phase 2.2)
        min_fee_cash: 최소 수수료 금액
        rounding_pct: 반올림 비용 비율
        quote_ccy: 기준 통화

    Returns:
        pd.DataFrame: sensitivity 분석 결과
    """
    if charging_modes is None:
        charging_modes = DEFAULT_CHARGING_MODES
    if eps_trade_grid is None:
        eps_trade_grid = [1e-12]  # 기본값으로 단순화
    if min_notional_grid is None:
        min_notional_grid = [0.0]
    if netting_options is None:
        netting_options = [False]

    rows = []
    combinations = list(product(charging_modes, eps_trade_grid, min_notional_grid, netting_options))
    total = len(combinations)

    print(f"\nRunning sensitivity analysis: {total} combinations")
    print(f"  charging_modes: {charging_modes}")
    print(f"  eps_trade_grid: {eps_trade_grid}")
    print(f"  min_notional_grid: {min_notional_grid}")
    print(f"  netting_options: {netting_options}")
    print()

    for i, (mode, eps, notional, netting) in enumerate(combinations):
        net_str = "net" if netting else "no_net"
        print(f"  [{i+1}/{total}] mode={mode}, eps={eps:.0e}, notional={notional}, {net_str}...", end=" ")

        row = run_sensitivity_single(
            phase1_result,
            charging_mode=mode,
            eps_trade=eps,
            min_notional_cash=notional,
            enable_netting=netting,
            min_fee_cash=min_fee_cash,
            rounding_pct=rounding_pct,
            quote_ccy=quote_ccy,
        )
        rows.append(row)
        print(f"NAV_P2={row.final_nav_p2:.2f}, events={row.min_fee_event_count}, ignored={row.total_ignored}")

    # DataFrame 생성
    df = pd.DataFrame([
        {
            "mode": r.mode,
            "eps_trade": r.eps_trade,
            "min_notional_cash": r.min_notional_cash,
            "enable_netting": r.enable_netting,
            "final_nav_p1": r.final_nav_p1,
            "final_nav_p2": r.final_nav_p2,
            "total_cost_p1": r.total_cost_p1,
            "total_extra_cost": r.total_extra_cost,
            "total_cost_p2": r.total_cost_p2,
            "min_fee_cost": r.min_fee_cost,
            "rounding_cost": r.rounding_cost,
            "min_fee_event_count": r.min_fee_event_count,
            "days_with_min_fee": r.days_with_min_fee,
            "avg_events_per_day": r.avg_events_per_day,
            # Phase 2.2 신규
            "ignored_small_trades": r.ignored_small_trades,
            "ignored_by_netting": r.ignored_by_netting,
            "total_ignored": r.total_ignored,
        }
        for r in rows
    ])

    return df


def run_sensitivity_from_loader(
    venue_path: str,
    charging_modes: List[str] = None,
    eps_trade_grid: List[float] = None,
    min_notional_grid: List[float] = None,
    netting_options: List[bool] = None,
    min_fee_cash: float = 1.0,
    rounding_pct: float = 0.0001,
    quote_ccy: str = "USDT",
    phase1_config: Phase1Config = None,
) -> Optional[pd.DataFrame]:
    """
    거래소 디렉토리에서 데이터를 로드하고 sensitivity 분석 실행
    """
    if phase1_config is None:
        phase1_config = Phase1Config()

    print(f"Loading data from {venue_path}...")
    phase1_result = run_phase1_from_loader(
        venue_path,
        config=phase1_config,
        min_days=phase1_config.min_history_days,
    )

    if phase1_result is None:
        print("[Error] Failed to run Phase 1")
        return None

    print(f"Phase 1 complete: {len(phase1_result.timeseries)} days")

    return run_sensitivity_grid(
        phase1_result,
        charging_modes=charging_modes,
        eps_trade_grid=eps_trade_grid,
        min_notional_grid=min_notional_grid,
        netting_options=netting_options,
        min_fee_cash=min_fee_cash,
        rounding_pct=rounding_pct,
        quote_ccy=quote_ccy,
    )


def save_sensitivity_results(
    df: pd.DataFrame,
    output_path: str = None,
    analysis_grid: Dict[str, Any] = None,
) -> str:
    """Sensitivity 분석 결과 저장"""
    if output_path is None:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        output_path = os.path.join(OUTPUT_DIR, "phase2_sensitivity_summary.csv")

    df.to_csv(output_path, index=False, encoding="utf-8-sig", float_format="%.10g")
    print(f"\nSaved to: {output_path}")

    # Manifest 업데이트
    if analysis_grid:
        import json
        manifest_path = os.path.join(OUTPUT_DIR, "real_run_manifest.json")
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
        else:
            manifest = {}

        manifest["sensitivity_analysis"] = {
            "charging_modes": analysis_grid.get("charging_modes", DEFAULT_CHARGING_MODES),
            "eps_trade_grid": analysis_grid.get("eps_trade_grid", DEFAULT_EPS_TRADE_GRID),
            "min_notional_grid": analysis_grid.get("min_notional_grid", [0.0]),
            "netting_options": analysis_grid.get("netting_options", [False]),
            "output_file": output_path,
            "phase": "2.2",
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"Updated manifest: {manifest_path}")

    return output_path


def print_sensitivity_summary(df: pd.DataFrame) -> None:
    """Sensitivity 분석 요약 출력"""
    print("\n" + "="*90)
    print(" Phase 2.2 Sensitivity Analysis Summary")
    print("="*90)

    # 전체 결과 테이블
    print("\n[Full Results]")
    display_cols = [
        "mode", "eps_trade", "min_notional_cash", "enable_netting",
        "final_nav_p2", "min_fee_cost", "min_fee_event_count", "total_ignored"
    ]
    print(df[display_cols].to_string(index=False))

    # NAV 양수인 조합만 필터링
    positive_nav = df[df["final_nav_p2"] > 0]
    if len(positive_nav) > 0:
        print(f"\n[Positive NAV combinations: {len(positive_nav)}/{len(df)}]")

        # 최고 NAV
        best = positive_nav.loc[positive_nav["final_nav_p2"].idxmax()]
        print(f"\n  Best NAV: mode={best['mode']}, min_notional={best['min_notional_cash']}")
        print(f"            NAV_P2={best['final_nav_p2']:.2f}, events={best['min_fee_event_count']}")
    else:
        print("\n[Warning] No positive NAV combinations found!")

    # 모드별 통계
    print("\n[Statistics by charging_mode]")
    mode_stats = df.groupby("mode").agg({
        "final_nav_p2": ["mean", "min", "max"],
        "min_fee_event_count": ["mean", "min", "max"],
        "total_ignored": "mean",
    }).round(2)
    print(mode_stats.to_string())

    # min_notional별 통계
    if df["min_notional_cash"].nunique() > 1:
        print("\n[Statistics by min_notional_cash]")
        notional_stats = df.groupby("min_notional_cash").agg({
            "final_nav_p2": ["mean", "min", "max"],
            "min_fee_event_count": ["mean"],
            "total_ignored": "mean",
        }).round(2)
        print(notional_stats.to_string())


def analyze_recommendation(df: pd.DataFrame) -> Dict[str, Any]:
    """권장 기본값 분석 및 근거 제시"""
    # 불변량 검증
    assert (df["final_nav_p2"] <= df["final_nav_p1"]).all(), "P2 NAV should be <= P1 NAV"

    cost_check = np.isclose(
        df["total_cost_p2"],
        df["total_cost_p1"] + df["total_extra_cost"],
        rtol=1e-6
    )
    assert cost_check.all(), "Cost sum mismatch"

    recommendations = []

    # 1. per_rebalance_day + eps=1e-12 (보수적)
    r1 = df[(df["mode"] == "per_rebalance_day") & (df["eps_trade"] == 1e-12) & (df["min_notional_cash"] == 0)]
    if not r1.empty:
        recommendations.append({
            "rank": 1,
            "mode": "per_rebalance_day",
            "eps_trade": 1e-12,
            "min_notional_cash": 0.0,
            "enable_netting": False,
            "rationale": "거래일당 min_fee 1회. 가장 보수적이고 안정적.",
            "nav_p2": r1.iloc[0]["final_nav_p2"],
            "min_fee_cost": r1.iloc[0]["min_fee_cost"],
            "event_count": r1.iloc[0]["min_fee_event_count"],
        })

    # 2. per_asset_day + min_notional=10 (현실적)
    r2 = df[(df["mode"] == "per_asset_day") & (df["min_notional_cash"] == 10.0)]
    if not r2.empty:
        r2_best = r2.loc[r2["final_nav_p2"].idxmax()]
        recommendations.append({
            "rank": 2,
            "mode": "per_asset_day",
            "eps_trade": r2_best["eps_trade"],
            "min_notional_cash": 10.0,
            "enable_netting": r2_best["enable_netting"],
            "rationale": "자산별 min_fee, 10 USDT 미만 거래 무시. 소액 거래 필터링.",
            "nav_p2": r2_best["final_nav_p2"],
            "min_fee_cost": r2_best["min_fee_cost"],
            "event_count": r2_best["min_fee_event_count"],
        })

    # 3. per_asset_day + min_notional=100 (엄격)
    r3 = df[(df["mode"] == "per_asset_day") & (df["min_notional_cash"] == 100.0)]
    if not r3.empty:
        r3_best = r3.loc[r3["final_nav_p2"].idxmax()]
        recommendations.append({
            "rank": 3,
            "mode": "per_asset_day",
            "eps_trade": r3_best["eps_trade"],
            "min_notional_cash": 100.0,
            "enable_netting": r3_best["enable_netting"],
            "rationale": "자산별 min_fee, 100 USDT 미만 거래 무시. 엄격한 필터링.",
            "nav_p2": r3_best["final_nav_p2"],
            "min_fee_cost": r3_best["min_fee_cost"],
            "event_count": r3_best["min_fee_event_count"],
        })

    return {
        "recommendations": recommendations,
        "invariants_verified": True,
        "total_combinations": len(df),
        "positive_nav_count": len(df[df["final_nav_p2"] > 0]),
    }


# ==============================================================================
# 메인
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 2.2 Sensitivity Analysis for charging_mode, eps_trade, min_notional"
    )
    parser.add_argument(
        "--venue", type=str, default="binance_futures",
        help="Venue to analyze (default: binance_futures)"
    )
    parser.add_argument(
        "--min-fee", type=float, default=1.0,
        help="Min fee amount (default: 1.0)"
    )
    parser.add_argument(
        "--rounding-pct", type=float, default=0.0001,
        help="Rounding cost percentage (default: 0.0001)"
    )
    parser.add_argument(
        "--quote-ccy", type=str, default="USDT",
        help="Quote currency (default: USDT)"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output CSV path"
    )
    # Phase 2.2 신규 옵션
    parser.add_argument(
        "--min-notional", type=float, nargs="+", default=[0.0, 10.0, 100.0],
        help="Min notional cash grid (default: 0 10 100)"
    )
    parser.add_argument(
        "--netting", type=str, choices=["on", "off", "both"], default="off",
        help="Netting option: on, off, or both (default: off)"
    )
    parser.add_argument(
        "--full-grid", action="store_true",
        help="Run full grid (all modes, eps, notional, netting)"
    )

    args = parser.parse_args()

    venue_path = os.path.join(BASE_DATA, args.venue)
    if not os.path.exists(venue_path):
        print(f"[Error] Path not found: {venue_path}")
        sys.exit(1)

    # Phase 1 설정
    phase1_config = Phase1Config(
        signal_delay=1,
        trend_ma_n=50,
        momentum_L=30,
        vol_lookback=30,
        regime_mode="tiered",
        vol_target_ann=0.30,
        max_gross_leverage=1.0,
        one_way_rate_bps=5.0,
        min_history_days=60,
        initial_nav=10000.0,
    )

    # netting 옵션 파싱
    if args.netting == "on":
        netting_options = [True]
    elif args.netting == "off":
        netting_options = [False]
    else:  # both
        netting_options = [False, True]

    # 그리드 설정
    if args.full_grid:
        charging_modes = DEFAULT_CHARGING_MODES
        eps_trade_grid = [1e-12]  # 단순화
        min_notional_grid = [0.0, 1.0, 10.0, 100.0]
        netting_options = [False, True]
    else:
        charging_modes = DEFAULT_CHARGING_MODES
        eps_trade_grid = [1e-12]
        min_notional_grid = args.min_notional

    # Sensitivity 분석 실행
    df = run_sensitivity_from_loader(
        venue_path,
        charging_modes=charging_modes,
        eps_trade_grid=eps_trade_grid,
        min_notional_grid=min_notional_grid,
        netting_options=netting_options,
        min_fee_cash=args.min_fee,
        rounding_pct=args.rounding_pct,
        quote_ccy=args.quote_ccy,
        phase1_config=phase1_config,
    )

    if df is None:
        sys.exit(1)

    # 결과 저장
    analysis_grid = {
        "charging_modes": charging_modes,
        "eps_trade_grid": eps_trade_grid,
        "min_notional_grid": min_notional_grid,
        "netting_options": netting_options,
    }
    save_sensitivity_results(df, args.output, analysis_grid)

    # 요약 출력
    print_sensitivity_summary(df)

    # 권장값 분석
    recommendation = analyze_recommendation(df)

    print("\n" + "="*90)
    print(" Phase 2.2 Recommendation Analysis")
    print("="*90)
    print(f"\n  Positive NAV: {recommendation['positive_nav_count']}/{recommendation['total_combinations']}")

    for rec in recommendation["recommendations"]:
        print(f"\n  Rank {rec['rank']}: mode={rec['mode']}, min_notional={rec['min_notional_cash']}")
        print(f"    Rationale: {rec['rationale']}")
        print(f"    NAV_P2={rec['nav_p2']:.2f}, min_fee_cost={rec['min_fee_cost']:.2f}")
        print(f"    event_count={rec['event_count']}")

    print(f"\n  Invariants verified: {recommendation['invariants_verified']}")
