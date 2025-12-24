"""
examples/run_realdata_phase2.py - 실제 데이터로 Phase 2 백테스트 실행 (Phase 2.1)

이 스크립트는 Phase 1과 Phase 2 결과를 비교하고,
비용 분해(breakdown)를 출력합니다.

Phase 2.1 변경사항:
- charging_mode별로 실행 지원 (CLI 옵션 또는 전체 실행)
- min_fee_charging_mode, eps_trade, quote_ccy 출력
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import pandas as pd

from phase1_anchor_engine import Phase1Config
from adapters.run_phase2_from_loader import (
    Phase2Config,
    run_phase2_from_loader,
    Phase2Result,
)


# ==============================================================================
# 설정
# ==============================================================================

BASE_DATA = r"E:\OneDrive\고형석\코인\data"
OUTPUT_DIR = r"C:\Users\고형석\outputs"

# 테스트할 거래소 목록
VENUES = [
    "binance_futures",
    "binance_spot",
    "upbit",
    "bithumb",
]

# 지원하는 charging_mode
CHARGING_MODES = [
    "per_asset_day",
    "per_asset_side_day",
    "per_rebalance_day",
]


def create_config(
    venue: str,
    charging_mode: str = "per_asset_day",
    eps_trade: float = 1e-12
) -> Phase2Config:
    """거래소별 설정 생성"""
    # 기본 Phase 1 설정
    phase1_config = Phase1Config(
        signal_delay=1,
        trend_ma_n=50,
        momentum_L=30,
        vol_lookback=30,
        regime_mode="tiered",
        regime_ma_n=50,
        vol_target_ann=0.30,
        max_gross_leverage=1.0,
        one_way_rate_bps=5.0,
        min_active_assets=1,
        min_history_days=60,
        initial_nav=10000.0,
    )

    # 거래소별 수수료 설정
    if venue == "binance_futures":
        min_fee = 0.5
        rounding_pct = 0.00005
        quote_ccy = "USDT"
    elif venue == "binance_spot":
        min_fee = 1.0
        rounding_pct = 0.0001
        quote_ccy = "USDT"
    else:  # 한국 거래소
        min_fee = 100
        rounding_pct = 0.0002
        quote_ccy = "KRW"

    return Phase2Config(
        phase1_config=phase1_config,
        enable_proportional=True,
        enable_min_fee=True,
        enable_rounding=True,
        min_fee_cash=min_fee,
        min_fee_charging_mode=charging_mode,
        min_fee_eps_trade=eps_trade,
        quote_ccy=quote_ccy,
        rounding_pct=rounding_pct,
        output_dir=OUTPUT_DIR,
        save_outputs=True,
    )


def print_comparison(result: Phase2Result, venue: str) -> None:
    """Phase 1 vs Phase 2 비교 출력"""
    summary = result.summary

    print(f"\n{'='*60}")
    print(f" {venue.upper()} - Phase 1 vs Phase 2 Comparison")
    print(f"{'='*60}")

    # NAV 비교
    print("\n[NAV Comparison]")
    print(f"  Initial NAV:     {summary['initial_nav']:>12,.2f}")
    print(f"  Final NAV (P1):  {summary['final_nav_phase1']:>12,.2f}")
    print(f"  Final NAV (P2):  {summary['final_nav_phase2']:>12,.2f}")
    print(f"  Difference:      {summary['nav_diff_phase1_vs_phase2']:>12,.2f}")

    # 수익률
    p1_return = (summary['final_nav_phase1'] / summary['initial_nav'] - 1) * 100
    p2_return = (summary['final_nav_phase2'] / summary['initial_nav'] - 1) * 100
    print(f"\n  Return (P1):     {p1_return:>11.2f}%")
    print(f"  Return (P2):     {p2_return:>11.2f}%")

    # 성과 지표
    print("\n[Performance Metrics - Phase 2]")
    print(f"  CAGR:            {summary.get('cagr_phase2', 0)*100:>11.2f}%")
    print(f"  Sharpe (P1):     {summary.get('sharpe_phase1', 0):>12.4f}")
    print(f"  Sharpe (P2):     {summary.get('sharpe_phase2', 0):>12.4f}")
    print(f"  Max DD (P1):     {summary.get('max_dd_phase1', 0)*100:>11.2f}%")
    print(f"  Max DD (P2):     {summary.get('max_dd_phase2', 0)*100:>11.2f}%")

    # 비용 분해
    print("\n[Cost Breakdown]")
    print(f"  Phase 1 (proportional): {summary['total_cost_cash_phase1']:>10,.2f}")

    breakdown = summary.get('cost_breakdown_by_model', {})
    for model, cost in breakdown.items():
        print(f"  + {model}:           {cost:>10,.2f}")

    print(f"  {'-'*35}")
    print(f"  Total Extra Cost:       {summary['total_extra_cost_cash']:>10,.2f}")
    print(f"  Phase 2 Total Cost:     {summary['total_cost_cash_phase2']:>10,.2f}")

    # Phase 2.1: min_fee 파라미터
    print("\n[Min Fee Config (Phase 2.1)]")
    print(f"  Charging Mode:   {summary.get('min_fee_charging_mode', 'N/A')}")
    print(f"  Eps Trade:       {summary.get('min_fee_eps_trade', 'N/A')}")
    print(f"  Quote CCY:       {summary.get('quote_ccy', 'N/A')}")

    # 거래 통계
    print("\n[Trading Statistics]")
    print(f"  Trading Days:    {summary['n_trading_days']:>12,d}")


def print_manifest(result: Phase2Result) -> None:
    """Manifest 정보 출력"""
    manifest = result.manifest

    print("\n[Manifest Info]")
    print(f"  Schema Version:  {manifest['schema_version']}")
    print(f"  Created:         {manifest['created_at']}")
    print(f"  Cost Models:     {', '.join(manifest['cost_models'])}")
    print(f"  Quote CCY:       {manifest.get('quote_ccy', 'N/A')}")


def run_single_venue(
    venue: str = "binance_futures",
    charging_mode: str = "per_asset_day",
    eps_trade: float = 1e-12
) -> None:
    """단일 거래소 실행"""
    venue_path = os.path.join(BASE_DATA, venue)
    if not os.path.exists(venue_path):
        print(f"[Error] Path not found: {venue_path}")
        return

    print(f"Running Phase 2.1 for {venue} (charging_mode={charging_mode})...")

    config = create_config(venue, charging_mode=charging_mode, eps_trade=eps_trade)
    result = run_phase2_from_loader(venue_path, config)

    if result is None:
        print("[Error] Failed to run Phase 2")
        return

    print_comparison(result, venue)
    print_manifest(result)

    # 타임시리즈 샘플 출력
    print("\n[Timeseries Sample - Last 5 days]")
    ts = result.timeseries[[
        "nav_post_open_phase1",
        "nav_post_open_phase2",
        "cost_cash_phase1",
        "extra_cost_cash",
        "turnover_L1"
    ]].tail(5)
    print(ts.to_string())


def run_all_modes(venue: str = "binance_futures") -> None:
    """모든 charging_mode로 실행 (비교용)"""
    venue_path = os.path.join(BASE_DATA, venue)
    if not os.path.exists(venue_path):
        print(f"[Error] Path not found: {venue_path}")
        return

    print(f"\n{'*'*70}")
    print(f" Comparing all charging_modes for {venue}")
    print(f"{'*'*70}")

    results = {}
    for mode in CHARGING_MODES:
        print(f"\n>>> Running with charging_mode={mode}...")
        config = create_config(venue, charging_mode=mode)
        config.save_outputs = False  # 비교용이므로 저장 안함
        result = run_phase2_from_loader(venue_path, config)

        if result is not None:
            results[mode] = result
            print(f"    NAV P2: {result.summary['final_nav_phase2']:.2f}")
            print(f"    min_fee cost: {result.summary['cost_breakdown_by_model'].get('min_fee', 0):.2f}")

    # 비교 테이블
    if results:
        print("\n" + "="*70)
        print(" Comparison Table: charging_mode impact")
        print("="*70)
        rows = []
        for mode, result in results.items():
            s = result.summary
            rows.append({
                "charging_mode": mode,
                "NAV_P2": s["final_nav_phase2"],
                "min_fee_cost": s["cost_breakdown_by_model"].get("min_fee", 0),
                "extra_cost": s["total_extra_cost_cash"],
            })
        df = pd.DataFrame(rows)
        print(df.to_string(index=False))


def run_all_venues(charging_mode: str = "per_asset_day") -> None:
    """모든 거래소에 대해 실행"""
    results = {}

    for venue in VENUES:
        venue_path = os.path.join(BASE_DATA, venue)
        if not os.path.exists(venue_path):
            print(f"\n[Skip] {venue}: path not found")
            continue

        print(f"\n{'*'*60}")
        print(f" Running Phase 2.1 for {venue} (mode={charging_mode})...")
        print(f"{'*'*60}")

        config = create_config(venue, charging_mode=charging_mode)
        result = run_phase2_from_loader(venue_path, config)

        if result is None:
            print(f"  [Error] Failed to run Phase 2 for {venue}")
            continue

        results[venue] = result
        print_comparison(result, venue)

    # 요약 테이블
    if results:
        print_summary_table(results)


def print_summary_table(results: dict) -> None:
    """모든 거래소 결과 요약 테이블"""
    print("\n")
    print("="*80)
    print(" Summary Table: Phase 1 vs Phase 2 across all venues")
    print("="*80)

    rows = []
    for venue, result in results.items():
        s = result.summary
        rows.append({
            "Venue": venue,
            "NAV_P1": s["final_nav_phase1"],
            "NAV_P2": s["final_nav_phase2"],
            "Cost_P1": s["total_cost_cash_phase1"],
            "Extra_Cost": s["total_extra_cost_cash"],
            "Cost_P2": s["total_cost_cash_phase2"],
            "Sharpe_P2": s.get("sharpe_phase2", 0),
            "Mode": s.get("min_fee_charging_mode", "N/A"),
        })

    df = pd.DataFrame(rows)
    df = df.set_index("Venue")
    print("\n")
    print(df.to_string())
    print("\n")


# ==============================================================================
# 메인
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Phase 2.1 backtest with charging_mode support"
    )
    parser.add_argument(
        "--venue", type=str, default=None,
        help="Specific venue to run (default: all)"
    )
    parser.add_argument(
        "--mode", type=str, default="per_asset_day",
        choices=CHARGING_MODES,
        help="Min fee charging mode (default: per_asset_day)"
    )
    parser.add_argument(
        "--eps", type=float, default=1e-12,
        help="Trade recognition threshold (default: 1e-12)"
    )
    parser.add_argument(
        "--compare-modes", action="store_true",
        help="Compare all charging modes for a single venue"
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all venues"
    )

    args = parser.parse_args()

    if args.compare_modes:
        venue = args.venue or "binance_futures"
        run_all_modes(venue)
    elif args.all:
        run_all_venues(charging_mode=args.mode)
    elif args.venue:
        run_single_venue(args.venue, charging_mode=args.mode, eps_trade=args.eps)
    else:
        # 기본: 모든 거래소
        run_all_venues(charging_mode=args.mode)
