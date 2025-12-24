"""
adapters/run_phase1_from_loader.py - Phase 1 실행기

데이터 로더와 Phase 1 Anchor Engine을 연결합니다.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_anchor_engine import (
    Phase1AnchorEngine,
    Phase1Config,
    Phase1Result,
)
from adapters.normalize import load_venue_data, normalize_inputs


def run_phase1_from_loader(
    venue_path: str,
    config: Optional[Phase1Config] = None,
    min_days: int = 60
) -> Optional[Phase1Result]:
    """
    거래소 디렉토리에서 데이터를 로드하고 Phase 1 백테스트 실행

    Args:
        venue_path: 거래소 데이터 디렉토리
        config: Phase1Config (None이면 기본값)
        min_days: 최소 필요 일수

    Returns:
        Phase1Result 또는 None
    """
    if config is None:
        config = Phase1Config()

    # 데이터 로드
    data = load_venue_data(venue_path, min_days=min_days)
    if not data:
        print(f"No valid data found in {venue_path}")
        return None

    # BTC, ETH 확인 (레짐 필터용)
    btc_df = None
    for key in ["BTC", "BTCUSDT", "BTCBUSD"]:
        if key in data and data[key] is not None:
            btc_df = data[key]
            break

    eth_df = None
    for key in ["ETH", "ETHUSDT", "ETHBUSD"]:
        if key in data and data[key] is not None:
            eth_df = data[key]
            break

    if btc_df is None or eth_df is None:
        print("BTC or ETH data not found for regime filter")
        return None

    btc_close = btc_df["close"]
    eth_close = eth_df["close"]

    # Phase 1 엔진 실행
    engine = Phase1AnchorEngine(config)
    result = engine.run(data, btc_close, eth_close)

    return result


def run_phase1_from_dict(
    price_data: Dict[str, pd.DataFrame],
    btc_close: pd.Series,
    eth_close: pd.Series,
    config: Optional[Phase1Config] = None
) -> Phase1Result:
    """
    딕셔너리 형태의 데이터로 Phase 1 백테스트 실행

    Args:
        price_data: {symbol: DataFrame with OHLCV}
        btc_close: BTC 종가 시리즈
        eth_close: ETH 종가 시리즈
        config: Phase1Config

    Returns:
        Phase1Result
    """
    if config is None:
        config = Phase1Config()

    engine = Phase1AnchorEngine(config)
    return engine.run(price_data, btc_close, eth_close)


if __name__ == "__main__":
    # 테스트 실행
    BASE_DATA = r"E:\OneDrive\고형석\코인\data"
    venues = ["binance_futures", "binance_spot", "upbit", "bithumb"]

    for venue in venues:
        venue_path = os.path.join(BASE_DATA, venue)
        if not os.path.exists(venue_path):
            continue

        print(f"\nRunning Phase 1 for {venue}...")
        result = run_phase1_from_loader(venue_path)
        if result:
            print(f"  Final NAV: {result.summary['final_nav_phase1']:.2f}")
            print(f"  Sharpe: {result.summary['sharpe_ratio']:.4f}")
            print(f"  MaxDD: {result.summary['max_drawdown']:.2%}")
