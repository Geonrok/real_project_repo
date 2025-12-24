"""
Phase 1 Anchor Engine - 핵심 백테스트 계산 엔진

이 모듈은 Phase 1의 핵심 계산 로직을 담고 있으며, 절대 수정되어서는 안 됩니다.
Phase 2 확장은 이 엔진의 결과를 "후처리"하는 방식으로 구현됩니다.

핵심 계산 규칙:
1. 회계 규칙: NAV는 비용 차감 전/후로 명확히 구분
2. 타이밍: t일 시그널 → t+1일 Open에서 체결 → t+1일 수익률 적용
3. 드리프트: 가격 변화로 인한 비중 드리프트 반영
4. L1 Turnover: |w_target - w_drift|의 합으로 계산
5. NaN 처리: 시그널 계산 불가 시 NaN → 0으로 대체
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ==============================================================================
# 기본 지표 함수들
# ==============================================================================

def sma(x: pd.Series, n: int) -> pd.Series:
    """Simple Moving Average"""
    return x.rolling(n, min_periods=n).mean()


def ema(x: pd.Series, n: int) -> pd.Series:
    """Exponential Moving Average"""
    return x.ewm(span=n, adjust=False, min_periods=n).mean()


def rolling_volatility(ret: pd.Series, n: int) -> pd.Series:
    """롤링 표준편차 (일간 기준)"""
    return ret.rolling(n, min_periods=n).std()


def tsmom(price: pd.Series, L: int) -> pd.Series:
    """Time-Series Momentum: 과거 L일 수익률"""
    return price / price.shift(L) - 1.0


def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    """안전한 나눗셈 (0으로 나누기 방지)"""
    return a / (b.replace(0, np.nan) + eps)


# ==============================================================================
# Phase 1 Config
# ==============================================================================

@dataclass
class Phase1Config:
    """Phase 1 백테스트 설정"""
    # 타이밍 규칙
    signal_delay: int = 1  # t일 시그널 → t+signal_delay일에 체결

    # 지표 파라미터
    trend_ma_n: int = 50
    momentum_L: int = 30
    vol_lookback: int = 30

    # 레짐 필터
    regime_mode: str = "tiered"  # gate_btc, or_gate, tiered
    regime_ma_n: int = 50

    # 포트폴리오 구성
    vol_target_ann: float = 0.30  # 연환산 목표 변동성
    max_gross_leverage: float = 1.0

    # 비용 설정 (Phase 1 기본)
    one_way_rate_bps: float = 5.0  # 편도 5bps (= 0.05%)

    # 최소 조건
    min_active_assets: int = 1
    min_history_days: int = 60

    # 기타
    initial_nav: float = 10000.0
    annualization_factor: float = 365.0  # 일봉 기준


# ==============================================================================
# Phase 1 결과 구조체
# ==============================================================================

@dataclass
class Phase1Result:
    """Phase 1 엔진 실행 결과"""
    # 메타데이터
    config: Phase1Config
    schema_version: str = "1.5.0"

    # 타임시리즈 (pd.DataFrame)
    timeseries: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 가중치 행렬
    w_target: pd.DataFrame = field(default_factory=pd.DataFrame)  # 목표 비중
    w_drift: pd.DataFrame = field(default_factory=pd.DataFrame)   # 드리프트 후 비중
    w_exec: pd.DataFrame = field(default_factory=pd.DataFrame)    # 체결 후 비중
    delta_w: pd.DataFrame = field(default_factory=pd.DataFrame)   # 리밸런싱 양

    # 요약 통계
    summary: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 레짐 필터
# ==============================================================================

def compute_regime_multiplier(
    btc_close: pd.Series,
    eth_close: pd.Series,
    mode: str = "tiered",
    ma_n: int = 50
) -> pd.Series:
    """
    레짐 멀티플라이어 계산

    mode:
      - "gate_btc": BTC>MA이면 1, 아니면 0
      - "or_gate": BTC>MA OR ETH>MA이면 1, 아니면 0
      - "tiered": 둘 다 1, 하나만 0.5, 둘 다 아니면 0
    """
    btc_ma = sma(btc_close, ma_n)
    eth_ma = sma(eth_close, ma_n)

    btc_above = btc_close > btc_ma
    eth_above = eth_close > eth_ma

    mode = mode.lower()

    if mode == "gate_btc":
        return btc_above.astype(float)

    if mode == "or_gate":
        return (btc_above | eth_above).astype(float)

    if mode == "tiered":
        out = pd.Series(0.0, index=btc_close.index)
        out[btc_above & eth_above] = 1.0
        out[btc_above ^ eth_above] = 0.5  # XOR: 하나만 True
        return out

    raise ValueError(f"Unknown regime mode: {mode}")


# ==============================================================================
# 시그널 생성
# ==============================================================================

def compute_entry_signal(
    close: pd.Series,
    trend_ma_n: int,
    momentum_L: int
) -> pd.Series:
    """
    진입 시그널: (close > MA) OR (momentum > 0)

    Returns:
        pd.Series: True/False 시그널 (NaN은 False로 처리)
    """
    ma = sma(close, trend_ma_n)
    trend_up = close > ma

    mom = tsmom(close, momentum_L)
    mom_up = mom > 0

    # 진입 조건: 트렌드 상승 OR 모멘텀 양수
    signal = trend_up | mom_up

    # NaN 처리: False로 대체
    signal = signal.fillna(False)

    return signal


# ==============================================================================
# 핵심 엔진: Phase 1 Anchor Engine
# ==============================================================================

class Phase1AnchorEngine:
    """
    Phase 1 Anchor Engine

    핵심 계산을 담당하며, 이 클래스의 내부 로직은 절대 변경되어서는 안 됩니다.
    Phase 2는 이 엔진의 결과를 "후처리"하는 방식으로 확장합니다.
    """

    def __init__(self, config: Phase1Config):
        self.config = config

    def run(
        self,
        price_data: Dict[str, pd.DataFrame],
        btc_close: pd.Series,
        eth_close: pd.Series,
        common_index: Optional[pd.DatetimeIndex] = None
    ) -> Phase1Result:
        """
        백테스트 실행

        Args:
            price_data: {symbol: DataFrame with columns [open, high, low, close, volume]}
            btc_close: BTC 종가 시리즈 (레짐 필터용)
            eth_close: ETH 종가 시리즈 (레짐 필터용)
            common_index: 공통 날짜 인덱스 (None이면 자동 생성)

        Returns:
            Phase1Result: 백테스트 결과
        """
        cfg = self.config

        # 공통 인덱스 생성
        if common_index is None:
            common_index = self._build_common_index(price_data, btc_close, eth_close)

        symbols = sorted(price_data.keys())
        n_days = len(common_index)
        n_assets = len(symbols)

        # 데이터 정렬 및 리인덱싱
        closes = pd.DataFrame(
            {s: price_data[s]["close"].reindex(common_index) for s in symbols}
        )
        opens = pd.DataFrame(
            {s: price_data[s]["open"].reindex(common_index) for s in symbols}
        )
        returns = closes.pct_change()

        # 레짐 멀티플라이어
        btc_al = btc_close.reindex(common_index).ffill()
        eth_al = eth_close.reindex(common_index).ffill()
        regime_mult = compute_regime_multiplier(
            btc_al, eth_al,
            mode=cfg.regime_mode,
            ma_n=cfg.regime_ma_n
        )

        # 종목별 진입 시그널
        entry_signals = pd.DataFrame(index=common_index, columns=symbols, dtype=bool)
        for s in symbols:
            entry_signals[s] = compute_entry_signal(
                closes[s],
                cfg.trend_ma_n,
                cfg.momentum_L
            )

        # 롤링 변동성
        rolling_vol = returns.rolling(cfg.vol_lookback, min_periods=cfg.vol_lookback).std()

        # ======================================================================
        # 핵심 루프: 비중 계산 및 NAV 추적
        # ======================================================================

        # 결과 저장용 배열
        w_target = np.zeros((n_days, n_assets))
        w_drift = np.zeros((n_days, n_assets))
        w_exec = np.zeros((n_days, n_assets))
        delta_w = np.zeros((n_days, n_assets))

        nav = np.zeros(n_days)
        nav_before_cost = np.zeros(n_days)
        turnover_L1 = np.zeros(n_days)
        cost_ratio = np.zeros(n_days)
        cost_cash = np.zeros(n_days)
        gross_leverage = np.zeros(n_days)

        # 초기화
        nav[0] = cfg.initial_nav
        nav_before_cost[0] = cfg.initial_nav

        one_way_rate = cfg.one_way_rate_bps / 10000.0

        for t in range(1, n_days):
            # ------------------------------------------------------------------
            # Step 1: 드리프트 계산 (전일 비중이 가격 변화로 변함)
            # ------------------------------------------------------------------
            if t >= 2:
                # 전일 체결 비중 × 오늘 수익률로 드리프트
                ret_today = returns.iloc[t].values
                ret_today = np.nan_to_num(ret_today, nan=0.0)

                # 드리프트된 비중 (수익률 반영)
                w_prev = w_exec[t-1].copy()
                w_drifted = w_prev * (1 + ret_today)

                # 정규화 (레버리지 고려)
                sum_drifted = np.sum(w_drifted)
                if sum_drifted > 0:
                    w_drift[t] = w_drifted / sum_drifted * np.sum(np.abs(w_prev))
                else:
                    w_drift[t] = np.zeros(n_assets)

            # ------------------------------------------------------------------
            # Step 2: 목표 비중 계산 (t-1일 시그널 기준)
            # ------------------------------------------------------------------
            t_signal = t - cfg.signal_delay  # 시그널 참조 시점

            if t_signal >= cfg.min_history_days:
                # 활성 종목 (시그널=True, 데이터 유효)
                active_mask = entry_signals.iloc[t_signal].values
                active_mask = np.array([bool(x) if pd.notna(x) else False for x in active_mask])

                # 유효한 수익률 데이터가 있는지 확인
                valid_data = np.isfinite(returns.iloc[t].values)
                active_mask = active_mask & valid_data

                n_active = int(np.sum(active_mask))
                regime_t = regime_mult.iloc[t_signal] if t_signal < len(regime_mult) else 0.0

                if n_active >= cfg.min_active_assets and regime_t > 0:
                    # 역변동성 가중
                    vol_t = rolling_vol.iloc[t_signal].values
                    vol_t = np.nan_to_num(vol_t, nan=1.0, posinf=1.0, neginf=1.0)
                    vol_t = np.maximum(vol_t, 1e-8)

                    inv_vol = np.zeros(n_assets)
                    inv_vol[active_mask] = 1.0 / vol_t[active_mask]

                    if np.sum(inv_vol) > 0:
                        w_raw = inv_vol / np.sum(inv_vol)
                    else:
                        w_raw = np.zeros(n_assets)

                    # 포트폴리오 변동성 계산 (대각 근사)
                    port_vol_daily = np.sqrt(np.sum((w_raw ** 2) * (vol_t ** 2)))
                    port_vol_ann = port_vol_daily * np.sqrt(cfg.annualization_factor)

                    # 목표 변동성으로 스케일링
                    eff_target = cfg.vol_target_ann * float(regime_t)
                    if port_vol_ann > 0:
                        scale = eff_target / port_vol_ann
                    else:
                        scale = 0.0

                    # 레버리지 제한
                    w_scaled = w_raw * scale
                    gross = np.sum(np.abs(w_scaled))
                    if gross > cfg.max_gross_leverage:
                        w_scaled = w_scaled * (cfg.max_gross_leverage / gross)

                    w_target[t] = w_scaled
                else:
                    w_target[t] = np.zeros(n_assets)
            else:
                w_target[t] = np.zeros(n_assets)

            # ------------------------------------------------------------------
            # Step 3: 턴오버 계산 (L1 norm)
            # ------------------------------------------------------------------
            # L1 Turnover = |w_target - w_drift|의 합
            delta_w[t] = w_target[t] - w_drift[t]
            turnover_L1[t] = np.sum(np.abs(delta_w[t]))

            # ------------------------------------------------------------------
            # Step 4: 체결 비중 = 목표 비중 (슬리피지 없는 완전 체결 가정)
            # ------------------------------------------------------------------
            w_exec[t] = w_target[t]
            gross_leverage[t] = np.sum(np.abs(w_exec[t]))

            # ------------------------------------------------------------------
            # Step 5: NAV 계산
            # ------------------------------------------------------------------
            # 비용 전 NAV: 전일 NAV × (1 + 포트폴리오 수익률)
            ret_today = returns.iloc[t].values
            ret_today = np.nan_to_num(ret_today, nan=0.0)

            # 전일 체결 비중으로 오늘 수익률 적용
            port_ret = np.sum(w_exec[t-1] * ret_today)
            nav_before_cost[t] = nav[t-1] * (1 + port_ret)

            # 비용 계산: L1 턴오버 × one_way_rate
            cost_ratio[t] = turnover_L1[t] * one_way_rate
            cost_cash[t] = nav_before_cost[t] * cost_ratio[t]

            # 비용 후 NAV
            nav[t] = nav_before_cost[t] * (1 - cost_ratio[t])

        # ======================================================================
        # 결과 DataFrame 생성
        # ======================================================================

        timeseries = pd.DataFrame({
            "nav_pre_open": nav_before_cost,
            "nav_post_open": nav,
            "nav_post_open_phase1": nav,  # Phase 1 NAV (Anchor)
            "turnover_L1": turnover_L1,
            "cost_ratio_phase1": cost_ratio,
            "cost_cash_phase1": cost_cash,
            "gross_leverage": gross_leverage,
            "regime_mult": regime_mult.values,
        }, index=common_index)

        # 일간 수익률 추가
        timeseries["daily_return_gross"] = (
            timeseries["nav_pre_open"] / timeseries["nav_pre_open"].shift(1) - 1
        )
        timeseries["daily_return_net"] = (
            timeseries["nav_post_open"] / timeseries["nav_post_open"].shift(1) - 1
        )

        # 비중 DataFrame
        w_target_df = pd.DataFrame(w_target, index=common_index, columns=symbols)
        w_drift_df = pd.DataFrame(w_drift, index=common_index, columns=symbols)
        w_exec_df = pd.DataFrame(w_exec, index=common_index, columns=symbols)
        delta_w_df = pd.DataFrame(delta_w, index=common_index, columns=symbols)

        # 요약 통계 계산
        summary = self._compute_summary(timeseries, cfg)

        return Phase1Result(
            config=cfg,
            schema_version="1.5.0",
            timeseries=timeseries,
            w_target=w_target_df,
            w_drift=w_drift_df,
            w_exec=w_exec_df,
            delta_w=delta_w_df,
            summary=summary
        )

    def _build_common_index(
        self,
        price_data: Dict[str, pd.DataFrame],
        btc_close: pd.Series,
        eth_close: pd.Series
    ) -> pd.DatetimeIndex:
        """공통 날짜 인덱스 생성"""
        all_idx = btc_close.index.intersection(eth_close.index)
        for s, df in price_data.items():
            all_idx = all_idx.union(df.index)
        return all_idx.sort_values()

    def _compute_summary(
        self,
        ts: pd.DataFrame,
        cfg: Phase1Config
    ) -> Dict[str, Any]:
        """요약 통계 계산"""
        nav = ts["nav_post_open"]
        daily_ret = ts["daily_return_net"].dropna()

        n_days = len(nav)
        ann_factor = cfg.annualization_factor

        # 기본 통계
        final_nav = float(nav.iloc[-1])
        total_return = final_nav / cfg.initial_nav - 1.0

        if n_days > 1:
            cagr = (final_nav / cfg.initial_nav) ** (ann_factor / (n_days - 1)) - 1.0
        else:
            cagr = 0.0

        ann_vol = daily_ret.std() * np.sqrt(ann_factor) if len(daily_ret) > 0 else 0.0
        sharpe = (daily_ret.mean() * ann_factor) / (ann_vol + 1e-12) if ann_vol > 0 else 0.0

        # 최대 낙폭
        peak = nav.cummax()
        drawdown = (nav / peak - 1.0)
        max_dd = float(drawdown.min())

        # 비용 통계
        total_turnover = float(ts["turnover_L1"].sum())
        total_cost_cash = float(ts["cost_cash_phase1"].sum())
        total_cost_ratio = float(ts["cost_ratio_phase1"].sum())

        return {
            "initial_nav": cfg.initial_nav,
            "final_nav_phase1": final_nav,
            "total_return": total_return,
            "cagr": cagr,
            "ann_volatility": ann_vol,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_dd,
            "total_turnover_L1": total_turnover,
            "total_cost_cash_phase1": total_cost_cash,
            "total_cost_ratio_phase1": total_cost_ratio,
            "n_trading_days": n_days,
            "avg_daily_turnover": total_turnover / max(n_days, 1),
        }


# ==============================================================================
# 편의 함수
# ==============================================================================

def run_phase1_backtest(
    price_data: Dict[str, pd.DataFrame],
    btc_close: pd.Series,
    eth_close: pd.Series,
    config: Optional[Phase1Config] = None
) -> Phase1Result:
    """
    Phase 1 백테스트 편의 함수

    Args:
        price_data: 종목별 OHLCV 데이터
        btc_close: BTC 종가
        eth_close: ETH 종가
        config: Phase1Config (None이면 기본값 사용)

    Returns:
        Phase1Result
    """
    if config is None:
        config = Phase1Config()

    engine = Phase1AnchorEngine(config)
    return engine.run(price_data, btc_close, eth_close)


if __name__ == "__main__":
    # 간단한 테스트용 실행
    print("Phase 1 Anchor Engine loaded successfully.")
    print(f"Default config: {Phase1Config()}")
