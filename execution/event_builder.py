"""
execution/event_builder.py - Trade Event Builder (Phase 2.5)

거래 이벤트를 정의하고 필터링하는 핵심 레이어입니다.

Phase 2.5 변경사항:
  - 일별 상세 필터링 지표 추가 (daily_filter_details)
  - worst-days 리포트를 위한 top_filtered_assets 계산
  - FilterImpactMetrics에 daily_details DataFrame 추가

Phase 2.4 변경사항:
  1) min_notional 필터 기준을 trade_notional_cash로 변경
  2) w_exec 조정: 생략된 주문은 CASH로 자동 흡수 (Σw=1 유지)
  3) netting 옵션 실제 구현
  4) 필터링 영향 리포트: pre/post turnover, L1 distance

사용법:
    builder = TradeEventBuilder(TradeEventConfig(
        eps_trade=1e-12,
        min_notional_cash=100.0,
        enable_netting=True,
    ))
    result = builder.build(delta_w, nav, w_exec_raw=w_exec)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


class TradeSide(str, Enum):
    """거래 방향"""
    BUY = "buy"
    SELL = "sell"


@dataclass
class TradeEvent:
    """
    단일 거래 이벤트

    Attributes:
        date: 거래 날짜
        symbol: 자산 심볼
        side: 거래 방향 (buy/sell)
        delta_w: 가중치 변화량
        notional_cash: 거래 금액 (현금 기준)
        is_valid: 유효한 거래인지 (임계값 통과 여부)
        ignore_reason: 무시된 경우 사유
    """
    date: pd.Timestamp
    symbol: str
    side: TradeSide
    delta_w: float
    notional_cash: float
    is_valid: bool = True
    ignore_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "symbol": self.symbol,
            "side": self.side.value,
            "delta_w": self.delta_w,
            "notional_cash": self.notional_cash,
            "is_valid": self.is_valid,
            "ignore_reason": self.ignore_reason,
        }


@dataclass
class FilterImpactMetrics:
    """
    Phase 2.5: 필터링 영향 지표 (일별 상세 포함)

    Attributes:
        pre_filter_turnover_L1: 필터링 전 총 L1 턴오버
        post_filter_turnover_L1: 필터링 후 총 L1 턴오버
        total_notional_raw: 필터링 전 총 거래 notional
        total_notional_filtered: 필터링으로 생략된 총 notional
        filter_ratio: 필터링 비율 (생략된 notional / 전체 notional)
        l1_distance_daily: w_exec_raw vs w_exec_filtered 일별 L1 거리
        l1_distance_mean: 평균 L1 거리
        l1_distance_max: 최대 L1 거리
        n_trades_filtered: 필터링된 거래 수
        n_trades_total: 전체 거래 수
        daily_details: Phase 2.5 - 일별 상세 필터링 지표 DataFrame
    """
    pre_filter_turnover_L1: float = 0.0
    post_filter_turnover_L1: float = 0.0
    total_notional_raw: float = 0.0
    total_notional_filtered: float = 0.0
    filter_ratio: float = 0.0
    l1_distance_daily: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    l1_distance_mean: float = 0.0
    l1_distance_max: float = 0.0
    n_trades_filtered: int = 0
    n_trades_total: int = 0
    # Phase 2.5: 일별 상세 지표
    daily_details: pd.DataFrame = field(default_factory=pd.DataFrame)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pre_filter_turnover_L1": self.pre_filter_turnover_L1,
            "post_filter_turnover_L1": self.post_filter_turnover_L1,
            "total_notional_raw": self.total_notional_raw,
            "total_notional_filtered": self.total_notional_filtered,
            "filter_ratio": self.filter_ratio,
            "l1_distance_mean": self.l1_distance_mean,
            "l1_distance_max": self.l1_distance_max,
            "n_trades_filtered": self.n_trades_filtered,
            "n_trades_total": self.n_trades_total,
            "turnover_reduction_ratio": 1 - (self.post_filter_turnover_L1 / self.pre_filter_turnover_L1)
                if self.pre_filter_turnover_L1 > 0 else 0.0,
        }

    def get_worst_days(self, top_n: int = 20) -> pd.DataFrame:
        """
        Phase 2.5: L1 distance 상위 N일 반환

        Returns:
            DataFrame with columns: date, l1_distance, pre_filter_turnover_L1,
            post_filter_turnover_L1, filter_ratio_day, filtered_notional_day,
            total_notional_day, events_day, top_filtered_assets
        """
        if self.daily_details.empty:
            return pd.DataFrame()

        # L1 distance 기준 상위 N일
        worst = self.daily_details.nlargest(top_n, "l1_distance")
        return worst.reset_index().rename(columns={"index": "date"})


@dataclass
class TradeEventResult:
    """
    거래 이벤트 빌더 결과 (Phase 2.4)

    Attributes:
        events_per_day: 일자별 유효 이벤트 수 (Series)
        events_per_asset_day: 일자+자산별 이벤트 수 (DataFrame)
        buy_events_per_day: 일자별 매수 이벤트 수
        sell_events_per_day: 일자별 매도 이벤트 수
        total_events: 총 유효 이벤트 수
        ignored_events: 무시된 이벤트 수
        days_with_events: 이벤트 발생 일수
        ignored_by_eps: eps_trade 미만으로 무시된 수
        ignored_by_notional: min_notional 미만으로 무시된 수
        ignored_by_netting: netting으로 상쇄된 수
        config: 사용된 설정
        all_events: 모든 이벤트 목록 (디버깅용)
        delta_w_filtered: 필터링된 delta_w (Phase 2.4)
        w_exec_filtered: 필터링된 w_exec (Phase 2.4)
        filter_impact: 필터링 영향 지표 (Phase 2.4)
    """
    events_per_day: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    events_per_asset_day: pd.DataFrame = field(default_factory=pd.DataFrame)
    buy_events_per_day: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    sell_events_per_day: pd.Series = field(default_factory=lambda: pd.Series(dtype=int))
    total_events: int = 0
    ignored_events: int = 0
    days_with_events: int = 0
    ignored_by_eps: int = 0
    ignored_by_notional: int = 0
    ignored_by_netting: int = 0
    config: Dict[str, Any] = field(default_factory=dict)
    all_events: List[TradeEvent] = field(default_factory=list)
    # Phase 2.4 신규
    delta_w_filtered: pd.DataFrame = field(default_factory=pd.DataFrame)
    w_exec_filtered: pd.DataFrame = field(default_factory=pd.DataFrame)
    filter_impact: Optional[FilterImpactMetrics] = None

    def to_summary_dict(self) -> Dict[str, Any]:
        """요약 딕셔너리 반환"""
        result = {
            "total_events": self.total_events,
            "ignored_events": self.ignored_events,
            "days_with_events": self.days_with_events,
            "ignored_by_eps": self.ignored_by_eps,
            "ignored_by_notional": self.ignored_by_notional,
            "ignored_by_netting": self.ignored_by_netting,
            "avg_events_per_day": float(self.events_per_day.mean()) if len(self.events_per_day) > 0 else 0.0,
            "config": self.config,
        }
        if self.filter_impact:
            result["filter_impact"] = self.filter_impact.to_dict()
        return result


@dataclass
class TradeEventConfig:
    """
    거래 이벤트 빌더 설정 (Phase 2.4)

    Attributes:
        eps_trade: 가중치 변화 임계값 (기본: 1e-12)
        min_notional_cash: 최소 거래 금액 임계값 (기본: 0, 비활성)
        enable_netting: 동일 자산 buy/sell 상쇄 활성화 (기본: False)
        exclude_cash: CASH 컬럼 제외 (기본: True)
        apply_filter_to_weights: 필터를 w_exec에 적용할지 (Phase 2.4)
    """
    eps_trade: float = 1e-12
    min_notional_cash: float = 0.0  # 0이면 비활성 (모든 거래 인정)
    enable_netting: bool = False
    exclude_cash: bool = True
    apply_filter_to_weights: bool = True  # Phase 2.4: w_exec 조정 여부

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eps_trade": self.eps_trade,
            "min_notional_cash": self.min_notional_cash,
            "enable_netting": self.enable_netting,
            "exclude_cash": self.exclude_cash,
            "apply_filter_to_weights": self.apply_filter_to_weights,
        }


class TradeEventBuilder:
    """
    거래 이벤트 빌더 (Phase 2.4)

    delta_w와 NAV를 입력받아 거래 이벤트 목록을 생성합니다.
    Phase 2.4에서는 min_notional 필터가 실제로 delta_w를 클립하고,
    w_exec를 조정하여 Σw=1을 유지합니다.
    """

    def __init__(self, config: TradeEventConfig = None):
        self.config = config or TradeEventConfig()

    def _exclude_cash_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """CASH 컬럼 제외"""
        if not self.config.exclude_cash:
            return df
        cols_to_use = [c for c in df.columns if c.upper() != "CASH"]
        return df[cols_to_use]

    def _get_risk_columns(self, df: pd.DataFrame) -> List[str]:
        """CASH 제외한 컬럼 목록 반환"""
        return [c for c in df.columns if c.upper() != "CASH"]

    def _compute_trade_notional(
        self,
        delta_w: pd.DataFrame,
        nav: pd.Series,
    ) -> pd.DataFrame:
        """
        Phase 2.4: trade_notional_cash 계산

        trade_notional_cash[t, asset] = NAV[t] * abs(delta_w[t, asset])
        """
        # NAV를 delta_w와 같은 인덱스로 정렬
        nav_aligned = nav.reindex(delta_w.index).fillna(0)

        # 각 셀에 NAV 곱하기
        notional = delta_w.abs().multiply(nav_aligned, axis=0)
        return notional

    def _apply_min_notional_filter(
        self,
        delta_w: pd.DataFrame,
        trade_notional: pd.DataFrame,
        min_notional: float,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Phase 2.4: min_notional 필터 적용

        trade_notional < min_notional 인 셀의 delta_w를 0으로 클립

        Returns:
            (delta_w_filtered, filter_mask)
            - delta_w_filtered: 필터링된 delta_w
            - filter_mask: True = 필터링됨 (생략됨)
        """
        if min_notional <= 0:
            # 필터링 없음
            return delta_w.copy(), pd.DataFrame(False, index=delta_w.index, columns=delta_w.columns)

        # 필터 마스크: True = notional이 임계값 미만
        filter_mask = trade_notional < min_notional

        # 필터링된 delta_w: 마스크가 True인 곳은 0
        delta_w_filtered = delta_w.copy()
        delta_w_filtered[filter_mask] = 0.0

        return delta_w_filtered, filter_mask

    def _adjust_w_exec_for_cash(
        self,
        w_exec_raw: pd.DataFrame,
        delta_w_raw: pd.DataFrame,
        delta_w_filtered: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Phase 2.4: w_exec 조정 - 생략된 주문을 CASH로 흡수

        w_exec_filtered = w_exec_raw + (delta_w_raw - delta_w_filtered)의 CASH 흡수

        원리:
        1. delta_w_raw - delta_w_filtered = 생략된 가중치 변화
        2. 생략된 가중치는 CASH로 이동
        3. Σw = 1 유지
        """
        if w_exec_raw is None or w_exec_raw.empty:
            return pd.DataFrame()

        risk_cols = self._get_risk_columns(w_exec_raw)

        # 생략된 delta_w 계산
        delta_skipped = delta_w_raw[risk_cols] - delta_w_filtered[risk_cols]

        # 위험자산 비중 조정 (한 번에 처리)
        adjusted_risk = w_exec_raw[risk_cols] - delta_skipped

        # CASH 조정: Σw=1 유지
        risk_sum = adjusted_risk.sum(axis=1)
        cash_col = 1.0 - risk_sum

        # 모든 컬럼을 한 번에 결합 (PerformanceWarning 방지)
        w_exec_filtered = pd.concat([adjusted_risk, cash_col.rename("CASH")], axis=1)

        return w_exec_filtered

    def _compute_filter_impact(
        self,
        delta_w_raw: pd.DataFrame,
        delta_w_filtered: pd.DataFrame,
        w_exec_raw: pd.DataFrame,
        w_exec_filtered: pd.DataFrame,
        trade_notional: pd.DataFrame,
        filter_mask: pd.DataFrame,
        nav: pd.Series,
        events_per_day: pd.Series = None,
    ) -> FilterImpactMetrics:
        """
        Phase 2.5: 필터링 영향 지표 계산 (일별 상세 포함)
        """
        risk_cols = self._get_risk_columns(delta_w_raw)
        index = delta_w_raw.index

        # 일별 지표 계산
        pre_filter_daily = delta_w_raw[risk_cols].abs().sum(axis=1)
        post_filter_daily = delta_w_filtered[risk_cols].abs().sum(axis=1)
        notional_raw_daily = trade_notional[risk_cols].sum(axis=1)
        notional_filtered_daily = trade_notional[risk_cols].where(filter_mask[risk_cols], 0).sum(axis=1)

        # 총합
        pre_filter_turnover = float(pre_filter_daily.sum())
        post_filter_turnover = float(post_filter_daily.sum())
        total_notional_raw = float(notional_raw_daily.sum())
        notional_filtered = float(notional_filtered_daily.sum())

        # 필터 비율
        filter_ratio = notional_filtered / total_notional_raw if total_notional_raw > 0 else 0.0

        # 거래 수
        n_trades_total = int((delta_w_raw[risk_cols].abs() > 0).sum().sum())
        n_trades_filtered = int(filter_mask[risk_cols].sum().sum())

        # L1 distance between w_exec_raw and w_exec_filtered
        if w_exec_raw is not None and not w_exec_raw.empty and not w_exec_filtered.empty:
            common_cols = list(set(w_exec_raw.columns) & set(w_exec_filtered.columns))
            w_diff = (w_exec_raw[common_cols] - w_exec_filtered[common_cols]).abs()
            l1_distance_daily = w_diff.sum(axis=1)
            l1_distance_mean = float(l1_distance_daily.mean())
            l1_distance_max = float(l1_distance_daily.max())
        else:
            l1_distance_daily = pd.Series(0.0, index=index)
            l1_distance_mean = 0.0
            l1_distance_max = 0.0

        # Phase 2.5: 일별 상세 지표 DataFrame 생성
        daily_filter_ratio = notional_filtered_daily / notional_raw_daily.replace(0, np.nan)
        daily_filter_ratio = daily_filter_ratio.fillna(0)

        # 일별 상위 필터링 자산 계산
        top_filtered_assets_list = []
        for date in index:
            day_notional = trade_notional[risk_cols].loc[date]
            day_mask = filter_mask[risk_cols].loc[date]
            filtered_notional = day_notional.where(day_mask, 0)

            # 상위 5개 필터링 자산
            top5 = filtered_notional.nlargest(5)
            top5_str = "; ".join([f"{k}:{v:.0f}" for k, v in top5.items() if v > 0])
            top_filtered_assets_list.append(top5_str if top5_str else "")

        daily_details = pd.DataFrame({
            "l1_distance": l1_distance_daily,
            "pre_filter_turnover_L1": pre_filter_daily,
            "post_filter_turnover_L1": post_filter_daily,
            "filter_ratio_day": daily_filter_ratio,
            "filtered_notional_day": notional_filtered_daily,
            "total_notional_day": notional_raw_daily,
            "events_day": events_per_day if events_per_day is not None else 0,
            "top_filtered_assets": top_filtered_assets_list,
        }, index=index)

        return FilterImpactMetrics(
            pre_filter_turnover_L1=pre_filter_turnover,
            post_filter_turnover_L1=post_filter_turnover,
            total_notional_raw=total_notional_raw,
            total_notional_filtered=notional_filtered,
            filter_ratio=float(filter_ratio),
            l1_distance_daily=l1_distance_daily,
            l1_distance_mean=l1_distance_mean,
            l1_distance_max=l1_distance_max,
            n_trades_filtered=n_trades_filtered,
            n_trades_total=n_trades_total,
            daily_details=daily_details,
        )

    def _apply_netting(
        self,
        delta_w: pd.DataFrame,
        nav: pd.Series,
    ) -> Tuple[pd.DataFrame, int]:
        """
        Phase 2.4: Netting 적용

        동일 날짜에서 자산들 간의 buy/sell을 netting할 수 없음
        (각 자산의 delta_w는 이미 그 날의 순변화량)

        실제로 netting이 의미있는 경우:
        - 일중 여러 신호가 있을 때 (현재 모델에서는 해당 없음)

        Phase 2.4에서 netting의 의미:
        - 포트폴리오 레벨에서 buy 총액과 sell 총액을 상계
        - 순거래 금액만 이벤트로 카운트

        Returns:
            (net_delta_w, n_netted)
            - 현재는 delta_w 그대로 반환 (자산별 netting은 이미 적용됨)
            - n_netted: netting된 거래 수 (포트폴리오 레벨)
        """
        # 현재 모델에서 delta_w는 이미 자산별 순변화량
        # 포트폴리오 레벨 netting은 이벤트 카운트에서만 적용
        return delta_w.copy(), 0

    def build(
        self,
        delta_w: pd.DataFrame,
        nav: pd.Series,
        w_exec_raw: pd.DataFrame = None,
        **kwargs
    ) -> TradeEventResult:
        """
        거래 이벤트 생성 (Phase 2.4)

        Args:
            delta_w: 가중치 변화 DataFrame (index=날짜, columns=자산)
            nav: NAV 시리즈 (notional 계산 기준)
            w_exec_raw: 원본 w_exec (Phase 2.4)
            **kwargs: 추가 파라미터

        Returns:
            TradeEventResult
        """
        cfg = self.config
        risk_cols = self._get_risk_columns(delta_w)

        # delta_w에서 CASH 제외 버전
        delta_w_risk = delta_w[risk_cols] if risk_cols else delta_w.copy()

        index = delta_w.index
        symbols = risk_cols

        # Phase 2.4: trade_notional 계산
        trade_notional = self._compute_trade_notional(delta_w_risk, nav)

        # Phase 2.4: min_notional 필터 적용
        delta_w_filtered_risk, filter_mask = self._apply_min_notional_filter(
            delta_w_risk, trade_notional, cfg.min_notional_cash
        )

        # Phase 2.4: 전체 delta_w_filtered 생성 (CASH 포함)
        delta_w_filtered = delta_w.copy()
        for col in risk_cols:
            delta_w_filtered[col] = delta_w_filtered_risk[col]

        # Phase 2.4: w_exec 조정
        w_exec_filtered = pd.DataFrame()
        if w_exec_raw is not None and not w_exec_raw.empty and cfg.apply_filter_to_weights:
            w_exec_filtered = self._adjust_w_exec_for_cash(
                w_exec_raw, delta_w_risk, delta_w_filtered_risk
            )

        # 이벤트 생성 (필터링된 delta_w 기준)
        all_events: List[TradeEvent] = []
        ignored_by_eps = 0
        ignored_by_notional = 0
        ignored_by_netting = 0

        events_per_day_data = []
        buy_events_per_day_data = []
        sell_events_per_day_data = []

        for date in index:
            day_nav = nav.loc[date] if date in nav.index else 0.0
            day_delta_w_raw = delta_w_risk.loc[date]
            day_delta_w_filtered = delta_w_filtered_risk.loc[date]
            day_filter_mask = filter_mask.loc[date]

            day_events: List[TradeEvent] = []
            day_buy_count = 0
            day_sell_count = 0

            # Netting을 위한 buy/sell 집계
            total_buy_notional = 0.0
            total_sell_notional = 0.0

            for symbol in symbols:
                dw_raw = day_delta_w_raw[symbol]
                dw_filtered = day_delta_w_filtered[symbol]
                is_filtered = day_filter_mask[symbol]

                # 1) eps_trade 체크 (원본 기준)
                if abs(dw_raw) <= cfg.eps_trade:
                    ignored_by_eps += 1
                    continue

                # 방향 결정
                side = TradeSide.BUY if dw_raw > 0 else TradeSide.SELL
                notional_raw = abs(dw_raw) * day_nav
                notional_filtered = abs(dw_filtered) * day_nav

                # 2) min_notional 필터링 여부
                if is_filtered:
                    ignored_by_notional += 1
                    all_events.append(TradeEvent(
                        date=date,
                        symbol=symbol,
                        side=side,
                        delta_w=dw_raw,
                        notional_cash=notional_raw,
                        is_valid=False,
                        ignore_reason="below_min_notional",
                    ))
                    continue

                # 유효한 거래
                event = TradeEvent(
                    date=date,
                    symbol=symbol,
                    side=side,
                    delta_w=dw_filtered,
                    notional_cash=notional_filtered,
                    is_valid=True,
                )
                all_events.append(event)

                # Netting 계산을 위한 집계
                if cfg.enable_netting:
                    if side == TradeSide.BUY:
                        total_buy_notional += notional_filtered
                    else:
                        total_sell_notional += notional_filtered
                else:
                    # Netting 비활성: 각 거래를 개별 이벤트로 카운트
                    day_events.append(event)
                    if side == TradeSide.BUY:
                        day_buy_count += 1
                    else:
                        day_sell_count += 1

            # 3) Netting 적용 (활성화된 경우)
            if cfg.enable_netting:
                # 포트폴리오 레벨 netting
                net_notional = abs(total_buy_notional - total_sell_notional)

                if total_buy_notional > 0 or total_sell_notional > 0:
                    if net_notional >= cfg.min_notional_cash or cfg.min_notional_cash <= 0:
                        # 순거래가 min_notional 이상이면 1 이벤트로 카운트
                        if total_buy_notional > total_sell_notional:
                            day_buy_count = 1
                            day_sell_count = 0
                        elif total_sell_notional > total_buy_notional:
                            day_buy_count = 0
                            day_sell_count = 1
                        else:
                            # 완전 상쇄
                            day_buy_count = 0
                            day_sell_count = 0
                            ignored_by_netting += 1
                    else:
                        # 순거래가 min_notional 미만
                        ignored_by_netting += 1

            events_per_day_data.append(day_buy_count + day_sell_count)
            buy_events_per_day_data.append(day_buy_count)
            sell_events_per_day_data.append(day_sell_count)

        # Series 생성
        events_per_day = pd.Series(events_per_day_data, index=index, dtype=int)
        buy_events_per_day = pd.Series(buy_events_per_day_data, index=index, dtype=int)
        sell_events_per_day = pd.Series(sell_events_per_day_data, index=index, dtype=int)

        total_events = int(events_per_day.sum())
        ignored_events = ignored_by_eps + ignored_by_notional + ignored_by_netting
        days_with_events = int((events_per_day > 0).sum())

        # Phase 2.5: 필터링 영향 지표 계산 (events_per_day 포함)
        filter_impact = self._compute_filter_impact(
            delta_w_risk, delta_w_filtered_risk,
            w_exec_raw, w_exec_filtered,
            trade_notional, filter_mask, nav,
            events_per_day=events_per_day,
        )

        return TradeEventResult(
            events_per_day=events_per_day,
            events_per_asset_day=pd.DataFrame(),
            buy_events_per_day=buy_events_per_day,
            sell_events_per_day=sell_events_per_day,
            total_events=total_events,
            ignored_events=ignored_events,
            days_with_events=days_with_events,
            ignored_by_eps=ignored_by_eps,
            ignored_by_notional=ignored_by_notional,
            ignored_by_netting=ignored_by_netting,
            config=cfg.to_dict(),
            all_events=all_events,
            delta_w_filtered=delta_w_filtered,
            w_exec_filtered=w_exec_filtered,
            filter_impact=filter_impact,
        )

    def build_for_charging_mode(
        self,
        delta_w: pd.DataFrame,
        nav: pd.Series,
        charging_mode: str,
        w_exec_raw: pd.DataFrame = None,
    ) -> pd.Series:
        """
        charging_mode에 맞는 이벤트 수 시리즈 반환

        Args:
            delta_w: 가중치 변화 DataFrame
            nav: NAV 시리즈
            charging_mode: per_asset_day | per_asset_side_day | per_rebalance_day
            w_exec_raw: 원본 w_exec (Phase 2.4)

        Returns:
            pd.Series: 일자별 이벤트 수 (min_fee 계산에 직접 사용)
        """
        result = self.build(delta_w, nav, w_exec_raw=w_exec_raw)

        if charging_mode == "per_asset_day":
            # 자산별 이벤트 수 (buy/sell 합산)
            return result.events_per_day

        elif charging_mode == "per_asset_side_day":
            # buy + sell 별도 카운트
            return result.buy_events_per_day + result.sell_events_per_day

        elif charging_mode == "per_rebalance_day":
            # 이벤트가 1개라도 있으면 1
            return (result.events_per_day > 0).astype(int)

        else:
            raise ValueError(f"Unknown charging_mode: {charging_mode}")
