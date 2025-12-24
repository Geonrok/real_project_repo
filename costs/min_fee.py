"""
costs/min_fee.py - 최소 수수료 모델 (Phase 2.2)

거래가 발생한 날에 최소 수수료를 부과합니다.
이는 소액 거래 시 비례 비용보다 최소 수수료가 더 큰 경우를 반영합니다.

Phase 2.2 변경사항:
  - TradeEventBuilder 레이어 기반으로 이벤트 산출
  - min_notional_cash: 거래 금액 임계값 (미만 거래 무시)
  - enable_netting: 동일 자산 buy/sell 상쇄

charging_mode 옵션:
  A) "per_asset_day": (기본값) 자산별로 그날 해당 자산에서 거래가 발생하면 min_fee 1회
  B) "per_asset_side_day": 자산별/사이드별로 거래 발생 시 min_fee 부과
     - buy_trade: Δw_asset > eps_trade AND notional >= min_notional_cash
     - sell_trade: Δw_asset < -eps_trade AND notional >= min_notional_cash
  C) "per_rebalance_day": 그날 포트폴리오에서 거래가 1건이라도 있으면 min_fee 1회만

eps_trade: 거래 인식 임계값 (기본 1e-12)
min_notional_cash: 최소 거래 금액 임계값 (기본 0, 비활성)
enable_netting: 동일 자산 buy/sell 상쇄 (기본 False)
CASH 제외: delta_w 컬럼에 'CASH'가 있으면 제외
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

# 경로 설정
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .base import CostModel, CostModelResult

# Phase 2.2: TradeEventBuilder 사용
try:
    from execution import TradeEventBuilder, TradeEventConfig, TradeEventResult
    HAS_EVENT_BUILDER = True
except ImportError:
    HAS_EVENT_BUILDER = False


class ChargingMode(str, Enum):
    """최소 수수료 부과 모드"""
    PER_ASSET_DAY = "per_asset_day"
    PER_ASSET_SIDE_DAY = "per_asset_side_day"
    PER_REBALANCE_DAY = "per_rebalance_day"


@dataclass
class MinFeeCostConfig:
    """
    최소 수수료 설정 (Phase 2.2)

    Attributes:
        min_fee_cash: 거래당 최소 수수료 (통화 단위)
        quote_ccy: 통화 단위 (quote currency)
        charging_mode: 부과 모드
        eps_trade: 거래 인식 임계값 (가중치 기준)
        min_notional_cash: 최소 거래 금액 임계값 (Phase 2.2)
        enable_netting: 동일 자산 buy/sell 상쇄 (Phase 2.2)
    """
    min_fee_cash: float = 1.0
    quote_ccy: str = "USDT"
    charging_mode: str = "per_asset_day"
    eps_trade: float = 1e-12
    # Phase 2.2 신규 파라미터
    min_notional_cash: float = 0.0  # 0이면 비활성 (기존 동작 유지)
    enable_netting: bool = False

    # 하위 호환성을 위한 별칭 (deprecated)
    @property
    def currency(self) -> str:
        return self.quote_ccy

    @property
    def threshold_turnover(self) -> float:
        return self.eps_trade


class MinFeeCostModel(CostModel):
    """
    최소 수수료 모델 (Phase 2.2)

    거래가 발생한 날에 최소 수수료를 부과합니다.

    Phase 2.2 개선:
      - TradeEventBuilder를 사용하여 이벤트 산출
      - min_notional_cash: 소액 거래 필터링
      - enable_netting: 동일 자산 상쇄

    charging_mode:
      - per_asset_day: 자산별로 거래 발생 시 min_fee 1회 (기본값)
      - per_asset_side_day: 자산별/사이드별로 min_fee 부과
      - per_rebalance_day: 포트폴리오에서 거래 발생 시 min_fee 1회만
    """

    VALID_MODES = {
        "per_asset_day",
        "per_asset_side_day",
        "per_rebalance_day",
    }

    def __init__(self, config: MinFeeCostConfig | None = None):
        self._config = config or MinFeeCostConfig()
        self._validate_config()
        self._event_result: Optional[TradeEventResult] = None

    def _validate_config(self) -> None:
        """설정 검증"""
        if self._config.charging_mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid charging_mode: {self._config.charging_mode}. "
                f"Valid modes: {self.VALID_MODES}"
            )
        if self._config.eps_trade < 0:
            raise ValueError("eps_trade must be non-negative")
        if self._config.min_fee_cash < 0:
            raise ValueError("min_fee_cash must be non-negative")
        if self._config.min_notional_cash < 0:
            raise ValueError("min_notional_cash must be non-negative")

    @property
    def name(self) -> str:
        return "min_fee"

    @property
    def event_result(self) -> Optional[TradeEventResult]:
        """마지막 계산의 이벤트 결과 (진단용)"""
        return self._event_result

    def _exclude_cash(self, delta_w: pd.DataFrame) -> pd.DataFrame:
        """CASH 컬럼 제외"""
        cols_to_use = [c for c in delta_w.columns if c.upper() != "CASH"]
        return delta_w[cols_to_use]

    def _compute_legacy(
        self,
        nav: pd.Series,
        delta_w: pd.DataFrame,
    ) -> tuple:
        """
        Phase 2.1 레거시 로직 (TradeEventBuilder 없이)
        하위 호환성 유지용
        """
        cfg = self._config
        index = nav.index
        mode = cfg.charging_mode
        eps = cfg.eps_trade

        delta_w_ex_cash = self._exclude_cash(delta_w)

        if mode == "per_asset_day":
            events_per_day = (delta_w_ex_cash.abs() > eps).sum(axis=1)
        elif mode == "per_asset_side_day":
            buy_trades = (delta_w_ex_cash > eps).sum(axis=1)
            sell_trades = (delta_w_ex_cash < -eps).sum(axis=1)
            events_per_day = buy_trades + sell_trades
        elif mode == "per_rebalance_day":
            events_per_day = (delta_w_ex_cash.abs() > eps).any(axis=1).astype(int)
        else:
            raise ValueError(f"Unknown charging_mode: {mode}")

        events_per_day = pd.Series(events_per_day, index=index, dtype=int)
        total_event_count = int(events_per_day.sum())
        days_with_events = int((events_per_day > 0).sum())

        return events_per_day, total_event_count, days_with_events, 0, 0, 0

    def _compute_with_event_builder(
        self,
        nav: pd.Series,
        delta_w: pd.DataFrame,
    ) -> tuple:
        """
        Phase 2.2 이벤트 빌더 기반 로직
        """
        cfg = self._config
        mode = cfg.charging_mode

        # TradeEventBuilder 설정
        event_config = TradeEventConfig(
            eps_trade=cfg.eps_trade,
            min_notional_cash=cfg.min_notional_cash,
            enable_netting=cfg.enable_netting,
            exclude_cash=True,
        )

        builder = TradeEventBuilder(event_config)
        self._event_result = builder.build(delta_w, nav)

        # charging_mode에 따른 이벤트 수 산출
        events_per_day = builder.build_for_charging_mode(delta_w, nav, mode)

        total_event_count = int(events_per_day.sum())
        days_with_events = int((events_per_day > 0).sum())
        ignored_by_eps = self._event_result.ignored_by_eps
        ignored_by_notional = self._event_result.ignored_by_notional
        ignored_by_netting = self._event_result.ignored_by_netting

        return (
            events_per_day,
            total_event_count,
            days_with_events,
            ignored_by_eps,
            ignored_by_notional,
            ignored_by_netting,
        )

    def compute(
        self,
        nav: pd.Series,
        turnover_L1: pd.Series,
        delta_w: pd.DataFrame,
        **kwargs
    ) -> CostModelResult:
        """
        최소 수수료 계산

        Args:
            nav: NAV 시리즈 (비용 계산 기준)
            turnover_L1: L1 턴오버 시리즈
            delta_w: 리밸런싱 양 DataFrame (columns=symbols)

        Returns:
            CostModelResult
        """
        self.validate_inputs(nav, turnover_L1, delta_w)

        cfg = self._config
        index = nav.index
        mode = cfg.charging_mode

        # Phase 2.2: 이벤트 빌더 사용 여부 결정
        # min_notional_cash > 0 또는 enable_netting이면 새 로직 사용
        use_event_builder = (
            HAS_EVENT_BUILDER and
            (cfg.min_notional_cash > 0 or cfg.enable_netting)
        )

        if use_event_builder:
            (
                events_per_day,
                total_event_count,
                days_with_events,
                ignored_by_eps,
                ignored_by_notional,
                ignored_by_netting,
            ) = self._compute_with_event_builder(nav, delta_w)
        else:
            (
                events_per_day,
                total_event_count,
                days_with_events,
                ignored_by_eps,
                ignored_by_notional,
                ignored_by_netting,
            ) = self._compute_legacy(nav, delta_w)

        # 비용 계산
        cost_cash = events_per_day * cfg.min_fee_cash
        cost_cash = pd.Series(cost_cash, index=index, dtype=float)

        # 비율 계산 (NAV 대비)
        cost_ratio = cost_cash / nav.replace(0, np.nan)
        cost_ratio = cost_ratio.fillna(0.0)

        # 평균 이벤트 수
        avg_events_per_day = float(events_per_day.mean()) if len(events_per_day) > 0 else 0.0

        # 무시된 총 이벤트 수
        total_ignored = ignored_by_eps + ignored_by_notional + ignored_by_netting

        return CostModelResult(
            model_name=self.name,
            cost_cash=cost_cash,
            cost_ratio=cost_ratio,
            parameters=self.get_parameters(),
            metadata={
                "description": f"Minimum fee per trade (mode={mode})",
                "quote_ccy": cfg.quote_ccy,
                "charging_mode": mode,
                "eps_trade": cfg.eps_trade,
                "approximation": False,
                # Phase 2.1 호환
                "min_fee_event_count": total_event_count,
                "days_with_min_fee": days_with_events,
                "avg_events_per_day": avg_events_per_day,
                "events_per_day": events_per_day,
                # Phase 2.2 신규 진단 정보
                "min_notional_cash": cfg.min_notional_cash,
                "enable_netting": cfg.enable_netting,
                "min_fee_event_count_effective": total_event_count,
                "min_fee_ignored_small_trades": ignored_by_notional,
                "min_fee_ignored_by_eps": ignored_by_eps,
                "min_fee_ignored_by_netting": ignored_by_netting,
                "min_fee_total_ignored": total_ignored,
            }
        )

    def get_parameters(self) -> Dict[str, Any]:
        cfg = self._config
        return {
            "min_fee_cash": cfg.min_fee_cash,
            "quote_ccy": cfg.quote_ccy,
            "charging_mode": cfg.charging_mode,
            "eps_trade": cfg.eps_trade,
            # Phase 2.2 신규
            "min_notional_cash": cfg.min_notional_cash,
            "enable_netting": cfg.enable_netting,
        }
