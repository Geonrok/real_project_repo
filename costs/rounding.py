"""
costs/rounding.py - 라운딩 비용 근사 모델

주문 수량/호가 라운딩으로 인한 추가 비용을 근사합니다.

실제로는:
  - 최소 주문 단위 (lot size)
  - 호가 단위 (tick size)
  - 최소 notional 요구사항

이러한 제약으로 인해 목표 비중과 실제 체결 비중 사이에 차이가 발생합니다.
이 모델은 이를 단순화하여 "평균 라운딩 오차"로 비용을 근사합니다.

[주의] 이 모델은 근사(approximation)입니다. 실제 체결 시뮬레이션과 다를 수 있습니다.

비용 계산 (1차 근사):
  rounding_cost_ratio[t] = avg_rounding_pct * turnover_L1[t]

여기서 avg_rounding_pct는 "평균적으로 라운딩으로 인해 발생하는 추가 비용 비율"입니다.
보수적으로 0.5 * tick_size / avg_price로 추정할 수 있습니다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import CostModel, CostModelResult


@dataclass
class RoundingCostConfig:
    """라운딩 비용 설정"""
    # 단순화 모드: 고정 비율 사용
    avg_rounding_pct: float = 0.0001  # 0.01% (1 bps) 기본값

    # 고급 모드: tick_size / lot_size 기반 계산 (Phase 2.1 이후)
    use_tick_size: bool = False
    tick_size_bps: float = 1.0  # 호가 단위 (가격의 bps)
    lot_size_pct: float = 0.001  # 최소 주문 단위 (notional의 %)


class RoundingCostModel(CostModel):
    """
    라운딩 비용 근사 모델

    [주의] 이 모델은 1차 근사입니다.
    실제 체결 시뮬레이션과 다를 수 있으며, 파라미터는 경험적으로 조정이 필요합니다.
    """

    def __init__(self, config: RoundingCostConfig | None = None):
        self._config = config or RoundingCostConfig()

    @property
    def name(self) -> str:
        return "rounding"

    def compute(
        self,
        nav: pd.Series,
        turnover_L1: pd.Series,
        delta_w: pd.DataFrame,
        **kwargs
    ) -> CostModelResult:
        """
        라운딩 비용 계산

        Args:
            nav: NAV 시리즈
            turnover_L1: L1 턴오버 시리즈
            delta_w: 리밸런싱 양 DataFrame

        Returns:
            CostModelResult
        """
        self.validate_inputs(nav, turnover_L1, delta_w)

        cfg = self._config
        index = nav.index

        if cfg.use_tick_size:
            # 고급 모드: tick_size/lot_size 기반 계산
            # 라운딩 오차 = 0.5 * (tick_size + lot_size)
            # 이를 turnover에 비례하는 비용으로 환산
            tick_rate = cfg.tick_size_bps / 10000.0
            lot_rate = cfg.lot_size_pct / 100.0
            avg_rounding = 0.5 * (tick_rate + lot_rate)
        else:
            avg_rounding = cfg.avg_rounding_pct

        # 거래가 있는 날에만 라운딩 비용 발생
        # cost_ratio = avg_rounding * turnover_L1
        cost_ratio = turnover_L1 * avg_rounding
        cost_cash = nav * cost_ratio

        return CostModelResult(
            model_name=self.name,
            cost_cash=cost_cash,
            cost_ratio=cost_ratio,
            parameters=self.get_parameters(),
            metadata={
                "description": "Rounding cost approximation (tick/lot size)",
                "approximation": True,
                "approximation_method": "avg_rounding * turnover_L1",
                "notes": "This is a first-order approximation. Actual execution may differ.",
            }
        )

    def get_parameters(self) -> Dict[str, Any]:
        cfg = self._config
        return {
            "avg_rounding_pct": cfg.avg_rounding_pct,
            "use_tick_size": cfg.use_tick_size,
            "tick_size_bps": cfg.tick_size_bps,
            "lot_size_pct": cfg.lot_size_pct,
        }
