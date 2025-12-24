"""
costs/proportional.py - 비례 비용 모델 (Phase 1 래퍼)

Phase 1의 비례 비용 계산을 래핑합니다.
Phase 1 로직을 건드리지 않고 동일한 계산을 수행합니다.

비용 계산:
  cost_ratio[t] = turnover_L1[t] * one_way_rate
  cost_cash[t] = nav_before_cost[t] * cost_ratio[t]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from .base import CostModel, CostModelResult


@dataclass
class ProportionalCostConfig:
    """비례 비용 설정"""
    one_way_rate_bps: float = 5.0  # 편도 5bps (= 0.05%)


class ProportionalCostModel(CostModel):
    """
    비례 비용 모델 (Phase 1 래퍼)

    Phase 1에서 사용하는 turnover_L1 * rate 방식을 그대로 래핑합니다.
    이 모델은 Phase 1 결과와 동일한 비용을 계산해야 합니다.
    """

    def __init__(self, config: ProportionalCostConfig | None = None):
        self._config = config or ProportionalCostConfig()

    @property
    def name(self) -> str:
        return "proportional"

    def compute(
        self,
        nav: pd.Series,
        turnover_L1: pd.Series,
        delta_w: pd.DataFrame,
        **kwargs
    ) -> CostModelResult:
        """
        비례 비용 계산

        Args:
            nav: NAV 시리즈 (비용 차감 전)
            turnover_L1: L1 턴오버 시리즈
            delta_w: 리밸런싱 양 (이 모델에서는 사용 안 함)

        Returns:
            CostModelResult
        """
        self.validate_inputs(nav, turnover_L1, delta_w)

        one_way_rate = self._config.one_way_rate_bps / 10000.0

        # 비용 계산
        cost_ratio = turnover_L1 * one_way_rate
        cost_cash = nav * cost_ratio

        return CostModelResult(
            model_name=self.name,
            cost_cash=cost_cash,
            cost_ratio=cost_ratio,
            parameters=self.get_parameters(),
            metadata={
                "description": "Proportional cost: turnover_L1 * one_way_rate",
                "unit": "ratio",
            }
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "one_way_rate_bps": self._config.one_way_rate_bps,
            "one_way_rate": self._config.one_way_rate_bps / 10000.0,
        }
