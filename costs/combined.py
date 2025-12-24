"""
costs/combined.py - 여러 비용 모델 합성

여러 CostModel을 조합하여 총 비용을 계산합니다.

비용 합산:
  total_cost_cash = sum(model.cost_cash for model in models)
  total_cost_ratio = sum(model.cost_ratio for model in models)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .base import CostModel, CostModelResult


@dataclass
class CombinedCostResult:
    """
    합성 비용 모델 결과

    개별 모델 결과와 합계를 모두 포함합니다.
    """
    model_name: str = "combined"
    individual_results: List[CostModelResult] = field(default_factory=list)
    total_cost_cash: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    total_cost_ratio: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))

    def sum_cost_cash(self) -> float:
        """총 비용 (현금)"""
        return float(self.total_cost_cash.sum())

    def sum_cost_ratio(self) -> float:
        """총 비용 비율"""
        return float(self.total_cost_ratio.sum())

    def breakdown_by_model(self) -> Dict[str, float]:
        """모델별 비용 분해"""
        return {
            r.model_name: r.total_cost_cash()
            for r in self.individual_results
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_cost_cash": self.sum_cost_cash(),
            "total_cost_ratio": self.sum_cost_ratio(),
            "breakdown": self.breakdown_by_model(),
            "individual_models": [r.to_dict() for r in self.individual_results],
        }


class CombinedCostModel(CostModel):
    """
    합성 비용 모델

    여러 CostModel을 조합하여 총 비용을 계산합니다.

    Example:
        combined = CombinedCostModel([
            ProportionalCostModel(ProportionalCostConfig(one_way_rate_bps=5.0)),
            MinFeeCostModel(MinFeeCostConfig(min_fee_cash=1.0)),
            RoundingCostModel(RoundingCostConfig(avg_rounding_pct=0.0001)),
        ])
    """

    def __init__(self, models: List[CostModel]):
        if not models:
            raise ValueError("At least one cost model must be provided")
        self._models = models

    @property
    def name(self) -> str:
        return "combined"

    @property
    def models(self) -> List[CostModel]:
        """내부 모델 리스트"""
        return self._models

    def compute(
        self,
        nav: pd.Series,
        turnover_L1: pd.Series,
        delta_w: pd.DataFrame,
        **kwargs
    ) -> CombinedCostResult:
        """
        합성 비용 계산

        Args:
            nav: NAV 시리즈
            turnover_L1: L1 턴오버 시리즈
            delta_w: 리밸런싱 양 DataFrame
            **kwargs: 추가 데이터

        Returns:
            CombinedCostResult
        """
        self.validate_inputs(nav, turnover_L1, delta_w)

        index = nav.index
        individual_results: List[CostModelResult] = []
        total_cost_cash = pd.Series(0.0, index=index)
        total_cost_ratio = pd.Series(0.0, index=index)

        for model in self._models:
            result = model.compute(nav, turnover_L1, delta_w, **kwargs)
            individual_results.append(result)
            total_cost_cash += result.cost_cash.fillna(0.0)
            total_cost_ratio += result.cost_ratio.fillna(0.0)

        return CombinedCostResult(
            model_name=self.name,
            individual_results=individual_results,
            total_cost_cash=total_cost_cash,
            total_cost_ratio=total_cost_ratio,
        )

    def get_parameters(self) -> Dict[str, Any]:
        return {
            "models": [m.name for m in self._models],
            "model_params": {m.name: m.get_parameters() for m in self._models},
        }

    def compute_breakdown(
        self,
        nav: pd.Series,
        turnover_L1: pd.Series,
        delta_w: pd.DataFrame,
        **kwargs
    ) -> Dict[str, CostModelResult]:
        """
        모델별 비용 분해 계산

        Returns:
            Dict[model_name, CostModelResult]
        """
        result = self.compute(nav, turnover_L1, delta_w, **kwargs)
        return {r.model_name: r for r in result.individual_results}
