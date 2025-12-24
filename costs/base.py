"""
costs/base.py - CostModel 인터페이스 정의

모든 비용 모델은 이 인터페이스를 구현해야 합니다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class CostModelResult:
    """
    비용 모델 계산 결과

    Attributes:
        model_name: 모델 이름
        cost_cash: 비용 (현금 단위, 시리즈)
        cost_ratio: 비용 비율 (NAV 대비, 시리즈)
        parameters: 사용된 파라미터 기록
        metadata: 추가 메타데이터
    """
    model_name: str
    cost_cash: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    cost_ratio: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def total_cost_cash(self) -> float:
        """총 비용 (현금)"""
        return float(self.cost_cash.sum())

    def total_cost_ratio(self) -> float:
        """총 비용 비율"""
        return float(self.cost_ratio.sum())

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "model_name": self.model_name,
            "total_cost_cash": self.total_cost_cash(),
            "total_cost_ratio": self.total_cost_ratio(),
            "parameters": self.parameters,
            "metadata": self.metadata,
        }


class CostModel(ABC):
    """
    비용 모델 추상 인터페이스 (Protocol/ABC)

    모든 Phase 2 비용 모델은 이 인터페이스를 구현해야 합니다.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """모델 이름"""
        pass

    @abstractmethod
    def compute(
        self,
        nav: pd.Series,
        turnover_L1: pd.Series,
        delta_w: pd.DataFrame,
        **kwargs
    ) -> CostModelResult:
        """
        비용 계산

        Args:
            nav: NAV 시리즈 (비용 차감 전 기준)
            turnover_L1: L1 턴오버 시리즈
            delta_w: 리밸런싱 양 DataFrame (columns=symbols)
            **kwargs: 추가 데이터 (가격 등)

        Returns:
            CostModelResult: 비용 계산 결과
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """사용된 파라미터 반환"""
        pass

    def validate_inputs(
        self,
        nav: pd.Series,
        turnover_L1: pd.Series,
        delta_w: pd.DataFrame
    ) -> None:
        """
        입력 검증 (공통)

        Raises:
            ValueError: 입력이 유효하지 않은 경우
        """
        if len(nav) != len(turnover_L1):
            raise ValueError("nav and turnover_L1 must have same length")
        if len(nav) != len(delta_w):
            raise ValueError("nav and delta_w must have same length")
        if not nav.index.equals(turnover_L1.index):
            raise ValueError("nav and turnover_L1 must have same index")
        if not nav.index.equals(delta_w.index):
            raise ValueError("nav and delta_w must have same index")
