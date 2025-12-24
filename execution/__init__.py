"""
execution/ - Trade Event Builder 모듈 (Phase 2.4)

거래 이벤트를 정의하고 필터링하는 레이어입니다.

Phase 2.4 변경사항:
  - min_notional 필터 기준을 trade_notional_cash로 변경
  - trade_notional_cash < min_notional_cash 이면 delta_w를 0으로 클립
  - w_exec 조정: 생략된 주문은 CASH로 자동 흡수 (Σw=1 유지)
  - netting 옵션 실제 구현: buy/sell 상계 후 순주문만 이벤트로 카운트
  - FilterImpactMetrics: 필터링 영향 리포트 (pre/post turnover, L1 distance)
"""

from .event_builder import (
    TradeSide,
    TradeEvent,
    TradeEventResult,
    TradeEventBuilder,
    TradeEventConfig,
    FilterImpactMetrics,
)

__all__ = [
    "TradeSide",
    "TradeEvent",
    "TradeEventResult",
    "TradeEventBuilder",
    "TradeEventConfig",
    "FilterImpactMetrics",
]
