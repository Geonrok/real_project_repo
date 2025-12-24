"""
adapters/run_phase2_from_loader.py - Phase 2 실행기 (Phase 2.5)

Phase 1 결과를 받아서 추가 비용 모델을 적용하고 Phase 2 NAV를 계산합니다.

원칙:
- Phase 1 NAV는 'Anchor'로 보존
- Phase 2 NAV는 nav_post_open_phase2로 별도 생성
- 추가 비용은 cash subtraction 형태로 적용

Phase 2.5 변경사항 (Execution Fidelity & Distortion Control):
- Worst-days 리포트: L1 distance 상위 20일 추출 (phase25_distortion_top20.csv)
- Distortion guardrail: l1_distance_max/mean 임계값 초과 시 warning/error
- min_notional_cash sensitivity sweep (phase25_min_notional_sweep.csv)
- schema_version 2.6.0

Phase 2.4 변경사항:
- min_notional 필터 기준을 trade_notional_cash로 변경
- w_exec 조정: 생략된 주문은 CASH로 자동 흡수 (Σw=1 유지)
- 필터링 영향 리포트: pre/post turnover, L1 distance
"""

from __future__ import annotations

import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from phase1_anchor_engine import Phase1Config, Phase1Result
from adapters.run_phase1_from_loader import run_phase1_from_loader, run_phase1_from_dict
from costs import (
    CostModel,
    CostModelResult,
    CombinedCostModel,
    ProportionalCostModel,
    MinFeeCostModel,
    RoundingCostModel,
)
from costs.proportional import ProportionalCostConfig
from costs.min_fee import MinFeeCostConfig
from costs.rounding import RoundingCostConfig
from costs.combined import CombinedCostResult

# Phase 2.4: FilterImpactMetrics import
from execution import TradeEventBuilder, TradeEventConfig, FilterImpactMetrics

# 로깅 설정
logger = logging.getLogger(__name__)


# ==============================================================================
# Phase 2.5 기본값 상수
# ==============================================================================

# Phase 2.5 봉인된 기본값 (권장 설정)
PHASE25_DEFAULT_CONFIG = {
    "charging_mode": "per_rebalance_day",
    "eps_trade": 1e-12,
    "min_notional_cash": 100.0,
    "enable_netting": False,
    "min_fee_cash": 1.0,
    "quote_ccy": "USDT",
}

# 하위 호환성
PHASE24_DEFAULT_CONFIG = PHASE25_DEFAULT_CONFIG
PHASE23_DEFAULT_CONFIG = PHASE25_DEFAULT_CONFIG

# Guardrail 임계값
GUARDRAIL_EVENTS_PER_DAY_THRESHOLD = 100
GUARDRAIL_NAV_COLLAPSE_ENABLED = True

# Phase 2.5: Distortion Guardrail 임계값
GUARDRAIL_L1_DISTANCE_MAX = 0.8
GUARDRAIL_L1_DISTANCE_MEAN = 0.2
GUARDRAIL_DISTORTION_MODE = "warning"  # "warning" or "error"

# Phase 2.5: min_notional_cash sweep 그리드
MIN_NOTIONAL_SWEEP_GRID = [0, 50, 100, 200, 500]

# Phase 2.6: Default Selection & Pareto Report
PHASE26_SCHEMA_VERSION = "2.6.0"
PHASE26_TIE_TOLERANCE = 0.001
PHASE26_DEFAULT_CONSTRAINTS = {
    "require_positive_nav": True,
    "guardrail_l1_distance_max": GUARDRAIL_L1_DISTANCE_MAX,
    "guardrail_l1_distance_mean": GUARDRAIL_L1_DISTANCE_MEAN,
}

PHASE26_RECOMMENDATION_RULE = (
    "Select the feasible grid point with final_nav_p2 maximized (default), "
    "subject to final_nav_p2 > 0, l1_distance_max <= guardrail_l1_distance_max, "
    "and l1_distance_mean <= guardrail_l1_distance_mean; if multiple points are within 0.1% "
    "of the best objective, break ties by lower l1_distance_mean, then lower filter_ratio, "
    "then smaller min_notional_cash."
)


# ==============================================================================
# Exceptions
# ==============================================================================

class Phase2NAVCollapseError(RuntimeError):
    """Phase 2 NAV가 음수로 붕괴했을 때 발생하는 예외"""

    def __init__(
        self,
        final_nav: float,
        charging_mode: str,
        eps_trade: float,
        min_notional_cash: float,
        enable_netting: bool,
        total_events: int,
        n_days: int,
        filtered_ratio: float,
        events_per_day_max: int,
    ):
        self.final_nav = final_nav
        self.charging_mode = charging_mode
        self.eps_trade = eps_trade
        self.min_notional_cash = min_notional_cash
        self.enable_netting = enable_netting
        self.total_events = total_events
        self.n_days = n_days
        self.filtered_ratio = filtered_ratio
        self.events_per_day_max = events_per_day_max

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        return (
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  Phase 2 NAV Collapse Detected!                                  ║\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Final NAV: {self.final_nav:>15.2f} (< 0)                         ║\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Cause Analysis:                                                 ║\n"
            f"║    charging_mode:     {self.charging_mode:<20}                ║\n"
            f"║    eps_trade:         {self.eps_trade:<20g}                ║\n"
            f"║    min_notional_cash: {self.min_notional_cash:<20.2f}                ║\n"
            f"║    enable_netting:    {str(self.enable_netting):<20}                ║\n"
            f"║    total_events:      {self.total_events:<20,}                ║\n"
            f"║    n_days:            {self.n_days:<20,}                ║\n"
            f"║    events_per_day_max:{self.events_per_day_max:<20,}                ║\n"
            f"║    filtered_ratio:    {self.filtered_ratio:<20.2%}                ║\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Recommendations:                                                ║\n"
            f"║    1. Use charging_mode='per_rebalance_day'                      ║\n"
            f"║    2. Increase min_notional_cash (e.g., 100 USDT)                ║\n"
            f"║    3. Enable netting to reduce event count                       ║\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_nav": self.final_nav,
            "charging_mode": self.charging_mode,
            "eps_trade": self.eps_trade,
            "min_notional_cash": self.min_notional_cash,
            "enable_netting": self.enable_netting,
            "total_events": self.total_events,
            "n_days": self.n_days,
            "filtered_ratio": self.filtered_ratio,
            "events_per_day_max": self.events_per_day_max,
        }


class Phase2DistortionError(RuntimeError):
    """Phase 2.5: w_exec 왜곡이 임계값을 초과했을 때 발생하는 예외"""

    def __init__(
        self,
        l1_distance_max: float,
        l1_distance_mean: float,
        threshold_max: float,
        threshold_mean: float,
        min_notional_cash: float,
        worst_date: str,
    ):
        self.l1_distance_max = l1_distance_max
        self.l1_distance_mean = l1_distance_mean
        self.threshold_max = threshold_max
        self.threshold_mean = threshold_mean
        self.min_notional_cash = min_notional_cash
        self.worst_date = worst_date

        message = self._format_message()
        super().__init__(message)

    def _format_message(self) -> str:
        exceeded = []
        if self.l1_distance_max > self.threshold_max:
            exceeded.append(f"max={self.l1_distance_max:.4f} > {self.threshold_max}")
        if self.l1_distance_mean > self.threshold_mean:
            exceeded.append(f"mean={self.l1_distance_mean:.4f} > {self.threshold_mean}")

        return (
            f"\n"
            f"╔══════════════════════════════════════════════════════════════════╗\n"
            f"║  Phase 2.5 Distortion Guardrail Triggered!                       ║\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  L1 Distance Exceeded:                                           ║\n"
            f"║    {'; '.join(exceeded):<56} ║\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Current Settings:                                               ║\n"
            f"║    min_notional_cash: {self.min_notional_cash:<20.2f}                ║\n"
            f"║    worst_date:        {self.worst_date:<20}                ║\n"
            f"╠══════════════════════════════════════════════════════════════════╣\n"
            f"║  Recommendations:                                                ║\n"
            f"║    1. Decrease min_notional_cash to reduce w_exec distortion     ║\n"
            f"║    2. Increase guardrail thresholds if acceptable                ║\n"
            f"║    3. Review phase25_distortion_top20.csv for worst days         ║\n"
            f"╚══════════════════════════════════════════════════════════════════╝\n"
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "l1_distance_max": self.l1_distance_max,
            "l1_distance_mean": self.l1_distance_mean,
            "threshold_max": self.threshold_max,
            "threshold_mean": self.threshold_mean,
            "min_notional_cash": self.min_notional_cash,
            "worst_date": self.worst_date,
        }


# ==============================================================================
# Config & Result
# ==============================================================================

@dataclass
class Phase2Config:
    """
    Phase 2 설정 (Phase 2.5)

    Phase 2.5 봉인된 기본값:
        - charging_mode: per_rebalance_day
        - eps_trade: 1e-12
        - min_notional_cash: 100.0

    Phase 2.5 변경사항:
        - Distortion guardrail (l1_distance_max/mean)
        - Worst-days 리포트 생성
        - min_notional_cash sensitivity sweep
    """
    # Phase 1 설정 (상속)
    phase1_config: Phase1Config = field(default_factory=Phase1Config)

    # Phase 2 추가 비용 모델 설정
    enable_proportional: bool = True
    enable_min_fee: bool = True
    enable_rounding: bool = True

    # MinFee 모델 설정 (Phase 2.5 기본값 봉인)
    min_fee_cash: float = 1.0
    min_fee_charging_mode: str = "per_rebalance_day"
    min_fee_eps_trade: float = 1e-12
    quote_ccy: str = "USDT"

    # Phase 2.4 파라미터
    min_notional_cash: float = 100.0
    enable_netting: bool = False

    # Rounding 모델 설정
    rounding_pct: float = 0.0001

    # 출력 설정
    output_dir: str = r"C:\Users\고형석\outputs"
    save_outputs: bool = True

    # Guardrail 설정 (Phase 2.3)
    guardrail_events_threshold: int = GUARDRAIL_EVENTS_PER_DAY_THRESHOLD
    guardrail_nav_collapse: bool = GUARDRAIL_NAV_COLLAPSE_ENABLED

    # Phase 2.5: Distortion Guardrail 설정
    guardrail_l1_distance_max: float = GUARDRAIL_L1_DISTANCE_MAX
    guardrail_l1_distance_mean: float = GUARDRAIL_L1_DISTANCE_MEAN
    guardrail_distortion_mode: str = GUARDRAIL_DISTORTION_MODE  # "warning" or "error"

    # 하위 호환성 (deprecated)
    @property
    def min_fee_per_asset(self) -> bool:
        """deprecated: charging_mode 사용 권장"""
        return self.min_fee_charging_mode == "per_asset_day"

    def get_default_config_dict(self) -> Dict[str, Any]:
        """Phase 2.5 봉인된 기본값 반환"""
        return PHASE25_DEFAULT_CONFIG.copy()


@dataclass
class Phase2Result:
    """Phase 2 실행 결과 (Phase 2.5)"""
    # Phase 1 결과 (Anchor)
    phase1_result: Phase1Result

    # Phase 2 확장
    schema_version: str = PHASE26_SCHEMA_VERSION
    timeseries: pd.DataFrame = field(default_factory=pd.DataFrame)
    summary: Dict[str, Any] = field(default_factory=dict)

    # 비용 분해
    cost_breakdown: Dict[str, CostModelResult] = field(default_factory=dict)
    combined_cost: Optional[CombinedCostResult] = None

    # 이벤트 통계
    event_stats: Dict[str, Any] = field(default_factory=dict)
    event_stats_daily: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Phase 2.4: 필터링 영향 지표
    filter_impact: Optional[FilterImpactMetrics] = None
    w_exec_filtered: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Phase 2.5: Worst-days 리포트
    distortion_top20: pd.DataFrame = field(default_factory=pd.DataFrame)

    # 메타데이터
    manifest: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# Phase 2 Runner
# ==============================================================================

class Phase2Runner:
    """
    Phase 2 Runner (Phase 2.5)

    Phase 2.5 기능:
    - Worst-days 리포트 (L1 distance 상위 20일)
    - Distortion guardrail (l1_distance_max/mean)
    - min_notional 필터가 실제로 delta_w를 클립
    - w_exec 조정으로 Σw=1 유지
    - 필터링 영향 리포트 (pre/post turnover, L1 distance)
    """

    def __init__(self, config: Phase2Config):
        self.config = config
        self._cost_models: List[CostModel] = []
        self._build_cost_models()

    def _build_cost_models(self) -> None:
        """비용 모델 구성"""
        cfg = self.config

        if cfg.enable_min_fee:
            self._cost_models.append(MinFeeCostModel(MinFeeCostConfig(
                min_fee_cash=cfg.min_fee_cash,
                quote_ccy=cfg.quote_ccy,
                charging_mode=cfg.min_fee_charging_mode,
                eps_trade=cfg.min_fee_eps_trade,
                min_notional_cash=cfg.min_notional_cash,
                enable_netting=cfg.enable_netting,
            )))

        if cfg.enable_rounding:
            self._cost_models.append(RoundingCostModel(RoundingCostConfig(
                avg_rounding_pct=cfg.rounding_pct,
            )))

    def run(self, phase1_result: Phase1Result) -> Phase2Result:
        """Phase 2 비용 계산 및 NAV 조정"""
        ts1 = phase1_result.timeseries.copy()
        cfg = self.config

        # Phase 1 비용 및 NAV (Anchor)
        nav_phase1 = ts1["nav_post_open_phase1"].copy()
        cost_cash_phase1 = ts1["cost_cash_phase1"].copy()
        turnover_L1 = ts1["turnover_L1"].copy()

        # 비용 차감 전 NAV (Phase 1 기준)
        nav_before_cost = ts1["nav_pre_open"].copy()

        # Phase 2 추가 비용 계산
        delta_w = phase1_result.delta_w
        w_exec_raw = phase1_result.w_exec

        # Phase 2.4: TradeEventBuilder로 필터링 영향 계산
        event_builder = TradeEventBuilder(TradeEventConfig(
            eps_trade=cfg.min_fee_eps_trade,
            min_notional_cash=cfg.min_notional_cash,
            enable_netting=cfg.enable_netting,
            exclude_cash=True,
            apply_filter_to_weights=True,
        ))

        event_result = event_builder.build(delta_w, nav_before_cost, w_exec_raw=w_exec_raw)
        filter_impact = event_result.filter_impact
        w_exec_filtered = event_result.w_exec_filtered

        # 비용 모델 실행
        extra_cost_cash = pd.Series(0.0, index=ts1.index)
        extra_cost_ratio = pd.Series(0.0, index=ts1.index)
        cost_breakdown: Dict[str, CostModelResult] = {}

        combined_result = None
        if self._cost_models:
            combined = CombinedCostModel(self._cost_models)
            combined_result = combined.compute(nav_before_cost, turnover_L1, delta_w)
            extra_cost_cash = combined_result.total_cost_cash
            extra_cost_ratio = combined_result.total_cost_ratio

            for r in combined_result.individual_results:
                cost_breakdown[r.model_name] = r

        # Phase 2 총 비용
        cost_cash_phase2 = cost_cash_phase1 + extra_cost_cash
        cost_ratio_phase2 = ts1["cost_ratio_phase1"] + extra_cost_ratio

        # Phase 2 NAV 계산: 누적 추가 비용 차감
        cumulative_extra_cost = extra_cost_cash.cumsum()
        nav_phase2 = nav_phase1 - cumulative_extra_cost

        # 타임시리즈 확장
        ts2 = ts1.copy()
        ts2["nav_post_open_phase2"] = nav_phase2
        ts2["extra_cost_cash"] = extra_cost_cash
        ts2["extra_cost_ratio"] = extra_cost_ratio
        ts2["cost_cash_phase2"] = cost_cash_phase2
        ts2["cost_ratio_phase2"] = cost_ratio_phase2
        ts2["daily_return_phase2"] = ts2["nav_post_open_phase2"].pct_change()

        # 이벤트 통계 수집
        event_stats, event_stats_daily = self._collect_event_stats(
            cost_breakdown, cfg, ts1.index, turnover_L1, event_result
        )

        # Guardrail 체크 (Phase 2.3: events, nav collapse)
        self._check_guardrails(event_stats, nav_phase2, cfg)

        # Phase 2.5: Distortion guardrail 체크
        self._check_distortion_guardrails(filter_impact, cfg)

        # Phase 2.5: Worst-days 리포트 생성
        distortion_top20 = pd.DataFrame()
        if filter_impact is not None:
            distortion_top20 = filter_impact.get_worst_days(top_n=20)

        # 요약 통계
        summary = self._compute_summary(
            ts2, phase1_result, cost_breakdown, cfg, event_stats, filter_impact
        )

        # Manifest 생성
        manifest = self._create_manifest(phase1_result, cost_breakdown, cfg, filter_impact)

        result = Phase2Result(
            phase1_result=phase1_result,
            schema_version=PHASE26_SCHEMA_VERSION,
            timeseries=ts2,
            summary=summary,
            cost_breakdown=cost_breakdown,
            combined_cost=combined_result,
            event_stats=event_stats,
            event_stats_daily=event_stats_daily,
            filter_impact=filter_impact,
            w_exec_filtered=w_exec_filtered,
            distortion_top20=distortion_top20,
            manifest=manifest,
        )

        # 출력 저장
        if cfg.save_outputs:
            self._save_outputs(result)

        return result

    def _collect_event_stats(
        self,
        cost_breakdown: Dict[str, CostModelResult],
        cfg: Phase2Config,
        index: pd.DatetimeIndex,
        turnover_L1: pd.Series,
        event_result=None,
    ) -> tuple:
        """이벤트 통계 수집"""
        stats = {
            "min_notional_cash": cfg.min_notional_cash,
            "enable_netting": cfg.enable_netting,
            "eps_trade": cfg.min_fee_eps_trade,
            "charging_mode": cfg.min_fee_charging_mode,
        }

        events_per_day = pd.Series(0, index=index, dtype=int)
        ignored_per_day = pd.Series(0, index=index, dtype=int)

        if "min_fee" in cost_breakdown:
            min_fee_result = cost_breakdown["min_fee"]
            meta = min_fee_result.metadata

            total_events = meta.get("min_fee_event_count", 0)
            ignored_by_eps = meta.get("min_fee_ignored_by_eps", 0)
            ignored_by_notional = meta.get("min_fee_ignored_small_trades", 0)
            ignored_by_netting = meta.get("min_fee_ignored_by_netting", 0)
            total_ignored = meta.get("min_fee_total_ignored", 0)

            total_raw_trades = total_events + total_ignored
            filtered_ratio = total_ignored / total_raw_trades if total_raw_trades > 0 else 0.0

            stats.update({
                "min_fee_event_count": total_events,
                "min_fee_event_count_effective": meta.get("min_fee_event_count_effective", 0),
                "days_with_min_fee": meta.get("days_with_min_fee", 0),
                "avg_events_per_day": meta.get("avg_events_per_day", 0.0),
                "ignored_by_eps": ignored_by_eps,
                "ignored_by_notional": ignored_by_notional,
                "ignored_by_netting": ignored_by_netting,
                "total_ignored": total_ignored,
                "total_raw_trades": total_raw_trades,
                "min_fee_filtered_trade_ratio": filtered_ratio,
            })

            epd = meta.get("events_per_day")
            if epd is not None and hasattr(epd, "values"):
                events_per_day = epd.copy()
                top5_days = events_per_day.nlargest(5)
                stats["events_per_day_max"] = int(events_per_day.max())
                stats["events_per_day_mean"] = float(events_per_day.mean())
                stats["events_per_day_top5"] = [
                    {"date": str(d.date()), "events": int(v)}
                    for d, v in top5_days.items()
                ]

        # Phase 2.4: filter_impact 통계 추가
        if event_result and event_result.filter_impact:
            fi = event_result.filter_impact
            stats["filter_impact"] = fi.to_dict()

        event_stats_daily = pd.DataFrame({
            "events": events_per_day,
            "ignored_small_trades": ignored_per_day,
            "gross_turnover": turnover_L1,
        }, index=index)

        return stats, event_stats_daily

    def _check_guardrails(
        self,
        event_stats: Dict[str, Any],
        nav_phase2: pd.Series,
        cfg: Phase2Config,
    ) -> None:
        """Guardrail 체크"""
        max_events = event_stats.get("events_per_day_max", 0)
        if max_events > cfg.guardrail_events_threshold:
            warning_msg = (
                f"[Phase 2.4 Guardrail] events_per_day_max ({max_events:,}) "
                f"exceeds threshold ({cfg.guardrail_events_threshold:,}). "
                f"Consider using per_rebalance_day mode or increasing min_notional_cash."
            )
            warnings.warn(warning_msg, UserWarning)
            logger.warning(warning_msg)

        final_nav = float(nav_phase2.iloc[-1])
        if final_nav < 0 and cfg.guardrail_nav_collapse:
            filtered_ratio = event_stats.get("min_fee_filtered_trade_ratio", 0.0)
            total_events = event_stats.get("min_fee_event_count", 0)
            n_days = len(nav_phase2)

            raise Phase2NAVCollapseError(
                final_nav=final_nav,
                charging_mode=cfg.min_fee_charging_mode,
                eps_trade=cfg.min_fee_eps_trade,
                min_notional_cash=cfg.min_notional_cash,
                enable_netting=cfg.enable_netting,
                total_events=total_events,
                n_days=n_days,
                filtered_ratio=filtered_ratio,
                events_per_day_max=max_events,
            )

    def _check_distortion_guardrails(
        self,
        filter_impact: Optional[FilterImpactMetrics],
        cfg: Phase2Config,
    ) -> None:
        """Phase 2.5: Distortion guardrail 체크"""
        if filter_impact is None:
            return

        l1_max = filter_impact.l1_distance_max
        l1_mean = filter_impact.l1_distance_mean

        exceeded_max = l1_max > cfg.guardrail_l1_distance_max
        exceeded_mean = l1_mean > cfg.guardrail_l1_distance_mean

        if not exceeded_max and not exceeded_mean:
            return

        # Worst date 찾기
        worst_date = ""
        if not filter_impact.l1_distance_daily.empty:
            worst_idx = filter_impact.l1_distance_daily.idxmax()
            worst_date = str(worst_idx.date()) if hasattr(worst_idx, 'date') else str(worst_idx)

        if cfg.guardrail_distortion_mode == "error":
            raise Phase2DistortionError(
                l1_distance_max=l1_max,
                l1_distance_mean=l1_mean,
                threshold_max=cfg.guardrail_l1_distance_max,
                threshold_mean=cfg.guardrail_l1_distance_mean,
                min_notional_cash=cfg.min_notional_cash,
                worst_date=worst_date,
            )
        else:  # warning
            warning_parts = []
            if exceeded_max:
                warning_parts.append(f"max={l1_max:.4f} > {cfg.guardrail_l1_distance_max}")
            if exceeded_mean:
                warning_parts.append(f"mean={l1_mean:.4f} > {cfg.guardrail_l1_distance_mean}")

            warning_msg = (
                f"[Phase 2.5 Distortion Guardrail] L1 distance exceeded: "
                f"{'; '.join(warning_parts)}. "
                f"Worst date: {worst_date}. Consider decreasing min_notional_cash."
            )
            warnings.warn(warning_msg, UserWarning)
            logger.warning(warning_msg)

    def _compute_summary(
        self,
        ts: pd.DataFrame,
        p1_result: Phase1Result,
        cost_breakdown: Dict[str, CostModelResult],
        cfg: Phase2Config,
        event_stats: Dict[str, Any],
        filter_impact: Optional[FilterImpactMetrics] = None,
    ) -> Dict[str, Any]:
        """요약 통계 계산"""
        p1_summary = p1_result.summary

        nav_phase2 = ts["nav_post_open_phase2"]
        daily_ret = ts["daily_return_phase2"].dropna()
        ann_factor = cfg.phase1_config.annualization_factor

        n_days = len(nav_phase2)
        final_nav_phase2 = float(nav_phase2.iloc[-1])

        if n_days > 1:
            cagr_phase2 = (final_nav_phase2 / cfg.phase1_config.initial_nav) ** (ann_factor / (n_days - 1)) - 1
        else:
            cagr_phase2 = 0.0

        ann_vol = daily_ret.std() * np.sqrt(ann_factor) if len(daily_ret) > 0 else 0.0
        sharpe = (daily_ret.mean() * ann_factor) / (ann_vol + 1e-12) if ann_vol > 0 else 0.0

        peak = nav_phase2.cummax()
        max_dd = float((nav_phase2 / peak - 1).min())

        total_extra_cost = float(ts["extra_cost_cash"].sum())
        total_cost_phase1 = float(ts["cost_cash_phase1"].sum())
        total_cost_phase2 = float(ts["cost_cash_phase2"].sum())

        cost_by_model = {
            name: r.total_cost_cash()
            for name, r in cost_breakdown.items()
        }

        result = {
            # Phase 1 (Anchor)
            "initial_nav": cfg.phase1_config.initial_nav,
            "final_nav_phase1": p1_summary.get("final_nav_phase1", 0),
            "sharpe_phase1": p1_summary.get("sharpe_ratio", 0),
            "max_dd_phase1": p1_summary.get("max_drawdown", 0),

            # Phase 2
            "final_nav_phase2": final_nav_phase2,
            "cagr_phase2": cagr_phase2,
            "sharpe_phase2": sharpe,
            "max_dd_phase2": max_dd,

            # 비용
            "total_cost_cash_phase1": total_cost_phase1,
            "total_extra_cost_cash": total_extra_cost,
            "total_cost_cash_phase2": total_cost_phase2,
            "cost_breakdown_by_model": cost_by_model,

            # 설정
            "min_fee_charging_mode": cfg.min_fee_charging_mode,
            "min_fee_eps_trade": cfg.min_fee_eps_trade,
            "quote_ccy": cfg.quote_ccy,
            "min_notional_cash": cfg.min_notional_cash,
            "enable_netting": cfg.enable_netting,

            # 이벤트 통계
            "min_fee_event_count_effective": event_stats.get("min_fee_event_count_effective", 0),
            "min_fee_ignored_small_trades": event_stats.get("ignored_by_notional", 0),
            "min_fee_total_ignored": event_stats.get("total_ignored", 0),
            "min_fee_filtered_trade_ratio": event_stats.get("min_fee_filtered_trade_ratio", 0.0),
            "min_fee_total_raw_trades": event_stats.get("total_raw_trades", 0),

            "event_stats": {
                "events_per_day_mean": event_stats.get("events_per_day_mean", 0.0),
                "events_per_day_max": event_stats.get("events_per_day_max", 0),
                "events_per_day_top5": event_stats.get("events_per_day_top5", []),
            },

            "n_trading_days": n_days,
            "nav_diff_phase1_vs_phase2": p1_summary.get("final_nav_phase1", 0) - final_nav_phase2,
        }

        # Phase 2.4: filter_impact 추가
        if filter_impact:
            result["filter_impact"] = filter_impact.to_dict()

        return result

    def _create_manifest(
        self,
        p1_result: Phase1Result,
        cost_breakdown: Dict[str, CostModelResult],
        cfg: Phase2Config,
        filter_impact: Optional[FilterImpactMetrics] = None,
    ) -> Dict[str, Any]:
        """Manifest 생성"""
        cost_models_list = ["proportional"]
        cost_models_params = {
            "proportional": {
                "one_way_rate_bps": cfg.phase1_config.one_way_rate_bps
            }
        }

        for name, result in cost_breakdown.items():
            cost_models_list.append(name)
            cost_models_params[name] = result.parameters

        manifest = {
            "schema_version": PHASE26_SCHEMA_VERSION,
            "created_at": datetime.now().isoformat(),
            "phase1_schema_version": p1_result.schema_version,

            "default_config": PHASE25_DEFAULT_CONFIG,

            "phase1_config": {
                "signal_delay": cfg.phase1_config.signal_delay,
                "trend_ma_n": cfg.phase1_config.trend_ma_n,
                "momentum_L": cfg.phase1_config.momentum_L,
                "vol_lookback": cfg.phase1_config.vol_lookback,
                "regime_mode": cfg.phase1_config.regime_mode,
                "vol_target_ann": cfg.phase1_config.vol_target_ann,
                "max_gross_leverage": cfg.phase1_config.max_gross_leverage,
                "one_way_rate_bps": cfg.phase1_config.one_way_rate_bps,
            },

            "cost_models": cost_models_list,
            "cost_model_params": cost_models_params,

            "current_config": {
                "charging_mode": cfg.min_fee_charging_mode,
                "eps_trade": cfg.min_fee_eps_trade,
                "min_notional_cash": cfg.min_notional_cash,
                "enable_netting": cfg.enable_netting,
                "quote_ccy": cfg.quote_ccy,
            },

            "guardrails": {
                "events_threshold": cfg.guardrail_events_threshold,
                "nav_collapse_enabled": cfg.guardrail_nav_collapse,
                "l1_distance_max": cfg.guardrail_l1_distance_max,
                "l1_distance_mean": cfg.guardrail_l1_distance_mean,
                "distortion_mode": cfg.guardrail_distortion_mode,
            },

            "notes": {
                "phase1_anchor": "Phase 1 NAV is preserved as anchor",
                "phase2_nav": "Phase 2 NAV = Phase 1 NAV - extra costs",
                "rounding_approximation": cfg.enable_rounding,
                "phase24_filter": "Trade notional filter clips delta_w below min_notional_cash",
                "phase25_distortion": "Distortion guardrails monitor w_exec deviation from target",
            }
        }

        # Phase 2.4: filter_impact 추가
        if filter_impact:
            manifest["filter_impact_summary"] = filter_impact.to_dict()

        return manifest

    def _save_outputs(self, result: Phase2Result) -> None:
        """출력 파일 저장"""
        cfg = self.config
        out_dir = cfg.output_dir
        os.makedirs(out_dir, exist_ok=True)

        # Manifest
        manifest_path = os.path.join(out_dir, "real_run_manifest.json")
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(result.manifest, f, indent=2, ensure_ascii=False)

        # Summary
        summary_path = os.path.join(out_dir, "real_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result.summary, f, indent=2, ensure_ascii=False)

        # Timeseries
        ts_path = os.path.join(out_dir, "real_timeseries.csv")
        result.timeseries.to_csv(ts_path, encoding="utf-8-sig", float_format="%.16g")

        # w_exec
        wexec_path = os.path.join(out_dir, "real_w_exec.csv")
        result.phase1_result.w_exec.to_csv(wexec_path, encoding="utf-8-sig", float_format="%.16g")

        # 이벤트 통계 JSON
        event_stats_json_path = os.path.join(out_dir, "phase2_event_stats.json")
        event_stats_save = {
            k: v for k, v in result.event_stats.items()
            if k not in ("events_per_day_list",)
        }
        with open(event_stats_json_path, "w", encoding="utf-8") as f:
            json.dump(event_stats_save, f, indent=2, ensure_ascii=False)

        # 이벤트 통계 CSV
        event_stats_csv_path = os.path.join(out_dir, "phase2_event_stats.csv")
        result.event_stats_daily.to_csv(event_stats_csv_path, encoding="utf-8-sig", float_format="%.16g")

        # Phase 2.4: filter_impact_summary.json
        if result.filter_impact:
            filter_impact_path = os.path.join(out_dir, "phase24_filter_impact_summary.json")
            with open(filter_impact_path, "w", encoding="utf-8") as f:
                json.dump(result.filter_impact.to_dict(), f, indent=2, ensure_ascii=False)

        # Phase 2.4: w_exec_filtered.csv
        if result.w_exec_filtered is not None and not result.w_exec_filtered.empty:
            w_exec_filtered_path = os.path.join(out_dir, "real_w_exec_filtered.csv")
            result.w_exec_filtered.to_csv(w_exec_filtered_path, encoding="utf-8-sig", float_format="%.16g")

        # Phase 2.5: distortion_top20.csv (worst-days report)
        if result.distortion_top20 is not None and not result.distortion_top20.empty:
            distortion_path = os.path.join(out_dir, "phase25_distortion_top20.csv")
            result.distortion_top20.to_csv(distortion_path, index=False, encoding="utf-8-sig", float_format="%.6g")

        print(f"Outputs saved to {out_dir}")


# ==============================================================================
# Public API
# ==============================================================================

def run_phase2_from_loader(
    venue_path: str,
    config: Optional[Phase2Config] = None
) -> Optional[Phase2Result]:
    """거래소 디렉토리에서 데이터를 로드하고 Phase 2 백테스트 실행"""
    if config is None:
        config = Phase2Config()

    p1_result = run_phase1_from_loader(
        venue_path,
        config=config.phase1_config,
        min_days=config.phase1_config.min_history_days
    )

    if p1_result is None:
        return None

    runner = Phase2Runner(config)
    return runner.run(p1_result)


def run_phase2_from_phase1_result(
    phase1_result: Phase1Result,
    config: Optional[Phase2Config] = None
) -> Phase2Result:
    """Phase 1 결과를 받아서 Phase 2 실행"""
    if config is None:
        config = Phase2Config()

    runner = Phase2Runner(config)
    return runner.run(phase1_result)


def run_min_notional_sweep(
    phase1_result: Phase1Result,
    grid: Optional[List[float]] = None,
    base_config: Optional[Phase2Config] = None,
    output_dir: Optional[str] = None,
) -> pd.DataFrame:
    """
    Phase 2.5: min_notional_cash sensitivity sweep

    다양한 min_notional_cash 값에 대해 Phase 2를 실행하고
    filter_impact + final_nav + events 통계를 수집합니다.

    Args:
        phase1_result: Phase 1 결과
        grid: min_notional_cash 그리드 (기본: [0, 50, 100, 200, 500])
        base_config: 기본 설정 (min_notional_cash 제외)
        output_dir: 결과 저장 디렉토리 (None이면 저장 안 함)

    Returns:
        DataFrame with sweep results
    """
    if grid is None:
        grid = MIN_NOTIONAL_SWEEP_GRID

    if base_config is None:
        base_config = Phase2Config(save_outputs=False)

    results = []

    for min_notional in grid:
        # 설정 복사 및 min_notional_cash 변경
        cfg = Phase2Config(
            phase1_config=base_config.phase1_config,
            enable_proportional=base_config.enable_proportional,
            enable_min_fee=base_config.enable_min_fee,
            enable_rounding=base_config.enable_rounding,
            min_fee_cash=base_config.min_fee_cash,
            min_fee_charging_mode=base_config.min_fee_charging_mode,
            min_fee_eps_trade=base_config.min_fee_eps_trade,
            quote_ccy=base_config.quote_ccy,
            min_notional_cash=min_notional,
            enable_netting=base_config.enable_netting,
            rounding_pct=base_config.rounding_pct,
            save_outputs=False,  # sweep 중에는 개별 저장 안 함
            guardrail_nav_collapse=False,  # sweep 중에는 예외 발생 안 함
            guardrail_distortion_mode="warning",  # warning만
        )

        try:
            result = run_phase2_from_phase1_result(phase1_result, cfg)

            row = {
                "min_notional_cash": min_notional,
                "final_nav_phase2": result.summary["final_nav_phase2"],
                "total_extra_cost": result.summary["total_extra_cost_cash"],
                "min_fee_cost": result.summary.get("cost_breakdown_by_model", {}).get("min_fee", 0),
                "rounding_cost": result.summary.get("cost_breakdown_by_model", {}).get("rounding", 0),
                "events_total": result.event_stats.get("min_fee_event_count", 0),
                "events_per_day_mean": result.event_stats.get("events_per_day_mean", 0),
                "events_per_day_max": result.event_stats.get("events_per_day_max", 0),
            }

            if result.filter_impact:
                fi = result.filter_impact
                row.update({
                    "pre_filter_turnover": fi.pre_filter_turnover_L1,
                    "post_filter_turnover": fi.post_filter_turnover_L1,
                    "filter_ratio": fi.filter_ratio,
                    "l1_distance_mean": fi.l1_distance_mean,
                    "l1_distance_max": fi.l1_distance_max,
                    "turnover_reduction": 1 - (fi.post_filter_turnover_L1 / fi.pre_filter_turnover_L1)
                        if fi.pre_filter_turnover_L1 > 0 else 0,
                })

            results.append(row)

        except Exception as e:
            logger.warning(f"Sweep failed for min_notional={min_notional}: {e}")
            results.append({
                "min_notional_cash": min_notional,
                "error": str(e),
            })

    df = pd.DataFrame(results)

    # 결과 저장
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        sweep_path = os.path.join(output_dir, "phase25_min_notional_sweep.csv")
        df.to_csv(sweep_path, index=False, encoding="utf-8-sig", float_format="%.6g")
        print(f"Sweep results saved to {sweep_path}")

    return df


# ==============================================================================
# Phase 2.6 Recommendation & Pareto Report
# ==============================================================================

def _resolve_phase26_columns(sweep_df: pd.DataFrame) -> Dict[str, str]:
    nav_candidates = ["final_nav_phase2", "final_nav_p2"]
    cost_candidates = ["total_extra_cost", "total_extra_cost_cash"]
    l1_mean_candidates = ["l1_distance_mean"]
    l1_max_candidates = ["l1_distance_max"]
    filter_ratio_candidates = ["filter_ratio"]

    def pick(candidates: List[str], label: str) -> str:
        for col in candidates:
            if col in sweep_df.columns:
                return col
        raise KeyError(f"Missing required column for {label}: {candidates}")

    return {
        "nav": pick(nav_candidates, "final_nav_p2"),
        "cost": pick(cost_candidates, "total_extra_cost"),
        "l1_mean": pick(l1_mean_candidates, "l1_distance_mean"),
        "l1_max": pick(l1_max_candidates, "l1_distance_max"),
        "filter_ratio": pick(filter_ratio_candidates, "filter_ratio"),
    }


def _normalize_phase26_constraints(constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = PHASE26_DEFAULT_CONSTRAINTS.copy()
    if constraints:
        merged.update(constraints)
    return merged


def _apply_phase26_constraints(
    df: pd.DataFrame,
    cols: Dict[str, str],
    constraints: Dict[str, Any],
) -> pd.Series:
    nav_col = cols["nav"]
    l1_max_col = cols["l1_max"]
    l1_mean_col = cols["l1_mean"]

    require_positive = constraints.get("require_positive_nav", True)
    guardrail_l1_max = constraints.get("guardrail_l1_distance_max", GUARDRAIL_L1_DISTANCE_MAX)
    guardrail_l1_mean = constraints.get("guardrail_l1_distance_mean", GUARDRAIL_L1_DISTANCE_MEAN)

    mask = pd.Series(True, index=df.index)
    if require_positive:
        mask &= df[nav_col] > 0
    mask &= df[l1_max_col] <= guardrail_l1_max
    mask &= df[l1_mean_col] <= guardrail_l1_mean
    mask &= df[nav_col].notna()
    mask &= df[l1_max_col].notna()
    mask &= df[l1_mean_col].notna()

    return mask


def _sorted_phase26_feasible(
    df: pd.DataFrame,
    cols: Dict[str, str],
    objective: str,
) -> pd.DataFrame:
    nav_col = cols["nav"]
    cost_col = cols["cost"]
    l1_mean_col = cols["l1_mean"]
    filter_ratio_col = cols["filter_ratio"]

    if objective == "max_nav":
        sort_cols = [nav_col, l1_mean_col, filter_ratio_col, "min_notional_cash"]
        ascending = [False, True, True, True]
    elif objective == "min_cost":
        sort_cols = [cost_col, l1_mean_col, filter_ratio_col, "min_notional_cash"]
        ascending = [True, True, True, True]
    else:
        raise ValueError(f"Unknown objective: {objective}")

    return df.sort_values(sort_cols, ascending=ascending, kind="mergesort")


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return float(value)


def _extract_phase26_metrics(row: pd.Series, cols: Dict[str, str]) -> Dict[str, Any]:
    metrics = {
        "min_notional_cash": _safe_float(row.get("min_notional_cash")),
        "final_nav_p2": _safe_float(row.get(cols["nav"])),
        "total_extra_cost": _safe_float(row.get(cols["cost"])),
        "filter_ratio": _safe_float(row.get(cols["filter_ratio"])),
        "l1_distance_mean": _safe_float(row.get(cols["l1_mean"])),
        "l1_distance_max": _safe_float(row.get(cols["l1_max"])),
    }
    if "events_total" in row.index:
        metrics["events_total"] = _safe_float(row.get("events_total"))
    return metrics


def recommend_min_notional_from_sweep(
    sweep_df: pd.DataFrame,
    constraints: Optional[Dict[str, Any]] = None,
    objective: str = "max_nav",
) -> Dict[str, Any]:
    """
    Phase 2.6 recommendation from Phase 2.5 sweep.

    Rule: Select the feasible grid point with final_nav_p2 maximized (default), subject to final_nav_p2 > 0, l1_distance_max <= guardrail_l1_distance_max, and l1_distance_mean <= guardrail_l1_distance_mean; if multiple points are within 0.1% of the best objective, break ties by lower l1_distance_mean, then lower filter_ratio, then smaller min_notional_cash.
    """
    cols = _resolve_phase26_columns(sweep_df)
    constraints_used = _normalize_phase26_constraints(constraints)

    feasible_mask = _apply_phase26_constraints(sweep_df, cols, constraints_used)
    feasible_df = sweep_df.loc[feasible_mask].copy()
    ranking_df = _sorted_phase26_feasible(feasible_df, cols, objective) if not feasible_df.empty else feasible_df

    selected_value = None
    if not feasible_df.empty:
        nav_col = cols["nav"]
        cost_col = cols["cost"]
        l1_mean_col = cols["l1_mean"]
        filter_ratio_col = cols["filter_ratio"]

        if objective == "max_nav":
            best_value = feasible_df[nav_col].max()
            within_tol = feasible_df[nav_col] >= best_value * (1 - PHASE26_TIE_TOLERANCE)
        elif objective == "min_cost":
            best_value = feasible_df[cost_col].min()
            within_tol = feasible_df[cost_col] <= best_value * (1 + PHASE26_TIE_TOLERANCE)
        else:
            raise ValueError(f"Unknown objective: {objective}")

        tie_df = feasible_df.loc[within_tol].sort_values(
            [l1_mean_col, filter_ratio_col, "min_notional_cash"],
            ascending=[True, True, True],
            kind="mergesort",
        )
        if not tie_df.empty:
            selected_value = _safe_float(tie_df.iloc[0]["min_notional_cash"])

    ranking_top5 = []
    if not ranking_df.empty:
        for _, row in ranking_df.head(5).iterrows():
            ranking_top5.append(_extract_phase26_metrics(row, cols))

    return {
        "selected_min_notional_cash": selected_value,
        "constraints": constraints_used,
        "objective": objective,
        "ranking_top5": ranking_top5,
        "selection_rule": PHASE26_RECOMMENDATION_RULE,
    }


def build_phase26_pareto_df(
    sweep_df: pd.DataFrame,
    constraints: Optional[Dict[str, Any]] = None,
    objective: str = "max_nav",
) -> pd.DataFrame:
    cols = _resolve_phase26_columns(sweep_df)
    constraints_used = _normalize_phase26_constraints(constraints)

    pareto_df = sweep_df.copy()
    feasible_mask = _apply_phase26_constraints(pareto_df, cols, constraints_used)
    pareto_df["is_feasible"] = feasible_mask
    pareto_df["rank"] = np.nan

    feasible_df = pareto_df.loc[feasible_mask].copy()
    if not feasible_df.empty:
        sorted_df = _sorted_phase26_feasible(feasible_df, cols, objective)
        ranks = pd.Series(range(1, len(sorted_df) + 1), index=sorted_df.index)
        pareto_df.loc[ranks.index, "rank"] = ranks

    return pareto_df


def save_phase26_reports(
    sweep_df: pd.DataFrame,
    output_dir: str,
    constraints: Optional[Dict[str, Any]] = None,
    objective: str = "max_nav",
    selected_grid: Optional[List[float]] = None,
    save_plot: bool = False,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    recommendation = recommend_min_notional_from_sweep(
        sweep_df=sweep_df,
        constraints=constraints,
        objective=objective,
    )
    pareto_df = build_phase26_pareto_df(
        sweep_df=sweep_df,
        constraints=constraints,
        objective=objective,
    )

    recommendation_path = os.path.join(output_dir, "phase26_recommendation.json")
    with open(recommendation_path, "w", encoding="utf-8") as f:
        json.dump(recommendation, f, indent=2, ensure_ascii=False)

    pareto_path = os.path.join(output_dir, "phase26_pareto.csv")
    pareto_df.to_csv(pareto_path, index=False, encoding="utf-8-sig", float_format="%.6g")

    if save_plot:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            plt = None
        if plt is not None:
            cols = _resolve_phase26_columns(sweep_df)
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.scatter(
                pareto_df["min_notional_cash"],
                pareto_df[cols["nav"]],
                c=pareto_df["is_feasible"].map({True: "tab:green", False: "tab:red"}),
                alpha=0.8,
            )
            ax.set_xlabel("min_notional_cash")
            ax.set_ylabel(cols["nav"])
            ax.set_title("Phase 2.6 Tradeoff (NAV vs min_notional_cash)")
            fig.tight_layout()
            plot_path = os.path.join(output_dir, "phase26_tradeoff.png")
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

    manifest_path = os.path.join(output_dir, "real_run_manifest.json")
    if os.path.exists(manifest_path):
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["schema_version"] = PHASE26_SCHEMA_VERSION
        manifest["recommendation"] = {
            "min_notional_cash": recommendation.get("selected_min_notional_cash"),
            "objective": recommendation.get("objective"),
            "constraints": recommendation.get("constraints"),
            "selected_from_grid": selected_grid if selected_grid is not None else sweep_df["min_notional_cash"].tolist(),
        }
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)

    return {
        "recommendation": recommendation,
        "pareto_df": pareto_df,
    }


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    BASE_DATA = r"E:\OneDrive\고형석\코인\data"
    venue = "binance_futures"
    venue_path = os.path.join(BASE_DATA, venue)

    if os.path.exists(venue_path):
        print(f"Running Phase 2.5 for {venue}...")
        print(f"Default config: {PHASE25_DEFAULT_CONFIG}")

        try:
            result = run_phase2_from_loader(venue_path)
            if result:
                print(f"\n=== Phase 1 vs Phase 2 ===")
                print(f"Final NAV Phase1: {result.summary['final_nav_phase1']:.2f}")
                print(f"Final NAV Phase2: {result.summary['final_nav_phase2']:.2f}")
                print(f"NAV Difference: {result.summary['nav_diff_phase1_vs_phase2']:.2f}")

                print(f"\n=== Cost Breakdown ===")
                print(f"Phase1 Cost: {result.summary['total_cost_cash_phase1']:.2f}")
                print(f"Extra Cost: {result.summary['total_extra_cost_cash']:.2f}")
                print(f"Phase2 Total Cost: {result.summary['total_cost_cash_phase2']:.2f}")
                for model, cost in result.summary['cost_breakdown_by_model'].items():
                    print(f"  {model}: {cost:.2f}")

                print(f"\n=== Phase 2.5 Filter Impact ===")
                if result.filter_impact:
                    fi = result.filter_impact
                    print(f"Pre-filter Turnover L1: {fi.pre_filter_turnover_L1:.4f}")
                    print(f"Post-filter Turnover L1: {fi.post_filter_turnover_L1:.4f}")
                    print(f"Turnover Reduction: {1 - fi.post_filter_turnover_L1/fi.pre_filter_turnover_L1:.2%}")
                    print(f"Filter Ratio: {fi.filter_ratio:.2%}")
                    print(f"L1 Distance (mean): {fi.l1_distance_mean:.6f}")
                    print(f"L1 Distance (max): {fi.l1_distance_max:.6f}")
                    print(f"Trades Filtered: {fi.n_trades_filtered:,} / {fi.n_trades_total:,}")

                print(f"\n=== Phase 2.5 Worst-Days (Top 5) ===")
                if not result.distortion_top20.empty:
                    top5 = result.distortion_top20.head(5)
                    for _, row in top5.iterrows():
                        print(f"  {row['date']}: L1={row['l1_distance']:.4f}, "
                              f"filtered={row['filtered_notional_day']:.0f} USDT")

                # Phase 2.5: min_notional_cash sweep
                print(f"\n=== Running min_notional_cash Sweep ===")
                sweep_df = run_min_notional_sweep(
                    result.phase1_result,
                    grid=MIN_NOTIONAL_SWEEP_GRID,
                    output_dir=result.config.output_dir if hasattr(result, 'config') else r"C:\Users\고형석\outputs",
                )
                print(sweep_df[["min_notional_cash", "final_nav_phase2", "l1_distance_mean", "l1_distance_max"]].to_string(index=False))

        except Phase2NAVCollapseError as e:
            print(str(e))
        except Phase2DistortionError as e:
            print(str(e))
