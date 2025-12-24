# Phase 2.5: Execution Fidelity & Distortion Control

## Overview

Phase 2.5 completes the distortion monitoring and control layer for the backtester. It adds worst-days reporting, distortion guardrails, and min_notional sensitivity analysis to ensure execution fidelity is measurable and controllable.

## Why Phase 2.5? (5-Line Summary)

```
1. Phase 2.4의 min_notional 필터는 w_exec 왜곡을 유발하지만, 탐지/제어 수단이 부재했음
2. Worst-days 리포트: L1 distance 상위 20일을 추출하여 왜곡이 집중된 시점을 파악
3. Distortion guardrail: l1_distance_max/mean 임계값 초과 시 warning/error 발생
4. Sensitivity sweep: min_notional_cash 그리드별 NAV/L1 트레이드오프 분석
5. 이로써 필터 강도 ↔ 실행 충실도 간의 균형점을 정량적으로 선택 가능
```

## Key Changes

### 1. Worst-Days Report

L1 distance 상위 20일을 추출하여 `outputs/phase25_distortion_top20.csv` 저장:

| Column | Description |
|--------|-------------|
| date | 날짜 |
| l1_distance | w_exec_raw와 w_exec_filtered 간 L1 거리 |
| pre_filter_turnover_L1 | 필터 전 일별 턴오버 |
| post_filter_turnover_L1 | 필터 후 일별 턴오버 |
| filter_ratio_day | 일별 필터링 비율 |
| filtered_notional_day | 필터링된 notional (USDT) |
| total_notional_day | 총 notional (USDT) |
| events_day | 일별 이벤트 수 |
| top_filtered_assets | 상위 5개 필터링 자산 (자산명:금액) |

### 2. Distortion Guardrail

```python
# Phase2Config 파라미터
guardrail_l1_distance_max: float = 0.8   # 최대 L1 임계값
guardrail_l1_distance_mean: float = 0.2  # 평균 L1 임계값
guardrail_distortion_mode: str = "warning"  # "warning" or "error"
```

- **warning 모드**: 임계값 초과 시 UserWarning 발생, 실행 계속
- **error 모드**: 임계값 초과 시 Phase2DistortionError 예외 발생

### 3. min_notional_cash Sensitivity Sweep

```python
sweep_df = run_min_notional_sweep(
    phase1_result,
    grid=[0, 50, 100, 200, 500],  # 기본 그리드
    output_dir="outputs",
)
```

결과: `outputs/phase25_min_notional_sweep.csv`

| Column | Description |
|--------|-------------|
| min_notional_cash | 그리드 값 |
| final_nav_phase2 | 최종 NAV |
| total_extra_cost | 추가 비용 합계 |
| l1_distance_mean | 평균 L1 왜곡 |
| l1_distance_max | 최대 L1 왜곡 |
| filter_ratio | 필터링 비율 |
| turnover_reduction | 턴오버 감소율 |

### 4. FilterImpactMetrics.daily_details

Phase 2.5에서 추가된 일별 상세 지표:

```python
fi = result.filter_impact
daily = fi.daily_details  # DataFrame

# worst-days 조회
worst20 = fi.get_worst_days(top_n=20)
```

## Schema Changes

| Field | Version | Notes |
|-------|---------|-------|
| `schema_version` | `2.4.0` → `2.5.0` | |
| `guardrails.l1_distance_max` | NEW | 0.8 default |
| `guardrails.l1_distance_mean` | NEW | 0.2 default |
| `guardrails.distortion_mode` | NEW | "warning" default |
| `Phase2Result.distortion_top20` | NEW | DataFrame |
| `FilterImpactMetrics.daily_details` | NEW | DataFrame |

## Files Changed

```
execution/event_builder.py (MODIFIED)
  - FilterImpactMetrics.daily_details: pd.DataFrame
  - FilterImpactMetrics.get_worst_days() method
  - _compute_filter_impact(): daily_details 계산 추가

adapters/run_phase2_from_loader.py (MODIFIED)
  - Schema version 2.5.0
  - PHASE25_DEFAULT_CONFIG constant
  - Phase2DistortionError exception class
  - Phase2Config: guardrail_l1_distance_max/mean, guardrail_distortion_mode
  - Phase2Result.distortion_top20 field
  - Phase2Runner._check_distortion_guardrails() method
  - run_min_notional_sweep() function
  - _save_outputs(): phase25_distortion_top20.csv
  - Updated main block for Phase 2.5

tests/test_phase25_distortion.py (NEW)
  - 19 tests for Phase 2.5 features
  - TestWorstDaysReport: 5 tests
  - TestDistortionGuardrail: 4 tests
  - TestMinNotionalSweep: 5 tests
  - TestSchemaVersion25: 3 tests
  - TestDailyDetails: 2 tests

tests/test_phase23_guardrails.py (MODIFIED)
  - Schema version updated to 2.5.0

tests/test_phase2_integration_smoke.py (MODIFIED)
  - Schema version updated to 2.5.0

docs/PHASE25_CHANGELOG.md (NEW)
  - This documentation
```

## Test Results

```
============================= 125 passed in 28.62s =============================
  - Phase 2.2 event builder tests: 18
  - Phase 2.3 guardrail tests: 16
  - Phase 2.4 filter impact tests: 22
  - Phase 2.5 distortion tests: 19
  - Cost model handcalc tests: 20
  - Integration smoke tests: 15
  - Sensitivity smoke tests: 15
```

## New Output Files

| File | Description |
|------|-------------|
| `phase25_distortion_top20.csv` | L1 distance 상위 20일 리포트 |
| `phase25_min_notional_sweep.csv` | min_notional 그리드 sweep 결과 |

## Migration Notes

### From Phase 2.4 to 2.5

1. **Schema version changed**: `2.4.0` → `2.5.0`

2. **New config parameters**:
   ```python
   config = Phase2Config(
       guardrail_l1_distance_max=0.8,    # NEW
       guardrail_l1_distance_mean=0.2,   # NEW
       guardrail_distortion_mode="warning",  # NEW
   )
   ```

3. **New exception type**:
   - `Phase2DistortionError` may be raised if `guardrail_distortion_mode="error"`

4. **New result fields**:
   - `result.distortion_top20`: DataFrame with worst-days
   - `result.filter_impact.daily_details`: DataFrame with daily metrics

5. **Backward compatibility**:
   - `PHASE24_DEFAULT_CONFIG` is an alias for `PHASE25_DEFAULT_CONFIG`
   - All Phase 2.4 tests continue to pass

### Accessing Worst-Days

```python
result = run_phase2_from_phase1_result(phase1_result, config)

# Top 5 worst days
worst5 = result.distortion_top20.head(5)
for _, row in worst5.iterrows():
    print(f"{row['date']}: L1={row['l1_distance']:.4f}")

# Or via filter_impact
worst = result.filter_impact.get_worst_days(top_n=10)
```

### Running Sensitivity Sweep

```python
from adapters.run_phase2_from_loader import run_min_notional_sweep

sweep_df = run_min_notional_sweep(
    phase1_result,
    grid=[0, 25, 50, 75, 100, 150, 200, 300, 500],
    output_dir="outputs",
)

# Find optimal min_notional for target L1 distortion
target_l1 = 0.15
optimal = sweep_df[sweep_df["l1_distance_mean"] <= target_l1].iloc[-1]
print(f"Optimal: min_notional={optimal['min_notional_cash']}, NAV={optimal['final_nav_phase2']}")
```
