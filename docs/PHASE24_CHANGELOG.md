# Phase 2.4: Realistic Order Units (현실적 주문 단위)

## Overview

Phase 2.4 aligns the backtester with realistic order unit constraints by implementing trade_notional-based filtering that clips delta_w below threshold, adjusts w_exec to maintain Σw=1, and reports filter impact metrics.

## Key Changes

### 1. trade_notional Based Filtering

```python
# Filter formula
trade_notional_cash[t, asset] = NAV_before_cost[t] * abs(delta_w_risk[t, asset])

# If below threshold → clip delta_w to 0
if trade_notional_cash < min_notional_cash:
    delta_w_filtered[t, asset] = 0
```

This ensures small notional trades below the minimum threshold are excluded from execution.

### 2. w_exec Adjustment (Σw=1 Preservation)

```python
# Skipped orders absorbed by CASH
delta_skipped = delta_w_raw - delta_w_filtered
w_exec_filtered = w_exec_raw - delta_skipped  # for risk assets
w_exec_filtered["CASH"] = 1.0 - sum(risk_weights)
```

The weight invariant Σw=1 is maintained by absorbing filtered positions into CASH.

### 3. Portfolio-Level Netting

When `enable_netting=True`:
- Buy and sell totals are netted at portfolio level per day
- Only net order notional is counted as a single event
- Reduces event count for min_fee calculations

### 4. FilterImpactMetrics

New dataclass reporting filter effects:

```python
@dataclass
class FilterImpactMetrics:
    pre_filter_turnover_L1: float      # Total L1 turnover before filtering
    post_filter_turnover_L1: float     # Total L1 turnover after filtering
    total_notional_raw: float          # Total notional before filtering
    total_notional_filtered: float     # Notional amount filtered out
    filter_ratio: float                # filtered / total
    l1_distance_daily: pd.Series       # Daily L1 distance w_raw vs w_filtered
    l1_distance_mean: float            # Mean L1 distance
    l1_distance_max: float             # Max L1 distance
    n_trades_filtered: int             # Number of trades filtered
    n_trades_total: int                # Total number of trades
    turnover_reduction_ratio: float    # 1 - post/pre turnover
```

### 5. New Output Files

| File | Description |
|------|-------------|
| `phase24_filter_impact_summary.json` | Complete filter impact metrics |
| `real_w_exec_filtered.csv` | Adjusted weights with Σw=1 preserved |

## Schema Changes

| Field | Version | Notes |
|-------|---------|-------|
| `schema_version` | `2.3.0` → `2.4.0` | Major version bump |
| `filter_impact_summary` | NEW | In manifest |
| `filter_impact` | NEW | In summary |
| `PHASE24_DEFAULT_CONFIG` | NEW | Alias for sealed defaults |

## Files Changed

```
execution/event_builder.py (MODIFIED)
  - FilterImpactMetrics dataclass
  - TradeEventConfig.apply_filter_to_weights parameter
  - TradeEventBuilder._compute_trade_notional()
  - TradeEventBuilder._apply_min_notional_filter()
  - TradeEventBuilder._adjust_w_exec_for_cash()
  - TradeEventBuilder._compute_filter_impact()
  - Portfolio-level netting in build()

execution/__init__.py (MODIFIED)
  - Export FilterImpactMetrics

adapters/run_phase2_from_loader.py (MODIFIED)
  - Schema version 2.4.0
  - PHASE24_DEFAULT_CONFIG constant
  - Phase2Result.filter_impact field
  - Phase2Result.w_exec_filtered field
  - _save_outputs(): phase24_filter_impact_summary.json, real_w_exec_filtered.csv
  - filter_impact in manifest and summary

tests/test_phase24_filter_impact.py (NEW)
  - 22 tests for Phase 2.4 features
  - Handcalc tests for trade_notional filtering
  - Invariant tests: Σw=1, P2 NAV <= P1 NAV, cost_sum

tests/test_phase23_guardrails.py (MODIFIED)
  - Schema version updated to 2.4.0

tests/test_phase2_integration_smoke.py (MODIFIED)
  - Schema version updated to 2.4.0

docs/PHASE24_CHANGELOG.md (NEW)
  - This documentation
```

## Test Results

```
============================= 106 passed in 5.37s =============================
  - Phase 2.2 tests: 18
  - Phase 2.3 guardrail tests: 16
  - Phase 2.4 filter impact tests: 22
  - Cost model handcalc tests: 20
  - Integration smoke tests: 15
  - Sensitivity smoke tests: 15
```

## Realdata Summary (5 lines)

```
Final NAV Phase1: 11,658.33  |  Final NAV Phase2: 10,834.50
Turnover Reduction: 46.79% (335.87 → 178.71 L1)
Filter Ratio: 46.08% (1,349,239 / 2,928,348 USDT notional filtered)
L1 Distance: mean=0.105, max=0.978 (w_exec distortion)
Total Cost Phase2: 2,288.01 (min_fee=531, rounding=293)
```

## Invariants Verified

1. **Σw=1**: `w_exec_filtered.sum(axis=1) ≈ 1.0` for all rows
2. **P2 NAV ≤ P1 NAV**: `nav_phase2[t] <= nav_phase1[t]` for all t
3. **Cost Sum**: `cost_phase2 = cost_phase1 + extra_cost`
4. **Breakdown Sum**: `sum(breakdown) = total_extra_cost`

## Migration Notes

### From Phase 2.3 to 2.4

1. **Schema version changed**: `2.3.0` → `2.4.0`

2. **New fields in result**:
   - `filter_impact: FilterImpactMetrics`
   - `w_exec_filtered: pd.DataFrame`

3. **New output files**:
   - `phase24_filter_impact_summary.json`
   - `real_w_exec_filtered.csv`

4. **Backward compatibility**:
   - `PHASE23_DEFAULT_CONFIG` is an alias for `PHASE24_DEFAULT_CONFIG`
   - All Phase 2.3 tests continue to pass

### Accessing Filter Impact

```python
result = run_phase2_from_phase1_result(phase1_result, config)

# Access filter impact
fi = result.filter_impact
print(f"Turnover reduced by {fi.turnover_reduction_ratio:.2%}")
print(f"L1 distortion: mean={fi.l1_distance_mean:.4f}, max={fi.l1_distance_max:.4f}")

# Access filtered weights
w_filtered = result.w_exec_filtered
assert abs(w_filtered.sum(axis=1) - 1.0).max() < 1e-6  # Σw=1 invariant
```
