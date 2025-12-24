# Phase 2.3: Default Configuration & Guardrails

## Overview

Phase 2.3 seals recommended default configurations and adds safety guardrails to prevent NAV collapse and excessive event counts.

## Key Changes

### 1. Default Configuration Sealed

```python
PHASE23_DEFAULT_CONFIG = {
    "charging_mode": "per_rebalance_day",
    "eps_trade": 1e-12,
    "min_notional_cash": 100.0,
    "enable_netting": False,
    "min_fee_cash": 1.0,
    "quote_ccy": "USDT",
}
```

These defaults are now:
- Applied automatically when `Phase2Config()` is created
- Recorded in `manifest["default_config"]` for reproducibility
- Recommended for production use

### 2. Guardrails Added

#### NAV Collapse Prevention
```python
# If final_nav_phase2 < 0, raises Phase2NAVCollapseError
# Error includes:
#   - charging_mode, eps_trade, min_notional_cash
#   - total_events, filtered_ratio, events_per_day_max
#   - Recommendations for fixing
```

#### High Event Count Warning
```python
# If events_per_day_max > threshold (default 100), emits UserWarning
# Warning message includes:
#   - Current max events
#   - Threshold value
#   - Recommendation to use per_rebalance_day
```

### 3. Filtered Trade Ratio

New metric: `min_fee_filtered_trade_ratio`
```
filtered_ratio = total_ignored / total_raw_trades
```

Example from realdata:
- Total raw trades: 858,190
- Filtered: 857,659 (99.94%)
- Effective events: 531

### 4. Output Standardization

#### phase2_event_stats.csv (NEW)
```csv
date,events,ignored_small_trades,gross_turnover
2022-02-08,1,0,0.0123
2022-02-09,1,0,0.0145
...
```

#### real_summary.json (ENHANCED)
```json
{
  "event_stats": {
    "events_per_day_mean": 0.354,
    "events_per_day_max": 1,
    "events_per_day_top5": [
      {"date": "2022-02-08", "events": 1},
      ...
    ]
  },
  "min_fee_filtered_trade_ratio": 0.9993,
  "min_fee_total_raw_trades": 858190
}
```

#### real_run_manifest.json (ENHANCED)
```json
{
  "schema_version": "2.3.0",
  "default_config": {...},
  "current_config": {...},
  "guardrails": {
    "events_threshold": 100,
    "nav_collapse_enabled": true
  }
}
```

## New Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `guardrail_events_threshold` | int | 100 | events_per_day warning threshold |
| `guardrail_nav_collapse` | bool | True | Enable NAV<0 RuntimeError |

## Files Changed

```
adapters/run_phase2_from_loader.py (MODIFIED)
  - PHASE23_DEFAULT_CONFIG constant
  - Phase2NAVCollapseError exception
  - Phase2Config: new guardrail fields
  - Phase2Result: event_stats_daily DataFrame
  - Phase2Runner: _check_guardrails() method
  - _save_outputs: phase2_event_stats.csv

tests/test_phase23_guardrails.py (NEW)
  - 16 tests for guardrails, filtered ratio, defaults

tests/test_phase2_integration_smoke.py (MODIFIED)
  - schema_version updated to 2.3.0

docs/PHASE23_CHANGELOG.md (NEW)
  - This documentation
```

## Test Results

```
============================= 85 passed in 2.16s =============================
  - Phase 2.2 tests: 18
  - Phase 2.3 guardrail tests: 16
  - Existing tests: 51
```

## Realdata Summary (5 lines)

```
Final NAV Phase1: 11,658.33  |  Final NAV Phase2: 10,834.50
Filtered Ratio: 99.94% (857,659 / 858,190 trades filtered)
Effective Events: 531 (per_rebalance_day, min_notional=100)
Events/Day: mean=0.35, max=1
Total Cost Phase2: 2,288.01 (min_fee=531, rounding=292.83)
```

## Migration Notes

### From Phase 2.2 to 2.3

1. **Default values changed**:
   - `min_fee_charging_mode`: `per_asset_day` → `per_rebalance_day`
   - `min_notional_cash`: `0.0` → `100.0`

2. **New exception type**:
   - `Phase2NAVCollapseError` may be raised if NAV < 0
   - Catch this exception or set `guardrail_nav_collapse=False`

3. **Schema version**:
   - `2.2.0` → `2.3.0`

### Disable Guardrails (if needed)

```python
config = Phase2Config(
    guardrail_nav_collapse=False,  # Disable NAV collapse check
    guardrail_events_threshold=10000,  # High threshold
)
```
