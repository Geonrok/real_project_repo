# Phase 2.2: Trade Event Definition Layer

## Overview

Phase 2.2 introduces a **TradeEventBuilder** layer that filters trade events before applying minimum fees. This addresses the critical issue where `per_asset_day` and `per_asset_side_day` charging modes caused NAV collapse due to excessive micro-trade event counting.

## Problem Statement (Phase 2.1)

In Phase 2.1, the minimum fee cost model counted every weight change (`delta_w`) as a trade event. With 574 assets over 1,500+ trading days:

- **136,000+ events** were generated (most being micro-changes < 1 USDT)
- At 1 USDT min_fee per event: **136,000 USDT total cost**
- Initial NAV: 10,000 USDT
- Result: **NAV collapsed to -125,000 USDT**

```
Phase 2.1 Results (min_notional=0):
  per_asset_day:      NAV = -125,237  (136,603 events)
  per_asset_side_day: NAV = -125,237  (136,603 events)
  per_rebalance_day:  NAV = +10,510   (855 events)  <- stable
```

## Solution (Phase 2.2)

### 1. Trade Event Definition Layer

New module: `execution/event_builder.py`

```python
from execution import TradeEventBuilder, TradeEventConfig

config = TradeEventConfig(
    eps_trade=1e-12,           # Weight change threshold
    min_notional_cash=100.0,   # Minimum trade value (USDT)
    enable_netting=False,      # Same-asset buy/sell cancellation
    exclude_cash=True,         # Exclude CASH column
)

builder = TradeEventBuilder(config)
result = builder.build(delta_w, nav)
```

### 2. Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `eps_trade` | float | 1e-12 | Minimum weight change to consider as trade |
| `min_notional_cash` | float | 0.0 | Minimum trade value in quote currency |
| `enable_netting` | bool | False | Cancel out same-asset buy/sell on same day |
| `exclude_cash` | bool | True | Exclude CASH column from event counting |

### 3. TradeEventResult

```python
@dataclass
class TradeEventResult:
    events_per_day: pd.Series      # Valid events per day
    total_events: int              # Total valid events
    ignored_events: int            # Total ignored events
    ignored_by_eps: int            # Ignored by eps_trade
    ignored_by_notional: int       # Ignored by min_notional_cash
    ignored_by_netting: int        # Ignored by netting
    days_with_events: int          # Days with at least one event
```

## Results

### Sensitivity Analysis with min_notional_cash

```
Phase 2.2 Results:

min_notional=0 (legacy behavior):
  per_asset_day:      NAV = -125,237  (136,603 events, 0 ignored)

min_notional=10 USDT:
  per_asset_day:      NAV = -29,322   (40,688 events, 820,312 ignored)

min_notional=100 USDT:
  per_asset_day:      NAV = +8,024    (3,341 events, 857,659 ignored)  <- FIXED!

per_rebalance_day (any min_notional):
  NAV = +10,510 ~ +10,834  (531-855 events)  <- always stable
```

### Key Findings

1. **857,659 micro-trades** were below 100 USDT threshold (99.6% of all trades)
2. With `min_notional_cash=100`, `per_asset_day` becomes viable
3. `per_rebalance_day` remains the most conservative and stable option

## Usage

### CLI

```bash
# Run sensitivity with min_notional options
python examples/analyze_phase2_sensitivity.py \
    --venue binance_futures \
    --min-notional 0 10 100 \
    --netting off

# Full grid search
python examples/analyze_phase2_sensitivity.py \
    --venue binance_futures \
    --full-grid
```

### Python API

```python
from adapters.run_phase2_from_loader import Phase2Config, run_phase2_from_phase1_result

config = Phase2Config(
    enable_min_fee=True,
    min_fee_cash=1.0,
    charging_mode="per_asset_day",
    min_notional_cash=100.0,  # Phase 2.2
    enable_netting=False,     # Phase 2.2
)

result = run_phase2_from_phase1_result(phase1_result, config)
```

## Schema Changes

- Schema version: `2.0.0` -> `2.2.0`
- New fields in Phase2Result:
  - `event_stats`: Event statistics dictionary
- New metadata fields in CostModelResult:
  - `min_notional_cash`
  - `enable_netting`
  - `min_fee_event_count_effective`
  - `min_fee_ignored_small_trades`
  - `min_fee_ignored_by_eps`
  - `min_fee_ignored_by_netting`
  - `min_fee_total_ignored`

## Backward Compatibility

- When `min_notional_cash=0` and `enable_netting=False`, the legacy behavior is preserved
- All existing tests pass with schema version update
- Phase 1 Anchor Engine is **unchanged**

## Files Changed

```
execution/
  __init__.py           (new) - Module exports
  event_builder.py      (new) - TradeEventBuilder, TradeEventConfig, TradeEventResult

costs/
  min_fee.py            (modified) - Uses TradeEventBuilder when min_notional > 0

adapters/
  run_phase2_from_loader.py (modified) - Phase2Config extended

examples/
  analyze_phase2_sensitivity.py (modified) - CLI options added

tests/
  test_phase22_event_builder.py (new) - 18 new tests
  test_phase2_integration_smoke.py (modified) - Schema version updated
```

## Test Results

```
============================= 69 passed =============================
  - 18 new Phase 2.2 tests (event builder, min_notional, netting)
  - 51 existing tests (all passing)
```

## Recommendations

| Scenario | Recommended Settings |
|----------|---------------------|
| Conservative | `per_rebalance_day`, any min_notional |
| Balanced | `per_asset_day`, min_notional=100 |
| Aggressive | `per_asset_day`, min_notional=10 |

For most production use cases, `per_rebalance_day` with `min_notional_cash=0` provides the simplest and most predictable behavior.
