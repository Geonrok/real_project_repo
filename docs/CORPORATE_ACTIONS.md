# Corporate Actions Normalization

## Overview

Token redenominations, splits, and other corporate actions can cause extreme apparent price changes in historical data. Without adjustment, a 1:1000 token swap would appear as a -99.9% crash, triggering invalid backtest signals.

The corporate actions system normalizes historical prices so returns are computed on a consistent unit basis.

## Configuration

Corporate actions are defined in `configs/corporate_actions.yaml`:

```yaml
corporate_actions:
  - market: binance_spot
    symbol: QUICK
    date: "2023-07-21"
    action_type: redenomination
    ratio: 1000
    note: "QUICK token swap 1 OLD = 1000 NEW (QuickSwap migration)"
    source_refs:
      - "binance_announcement_2023-07-21"
```

### Schema

| Field | Type | Description |
|-------|------|-------------|
| `market` | string | Market identifier (e.g., `binance_spot`, `binance_futures`) |
| `symbol` | string | Symbol name as it appears in normalized data |
| `date` | string | Effective date of the action (YYYY-MM-DD) |
| `action_type` | string | Type of action: `redenomination`, `split`, `reverse_split` |
| `ratio` | number | Conversion ratio (e.g., 1000 means 1 old = 1000 new) |
| `note` | string | Human-readable description |
| `source_refs` | list | Reference sources (announcements, support articles) |

## Adjustment Logic

### Redenomination / Split

For `action_type` in `[redenomination, split]`:

- Dates **BEFORE** `action_date`: `close = close / ratio`
- This converts old units to new units for consistent returns

**Example (QUICK):**
- Before adjustment: 74.20 (old units) -> 0.05967 (new units) = **-99.92%**
- After adjustment: 0.0742 (normalized) -> 0.05967 = **-19.6%**

### Reverse Split

For `action_type == reverse_split`:

- Dates **BEFORE** `action_date`: `close = close * ratio`
- This converts old units to new units

## Usage

The adjustment is applied automatically in `load_symbol_data()`:

```python
from backtest_runner import load_symbol_data

# Automatic adjustment (default)
df = load_symbol_data(normalized_dir, "binance_spot", "QUICK")

# Skip adjustment (for debugging)
df = load_symbol_data(normalized_dir, "binance_spot", "QUICK",
                       apply_corporate_actions=False)
```

## Adding New Corporate Actions

1. Research the event: Find official announcements, support articles, or exchange notices
2. Add entry to `configs/corporate_actions.yaml` with all required fields
3. Run verification to confirm adjustment works:

```bash
python -c "
from scripts.backtest_runner import load_symbol_data
from pathlib import Path

df = load_symbol_data(Path('outputs/normalized_1d'), 'binance_spot', 'QUICK')
print(df[df['date'].dt.date.astype(str).isin(['2023-07-17', '2023-07-21'])])
"
```

4. Run tests: `pytest tests/test_corporate_actions.py -v`

## Known Corporate Actions

| Symbol | Market | Date | Action | Ratio | Note |
|--------|--------|------|--------|-------|------|
| QUICK | binance_spot | 2023-07-21 | redenomination | 1000 | QuickSwap migration |

## Related

- Stage4 event trace identified QUICK 2023-07-21 as the source of all 13 WARN strategies
- See `outputs/stage4_event_trace_v1/stage4_event_trace.md` for details
