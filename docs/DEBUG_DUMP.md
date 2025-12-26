# Debug Dump Feature

## Overview

The debug dump feature allows you to save detailed equity curve data for a specific strategy, enabling in-depth analysis and debugging of backtest results.

## Usage

### CLI Option

```bash
python scripts/backtest_runner.py \
    --markets configs/markets.yaml \
    --normalized-dir outputs/normalized_1d \
    --grid configs/grid_stage1.yaml \
    --out outputs/stage1 \
    --debug-dump "market=binance_futures,strategy_id=KAMA_TSMOM_30_ETH_GT_MA50_GATE"
```

### Output Location

The debug dump is saved to:
```
outputs/<out>/debug/<market>_<strategy_id>_equity.csv
```

Example:
```
outputs/stage1/debug/binance_futures_KAMA_TSMOM_30_ETH_GT_MA50_GATE_equity.csv
```

## Output Columns

| Column | Description |
|--------|-------------|
| `date` | Trading date (YYYY-MM-DD) |
| `close` | Closing price |
| `position` | Position size (0-1 for spot, can exceed 1 for futures) |
| `strat_ret` | Daily strategy return |
| `equity` | Cumulative equity curve |
| `costs` | Transaction costs applied |
| `pos_change` | Absolute position change |
| `entry_flag` | 1 if entry signal triggered, 0 otherwise |
| `exit_flag` | 1 if exit signal triggered, 0 otherwise |
| `regime_state` | Regime indicator value (merged via date join) |

## Example Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load debug dump
df = pd.read_csv("outputs/stage1/debug/binance_futures_KAMA_TSMOM_30_ETH_GT_MA50_GATE_equity.csv")

# Plot equity curve
plt.figure(figsize=(12, 6))
plt.plot(pd.to_datetime(df["date"]), df["equity"])
plt.title("Equity Curve")
plt.xlabel("Date")
plt.ylabel("Equity")
plt.grid(True)
plt.show()

# Analyze drawdowns
df["drawdown"] = df["equity"] / df["equity"].cummax() - 1
print(f"Max Drawdown: {df['drawdown'].min():.2%}")

# Entry/Exit analysis
print(f"Total Entries: {df['entry_flag'].sum()}")
print(f"Total Exits: {df['exit_flag'].sum()}")
```

## Invariant Validation

When running with `--fail-on-invariant`, the script will:

1. Validate metrics invariants **per-symbol** for each strategy
2. Save violations to `outputs/<out>/invariant_violations.csv`
3. Exclude invalid results from `stage1_top20.csv`
4. Exit with code 1 if any violations found (useful for CI)

### Invariants Checked (Option A: Strict Policy)

> **Note**: Option A (strict for all markets) is applied because the current stage has no liquidation model.
> Once a proper liquidation/margin-call model is implemented, futures markets may allow
> controlled negative equity during drawdowns.

| Invariant | Description | Applies To |
|-----------|-------------|------------|
| `equity_negative` | Equity must be >= 0 (with EPS_EQUITY=1e-12 tolerance) | All markets |
| `mdd_below_minus1` | MDD must be >= -1.0 | All markets |
| `mdd_positive` | MDD must be <= 0 | All markets |
| `equity_has_nan` | Equity must not contain NaN | All markets |
| `equity_has_inf` | Equity must not contain inf | All markets |
| `cagr_consistency` | If final_equity ≈ 0, CAGR must be -1.0 | All markets |

### Per-Symbol Validation

Invariants are validated **per-symbol**. A strategy is marked invalid if **any** symbol violates an invariant.

Violation format in `invariant_violations.csv`:
```
{symbol}:{violation_type}:{value}
```

Example: `BTCUSDT:equity_negative:-0.123456`

### Invariant Violations Schema

The `invariant_violations.csv` file contains:

| Column | Description |
|--------|-------------|
| `market` | Market name (e.g., binance_futures) |
| `strategy_id` | Strategy identifier |
| `violations` | Comma-separated list of violations |

Example:
```csv
market,strategy_id,violations
binance_futures,KAMA_TSMOM_30_ETH_GT_MA50_GATE,"BTCUSDT:equity_negative:-0.05,ETHUSDT:mdd_below_minus1:-1.23"
```

## Trades Definition

Trades are counted as:

1. **New entry**: flat (0) → non-zero position
   - Long entry: 0 → positive
   - Short entry: 0 → negative

2. **Sign flip**: direct position reversal without going flat
   - Long to short: positive → negative
   - Short to long: negative → positive

Formula: `entries = (prev==0 & pos!=0) | (prev*pos < 0)`

This ensures accurate trade counting for markets that allow both long and short positions.

## Market Constraints

The following constraints are enforced based on `markets.yaml`:

| Market Type | max_gross_leverage | allow_borrowing | allow_shorting |
|-------------|-------------------|-----------------|----------------|
| Spot | 1.0 | false | false |
| Futures | 2.0 | true | true |

These constraints are applied **before** returns are computed, preventing invalid states.

## Reproduction Commands

### Stage1 Full Run

```bash
# Full grid backtest across all markets
python scripts/backtest_runner.py \
    --markets configs/markets.yaml \
    --normalized-dir outputs/normalized_1d \
    --grid configs/grid_stage1.yaml \
    --out outputs/stage1 \
    --eval-mode both
```

### CI Mode with Invariant Check

```bash
# CI mode: exit non-zero on any invariant violation
python scripts/backtest_runner.py \
    --markets configs/markets.yaml \
    --normalized-dir outputs/normalized_1d \
    --grid configs/grid_stage1.yaml \
    --out outputs/stage1 \
    --eval-mode both \
    --fail-on-invariant
```

### Debug Dump for Specific Strategy

```bash
# Save detailed equity curve for debugging
python scripts/backtest_runner.py \
    --markets configs/markets.yaml \
    --normalized-dir outputs/normalized_1d \
    --grid configs/grid_stage1.yaml \
    --out outputs/stage1 \
    --debug-dump "market=binance_futures,strategy_id=KAMA_TSMOM_30_ETH_GT_MA50_GATE"
```

## Local Testing Notes

### Python 3.14 GIL Warning

When using Python 3.14 (free-threaded build), you may see:

```
RuntimeWarning: The global interpreter lock (GIL) has been enabled to load module 'pandas._libs.pandas_parser'...
```

This is a pandas/numpy compatibility warning with Python 3.14's free-threaded mode, not a code issue.

**Recommendation**: For local `-W error` strict testing, use Python 3.12 or 3.13 venv.
