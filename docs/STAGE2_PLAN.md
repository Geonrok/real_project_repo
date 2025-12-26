# Stage2: Robust Candidate Verification Plan

## Purpose

Stage2 validates the 18 robust candidates from Stage1 through sensitivity analysis,
period stress testing, and data quality diagnostics. The goal is to narrow down
to a final set of strategies that are truly robust across different conditions.

## Input

```
outputs/stage1_full_v2/candidates_robust_kof4_top20.csv
```

18 strategies that appeared in Top20 of 3+ markets (K>=3).

## Verification Tests (3 Types)

### 1. Cost Sensitivity Analysis

**Purpose:** Check if strategy performance degrades sharply with realistic cost variations.

**Parameters:**
- Base costs: fee=10bps, slippage=5bps (total 15bps per trade)
- Test grid: base * [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
- Metrics: sharpe, cagr, mdd at each cost level

**Pass criteria:**
- sharpe_ratio_at_2x_cost >= 0.5 * sharpe_ratio_at_base
- No strategy should flip from positive to deeply negative sharpe

**Output:** `outputs/stage2/stage2_sensitivity.csv`

Columns:
```
strategy_id, market, cost_multiplier, fee_bps, slippage_bps,
sharpe, cagr, mdd, trades, sharpe_delta_pct
```

### 2. Period Stress Testing (Window Splits)

**Purpose:** Verify performance consistency across different market regimes.

**Window definitions:**
- Full period: entire available data
- First half: first 50% of data by date
- Second half: last 50% of data by date
- Crisis window: identified high-volatility periods (e.g., COVID crash, 2022 bear)

**Metrics per window:**
- sharpe, cagr, mdd, win_rate, max_consecutive_loss

**Pass criteria:**
- sharpe >= 0 in at least 2 of 3 windows (excluding crisis)
- No window with mdd < -80% while others show mdd > -50%

**Output:** `outputs/stage2/stage2_window_stress.csv`

Columns:
```
strategy_id, market, window_name, start_date, end_date, days,
sharpe, cagr, mdd, trades, win_rate
```

### 3. Data Quality Diagnostics

**Purpose:** Identify strategies that rely on questionable data (price=0, ret=-1, etc.).

**Diagnostics to collect:**
- `clamp_count`: Number of bars where strat_ret was clipped to -1.0
- `ret_inf_count`: Number of bars where ret was inf/-inf (replaced with 0)
- `zero_close_count`: Number of bars with close=0
- `extreme_ret_count`: Number of bars with |ret| > 0.5 (50% daily move)

**Aggregation:** Per strategy-market pair

**Pass criteria:**
- clamp_count == 0 (no artificial equity preservation needed)
- zero_close_count == 0 preferred (indicates data quality issues)

**Output:** `outputs/stage2/stage2_data_quality.csv`

Columns:
```
strategy_id, market, total_bars, clamp_count, ret_inf_count,
zero_close_count, extreme_ret_count, data_quality_score
```

Where `data_quality_score = 1.0 - (clamp_count + zero_close_count) / total_bars`

## Execution Commands

```powershell
cd 'E:\repos\real_project_repo'

# 1. Cost sensitivity
.\.venv\Scripts\python.exe scripts/stage2_sensitivity.py `
  --candidates outputs/stage1_full_v2/candidates_robust_kof4_top20.csv `
  --markets configs/markets.yaml `
  --normalized-dir outputs/normalized_1d `
  --out outputs/stage2

# 2. Window stress
.\.venv\Scripts\python.exe scripts/stage2_window_stress.py `
  --candidates outputs/stage1_full_v2/candidates_robust_kof4_top20.csv `
  --markets configs/markets.yaml `
  --normalized-dir outputs/normalized_1d `
  --out outputs/stage2

# 3. Data quality
.\.venv\Scripts\python.exe scripts/stage2_data_quality.py `
  --candidates outputs/stage1_full_v2/candidates_robust_kof4_top20.csv `
  --markets configs/markets.yaml `
  --normalized-dir outputs/normalized_1d `
  --out outputs/stage2
```

## Final Candidate Selection

After Stage2, combine results:

```python
# Pseudocode
sensitivity_ok = sensitivity_df.groupby('strategy_id')['sharpe_delta_pct'].min() > -50
stress_ok = stress_df.groupby('strategy_id')['sharpe'].apply(lambda x: (x >= 0).sum() >= 2)
quality_ok = quality_df.groupby('strategy_id')['clamp_count'].sum() == 0

final_candidates = sensitivity_ok & stress_ok & quality_ok
```

## Expected Output Structure

```
outputs/stage2/
├── stage2_sensitivity.csv
├── stage2_window_stress.csv
├── stage2_data_quality.csv
├── stage2_final_candidates.csv
└── run_metadata.json
```

## Follow-up Items (Post-Stage2)

1. **clamp_count logging in backtest_runner.py**
   - Add diagnostic columns to summary DF: `clamp_count`, `ret_inf_count`
   - This enables data quality monitoring without separate scripts

2. **Symbol exclusion list**
   - If `zero_close_count > 0` for a symbol, consider adding to exclusion list
   - Alternative: penalize strategies that trade problematic symbols

3. **Stage3 (Optional): Walk-forward validation**
   - Train/test split with rolling windows
   - Out-of-sample sharpe as final gate
