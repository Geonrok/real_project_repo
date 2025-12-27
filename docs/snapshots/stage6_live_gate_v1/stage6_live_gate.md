# Stage6 Live Gate Report

## Configuration

- **MIN_TRADES_PER_WINDOW**: 10
- **REQUIRE_SHARPE_POSITIVE**: True
- **MAX_MDD**: -0.6
- **Research Mode**: False

## Overall Summary

| Status | Count |
|--------|-------|
| LIVE_PASS | 0 |
| LIVE_FAIL | 18 |
| LIVE_INCONCLUSIVE | 54 |

## Per-Market Summary

| Market | LIVE_PASS | LIVE_FAIL | LIVE_INCONCLUSIVE | Recommendation |
|--------|-----------|-----------|-------------------|----------------|
| binance_futures | 0 | 0 | 18 | NEEDS MORE DATA |
| binance_spot | 0 | 18 | 0 | EXCLUDE MARKET |
| bithumb_krw | 0 | 0 | 18 | NEEDS MORE DATA |
| upbit_krw | 0 | 0 | 18 | NEEDS MORE DATA |

## Market Exclusion Warnings

The following markets have **0 strategies passing live gate**:

- **binance_spot**: avg_sharpe=-0.030, avg_mdd=-0.936
  - Recommendation: Exclude from live trading or redesign strategy parameters

## LIVE_PASS Strategies

**No strategies passed all live gates.**

This indicates:
- Current strategy parameters may not be suitable for live trading
- Consider relaxing gate thresholds (research mode)
- Or redesign strategies with better risk/return characteristics

## LIVE_FAIL Analysis

| Failure Reason | Count |
|----------------|-------|
| sharpe=-0.039<=0 | 4 |
| sharpe=-0.035<=0 | 2 |
| mdd=-0.945<-0.6 | 2 |
| sharpe=-0.006<=0 | 2 |
| mdd=-0.977<-0.6 | 2 |
| mdd=-0.984<-0.6 | 2 |
| sharpe=-0.046<=0 | 2 |
| mdd=-0.928<-0.6 | 2 |
| mdd=-0.933<-0.6 | 2 |
| sharpe=-0.016<=0 | 1 |
| mdd=-0.891<-0.6 | 1 |
| sharpe=-0.047<=0 | 1 |
| mdd=-0.941<-0.6 | 1 |
| sharpe=-0.029<=0 | 1 |
| mdd=-0.876<-0.6 | 1 |
| sharpe=-0.036<=0 | 1 |
| mdd=-0.983<-0.6 | 1 |
| mdd=-0.934<-0.6 | 1 |
| sharpe=-0.028<=0 | 1 |
| mdd=-0.888<-0.6 | 1 |
| sharpe=-0.044<=0 | 1 |
| mdd=-0.937<-0.6 | 1 |
| sharpe=-0.011<=0 | 1 |
| mdd=-0.868<-0.6 | 1 |

## LIVE_INCONCLUSIVE Analysis

**54 strategy-market combinations have insufficient window stress data.**

These cannot be classified as PASS or FAIL because window stress testing
was not exercised (trades=0 in stress windows).

Options:
- Extend backtest period to include more stress events
- Use research mode (--research-mode) with relaxed thresholds
- Accept as research-only candidates
