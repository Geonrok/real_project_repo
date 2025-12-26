# Final Candidates

## Strict Pass (Live-Eligible)

Total: **5 strategies**

| Strategy ID |
|-------------|
| BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| VIDYA_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING |

## Research Candidates (Pass + Warn)

Total: **18 strategies** (5 PASS + 13 WARN)

| Strategy ID | Status | Reason |
|-------------|--------|--------|
| BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING | WARN | EXTREME_RETURN_CLAMP_ONLY |
| BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING | PASS | nan |
| KAMA_TSMOM_30_ETH_GT_MA50_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING | PASS | nan |
| SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING | WARN | EXTREME_RETURN_CLAMP_ONLY |
| SUPERTREND_TSMOM_30_INDEX_GT_MA50_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| SUPERTREND_TSMOM_30_INDEX_GT_MA50_SIZING | WARN | EXTREME_RETURN_CLAMP_ONLY |
| SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING | PASS | nan |
| VIDYA_TSMOM_30_ETH_GT_MA50_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| VIDYA_TSMOM_30_ETH_GT_MA50_SIZING | WARN | EXTREME_RETURN_CLAMP_ONLY |
| VIDYA_TSMOM_30_SCORE_AVG_GT_1_SIZING | PASS | nan |
| VMA_TSMOM_30_ETH_GT_MA50_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| VMA_TSMOM_30_ETH_GT_MA50_SIZING | WARN | EXTREME_RETURN_CLAMP_ONLY |
| VMA_TSMOM_30_SCORE_AVG_GT_1_GATE | WARN | EXTREME_RETURN_CLAMP_ONLY |
| VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING | PASS | nan |

## Policy

- **PASS**: Live-eligible. These strategies passed all data quality gates.
- **WARN**: Research-only. These strategies triggered extreme return clamps but remained viable.

## Stage4 Evidence

All 13 WARN strategies trace to a single event:

- **Symbol**: QUICK
- **Date**: 2023-07-21
- **Event**: Token redenomination (1 OLD = 1000 NEW)
- **Price**: 74.20 -> 0.05967 (-99.92%)

This was not a market crash but a token migration. The extreme return clamp prevented equity from going negative, which is correct model behavior. However, in live trading, such events could trigger margin calls or forced liquidation.

See `docs/snapshots/stage4_event_trace_v1/` for full event trace details.
