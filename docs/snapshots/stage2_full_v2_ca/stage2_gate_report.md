# Stage2 Gate Report

Generated: 2025-12-26T15:22:27.184196+00:00

## Summary

- Total candidates evaluated: 18
- **Final candidates (strict PASS): 5**
- **Final candidates (PASS + WARN): 18**

## Criteria Results

| Criterion | Passed | Warned | Failed | Skipped |
|-----------|--------|--------|--------|---------|
| sensitivity | 18 | - | 0 | No |
| window_stress | 18 | - | 0 | No |
| data_quality | 5 | 13 | 0 | No |

## Final Candidates (Strict PASS)

1. BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING
2. KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING
3. SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING
4. VIDYA_TSMOM_30_SCORE_AVG_GT_1_SIZING
5. VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING

## Final Candidates (PASS + WARN)

1. BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE ⚠️
2. BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING ⚠️
3. BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING
4. KAMA_TSMOM_30_ETH_GT_MA50_GATE ⚠️
5. KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING
6. SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE ⚠️
7. SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING ⚠️
8. SUPERTREND_TSMOM_30_INDEX_GT_MA50_GATE ⚠️
9. SUPERTREND_TSMOM_30_INDEX_GT_MA50_SIZING ⚠️
10. SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_GATE ⚠️
11. SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING
12. VIDYA_TSMOM_30_ETH_GT_MA50_GATE ⚠️
13. VIDYA_TSMOM_30_ETH_GT_MA50_SIZING ⚠️
14. VIDYA_TSMOM_30_SCORE_AVG_GT_1_SIZING
15. VMA_TSMOM_30_ETH_GT_MA50_GATE ⚠️
16. VMA_TSMOM_30_ETH_GT_MA50_SIZING ⚠️
17. VMA_TSMOM_30_SCORE_AVG_GT_1_GATE ⚠️
18. VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING

## Warnings (Data Quality)

| Strategy | Market | Reason | Clamp Count |
|----------|--------|--------|-------------|
| BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| KAMA_TSMOM_30_ETH_GT_MA50_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| SUPERTREND_TSMOM_30_INDEX_GT_MA50_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| SUPERTREND_TSMOM_30_INDEX_GT_MA50_SIZING | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| VIDYA_TSMOM_30_ETH_GT_MA50_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| VIDYA_TSMOM_30_ETH_GT_MA50_SIZING | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| VMA_TSMOM_30_ETH_GT_MA50_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| VMA_TSMOM_30_ETH_GT_MA50_SIZING | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |
| VMA_TSMOM_30_SCORE_AVG_GT_1_GATE | binance_spot | EXTREME_RETURN_CLAMP_ONLY | 1 |

### WARN Policy Note

- **EXTREME_RETURN_CLAMP_ONLY**: Strategies experienced extreme single-bar returns
  that got clipped to -1, but equity curve remained positive (no `cumret_clip`).
- Treated as WARN (not FAIL) for research candidate selection based on Stage3 analysis.
- **Revisit this policy for live trading gates.**


## Top Failures
