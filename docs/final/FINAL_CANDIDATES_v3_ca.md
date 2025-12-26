# Final Candidates (v3_ca - Corporate Actions Applied)

## Summary

This version was generated with **Stage5 corporate actions normalization** applied during full Stage2 backtest re-run.

| Metric | v2_ca | v3_ca |
|--------|-------|-------|
| Total | 18 | 18 |
| Strict PASS | 5 | **18** |
| WARN | 13 | **0** |

## Strict Pass (Live-Eligible)

Total: **18 strategies**

| # | Strategy ID |
|---|-------------|
| 1 | BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE |
| 2 | BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING |
| 3 | BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| 4 | KAMA_TSMOM_30_ETH_GT_MA50_GATE |
| 5 | KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| 6 | SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE |
| 7 | SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING |
| 8 | SUPERTREND_TSMOM_30_INDEX_GT_MA50_GATE |
| 9 | SUPERTREND_TSMOM_30_INDEX_GT_MA50_SIZING |
| 10 | SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_GATE |
| 11 | SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| 12 | VIDYA_TSMOM_30_ETH_GT_MA50_GATE |
| 13 | VIDYA_TSMOM_30_ETH_GT_MA50_SIZING |
| 14 | VIDYA_TSMOM_30_SCORE_AVG_GT_1_SIZING |
| 15 | VMA_TSMOM_30_ETH_GT_MA50_GATE |
| 16 | VMA_TSMOM_30_ETH_GT_MA50_SIZING |
| 17 | VMA_TSMOM_30_SCORE_AVG_GT_1_GATE |
| 18 | VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING |

## Policy

- **PASS**: Live-eligible. These strategies passed all data quality gates.

## Corporate Actions Resolution

In v2_ca (gate-only), 13 strategies had WARN status due to:

- **Symbol**: QUICK
- **Date**: 2023-07-21
- **Event**: Token redenomination (1 OLD = 1000 NEW)
- **Apparent Price Drop**: 74.20 -> 0.05967 (-99.92%)

With **Stage5 corporate actions normalization**:
- Pre-redenomination prices are scaled by factor 1/1000
- The apparent -99.92% drop becomes a normal ~-19.6% return
- No extreme return clamps triggered
- **All 13 WARN strategies promoted to PASS**

See `configs/corporate_actions.yaml` and `docs/CORPORATE_ACTIONS.md` for implementation details.
