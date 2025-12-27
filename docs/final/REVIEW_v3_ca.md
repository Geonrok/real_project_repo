# v3_ca Live-Readiness Review

Generated from Stage2 v3_ca artifacts with corporate actions applied.

## A) Market-Level Baseline Summary (fee_mult=1, slippage_mult=1)

| Market | Sharpe Min | Sharpe Median | Sharpe Max | Sharpe>0 | MDD Min | MDD Median | MDD Max | Trades Min | Trades Median | Trades Max |
|--------|------------|---------------|------------|----------|---------|------------|---------|------------|---------------|------------|
| binance_futures | -0.122 | -0.095 | -0.052 | 0/18 | -0.990 | -0.967 | -0.932 | 7431 | 11586 | 13395 |
| binance_spot | -0.047 | -0.035 | 0.003 | 1/18 | -0.984 | -0.935 | -0.868 | 9972 | 14948 | 17030 |
| bithumb_krw | 0.117 | 0.176 | 0.320 | 18/18 | -1.000 | -0.998 | -0.880 | 6539 | 9715 | 11337 |
| upbit_krw | -0.048 | -0.017 | 0.016 | 4/18 | -0.950 | -0.841 | -0.762 | 3351 | 4886 | 5726 |

## B) Top 5 Strategies by Sharpe (per Market)

### binance_futures

| Rank | Strategy ID | Sharpe | MDD | Trades |
|------|-------------|--------|-----|--------|
| 1 | SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE | -0.052 | -0.983 | 7431 |
| 2 | SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING | -0.052 | -0.983 | 7431 |
| 3 | KAMA_TSMOM_30_ETH_GT_MA50_GATE | -0.073 | -0.968 | 12027 |
| 4 | VMA_TSMOM_30_ETH_GT_MA50_GATE | -0.076 | -0.966 | 11841 |
| 5 | VMA_TSMOM_30_ETH_GT_MA50_SIZING | -0.076 | -0.966 | 11841 |

### binance_spot

| Rank | Strategy ID | Sharpe | MDD | Trades |
|------|-------------|--------|-----|--------|
| 1 | SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING | 0.003 | -0.934 | 10498 |
| 2 | SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE | -0.006 | -0.977 | 9972 |
| 3 | SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING | -0.006 | -0.977 | 9972 |
| 4 | VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING | -0.011 | -0.868 | 16572 |
| 5 | BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING | -0.016 | -0.891 | 16321 |

### bithumb_krw

| Rank | Strategy ID | Sharpe | MDD | Trades |
|------|-------------|--------|-----|--------|
| 1 | BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING | 0.320 | -0.998 | 10673 |
| 2 | VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING | 0.280 | -0.997 | 10710 |
| 3 | KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING | 0.263 | -0.998 | 11337 |
| 4 | SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING | 0.222 | -0.880 | 7117 |
| 5 | BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE | 0.218 | -1.000 | 9715 |

### upbit_krw

| Rank | Strategy ID | Sharpe | MDD | Trades |
|------|-------------|--------|-----|--------|
| 1 | SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE | 0.016 | -0.831 | 3351 |
| 2 | SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING | 0.016 | -0.831 | 3351 |
| 3 | KAMA_TSMOM_30_ETH_GT_MA50_GATE | 0.016 | -0.950 | 5127 |
| 4 | SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING | 0.004 | -0.769 | 3692 |
| 5 | VMA_TSMOM_30_ETH_GT_MA50_GATE | -0.007 | -0.930 | 4931 |

## C) Red Flags

### Deep MDD (<=−0.9)

- Deep MDD: binance_futures / BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.975
- Deep MDD: binance_futures / BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.975
- Deep MDD: binance_futures / BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.952
- Deep MDD: binance_futures / KAMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.968
- Deep MDD: binance_futures / KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.951
- Deep MDD: binance_futures / SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.983
- Deep MDD: binance_futures / SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.983
- Deep MDD: binance_futures / SUPERTREND_TSMOM_30_INDEX_GT_MA50_GATE: mdd=-0.990
- Deep MDD: binance_futures / SUPERTREND_TSMOM_30_INDEX_GT_MA50_SIZING: mdd=-0.990
- Deep MDD: binance_futures / SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_GATE: mdd=-0.989
- Deep MDD: binance_futures / SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.946
- Deep MDD: binance_futures / VIDYA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.941
- Deep MDD: binance_futures / VIDYA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.941
- Deep MDD: binance_futures / VIDYA_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.932
- Deep MDD: binance_futures / VMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.966
- Deep MDD: binance_futures / VMA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.966
- Deep MDD: binance_futures / VMA_TSMOM_30_SCORE_AVG_GT_1_GATE: mdd=-0.979
- Deep MDD: binance_futures / VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.954
- Deep MDD: binance_spot / BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.945
- Deep MDD: binance_spot / BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.945
- Deep MDD: binance_spot / KAMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.941
- Deep MDD: binance_spot / SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.977
- Deep MDD: binance_spot / SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.977
- Deep MDD: binance_spot / SUPERTREND_TSMOM_30_INDEX_GT_MA50_GATE: mdd=-0.984
- Deep MDD: binance_spot / SUPERTREND_TSMOM_30_INDEX_GT_MA50_SIZING: mdd=-0.984
- Deep MDD: binance_spot / SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_GATE: mdd=-0.983
- Deep MDD: binance_spot / SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.934
- Deep MDD: binance_spot / VIDYA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.928
- Deep MDD: binance_spot / VIDYA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.928
- Deep MDD: binance_spot / VMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.933
- Deep MDD: binance_spot / VMA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.933
- Deep MDD: binance_spot / VMA_TSMOM_30_SCORE_AVG_GT_1_GATE: mdd=-0.937
- Deep MDD: bithumb_krw / BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE: mdd=-1.000
- Deep MDD: bithumb_krw / BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-1.000
- Deep MDD: bithumb_krw / BOLL_MID_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.998
- Deep MDD: bithumb_krw / KAMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-1.000
- Deep MDD: bithumb_krw / KAMA_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.998
- Deep MDD: bithumb_krw / SUPERTREND_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.955
- Deep MDD: bithumb_krw / SUPERTREND_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.955
- Deep MDD: bithumb_krw / SUPERTREND_TSMOM_30_INDEX_GT_MA50_GATE: mdd=-0.954
- Deep MDD: bithumb_krw / SUPERTREND_TSMOM_30_INDEX_GT_MA50_SIZING: mdd=-0.954
- Deep MDD: bithumb_krw / SUPERTREND_TSMOM_30_SCORE_AVG_GT_1_GATE: mdd=-0.954
- Deep MDD: bithumb_krw / VIDYA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-1.000
- Deep MDD: bithumb_krw / VIDYA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-1.000
- Deep MDD: bithumb_krw / VIDYA_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.997
- Deep MDD: bithumb_krw / VMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-1.000
- Deep MDD: bithumb_krw / VMA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-1.000
- Deep MDD: bithumb_krw / VMA_TSMOM_30_SCORE_AVG_GT_1_GATE: mdd=-1.000
- Deep MDD: bithumb_krw / VMA_TSMOM_30_SCORE_AVG_GT_1_SIZING: mdd=-0.997
- Deep MDD: upbit_krw / BOLL_MID_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.918
- Deep MDD: upbit_krw / BOLL_MID_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.918
- Deep MDD: upbit_krw / KAMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.950
- Deep MDD: upbit_krw / VIDYA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.911
- Deep MDD: upbit_krw / VIDYA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.911
- Deep MDD: upbit_krw / VMA_TSMOM_30_ETH_GT_MA50_GATE: mdd=-0.930
- Deep MDD: upbit_krw / VMA_TSMOM_30_ETH_GT_MA50_SIZING: mdd=-0.930
- Deep MDD: upbit_krw / VMA_TSMOM_30_SCORE_AVG_GT_1_GATE: mdd=-0.939

### Markets with No Positive Sharpe

- No positive sharpe: binance_futures has 0/18 strategies with sharpe>0

### Window Stress Issues

**Window Stress Not Exercised:**

- binance_futures / crisis: 18 strategies with trades=0
- binance_futures / early: 18 strategies with trades=0
- binance_futures / late: 18 strategies with trades=0
- bithumb_krw / crisis: 18 strategies with trades=0
- bithumb_krw / early: 18 strategies with trades=0
- bithumb_krw / late: 18 strategies with trades=0
- upbit_krw / crisis: 18 strategies with trades=0
- upbit_krw / early: 18 strategies with trades=0
- upbit_krw / late: 18 strategies with trades=0
- binance_spot missing windows: {'late', 'early'}

## D) Data Quality Confirmation

| Metric | Sum |
|--------|-----|
| zero_close_count | 0 |
| ret_inf_count | 0 |
| strat_ret_clamp_count | 0 |
| cumret_clip_count | 0 |

**All data quality metrics sum to zero.**

## Summary

- **Total strategies**: 18
- **Markets**: binance_futures, binance_spot, bithumb_krw, upbit_krw
- **Data quality clean**: YES
- **Window stress issues**: YES
- **Deep MDD observed**: YES
