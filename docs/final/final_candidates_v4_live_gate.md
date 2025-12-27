# Final Candidates v4 (Live Gate Applied)

## Summary

This version applies strict live-readiness criteria:

- **Sharpe > 0** required
- **MDD >= -0.60** required
- **Window stress exercised** (trades >= 10 per window)

| Status | Count |
|--------|-------|
| LIVE_PASS | 0 |
| LIVE_FAIL | 18 |
| LIVE_INCONCLUSIVE | 54 |

## Per-Market Results

| Market | LIVE_PASS | LIVE_FAIL | LIVE_INCONCLUSIVE |
|--------|-----------|-----------|-------------------|
| binance_futures | 0 | 0 | 18 |
| binance_spot | 0 | 18 | 0 |
| bithumb_krw | 0 | 0 | 18 |
| upbit_krw | 0 | 0 | 18 |

## Result

**No strategies passed all live gate criteria.**

### Key Issues

1. **Window Stress Not Exercised**: 3/4 markets have trades=0 in stress windows
2. **binance_spot**: All 18 strategies fail due to negative Sharpe and deep MDD

### Recommendations

1. Extend backtest data to include more stress periods
2. Review strategy parameters for binance_spot market
3. Consider market-specific strategies rather than universal application
4. Reduce position sizing to limit MDD

## Comparison with v3_ca

| Version | Criteria | PASS Count | Note |
|---------|----------|------------|------|
| v3_ca | Data Quality Only | 18 | All strategies have clean data |
| v4 | Live Gate (Sharpe/MDD/Window) | 0 | Strict live-readiness |

**Conclusion**: v3_ca 'PASS' means data is clean, not that strategies are profitable or safe.
v4 applies real trading constraints and finds no strategies currently suitable for live trading.
