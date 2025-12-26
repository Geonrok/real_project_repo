# Stage2 Full Results (v3_ca - Corporate Actions Applied)

Generated after Stage5 corporate actions normalization with FULL backtest re-run.

## Summary

- **Total candidates**: 18
- **Strict PASS**: 18 (live-eligible)
- **WARN**: 0
- **FAIL**: 0

## What's New in v3_ca

This snapshot was generated with a **FULL Stage2 backtest re-run** after Stage5 corporate actions merge:

- `configs/corporate_actions.yaml` contains price adjustment rules
- `load_symbol_data()` applies corporate action normalization by default
- QUICK 2023-07-21 redenomination (1:1000) now properly handled

### Improvement over v2_ca

| Metric | v2_ca (gate-only) | v3_ca (full re-run) |
|--------|-------------------|---------------------|
| Strict PASS | 5 | **18** (+13) |
| WARN | 13 | **0** (-13) |
| FAIL | 0 | 0 |

**All 13 WARN strategies are now PASS** because:
- The QUICK token redenomination (74.20 -> 0.05967) was the sole cause of extreme return clamps
- With price normalization, pre-redenomination prices are scaled by 1/1000
- The apparent -99.92% drop becomes a normal ~-19.6% return
- No clamp triggers, no WARN flags

## Reproduce

```powershell
cd E:\repos\real_project_repo

# Full Stage2 with corporate actions
.\.venv\Scripts\python.exe scripts\stage2_sensitivity.py --stage1-dir outputs\stage1_full_v2 --out outputs\stage2_full_v3_ca
.\.venv\Scripts\python.exe scripts\stage2_window_stress.py --stage1-dir outputs\stage1_full_v2 --out outputs\stage2_full_v3_ca
.\.venv\Scripts\python.exe scripts\stage2_data_quality.py --stage2-dir outputs\stage2_full_v3_ca --out outputs\stage2_full_v3_ca
.\.venv\Scripts\python.exe scripts\stage2_gate_report.py --stage2-dir outputs\stage2_full_v3_ca --out outputs\stage2_full_v3_ca
.\.venv\Scripts\python.exe scripts\stage2_finalize.py --stage2-dir outputs\stage2_full_v3_ca
```

## Files

- `run_metadata.json` - Generation metadata
- `sha256_manifest.txt` - SHA256 hashes
- `stage2_gate_report.md` - Gate report summary
- `stage2_gate_report.json` - Machine-readable gate report
- `stage2_final_pass.csv` - 18 strict PASS strategies
- `stage2_final_pass_with_warn.csv` - 18 PASS+WARN strategies (same as above, 0 WARN)
- `stage2_sensitivity.csv` - Cost sensitivity analysis (288 rows)
- `stage2_window_stress.csv` - Window stress testing (180 rows)
- `stage2_data_quality.csv` - Data quality diagnostics (72 rows)
