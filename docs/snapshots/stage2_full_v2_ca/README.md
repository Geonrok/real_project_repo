# Stage2 Full Results (v2_ca - Corporate Actions)

Generated after Stage5 corporate actions merge.

## Summary

- **Total candidates**: 18
- **Strict PASS**: 5 (live-eligible)
- **WARN**: 13 (EXTREME_RETURN_CLAMP_ONLY)
- **FAIL**: 0

## What's New in v2_ca

This snapshot was generated after merging Stage5 corporate actions normalization:

- `configs/corporate_actions.yaml` now available
- `load_symbol_data()` can apply price adjustments for redenominations/splits
- QUICK 2023-07-21 (1:1000 redenomination) documented

## Important Note

The WARN count remains unchanged (13) because this gate-only run regenerated reports from existing backtest CSVs. The corporate actions normalization affects data loading, so a **full Stage2 backtest re-run** would be needed to see the QUICK WARN potentially resolved.

Expected effect of full re-run:
- QUICK-related clamp events should be eliminated
- 13 WARN strategies may reduce (those triggered by QUICK 2023-07-21)

## Reproduce

```powershell
cd E:\repos\real_project_repo

# Gate-only (regenerate reports from existing CSVs)
scripts\run_stage2_gate_only.cmd

# Full re-run (would apply corporate actions to backtests)
# scripts\run_stage2_all.cmd
```

## Files

- `run_metadata.json` - Generation metadata
- `sha256_manifest.txt` - SHA256 hashes
- `stage2_gate_report.md` - Gate report summary
- `stage2_final_pass.csv` - 5 strict PASS strategies
- `stage2_final_pass_with_warn.csv` - 18 PASS+WARN strategies
