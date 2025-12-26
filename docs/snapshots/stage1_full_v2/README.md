# Stage1 Full V2 Snapshot

This snapshot provides reproducibility evidence for Stage1 backtest results.

## What This Contains

- `run_metadata.json`: Execution context (git commit, command, environment, results summary)
- `sha256_manifest.txt`: SHA256 hashes of all output CSV files

## Why outputs/ Is Not in Git

Per `.gitignore` policy, `outputs/` is excluded from version control to:
- Prevent large binary/CSV files from bloating the repository
- Avoid merge conflicts when re-running experiments
- Keep the repo clean for code-focused reviews

## How to Verify Local Results

After running Stage1, verify your local outputs match this snapshot:

```powershell
cd E:\repos\real_project_repo

# Generate SHA256 for your local outputs
.\.venv\Scripts\python.exe -c @"
import hashlib
from pathlib import Path

out_dir = Path('outputs/stage1_full_v2')
for f in sorted(out_dir.glob('*.csv')):
    sha = hashlib.sha256(f.read_bytes()).hexdigest()
    print(f'{sha}  {f.name}')
"@

# Compare with docs/snapshots/stage1_full_v2/sha256_manifest.txt
```

## Reproduction Command

See `docs/DEBUG_DUMP.md` for full execution instructions, or run:

```powershell
.\.venv\Scripts\python.exe scripts/backtest_runner.py `
  --markets configs/markets.yaml `
  --normalized-dir outputs/normalized_1d `
  --grid configs/grid_stage1.yaml `
  --out outputs/stage1_full_v2 `
  --eval-mode both
```

## Key Results

| Metric | Value |
|--------|-------|
| Markets | 4 (upbit_krw, bithumb_krw, binance_spot, binance_futures) |
| Total strategies | 280 |
| Valid strategies | 280 |
| Invariant violations | 0 |
| Top20 rows | 80 (20 per market) |
| Robust candidates (K>=3) | 18 |
