# Stage3 Root-Cause Analysis Snapshot

## Summary

- **Strategies analyzed**: 13
- **Market**: binance_spot
- **Root cause distribution**: EXTREME_RETURN_CLAMP_ONLY: 13
- **Recommendation**: EXTREME_RETURN_CLAMP_ONLY → WARN (review), DATA_* → FAIL (exclude symbol)

## What This Contains

- `run_metadata.json`: Execution context (git commit, command, environment, results summary)
- `sha256_manifest.txt`: SHA256 hashes of all output CSV/JSON/MD files

## Reproduction

```powershell
cd E:\repos\real_project_repo
scripts\run_stage3_all.cmd

# Compare SHA256
.\.venv\Scripts\python.exe -c "
import hashlib
from pathlib import Path

for f in sorted(Path('outputs/stage3_rootcause_v1').glob('*')):
    if f.is_file() and f.suffix in ['.csv', '.json', '.md']:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f'{sha}  {f.name}')
"
```

## Manifest Location

- `docs/snapshots/stage3_rootcause_v1/sha256_manifest.txt`
