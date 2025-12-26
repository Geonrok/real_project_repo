# Stage2 Full V1 Snapshot

This snapshot provides reproducibility evidence for Stage2 verification results.

## What This Contains

- `run_metadata.json`: Execution context (git commit, command, environment, results summary)
- `sha256_manifest.txt`: SHA256 hashes of all output CSV/JSON/MD files

## Verification

After running Stage2, verify your local outputs match this snapshot:

```powershell
cd E:\repos\real_project_repo
scripts\run_stage2_all.cmd

# Compare SHA256
..\.venv\Scripts\python.exe -c "
import hashlib
from pathlib import Path

for f in sorted(Path('outputs/stage2_full_v1').glob('*')):
    if f.is_file() and f.suffix in ['.csv', '.json', '.md']:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f'{sha}  {f.name}')
"
```

## Reproduction Command

```powershell
cd E:\repos\real_project_repo
scripts\run_stage2_all.cmd
```
