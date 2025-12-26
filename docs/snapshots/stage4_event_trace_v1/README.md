# Stage4 Event Trace V1 Snapshot

This snapshot provides reproducibility evidence for Stage4 clamp event tracing.

## What This Contains

- `run_metadata.json`: Execution context (git commit, command, environment, results summary)
- `sha256_manifest.txt`: SHA256 hashes of all output CSV/JSON/MD files

## Purpose

Stage4 traces WARN strategies (EXTREME_RETURN_CLAMP_ONLY) from Stage2/Stage3 to:
- Identify the exact symbol and date where each clamp event occurred
- Extract bar-level details: close, prev_close, raw return, strat_ret before/after clip
- Provide concrete evidence for the "research WARN" classification

## Reproduction

```powershell
cd E:\repos\real_project_repo
scripts\run_stage4_trace.cmd

# Compare SHA256
.\.venv\Scripts\python.exe -c "
import hashlib
from pathlib import Path

for f in sorted(Path('outputs/stage4_event_trace_v1').glob('*')):
    if f.is_file() and f.suffix in ['.csv', '.json', '.md']:
        sha = hashlib.sha256(f.read_bytes()).hexdigest()
        print(f'{sha}  {f.name}')
"
```

## Key Results

See `run_metadata.json` for:
- Total clamp events traced
- TRACED vs REVIEW counts
- Strategies and symbols involved
