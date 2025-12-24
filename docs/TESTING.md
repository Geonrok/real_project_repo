# Testing & Environment Standardization

> **IMPORTANT**: 이 문서의 모든 명령어는 **cmd.exe** 기준입니다. PowerShell에서는 glob 패턴 문제가 있으니 반드시 cmd.exe를 사용하세요.

## Quick Commands (cmd.exe)

```cmd
REM Run unit/integration tests
python run_tests.py

REM Run pytest (always via interpreter - NEVER use pytest.exe directly)
python -m pytest -q

REM Targeted pytest example
python -m pytest -q tests/test_phase26_recommendation.py

REM Windows helper scripts
scripts\run_pytest.cmd
scripts\run_ci_chunks.cmd [chunk]
```

---

## Step7/Step8 Latest Run (cmd.exe)

```cmd
REM 1) Run Step7 once (creates outputs_step7\run_* and latest_run.*)
run_step7_all.cmd

REM 2) Confirm latest_run.txt points to an existing folder
python -c "from pathlib import Path; p=Path('outputs_step7/latest_run.txt').read_text().strip(); print(p); print(Path(p).exists())"

REM 3) Run Step8 using the latest run pointer
python build_step8_selection.py --step7-dir "%USERPROFILE%\outputs_step7" --out-dir "%USERPROFILE%\outputs_step8\verify_latest" --use-latest-run
```

---

## Step8 Run Gate & Fallback (cmd.exe)

```cmd
REM 0) Smoke2 (pipeline wiring only): allow partial
set ALLOW_PARTIAL=1
set SKIP_BINANCE_FUTURES=1
set MAX_SYMBOLS=20
call run_step7_all.cmd
call run_step8_latest.cmd
set ALLOW_PARTIAL=
set SKIP_BINANCE_FUTURES=
set MAX_SYMBOLS=

REM 1) Default policy: incomplete run => FAIL-FAST (full run or split+seal only)
call run_step8_latest.cmd

REM 2) Allow incomplete run (explicit override)
set ALLOW_INCOMPLETE=1
call run_step8_latest.cmd

REM 3) Conditional fallback to last complete run
set FALLBACK_TO_LAST_COMPLETE=1
set MAX_FALLBACK_AGE_HOURS=24
call run_step8_latest.cmd

REM 4) Split run (spot -> futures -> seal)
set STEP7_OUTPUT_ROOT=C:\Users\???\outputs_step7\run_YYYYMMDD_HHMMSS_RAND
set STEP7_EXCHANGES=upbit bithumb binance_spot
set SKIP_BINANCE_FUTURES=1
set MAX_SYMBOLS=20
call run_step7_all.cmd
set ONLY_EXCHANGE=binance_futures
set SKIP_BINANCE_FUTURES=
call run_step7_all.cmd
set SEAL_ONLY=1
set STEP7_EXCHANGES=upbit bithumb binance_spot binance_futures
call run_step7_all.cmd
set SEAL_ONLY=

REM 5) Fallback test restore (overwrite, not rename)
powershell -NoProfile -Command "$p=C:\Users\???\outputs_step7\latest_run.txt; $bak=C:\Users\???\outputs_step7\latest_run.txt.bak; Get-Content -Raw $p | Set-Content $bak -Encoding UTF8"
call run_step8_latest.cmd
powershell -NoProfile -Command "$p=C:\Users\???\outputs_step7\latest_run.txt; $bak=C:\Users\???\outputs_step7\latest_run.txt.bak; Get-Content -Raw $bak | Set-Content $p -Encoding UTF8"
```

---

## Why `python -m pytest` (pytest.exe 금지)

On Windows, `pytest.exe` can resolve to a different interpreter than `python`, which leads to mismatched site-packages and binary wheels. **Always run pytest through the same interpreter.**

```cmd
REM CORRECT
python -m pytest -q tests/test_phase24_filter_impact.py

REM WRONG - may use different interpreter
pytest tests/test_phase24_filter_impact.py
```

---

## PowerShell Glob 패턴 금지

**PowerShell에서 glob 패턴(`*`, `?`)은 리터럴 문자열로 전달되어 pytest가 파일을 찾지 못합니다.**

```powershell
# WRONG - glob이 리터럴로 전달됨 (PowerShell 문제)
python -m pytest tests/test_phase2_*.py

# 결과: "no tests ran" 또는 "file not found"
```

**해결책**: cmd.exe를 사용하거나 명시적 파일 리스트/청크 스크립트 사용

```cmd
REM CORRECT - 명시적 파일 리스트
python -m pytest -q tests/test_phase2_cost_models_handcalc.py tests/test_phase2_integration_smoke.py

REM CORRECT - 청크 스크립트 사용 (권장)
scripts\run_ci_chunks.cmd core
```

---

## Interpreter Check (Windows cmd.exe)

```cmd
where.exe python
where.exe pytest
python -c "import sys; print(sys.executable); import numpy as np; print(np.__version__)"
python -m pytest --version
```

---

## NumPy Import Errors

If `python -m pytest ...` fails due to NumPy C-extension import errors:

```cmd
REM 1. Repair the current interpreter environment
python -m pip install -U pip setuptools wheel
python -m pip uninstall -y numpy
python -m pip install --no-cache-dir --force-reinstall numpy

REM 2. Re-check
python -c "import numpy as np; print(np.__version__)"
```

---

## Python Version Guidance

- Recommended: Python 3.12 or 3.13 for wheel availability and stability.
- If staying on Python 3.14, ensure NumPy 2.3.x+ wheels are installed (avoid source builds).

---

## 14s CI Chunks (Windows cmd.exe)

Full pytest exceeds 14s harness limit; use `scripts\run_ci_chunks.cmd` for chunked execution.

### CHUNK 사용법

```cmd
REM 전체 실행 (기존 호환)
scripts\run_ci_chunks.cmd
scripts\run_ci_chunks.cmd all

REM 선택 실행 (특정 청크만)
scripts\run_ci_chunks.cmd core
scripts\run_ci_chunks.cmd p22
scripts\run_ci_chunks.cmd p23
scripts\run_ci_chunks.cmd p24
scripts\run_ci_chunks.cmd p25
scripts\run_ci_chunks.cmd p26

REM 대소문자 무시 (P25 == p25)
scripts\run_ci_chunks.cmd P25
```

### 청크별 파일 매핑

| Chunk  | Files                                                                                                              |
|--------|--------------------------------------------------------------------------------------------------------------------|
| `core` | `test_phase2_cost_models_handcalc.py`, `test_phase2_integration_smoke.py`, `test_phase2_sensitivity_smoke.py`     |
| `p22`  | `test_phase22_event_builder.py`                                                                                    |
| `p23`  | `test_phase23_guardrails.py`                                                                                       |
| `p24`  | `test_phase24_filter_impact.py`                                                                                    |
| `p25`  | `test_phase25_distortion.py`                                                                                       |
| `p26`  | `test_phase26_recommendation.py`                                                                                   |

### 로그 리다이렉션 (cmd.exe)

```cmd
REM 로그 파일로 저장 (stdout + stderr)
scripts\run_ci_chunks.cmd p22 > outputs\ci_chunks_log.txt 2>&1

REM 로그 확인
type outputs\ci_chunks_log.txt
```

### Python 선택 규칙

1. `%ROOT%\.venv\Scripts\python.exe` 우선 사용
2. 없으면 기본은 **FAIL-FAST** (즉시 종료)
3. `ALLOW_SYSTEM_PYTHON=1` 설정 시에만 system python fallback 허용

```cmd
REM system python fallback 허용 (권장하지 않음)
set ALLOW_SYSTEM_PYTHON=1
scripts\run_ci_chunks.cmd all
```

### 14s 하네스에서 유효한 이유 (3줄 요약)

1. **개별 청크 선택**으로 실패한 Phase만 재실행 → 디버깅 시 전체 ~90s 대신 ~14s 단위로 빠른 피드백
2. **기존 all 모드와 완전 호환** 유지로 CI 파이프라인 변경 없이 인자만 추가하면 선택 실행 활성화
3. **`call :CHUNK_*` 서브루틴 패턴**으로 코드 중복 제거 + fail-fast 동작이 all/선택 모두에서 일관되게 동작

---

## CI에서 청크 실행 (GitHub Actions)

CI 환경에서는 **14s 하네스 제한**을 준수하기 위해 `all` 모드 대신 **청크별 병렬 실행**을 사용합니다.

### CI 워크플로 구조

```yaml
# .github/workflows/ci_chunks_windows.yml
strategy:
  fail-fast: false
  matrix:
    chunk: [core, p22, p23, p24, p25, p26]  # 6개 병렬 잡
```

- **각 청크는 독립된 GitHub Actions 잡**으로 실행 (병렬)
- 각 잡에서 `scripts\run_ci_chunks.cmd <chunk>` 호출
- `fail-fast: false` 설정으로 하나가 실패해도 나머지 청크 결과 확인 가능
- **Python 3.14** 기본 (로컬 개발 환경과 동일)

### 로컬 vs CI 사용 구분

| 환경 | 권장 모드 | Python | 이유 |
|------|-----------|--------|------|
| **로컬 개발** | `all` | 3.14 | 타임아웃 제한 없음, 전체 검증 가능 |
| **CI/하네스** | 개별 청크 | 3.14 | 14s 제한 준수, 병렬 실행으로 총 시간 단축 |

### CI 트리거 조건 (paths 필터)

다음 파일/폴더 변경 시 CI가 자동 실행됩니다:

- `scripts/**`, `tests/**`, `pytest.ini`, `run_tests.py`
- `execution/**`, `adapters/**`, `examples/**`, `costs/**`
- `phase1_anchor_engine.py` (엔진 단일 파일)
- `docs/**`, `requirements.txt`, `requirements-dev.txt`, `pyproject.toml`, `setup.py`, `setup.cfg`

**경고(강함):** 현재 push.paths 트리거는 `requirements.txt`와 `requirements-dev.txt` **두 개만** 명시되어 있습니다.  
향후 `requirements-*.txt` 파일을 추가하면 workflow의 push.paths를 함께 갱신하지 않는 한 CI가 누락되어 실운영 리스크가 발생할 수 있습니다.  
새 requirements 파일 추가 시 push.paths 업데이트를 반드시 동반하십시오.

**강한 반박/대안:** glob 제거는 새 requirements 파일 추가 시 CI 누락을 부르는 실운영 리스크가 큽니다.  
**참고(대안):** (1) paths는 `requirements*.txt` glob 유지 + cache-dependency-path는 명시 리스트 유지 (2) paths에 `requirements*.txt` + `requirements.txt` + `requirements-dev.txt` 병기(중복 허용)

PR 대상 브랜치: `main`, `master`, `develop`

### CI 수동 트리거 (workflow_dispatch)

GitHub Actions에서 수동 실행 시 옵션 선택 가능:

1. Actions 탭 → "CI Chunks (Windows)" 선택
2. "Run workflow" 버튼 클릭
3. 입력 옵션:

| 입력 | 기본값 | 설명 |
|------|--------|------|
| `python_version` | `3.14` | 테스트할 Python 버전 (`3.14`, `3.13`, `3.12`) |
| `enable_utf8` | `false` | `PYTHONUTF8=1` 활성화 (인코딩 디버깅 시만 사용) |

### 주의사항

- **PowerShell glob 금지**: CI 스크립트에서도 `*`, `?` 패턴 사용 금지
- **PYTHONUTF8**: 기본 OFF (부작용 방지), `workflow_dispatch`에서만 선택적 활성화
- **.venv 필수**: CI는 반드시 `.venv`를 생성하고 pytest를 설치해야 함 (`run_ci_chunks.cmd`가 `.venv` 우선 사용)

---

## Test Principles

### Fixtures 기반 결정론 검증

- 모든 테스트는 **fixtures 기반 결정론적 검증**을 따릅니다.
- 엔진 로직 변경 시 골든 파일 갱신 규칙을 준수합니다.
- Phase 1 anchor 및 비용/엔진 계산 로직은 테스트 하네스에서 변경하지 않습니다.

### 골든 파일 갱신 규칙

1. 엔진 로직 변경이 **의도된 변경**인지 확인
2. 새 골든 값이 수학적으로 올바른지 hand-calculation으로 검증
3. 검증 후 `tests/golden/` 파일 갱신
4. 변경 사유를 커밋 메시지에 명시
