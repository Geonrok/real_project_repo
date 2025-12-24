# Phase 2.6 Changelog

## Why This Is Needed (Decision Sealed)
- Sweep 결과를 한 번의 규칙 기반 선택으로 봉인하여 재현성과 책임성을 확보한다.
- Distortion guardrail과 NAV 양수 조건을 동시에 만족하는 기본값을 강제한다.
- 선택 근거와 트레이드오프를 표준 리포트로 남겨 의사결정 기록을 남긴다.
- 동일한 입력에서 항상 동일한 추천이 나오도록 결정론을 보장한다.
- Phase 2 실행 로직을 변경하지 않고 분석/선택/리포팅 범위에 한정한다.

## Recommendation Rule
Select the feasible grid point with final_nav_p2 maximized (default), subject to final_nav_p2 > 0, l1_distance_max <= guardrail_l1_distance_max, and l1_distance_mean <= guardrail_l1_distance_mean; if multiple points are within 0.1% of the best objective, break ties by lower l1_distance_mean, then lower filter_ratio, then smaller min_notional_cash.

## Outputs
- `outputs/phase26_recommendation.json`
- `outputs/phase26_pareto.csv`
- `outputs/phase26_tradeoff.png` (optional)
- `outputs/real_run_manifest.json` includes `recommendation` block

## API/CLI
- `adapters.run_phase2_from_loader.recommend_min_notional_from_sweep`
- `examples/analyze_phase26_recommendation.py`

## Schema
- `schema_version`: `2.6.0`
