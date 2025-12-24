"""
adapters/ - 입력 정규화 및 Phase 1/2 실행기

이 모듈은 다양한 데이터 소스에서 백테스트를 실행할 수 있도록
입력 정규화 및 실행기 함수를 제공합니다.

Modules:
- normalize: 입력 데이터 정규화 및 검증
- run_phase1_from_loader: Phase 1 실행기
- run_phase2_from_loader: Phase 2 실행기 (Phase 1 결과 + 추가 비용)
"""

from .normalize import normalize_inputs, validate_ohlcv
from .run_phase1_from_loader import run_phase1_from_loader
from .run_phase2_from_loader import run_phase2_from_loader, Phase2Result

__all__ = [
    "normalize_inputs",
    "validate_ohlcv",
    "run_phase1_from_loader",
    "run_phase2_from_loader",
    "Phase2Result",
]
