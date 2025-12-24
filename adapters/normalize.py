"""
adapters/normalize.py - 입력 데이터 정규화 및 검증

다양한 포맷의 OHLCV 데이터를 표준 형식으로 변환합니다.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# 필수 컬럼
REQUIRED_COLUMNS = {"open", "high", "low", "close", "volume"}


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 소문자로 정규화"""
    df = df.copy()
    df.columns = [c.lower().strip() for c in df.columns]
    return df


def parse_datetime_index(df: pd.DataFrame) -> pd.DataFrame:
    """날짜/시간 인덱스 파싱"""
    df = df.copy()

    # 이미 DatetimeIndex인 경우
    if isinstance(df.index, pd.DatetimeIndex):
        return df

    # date, timestamp, time 컬럼 찾기
    date_cols = ["date", "timestamp", "time", "datetime"]
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
            df = df.set_index(col)
            return df

    # 첫 번째 컬럼이 날짜일 수 있음
    try:
        first_col = df.columns[0]
        df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
        if df[first_col].notna().sum() > len(df) * 0.5:
            df = df.set_index(first_col)
    except Exception:
        pass

    return df


def validate_ohlcv(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    OHLCV 데이터 검증

    Returns:
        (is_valid, error_messages)
    """
    errors = []

    # 컬럼 확인
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {missing}")

    # 인덱스 확인
    if not isinstance(df.index, pd.DatetimeIndex):
        errors.append("Index must be DatetimeIndex")

    # 데이터 타입 확인
    for col in REQUIRED_COLUMNS & set(df.columns):
        if not np.issubdtype(df[col].dtype, np.number):
            errors.append(f"Column '{col}' must be numeric")

    # OHLC 관계 확인 (high >= low)
    if "high" in df.columns and "low" in df.columns:
        violations = (df["high"] < df["low"]).sum()
        if violations > 0:
            errors.append(f"high < low in {violations} rows")

    # NaN 확인
    if "close" in df.columns:
        nan_ratio = df["close"].isna().sum() / len(df)
        if nan_ratio > 0.5:
            errors.append(f"Too many NaN in close: {nan_ratio:.1%}")

    return len(errors) == 0, errors


def normalize_inputs(
    df: pd.DataFrame,
    min_days: int = 60,
    drop_nan_close: bool = True
) -> Optional[pd.DataFrame]:
    """
    OHLCV 데이터 정규화

    Args:
        df: 원본 DataFrame
        min_days: 최소 필요 일수
        drop_nan_close: close가 NaN인 행 제거 여부

    Returns:
        정규화된 DataFrame 또는 None (유효하지 않은 경우)
    """
    if df is None or len(df) == 0:
        return None

    # 컬럼명 정규화
    df = normalize_column_names(df)

    # 인덱스 파싱
    df = parse_datetime_index(df)

    # 필수 컬럼 확인
    if not REQUIRED_COLUMNS.issubset(set(df.columns)):
        return None

    # 정렬
    df = df.sort_index()

    # NaN 처리
    if drop_nan_close:
        df = df.dropna(subset=["close"])

    # 최소 일수 확인
    if len(df) < min_days:
        return None

    # 필수 컬럼만 반환
    return df[["open", "high", "low", "close", "volume"]]


def load_ohlcv_file(
    path: str,
    min_days: int = 60
) -> Optional[pd.DataFrame]:
    """
    CSV 파일에서 OHLCV 데이터 로드 및 정규화

    Args:
        path: CSV 파일 경로
        min_days: 최소 필요 일수

    Returns:
        정규화된 DataFrame 또는 None
    """
    if not os.path.exists(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        try:
            df = pd.read_csv(path, index_col=0, parse_dates=True)
        except Exception:
            return None

    return normalize_inputs(df, min_days=min_days)


def load_venue_data(
    venue_path: str,
    min_days: int = 60
) -> Dict[str, pd.DataFrame]:
    """
    거래소 디렉토리에서 모든 심볼 데이터 로드

    Args:
        venue_path: 거래소 데이터 디렉토리
        min_days: 최소 필요 일수

    Returns:
        {symbol: DataFrame}
    """
    if not os.path.isdir(venue_path):
        return {}

    data = {}
    for f in os.listdir(venue_path):
        if not f.endswith(".csv"):
            continue

        df = load_ohlcv_file(os.path.join(venue_path, f), min_days=min_days)
        if df is None:
            continue

        symbol = f.replace(".csv", "").upper()
        data[symbol] = df

    return data
