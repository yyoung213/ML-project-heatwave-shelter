"""생활인구(A-1 단기외국인, A-2 내국인) CSV 통합 로더.

서울 열린데이터광장 원본은 wide-format (집단별 컬럼 분리).
여기서는 분석용 long-format 으로 통일:

    columns: dt(date), hour(int 0~23), dong(str8), group(str), pop(float)

group:
    'foreign_total'  단기외국인 전체           (A-1: 총생활인구수)
    'foreign_cn'     단기외국인 중 중국인       (A-1: 중국인체류인구수)
    'foreign_other'  단기외국인 중 비중국인     (A-1: 중국외외국인체류인구수)
    'domestic'       내국인 전체               (A-2: 총생활인구수)

A-1 CSV 의 trailing 쉼표 때문에 index_col=False 필수.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import pandas as pd

from ..config import RAW_DIR

FOREIGN_DIR = RAW_DIR / "seoul_lifepop" / "short_foreign"
DOMESTIC_DIR = RAW_DIR / "seoul_lifepop" / "domestic"

log = logging.getLogger(__name__)


def _parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, format="%Y%m%d").dt.date


def _load_foreign_month(path: Path) -> pd.DataFrame:
    """A-1 한 달치를 읽어 long-format 으로 변환."""
    df = pd.read_csv(
        path, encoding="utf-8-sig", index_col=False,
        usecols=[
            "기준일ID", "시간대구분", "행정동코드",
            "총생활인구수", "중국인체류인구수", "중국외외국인체류인구수",
        ],
        dtype={"기준일ID": str, "시간대구분": str, "행정동코드": str},
    )
    df = df.rename(columns={
        "기준일ID": "dt", "시간대구분": "hour", "행정동코드": "dong",
        "총생활인구수": "foreign_total",
        "중국인체류인구수": "foreign_cn",
        "중국외외국인체류인구수": "foreign_other",
    })
    df["dt"] = _parse_dt(df["dt"])
    df["hour"] = df["hour"].astype(int)
    for c in ("foreign_total", "foreign_cn", "foreign_other"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    long = df.melt(
        id_vars=["dt", "hour", "dong"],
        value_vars=["foreign_total", "foreign_cn", "foreign_other"],
        var_name="group", value_name="pop",
    )
    return long


def _load_domestic_month(path: Path) -> pd.DataFrame:
    """A-2 한 달치 → long-format. 성·연령별 컬럼은 무시(총생활인구수만)."""
    df = pd.read_csv(
        path, encoding="utf-8-sig", index_col=False,
        usecols=["기준일ID", "시간대구분", "행정동코드", "총생활인구수"],
        dtype={"기준일ID": str, "시간대구분": str, "행정동코드": str},
    )
    df = df.rename(columns={
        "기준일ID": "dt", "시간대구분": "hour", "행정동코드": "dong",
        "총생활인구수": "pop",
    })
    df["dt"] = _parse_dt(df["dt"])
    df["hour"] = df["hour"].astype(int)
    df["pop"] = pd.to_numeric(df["pop"], errors="coerce")
    df["group"] = "domestic"
    return df[["dt", "hour", "dong", "group", "pop"]]


def load_lifepop(months: list[str] | None = None) -> pd.DataFrame:
    """A-1·A-2 통합 long-format 로드.

    Args:
        months: ['202306', '202307', ...]. None 이면 raw/ 에 있는 모든 월.
    """
    if months is None:
        foreign_files = sorted(FOREIGN_DIR.glob("TEMP_FOREIGNER_DONG_*.csv"))
        months = sorted({p.stem.split("_")[-1] for p in foreign_files})

    frames = []
    for ym in months:
        f1 = FOREIGN_DIR / f"TEMP_FOREIGNER_DONG_{ym}.csv"
        f2 = DOMESTIC_DIR / f"LOCAL_PEOPLE_DONG_{ym}.csv"
        if not f1.exists():
            log.warning("missing A-1 %s — skip", ym)
            continue
        if not f2.exists():
            log.warning("missing A-2 %s — skip", ym)
            continue
        frames.append(_load_foreign_month(f1))
        frames.append(_load_domestic_month(f2))
        log.info("loaded %s", ym)

    if not frames:
        raise RuntimeError("no lifepop files matched")
    out = pd.concat(frames, ignore_index=True)
    # 결측 마스킹("*" 셀)은 to_numeric 단계에서 이미 NaN 처리됨
    return out
