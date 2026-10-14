"""ASOS 시간자료 → 분석용 패널 + 체감온도·폭염더미.

서울 단일 관측소(108) 사용. 행정동별 차이는 무시(현 단계에서는 동질 가정).

체감온도: 기상청 2020 개정 여름철 체감온도 (Stull 습구온도 기반).
참고: https://www.weather.go.kr (생활기상지수 산출식)
"""
from __future__ import annotations

import json
import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import RAW_DIR

ASOS_DIR = RAW_DIR / "weather" / "asos_hourly"
log = logging.getLogger(__name__)


def _load_asos_month(path: Path) -> pd.DataFrame:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for it in data["items"]:
        rows.append({
            "tm": it.get("tm"),
            "ta": it.get("ta"),
            "hm": it.get("hm"),
            "rn": it.get("rn"),
            "ws": it.get("ws"),
        })
    df = pd.DataFrame(rows)
    df["tm"] = pd.to_datetime(df["tm"])
    df["dt"] = df["tm"].dt.date
    df["hour"] = df["tm"].dt.hour
    for c in ("ta", "hm", "rn", "ws"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["rn"] = df["rn"].fillna(0.0)  # 강수 빈값 = 비 안 옴
    return df[["dt", "hour", "ta", "hm", "rn", "ws"]]


def load_asos(stn_id: int = 108) -> pd.DataFrame:
    files = sorted(ASOS_DIR.glob(f"stn{stn_id}_*.json"))
    if not files:
        raise RuntimeError(f"no ASOS files in {ASOS_DIR}")
    frames = [_load_asos_month(p) for p in files]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["dt", "hour"]).reset_index(drop=True)
    # 시간 보간(ta/hm/ws) — 단발 결측만. 강수는 0 fill.
    for c in ("ta", "hm", "ws"):
        df[c] = df[c].interpolate(method="linear", limit=3)
    return df


def _stull_wet_bulb(t: pd.Series, rh: pd.Series) -> pd.Series:
    """Stull (2011) 습구온도 근사. T[°C], RH[%]. 1기압 가정."""
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * np.power(rh, 1.5) * np.arctan(0.023101 * rh)
        - 4.686035
    )


def apparent_temp_summer(ta: pd.Series, hm: pd.Series) -> pd.Series:
    """기상청 여름철 체감온도 (2020 개정, Stull 습구온도 기반)."""
    tw = _stull_wet_bulb(ta, hm)
    return (
        -0.2442
        + 0.55399 * tw
        + 0.45535 * ta
        - 0.0022 * tw**2
        + 0.00278 * tw * ta
        + 3.0
    )


def add_derived(weather: pd.DataFrame) -> pd.DataFrame:
    """체감온도 + 일별 폭염 더미 + 강수 더미."""
    df = weather.copy()
    df["apt"] = apparent_temp_summer(df["ta"], df["hm"])  # apparent temperature
    df["rain"] = (df["rn"] > 0).astype(int)

    # 일별 최고기온·최고체감온도 → 폭염더미
    daily = df.groupby("dt").agg(
        ta_max=("ta", "max"),
        apt_max=("apt", "max"),
        rn_sum=("rn", "sum"),
    ).reset_index()
    daily["heatwave_ta"] = (daily["ta_max"] >= 33).astype(int)        # 기상청 기온 기준
    daily["heatwave_apt"] = (daily["apt_max"] >= 33).astype(int)      # 체감온도 기준 (2020~ 공식)
    daily["heatwave_warn_apt"] = (daily["apt_max"] >= 35).astype(int) # 폭염경보 기준
    daily["rainy_day"] = (daily["rn_sum"] >= 1).astype(int)

    return df.merge(daily, on="dt", how="left")
