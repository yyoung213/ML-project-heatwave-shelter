"""패널 조인 + 평년 정규화 + gap 지표.

핵심 산출 컬럼:
    dt, hour, dong, dow(요일 0=월), year, month,
    foreign_total, foreign_cn, foreign_other, domestic,
    base_foreign, base_domestic,        # 동·요일·시간대 중앙값 (3년 풀)
    ratio_foreign, ratio_domestic,      # 실제/평년
    gap,                                # ratio_foreign - ratio_domestic (분석의 핵심)
    ta, hm, rn, ws, apt,                # 기상
    rain, heatwave_ta, heatwave_apt, heatwave_warn_apt, rainy_day,
    missing_day                         # 원본 결측 일자 플래그 (2025-07-08, 09)

설계 메모:
- baseline 은 폭염 일자 포함하여 풀 중앙값. 분리하면 baseline 자체가 분석 결과에
  의존하므로 단순 풀 채택.
- baseline=0 이거나 NaN 이면 ratio 도 NaN — gap 계산에서 자동 제외.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


def to_wide(lifepop_long: pd.DataFrame) -> pd.DataFrame:
    """long → wide. (dt, hour, dong) 행, group → 컬럼."""
    wide = lifepop_long.pivot_table(
        index=["dt", "hour", "dong"], columns="group", values="pop", aggfunc="first",
    ).reset_index()
    wide.columns.name = None
    return wide


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.to_datetime(df["dt"])
    df = df.copy()
    df["year"] = dt.dt.year.astype(int)
    df["month"] = dt.dt.month.astype(int)
    df["dow"] = dt.dt.dayofweek.astype(int)  # 0=월
    return df


BASELINE_RENAME = {"foreign_total": "base_foreign", "domestic": "base_domestic"}


def compute_baseline(panel: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """동 × 요일 × 시간대 중앙값. 3년 풀 사용."""
    base = panel.groupby(["dong", "dow", "hour"])[cols].median().reset_index()
    return base.rename(columns=BASELINE_RENAME)


def build_panel(lifepop_long: pd.DataFrame, weather_with_derived: pd.DataFrame) -> pd.DataFrame:
    """전체 패널 산출."""
    wide = to_wide(lifepop_long)
    wide = add_calendar(wide)
    log.info("wide panel: %d rows", len(wide))

    # 결측 일자 플래그 (생활인구 원본 결측)
    full_days = pd.date_range(wide["dt"].min(), wide["dt"].max(), freq="D").date
    present = set(wide["dt"].unique())
    missing_days = set(full_days) - present
    # 분석 기간(여름 6~8월) 외 일자는 결측이 아님. 6~8월 일자만 체크.
    missing_summer = {d for d in missing_days if d.month in (6, 7, 8)}
    if missing_summer:
        log.warning("원본 결측 일자(여름): %s", sorted(missing_summer))

    # baseline: 동·요일·시간대 중앙값 (foreign_total, domestic 두 핵심만 우선)
    baseline = compute_baseline(wide, ["foreign_total", "domestic"])
    panel = wide.merge(baseline, on=["dong", "dow", "hour"], how="left")

    # 비율 + gap
    eps = 1e-6
    panel["ratio_foreign"] = panel["foreign_total"] / panel["base_foreign"].replace(0, np.nan)
    panel["ratio_domestic"] = panel["domestic"] / panel["base_domestic"].replace(0, np.nan)
    panel["gap"] = panel["ratio_foreign"] - panel["ratio_domestic"]

    # 기상 조인 (dt, hour)
    panel = panel.merge(
        weather_with_derived,
        on=["dt", "hour"], how="left",
    )

    # 결측 플래그
    panel["missing_day"] = 0  # 원본에 있는 일자는 모두 0 (없는 일자는 wide에 안 들어옴)

    return panel
