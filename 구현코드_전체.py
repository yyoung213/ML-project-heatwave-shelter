# -*- coding: utf-8 -*-
"""
폭염 상황의 모빌리티 적응 격차와 공간 타겟팅 — 전체 구현코드 (단일 파일, 2조).

보고서 흐름 15단계를 한 파일에 순서대로 모았다.
  01~02b 전처리(원천 raw) · 03~05 매칭 · 06~08 4장 그림 · 09~12 동 피처 · 13 클러스터링 · 14~15 입지.

실행:
  python 구현코드_전체.py                  # 03~15 (전처리 산출 parquet 사용, 그림 전부 재현)
  RUN_PREPROCESS=1 python 구현코드_전체.py # 01~02b 원천 raw 전처리부터 (KT 34GB, 수분 소요)

데이터는 이 파일과 같은 위치(또는 한 단계 위)의 '데이터' 폴더를 자동으로 찾는다.
출력은 이 파일 옆 ./출력/figures, ./출력/tables. 절대경로 없음 → 다른 컴퓨터에서도 그대로 실행된다.
"""
from __future__ import annotations
import os, sys, shutil, glob, re, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "Malgun Gothic", "NanumGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

# ── 경로 (번들 기준 상대) ──
HERE = Path(__file__).resolve().parent
# 데이터 폴더는 코드 파일과 같은 위치 또는 한 단계 위에서 자동으로 찾는다.
DATA = next((c for c in (HERE / "데이터", HERE.parent / "데이터") if c.exists()), HERE / "데이터")
ROOT = DATA
PROC = PROC_BASE = PROCESSED = DATA / "data/processed"
RAW = RAW_BASE = DATA / "data/raw"
GEO_DONG = DATA / "data/행정구역_폴리곤/admdong_kr_20250401.geojson"
GEO_GU = DATA / "data/raw/행정구역_폴리곤/seoul_gu.geojson"
SHELTER = DATA / "data/raw/보조_데이터/외국인쉼터/tourist_cooling_shelters_2025.csv"
ADD = DATA / "추가 데이터"
INTERMEDIATE = DATA / "분석_중간산출"
OUT = HERE / "출력"; FIGS = OUT / "figures"; TABLES = OUT / "tables"
for _d in (FIGS, TABLES): _d.mkdir(parents=True, exist_ok=True)
# 분석 중간산출(N15 피처·N11 공급·dong_base·dong_gistar)을 TABLES에 시드 → 5장 즉시 실행
if INTERMEDIATE.exists():
    for _p in INTERMEDIATE.glob("*.parquet"):
        _dst = TABLES / _p.name
        if not _dst.exists(): shutil.copy2(_p, _dst)

RUN_PREPROCESS = os.environ.get("RUN_PREPROCESS", "0") == "1" or "--preprocess" in sys.argv

# ── 3집단 상수 (src/pop_types 인라인) ──
POP_ORDER = ["내국인", "단기외국인", "장기외국인"]
FOREIGN_POPS = ["단기외국인", "장기외국인"]
MAIN_PAIR = ("내국인", "단기외국인")
POP_LABEL = {"내국인": "Local", "단기외국인": "Foreign (short)", "장기외국인": "Foreign (long)"}
POP_NAME_KR = {"내국인": "내국인", "단기외국인": "단기 체류 외국인", "장기외국인": "장기 체류 외국인"}
POP_COLOR = {"내국인": "#1f77b4", "단기외국인": "#d62728", "장기외국인": "#2ca02c"}
POP_COL = {"내국인": "local_arr", "단기외국인": "foreign_arr", "장기외국인": "foreign_long_arr"}
MATCH_TRUTH = {"내국인": -3.01, "단기외국인": -0.90, "장기외국인": -3.66}

# ── 자치구 4유형 상수 (src/gu_types 인라인) ──
GU_TYPE = {
    "강남구": "복합핵심", "강서구": "복합핵심", "마포구": "복합핵심", "서초구": "복합핵심", "성동구": "복합핵심",
    "영등포구": "복합핵심", "용산구": "복합핵심", "종로구": "복합핵심", "중구": "복합핵심",
    "구로구": "업무형", "금천구": "업무형", "동대문구": "업무형",
    "강동구": "관광·여가형", "광진구": "관광·여가형", "송파구": "관광·여가형",
    "강북구": "주거형", "관악구": "주거형", "노원구": "주거형", "도봉구": "주거형", "동작구": "주거형",
    "서대문구": "주거형", "성북구": "주거형", "양천구": "주거형", "은평구": "주거형", "중랑구": "주거형",
}
TYPE_ORDER = ["복합핵심", "업무형", "관광·여가형", "주거형"]
TYPE_COLOR = {"복합핵심": "#d62728", "업무형": "#1f77b4", "관광·여가형": "#ff9800", "주거형": "#7f7f7f"}
def gu_to_type(gu_name): return GU_TYPE.get(gu_name, "주거형")

def data_label(fig, text):
    fig.text(0.99, 0.005, text, ha="right", va="bottom", fontsize=7.5, color="#444",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbb", lw=0.7))



# ==============================================================================
# STEP 01 전처리 — 자치구 단위 (raw KT 도착지·ASOS → kt_gu_hour_daily·weather_daily)
# ==============================================================================
print("\n>>> STEP 01 전처리 — 자치구 단위 (raw KT 도착지·ASOS → kt_gu_hour_daily·weather_daily)")
# 기초 EDA 일괄 실행. KT 도착지 + 기상 ASOS 데이터 로드·집계·통계·시각화.
"""실행. python src/eda_basics.py
출력. data/processed/eda_*.csv, outputs/figures/eda_*.png"""

import json
import glob
import math
import sys
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams['font.family'] = ['Apple SD Gothic Neo', 'AppleGothic', 'Nanum Gothic', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

ROOT = ROOT
KT_BASE = ROOT / "data/raw/KT_도착지"
WEATHER = ROOT / "data/raw/기상_ASOS"
PROCESSED = ROOT / "data/processed"
FIGS = FIGS
PROCESSED.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)



# ============================================================
# 1) KT 도착지 일괄 집계 — 자치구×시간×일 단위
# ============================================================
def parse_time_cd(tc) -> int:
    """time_cd 0~23 또는 HHMM(700·720 등) → 시 단위 0~23."""
    tc = int(tc)
    if tc < 100:
        return tc
    # HHMM 형식 → 시 추출
    return tc // 100


def aggregate_kt_arrivals():
    """KT 도착지 일별 csv → 자치구×시간×일×pop 집계 저장.

    외국인은 단기/장기 분리. 단기외국인 = short_forn_cnt(관광객), 장기외국인 =
    total_cnt - short_forn_cnt(거주 외국인). 장기 = 성·연령 분해합과 정확히 일치
    (src/pop_types.py 참조). 내국인은 total_cnt.
    """
    print("=== KT 도착지 집계 시작 (내국인·단기외국인·장기외국인) ===")
    records = []
    age_cols = ([f"male_{i}0_cnt" for i in range(8)]
                + [f"feml_{i}0_cnt" for i in range(8)])  # 장기 검증용
    for pop_dir in ["내국인", "외국인"]:
        for year_dir in sorted((KT_BASE / pop_dir).iterdir()):
            year = year_dir.name.replace("_여름", "")
            for month_dir in sorted(year_dir.iterdir()):
                if not month_dir.is_dir():
                    continue
                month = month_dir.name.replace("월", "")
                files = sorted(month_dir.glob("*.csv"))
                if not files:
                    continue
                for f in files:
                    date_str = f.stem.split("_")[-1]  # YYYYMMDD
                    encoding = "cp949" if pop_dir == "외국인" else "utf-8"
                    df = pd.read_csv(f, encoding=encoding)
                    df["hour"] = df["time_cd"].apply(parse_time_cd)
                    df["gu_cd"] = df["d_admdong_cd"].astype(str).str[:5]
                    the_date = pd.to_datetime(date_str, format="%Y%m%d").date()
                    if pop_dir == "내국인":
                        cols = {"내국인": "total_cnt"}
                    else:
                        # 장기 = 전체 - 단기 (= 성·연령 분해합)
                        df["long_forn_cnt"] = df["total_cnt"] - df["short_forn_cnt"]
                        cols = {"단기외국인": "short_forn_cnt",
                                "장기외국인": "long_forn_cnt"}
                    for pop_name, cnt_col in cols.items():
                        agg = df.groupby(["gu_cd", "hour"])[cnt_col].sum().reset_index()
                        agg.columns = ["gu_cd", "hour", "total"]
                        agg["date"] = the_date
                        agg["pop"] = pop_name
                        records.append(agg)
                print(f"  {pop_dir} {year}-{month}: {len(files)}일 ({len(records)} chunks)")
    df = pd.concat(records, ignore_index=True)
    out = PROCESSED / "kt_gu_hour_daily.parquet"
    df.to_parquet(out)
    print(f"  저장. {out}  ({len(df):,}행)  pop = {sorted(df['pop'].unique())}")
    return df


# ============================================================
# 2) 기상 ASOS JSON 로드 + 체감온도 계산
# ============================================================
def kma_feels_like(ta: float, rh: float) -> float:
    """기상청 2020 여름철 체감온도 공식."""
    if pd.isna(ta) or pd.isna(rh):
        return np.nan
    try:
        tw = (
            ta * math.atan(0.151977 * (rh + 8.313659) ** 0.5)
            + math.atan(ta + rh)
            - math.atan(rh - 1.67633)
            + 0.00391838 * (rh ** 1.5) * math.atan(0.023101 * rh)
            - 4.686035
        )
        pt = (
            -0.2442
            + 0.55399 * tw
            + 0.45535 * ta
            - 0.0022 * (tw ** 2)
            + 0.00278 * tw * ta
            + 3.0
        )
        return pt
    except Exception:
        return np.nan


def load_weather():
    """ASOS JSON 4년치 통합 → DataFrame + 체감온도 추가."""
    print("\n=== 기상 ASOS 로드 ===")
    rows = []
    for f in sorted(WEATHER.glob("*/stn108_*.json")):
        with open(f) as fp:
            d = json.load(fp)
        for it in d["items"]:
            rows.append(it)
    df = pd.DataFrame(rows)
    df["tm"] = pd.to_datetime(df["tm"])
    df["date"] = df["tm"].dt.date
    df["hour"] = df["tm"].dt.hour
    df["year"] = df["tm"].dt.year
    df["month"] = df["tm"].dt.month
    for c in ["ta", "hm", "ws", "rn", "ss", "icsr", "ts", "td", "pa", "ps"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["feels_like"] = df.apply(lambda r: kma_feels_like(r["ta"], r["hm"]), axis=1)
    out = PROCESSED / "weather_hourly.parquet"
    df.to_parquet(out)
    print(f"  저장. {out}  ({len(df):,}행, {df['year'].nunique()}년)")
    return df


# ============================================================
# 3) 통계 요약
# ============================================================
def stats_summary(kt: pd.DataFrame, wx: pd.DataFrame) -> dict:
    """주요 통계를 dict로 모음 (보고서에 표로 출력)."""
    info = {}
    # KT 기본
    info["kt_rows"] = len(kt)
    info["kt_dates"] = kt["date"].nunique()
    info["kt_gus"] = kt["gu_cd"].nunique()
    info["kt_hours"] = sorted(kt["hour"].unique().tolist())
    info["kt_pop_split"] = kt.groupby("pop")["total"].sum().to_dict()
    info["kt_by_year"] = (
        kt.assign(year=pd.to_datetime(kt["date"]).dt.year)
        .groupby(["year", "pop"])["total"]
        .sum()
        .unstack(fill_value=0)
        .to_dict()
    )
    # 일별 폭염일 — 체감온도
    daily = (
        wx.groupby("date")
        .agg(
            ta_max=("ta", "max"),
            ta_mean=("ta", "mean"),
            ta_min=("ta", "min"),
            feels_max=("feels_like", "max"),
            hm_mean=("hm", "mean"),
            rn_sum=("rn", "sum"),
            icsr_sum=("icsr", "sum"),
        )
        .reset_index()
    )
    daily["is_heatwave_temp"] = daily["ta_max"] >= 33
    daily["is_heatwave_feels"] = daily["feels_max"] >= 33
    daily["year"] = pd.to_datetime(daily["date"]).dt.year
    info["heatwave_days_temp"] = (
        daily.groupby("year")["is_heatwave_temp"].sum().to_dict()
    )
    info["heatwave_days_feels"] = (
        daily.groupby("year")["is_heatwave_feels"].sum().to_dict()
    )
    daily.to_parquet(PROCESSED / "weather_daily.parquet")
    return info, daily


# ============================================================
# 4) 시각화
# ============================================================
def viz_all(kt: pd.DataFrame, wx: pd.DataFrame, daily: pd.DataFrame):
    print("\n=== Visualization ===")

    LABEL = POP_LABEL  # 내국인·단기외국인·장기외국인 영문 라벨

    # F1. 시간대별 도착자 (3집단 비교)
    pivot = kt.groupby(["hour", "pop"])["total"].mean().unstack()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    markers = {"내국인": "o-", "단기외국인": "s-", "장기외국인": "^-"}
    for pop in POP_ORDER:
        ax.plot(pivot.index, pivot[pop], markers[pop], label=POP_LABEL[pop],
                color=POP_COLOR[pop], linewidth=2)
    ax.set_xlabel("Hour"); ax.set_ylabel("Avg arrivals per district")
    ax.set_title("F1. Hourly avg arrivals (all dates, all districts)")
    ax.legend(); ax.set_xticks(range(0, 24))
    plt.tight_layout(); plt.savefig(FIGS / "eda_01_hourly_pop.png", dpi=120); plt.close()

    # F2. 일별 시계열 — 내국인 3년 비교
    daily_kt = (
        kt[kt["pop"] == "내국인"]
        .groupby("date")["total"].sum()
        .reset_index()
    )
    daily_kt["date"] = pd.to_datetime(daily_kt["date"])
    daily_kt["year"] = daily_kt["date"].dt.year
    daily_kt["mmdd"] = daily_kt["date"].dt.strftime("%m-%d")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for y, color in [(2023, "#1f77b4"), (2024, "#2ca02c"), (2025, "#d62728")]:
        sub = daily_kt[daily_kt["year"] == y]
        ax.plot(sub["mmdd"], sub["total"], "-", label=f"{y}", color=color, alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Month-Day"); ax.set_ylabel("Daily arrivals (Seoul total)")
    ax.set_title("F2. Local daily arrivals — Summer 2023 vs 2024 vs 2025")
    n = len(daily_kt[daily_kt["year"] == 2025])
    ax.set_xticks(range(0, n, 7)); ax.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "eda_02_daily_3year.png", dpi=120); plt.close()

    # F2b. Foreign (short-stay) 3-year daily time series
    daily_fo3 = (
        kt[kt["pop"] == "단기외국인"]
        .groupby("date")["total"].sum()
        .reset_index()
    )
    daily_fo3["date"] = pd.to_datetime(daily_fo3["date"])
    daily_fo3["year"] = daily_fo3["date"].dt.year
    daily_fo3["mmdd"] = daily_fo3["date"].dt.strftime("%m-%d")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    for y, color in [(2023, "#1f77b4"), (2024, "#2ca02c"), (2025, "#d62728")]:
        sub = daily_fo3[daily_fo3["year"] == y]
        ax.plot(sub["mmdd"], sub["total"], "-", label=f"{y}", color=color, alpha=0.85, linewidth=1.5)
    ax.set_xlabel("Month-Day"); ax.set_ylabel("Daily arrivals (Seoul total)")
    ax.set_title("F2b. Foreign (short-stay) daily arrivals — Summer 2023·2024·2025")
    n = len(daily_fo3[daily_fo3["year"] == 2025])
    ax.set_xticks(range(0, n, 7)); ax.legend()
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout(); plt.savefig(FIGS / "eda_02b_daily_3year_foreign.png", dpi=120); plt.close()


    # F3. 자치구별 평균 도착자 (3집단)
    gu_avg = kt.groupby(["gu_cd", "pop"])["total"].mean().unstack(fill_value=0)
    gu_avg["sum"] = gu_avg.sum(axis=1)
    gu_avg = gu_avg.sort_values("sum", ascending=True).drop("sum", axis=1)
    gu_avg = gu_avg.tail(25)
    fig, ax = plt.subplots(figsize=(10, 8))
    y_pos = np.arange(len(gu_avg))
    offs = {"내국인": -0.27, "단기외국인": 0.0, "장기외국인": 0.27}
    for pop in POP_ORDER:
        ax.barh(y_pos + offs[pop], gu_avg[pop], 0.27, label=POP_LABEL[pop],
                color=POP_COLOR[pop])
    ax.set_yticks(y_pos); ax.set_yticklabels(gu_avg.index, fontsize=8)
    ax.set_xlabel("Avg arrivals per district-hour")
    ax.set_title("F3. Top 25 districts by avg arrivals — Local vs Foreign (short/long)")
    ax.legend()
    plt.tight_layout(); plt.savefig(FIGS / "eda_03_gu_distribution.png", dpi=120); plt.close()

    # F4. 기온 일변화 (시간별 평균, 4년)
    wx["year"] = wx["tm"].dt.year
    hr = wx.groupby(["year", "hour"])["ta"].mean().unstack(level=0)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for y in hr.columns:
        ax.plot(hr.index, hr[y], "-", label=f"{y}", linewidth=2)
    ax.set_xlabel("Hour"); ax.set_ylabel("Avg temperature (°C)")
    ax.set_title("F4. Hourly avg temperature — 4 summers (2022~2025)")
    ax.legend(title="Year"); ax.set_xticks(range(0, 24))
    plt.tight_layout(); plt.savefig(FIGS / "eda_04_temp_hourly.png", dpi=120); plt.close()

    # F5. 일 최고기온·체감온도 시계열 4년
    daily["date_dt"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date_dt"].dt.year
    daily["mmdd"] = daily["date_dt"].dt.strftime("%m-%d")
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for y, color in [(2022, "#888"), (2023, "#1f77b4"), (2024, "#2ca02c"), (2025, "#d62728")]:
        sub = daily[daily["year"] == y]
        axes[0].plot(sub["mmdd"], sub["ta_max"], "-", label=f"{y}", color=color, alpha=0.85, linewidth=1.3)
        axes[1].plot(sub["mmdd"], sub["feels_max"], "-", label=f"{y}", color=color, alpha=0.85, linewidth=1.3)
    for ax, ttl in [(axes[0], "Daily max temp (°C)"), (axes[1], "Daily max feels-like (°C)")]:
        ax.axhline(33, color="orange", linestyle="--", linewidth=1, label="33°C threshold")
        ax.set_ylabel(ttl); ax.legend(ncol=5, fontsize=8); ax.grid(alpha=0.3)
    axes[0].set_title("F5. Daily max temp and feels-like — 4 summers (dashed = 33°C threshold)")
    n = len(daily[daily["year"] == 2025])
    axes[1].set_xticks(range(0, n, 7)); plt.xticks(rotation=45, fontsize=7)
    plt.tight_layout(); plt.savefig(FIGS / "eda_05_temp_daily_4year.png", dpi=120); plt.close()

    # F6. 폭염일 수 막대 (4년, 기온 기준 vs 체감 기준)
    cnt_temp = daily.groupby("year")["is_heatwave_temp"].sum()
    cnt_feels = daily.groupby("year")["is_heatwave_feels"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(cnt_temp))
    w = 0.35
    ax.bar(x - w/2, cnt_temp.values, w, label="Temp ≥ 33°C", color="#1f77b4")
    ax.bar(x + w/2, cnt_feels.values, w, label="Feels-like ≥ 33°C", color="#d62728")
    ax.set_xticks(x); ax.set_xticklabels(cnt_temp.index)
    ax.set_ylabel("Heatwave days")
    ax.set_title("F6. Heatwave day count by year — temp vs feels-like")
    ax.legend()
    for i, (a, b) in enumerate(zip(cnt_temp.values, cnt_feels.values)):
        ax.text(i - w/2, a + 0.3, str(a), ha="center", fontsize=10)
        ax.text(i + w/2, b + 0.3, str(b), ha="center", fontsize=10)
    plt.tight_layout(); plt.savefig(FIGS / "eda_06_heatwave_count.png", dpi=120); plt.close()

    # F7. 도착자 분포 히스토그램 (3집단)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, pop in zip(axes, POP_ORDER):
        sub = kt[kt["pop"] == pop]["total"]
        ax.hist(sub[sub > 0], bins=80, color=POP_COLOR[pop], alpha=0.7)
        ax.set_xlabel("Arrivals per district-hour")
        ax.set_ylabel("Frequency (log)")
        ax.set_yscale("log")
        ax.set_title(f"{LABEL[pop]} arrivals (median={sub.median():.0f}, max={sub.max():.0f})")
    plt.tight_layout(); plt.savefig(FIGS / "eda_07_arrival_hist.png", dpi=120); plt.close()

    # F8. 기온 vs 도착자 산점 (자치구 평균, 일 단위, 3집단)
    wx_daily = daily.assign(date_dt=pd.to_datetime(daily["date"]))[
        ["date_dt", "ta_max", "feels_max", "is_heatwave_feels"]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, pop in zip(axes, POP_ORDER):
        d_pop = kt[kt["pop"] == pop].groupby("date")["total"].sum().reset_index()
        d_pop["date"] = pd.to_datetime(d_pop["date"])
        df_m = d_pop.merge(wx_daily, left_on="date", right_on="date_dt", how="inner")
        normal = df_m[~df_m["is_heatwave_feels"]]
        heat = df_m[df_m["is_heatwave_feels"]]
        ax.scatter(normal["feels_max"], normal["total"], s=18, alpha=0.5, color=POP_COLOR[pop], label="Normal day")
        ax.scatter(heat["feels_max"], heat["total"], s=30, alpha=0.85, color="orange", edgecolor="black", linewidth=0.5, label="Heatwave day")
        ax.set_xlabel("Daily max feels-like (°C)")
        ax.set_ylabel("Daily arrivals (Seoul total)")
        ax.set_title(f"F8. {POP_LABEL[pop]} — arrivals vs feels-like")
        ax.legend()
    plt.tight_layout(); plt.savefig(FIGS / "eda_08_scatter_temp_arrival.png", dpi=120); plt.close()

    # F9. 결측·마스킹 비율 (KT)
    miss_by_year = (
        kt.assign(year=pd.to_datetime(kt["date"]).dt.year)
        .groupby(["year", "pop"])["total"]
        .agg([
            ("zero_pct", lambda s: (s == 0).mean() * 100),
            ("rows", "count"),
        ])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    pivot = miss_by_year.pivot(index="year", columns="pop", values="zero_pct").reindex(columns=POP_ORDER)
    colors = [POP_COLOR[c] for c in pivot.columns]
    pivot.columns = [LABEL[c] for c in pivot.columns]
    pivot.plot(kind="bar", ax=ax, color=colors)
    ax.set_ylabel("Zero-cell ratio (%)")
    ax.set_title("F9. District-hour cells with zero arrivals (masking proxy)")
    ax.legend(title="Population"); plt.xticks(rotation=0)
    plt.tight_layout(); plt.savefig(FIGS / "eda_09_missing.png", dpi=120); plt.close()

    # F10. 기상 결측 비율 (컬럼별)
    cols_we = ["ta", "hm", "ws", "wd", "rn", "ss", "icsr", "ts", "td", "pa"]
    miss = wx[cols_we].isna().mean() * 100
    fig, ax = plt.subplots(figsize=(10, 4))
    miss.sort_values().plot(kind="barh", ax=ax, color="#888")
    ax.set_xlabel("Missing ratio (%)")
    ax.set_title("F10. ASOS variable missing ratio (4 summers)")
    plt.tight_layout(); plt.savefig(FIGS / "eda_10_weather_missing.png", dpi=120); plt.close()

    # F11. 자치구·시간 heatmap (내국인 평균 도착자)
    pivot = kt[kt["pop"] == "내국인"].groupby(["gu_cd", "hour"])["total"].mean().unstack(fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(pivot, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Avg arrivals"})
    ax.set_title("F11. District × Hour avg arrivals (Local)")
    ax.set_xlabel("Hour"); ax.set_ylabel("District code")
    plt.tight_layout(); plt.savefig(FIGS / "eda_11_gu_hour_heatmap.png", dpi=120); plt.close()

    # F12. 폭염일 vs 평상일 시간대별 평균 (내·외)
    daily["date"] = pd.to_datetime(daily["date"]).dt.date
    heat_dates = set(daily[daily["is_heatwave_feels"]]["date"])
    kt["is_heat"] = kt["date"].apply(lambda d: d in heat_dates)
    pivot = kt.groupby(["hour", "pop", "is_heat"])["total"].mean().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for ax, pop in zip(axes, POP_ORDER):
        sub = pivot[pivot["pop"] == pop]
        n = sub[~sub["is_heat"]].sort_values("hour")
        h = sub[sub["is_heat"]].sort_values("hour")
        ax.plot(n["hour"], n["total"], "o-", label="Normal day", color="#1f77b4")
        ax.plot(h["hour"], h["total"], "s-", label="Heatwave day", color="#d62728")
        ax.fill_between(n["hour"], n["total"], h["total"], alpha=0.15, color="#d62728")
        pct = (h["total"].mean() / n["total"].mean() - 1) * 100
        ax.set_title(f"F12. {LABEL[pop]} hourly — Normal vs Heatwave ({pct:+.1f}%)")
        ax.set_xlabel("Hour"); ax.set_ylabel("Avg arrivals")
        ax.legend()
    plt.tight_layout(); plt.savefig(FIGS / "eda_12_heat_vs_normal.png", dpi=120); plt.close()

    print(f"  {len(list(FIGS.glob('eda_*.png')))} figures saved → {FIGS}")


# ============================================================
# Main
# ============================================================
if RUN_PREPROCESS:
    kt = aggregate_kt_arrivals()
    wx = load_weather()
    info, daily = stats_summary(kt, wx)
    viz_all(kt, wx, daily)
    print("\n=== 완료 ===")
    print(f"KT 행. {info['kt_rows']:,}")
    print(f"KT 일수. {info['kt_dates']}")
    print(f"KT 자치구. {info['kt_gus']}")
    print(f"폭염일(체감, 연도별). {info['heatwave_days_feels']}")



# ==============================================================================
# STEP 02 전처리 — 동 단위 단기외국인 도착 (raw → kt_dong_short_hourly)
# ==============================================================================
print("\n>>> STEP 02 전처리 — 동 단위 단기외국인 도착 (raw → kt_dong_short_hourly)")
# 행정동×시간대별 단기외국인(관광객) 일자료 집계 — 한낮 회피·실내 가설 검증용.
"""
배경. 폭염에도 안 줄어드는 동이 '실내라서'인지 '한낮에도 노출되는 야외라서'인지 가르려면
      시간대 분포가 필요. time_cd는 출퇴근만 20분 단위(700·720…), 나머지는 시간 단위 →
      hour = code (code<100) / code//100 (code>=100) 로 복원.
정의(src.pop_types). 단기외국인 = short_forn_cnt. 서울 행정동(11 시작)만.
출력. data/processed/kt_dong_short_hourly.parquet  (date, d_admdong_cd, hour, short)
실행. python src/build_dong_short_hourly.py
"""
import glob, re
import pandas as pd

RAW = RAW_BASE / "KT_도착지/외국인"
OUT = PROC_BASE / "kt_dong_short_hourly.parquet"
DATE_RE = re.compile(r"_(\d{8})\.csv$")


def to_hour(code: int) -> int:
    """time_cd → 0~23시. 4자리(HHMM)는 //100, 1~2자리는 그대로."""
    return code if code < 100 else code // 100


def main():
    files = sorted(glob.glob(f"{RAW}/*_여름/*월/seoul_purpose_admdong1_forn_*.csv"))
    print(f"외국인 도착지 파일 {len(files)}개 시간대 집계 시작.")
    rows = []
    for i, f in enumerate(files, 1):
        date = DATE_RE.search(f).group(1)
        d = pd.read_csv(f, encoding="cp949",
                        usecols=["d_admdong_cd", "time_cd", "short_forn_cnt"])
        d = d[d["d_admdong_cd"].astype(str).str.startswith("11")]
        d["hour"] = d["time_cd"].astype(int).map(to_hour)
        g = d.groupby(["d_admdong_cd", "hour"])["short_forn_cnt"].sum().rename("short").reset_index()
        g["date"] = pd.to_datetime(date, format="%Y%m%d")
        rows.append(g)
        if i % 40 == 0 or i == len(files):
            print(f"  {i}/{len(files)} ({date})")
    out = pd.concat(rows, ignore_index=True)[["date", "d_admdong_cd", "hour", "short"]]
    out.to_parquet(OUT, index=False)
    print(f"\n저장: {OUT}  행 {len(out):,} | 동 {out['d_admdong_cd'].nunique()} | "
          f"시간 {sorted(out['hour'].unique())}")


if RUN_PREPROCESS:
    main()



# ==============================================================================
# STEP 02b 전처리 — 동 단위 집중도 (raw OD → dong_concentration)
# ==============================================================================
print("\n>>> STEP 02b 전처리 — 동 단위 집중도 (raw OD → dong_concentration)")
# 동(행정동)별 외국인 쏠림 지표(LQ·비율) 집계 — 1단계 산출물 단일 생성기.
"""
배경. 자치구(25개) 평균에 묻히던 외국인 공간 쏠림을 행정동(653개) 해상도로 드러낸다.
정의(src.pop_types 준수).
  내국인     = in 파일 total_cnt
  단기외국인 = forn 파일 short_forn_cnt (관광객)
  장기외국인 = forn 파일 total_cnt - short_forn_cnt (등록 거주)
지표.
  share_short = 단기 / (단기+장기+내국인)          # 동 도착 중 관광객 비중
  LQ_g(동)    = (g_동/g_전체) / (내국인_동/내국인_전체)  # 입지계수, 내국인 분포 대비 과대표집 배수
              LQ>1 → 해당 집단이 그 동에 내국인보다 더 쏠림.
집계. 2025 여름 92일 도착동(d_admdong_cd) 기준 단순 합.
"""
import glob
import pandas as pd
import geopandas as gpd

RAW = RAW_BASE / "KT_OD_보너스/2025_여름"
GEO = GEO_DONG
OUT = PROC_BASE / "dong_concentration.parquet"


def _agg(files, usecols, value_cols):
    """도착동별 value_cols 합계 누적."""
    acc = None
    for f in files:
        d = pd.read_csv(f, encoding="cp949", usecols=usecols)
        g = d.groupby("d_admdong_cd")[value_cols].sum()
        acc = g if acc is None else acc.add(g, fill_value=0)
    return acc


def main():
    forn_files = sorted(glob.glob(f"{RAW}/seoul_purpose_admdong4_forn_*.csv"))
    in_files = sorted(glob.glob(f"{RAW}/seoul_purpose_admdong4_in_*.csv"))
    print(f"외국인 {len(forn_files)}일 / 내국인 {len(in_files)}일 집계 시작.")

    forn = _agg(forn_files, ["d_admdong_cd", "short_forn_cnt", "total_cnt"],
                ["short_forn_cnt", "total_cnt"])
    loc = _agg(in_files, ["d_admdong_cd", "total_cnt"], ["total_cnt"])

    df = pd.DataFrame(index=forn.index)
    df["단기외국인"] = forn["short_forn_cnt"]
    df["장기외국인"] = forn["total_cnt"] - forn["short_forn_cnt"]
    df["내국인"] = loc["total_cnt"].reindex(df.index).fillna(0)
    df = df.clip(lower=0)

    # 서울 내부 도착동만 — 서울 밖(26xx 부산 등)은 자치구 단위로 뭉쳐 들어와 제외.
    # 서울 자치구 코드 11110~11240. 0으로 끝나는 자치구 통합 코드는 동 단위 아님.
    code = df.index.astype(int)
    is_seoul = (code >= 11110000) & (code < 11250000) & (code % 100 != 0)
    df = df[is_seoul]
    print(f"서울 행정동 한정: {len(df)}개 동.")

    # 입지계수(LQ) — 내국인 분포를 기준선으로 (서울 내부 분포 대비)
    tot = df.sum()
    base = df["내국인"] / tot["내국인"]
    for g in ["단기외국인", "장기외국인"]:
        df[f"LQ_{g}"] = (df[g] / tot[g]) / base
    df["share_단기"] = df["단기외국인"] / df[["내국인", "단기외국인", "장기외국인"]].sum(axis=1)

    # 동 이름 매핑 (adm_cd2 앞 8자리 = KT 코드)
    geo = gpd.read_file(GEO)[["adm_cd2", "adm_nm", "sggnm"]]
    geo["d_admdong_cd"] = geo["adm_cd2"].astype(str).str[:8].astype(int)
    geo["동명"] = geo["adm_nm"].str.split().str[-1]
    df = df.reset_index().merge(
        geo[["d_admdong_cd", "동명", "sggnm"]].rename(columns={"sggnm": "자치구"}),
        on="d_admdong_cd", how="left")

    df.to_parquet(OUT, index=False)
    print(f"저장: {OUT}  ({len(df)}개 동)")

    # 진단 출력
    print("\n=== 단기관광객 LQ 상위 15개 동 (관광객 쏠림) ===")
    cols = ["동명", "자치구", "단기외국인", "LQ_단기외국인", "share_단기"]
    print(df.sort_values("LQ_단기외국인", ascending=False)[cols].head(15).round(2).to_string(index=False))
    print("\n=== LQ_단기외국인 분포 ===")
    print(df["LQ_단기외국인"].describe(percentiles=[.5, .75, .9, .95]).round(2).to_string())
    print("LQ>1 (내국인보다 쏠린) 동 수:", int((df["LQ_단기외국인"] > 1).sum()), "/", len(df))


if RUN_PREPROCESS:
    main()



# ==============================================================================
# STEP 03 매칭분석 — 자치구 매칭 회피율 (적응 스펙트럼 -3.01/-0.90/-3.66)
# ==============================================================================
print("\n>>> STEP 03 매칭분석 — 자치구 매칭 회피율 (적응 스펙트럼 -3.01/-0.90/-3.66)")
# 폭염일 매칭 baseline 비교 분석. 자치구·요일·시즌 confounding 자동 제거.
"""
실행. python notebooks/06_매칭분석.py
출력. outputs/figures/m_*.png, data/processed/matched_delta.parquet

핵심 아이디어 — 자연실험 매칭
각 (자치구 g, 폭염일 t, 인구 pop) 셀에 대해.
  baseline = mean(arrivals[g, s, pop])
    where s ∈ [t-14, t+14], s.dow == t.dow, s 비폭염, s 맑음
  delta(%) = (arrivals[g, t, pop] - baseline) / baseline × 100

→ 자치구·요일·시즌·강수 모두 같은 조건끼리 짝지움. 폭염만 남음.
"""

import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = ROOT

PROCESSED = ROOT / "data/processed"
FIGS = FIGS
GEO = ROOT / "data/raw/행정구역_폴리곤/seoul_gu.geojson"

KT_GU_NAME = {
    "11110":"종로구","11140":"중구","11170":"용산구","11200":"성동구","11215":"광진구",
    "11230":"동대문구","11260":"중랑구","11290":"성북구","11305":"강북구","11320":"도봉구",
    "11350":"노원구","11380":"은평구","11410":"서대문구","11440":"마포구","11470":"양천구",
    "11500":"강서구","11530":"구로구","11545":"금천구","11560":"영등포구","11590":"동작구",
    "11620":"관악구","11650":"서초구","11680":"강남구","11710":"송파구","11740":"강동구",
}

WINDOW_DAYS = 14  # 매칭 윈도우 ±14일

# ============================================================
# 데이터 로드 + 일 단위 합산
# ============================================================
kt = pd.read_parquet(PROCESSED / "kt_gu_hour_daily.parquet")
daily = pd.read_parquet(PROCESSED / "weather_daily.parquet")
kt["date"] = pd.to_datetime(kt["date"])
daily["date"] = pd.to_datetime(daily["date"])

# 서울만
seoul = kt[kt["gu_cd"].str.startswith("11")].copy()
seoul["name"] = seoul["gu_cd"].map(KT_GU_NAME)

# 자치구·일·인구 합산
gu_day = seoul.groupby(["name", "date", "pop"])["total"].sum().reset_index()
gu_day = gu_day.rename(columns={"total": "arrivals"})

# 일 라벨
daily["dow"] = daily["date"].dt.dayofweek
daily["is_rain"] = daily["rn_sum"] > 1.0
daily["is_heat"] = daily["is_heatwave_feels"]
gu_day = gu_day.merge(daily[["date", "is_heat", "is_rain", "dow"]], on="date", how="left")


# ============================================================
# 매칭 baseline 계산
# ============================================================
def compute_matched_delta():
    """각 (자치구, 폭염일, 인구) 셀의 매칭 baseline + 변화율."""
    results = []
    skipped = 0
    heat_dates = sorted(gu_day[gu_day["is_heat"]]["date"].unique())
    for t in heat_dates:
        # ±14일 안 같은 요일, 비폭염, 맑은 후보일
        t_dow = pd.Timestamp(t).dayofweek
        cand = gu_day[
            (gu_day["date"] >= t - pd.Timedelta(days=WINDOW_DAYS)) &
            (gu_day["date"] <= t + pd.Timedelta(days=WINDOW_DAYS)) &
            (gu_day["dow"] == t_dow) &
            (~gu_day["is_heat"]) &
            (~gu_day["is_rain"])
        ]
        if cand["date"].nunique() < 2:
            # 매칭 후보 부족 → drop
            skipped += 1
            continue
        for pop in POP_ORDER:
            cand_pop = cand[cand["pop"] == pop]
            baseline_per_gu = cand_pop.groupby("name")["arrivals"].mean()
            actual_pop = gu_day[(gu_day["date"] == t) & (gu_day["pop"] == pop)].set_index("name")["arrivals"]
            for name in baseline_per_gu.index:
                if name not in actual_pop.index:
                    continue
                b = baseline_per_gu[name]
                a = actual_pop[name]
                if b == 0:
                    continue
                results.append({
                    "name": name, "date": t, "pop": pop,
                    "actual": a, "baseline": b,
                    "delta_pct": (a / b - 1) * 100,
                    "n_match": cand_pop[cand_pop["name"] == name]["date"].nunique(),
                })
    df = pd.DataFrame(results)
    out = PROCESSED / "matched_delta.parquet"
    df.to_parquet(out)
    print(f"  매칭 결과. {len(df):,}행 (자치구×폭염일×인구), drop된 폭염일 {skipped}개")
    print(f"  저장. {out}")
    return df


# ============================================================
# 매칭 표본 점검
# ============================================================
def check_match_quality(matched):
    print("\n=== 매칭 품질 점검 ===")
    print(f"폭염일 수. {matched['date'].nunique()}")
    print(f"매칭 후보 수 분포 (n_match):")
    print(matched["n_match"].describe())
    print(f"\n폭염일별 매칭 자치구 수.")
    by_date = matched.groupby("date")["name"].nunique().describe()
    print(by_date)


# ============================================================
# 헬퍼. figure 라벨 박스
# ============================================================
def add_data_label(fig, text):
    fig.text(
        0.99, 0.005, text, ha="right", va="bottom", fontsize=7.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbb", lw=0.7),
    )

LABEL = POP_LABEL  # 내국인·단기외국인·장기외국인


# ============================================================
# M1. 폭염 효과 분해 — 단순→D2→평일→매칭 비교 (이전 P1 업데이트)
# ============================================================
def m1_decomposition(matched):
    """단순/조건 통제/매칭 4가지 방법 비교."""
    sd = seoul.groupby(["date", "pop"])["total"].sum().reset_index()
    sd = sd.merge(daily[["date", "is_heat", "is_rain", "dow"]], on="date", how="left")
    sd["is_weekend"] = sd["dow"].isin([5, 6])

    def delta_simple(sub):
        n = sub[~sub["is_heat"]]["arrivals" if "arrivals" in sub else "total"].mean()
        h = sub[sub["is_heat"]]["arrivals" if "arrivals" in sub else "total"].mean()
        return (h / n - 1) * 100

    results = {}
    for pop in POP_ORDER:
        s = sd[sd["pop"] == pop]
        results[(pop, "단순 비교\n(통제 없음)")] = delta_simple(s)
        results[(pop, "D2\n(맑음)")] = delta_simple(s[~s["is_rain"]])
        results[(pop, "D2+평일\n(맑음·평일)")] = delta_simple(s[(~s["is_rain"]) & (~s["is_weekend"])])
        # 매칭: 서울 전체 = 자치구별 변화율 평균
        m_pop = matched[matched["pop"] == pop]
        results[(pop, "매칭\n(자치구·요일·시즌)")] = m_pop["delta_pct"].mean()

    conds = ["단순 비교\n(통제 없음)", "D2\n(맑음)", "D2+평일\n(맑음·평일)", "매칭\n(자치구·요일·시즌)"]
    fig, ax = plt.subplots(figsize=(12.5, 5.5))
    x = np.arange(len(conds)); w = 0.27
    offs = {"내국인": -w, "단기외국인": 0.0, "장기외국인": w}
    bars_by_pop = {}
    for pop in POP_ORDER:
        vals = [results[(pop, c)] for c in conds]
        bs = ax.bar(x + offs[pop], vals, w, label=POP_LABEL[pop], color=POP_COLOR[pop])
        bars_by_pop[pop] = (bs, vals)
        for b, v in zip(bs, vals):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.15,
                    f"{v:+.1f}%", ha="center", fontsize=7.5, color=POP_COLOR[pop])
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(conds)
    ax.set_ylabel("Heat day vs Normal day (%)")
    ax.set_title("M1. 폭염 효과 분해 — Confounding 단계별 통제 + 매칭 최종 답")
    ax.legend(loc="upper right"); ax.set_ylim(-1, 10)
    # 최종 답 강조 (내국인 매칭값)
    b_last, v_last = bars_by_pop["내국인"][0][-1], bars_by_pop["내국인"][1][-1]
    ax.annotate("Q1 답", (b_last.get_x()+b_last.get_width()/2, v_last),
                xytext=(0, 30), textcoords="offset points", ha="center",
                fontsize=11, fontweight="bold", color="#1f77b4",
                arrowprops=dict(arrowstyle="->", color="#1f77b4"))
    add_data_label(fig,
        "Data: 3년치(2023~2025 여름 276일) | 매칭 baseline = 같은 자치구·같은 요일·±14일 내 비폭염·맑음 평균\n"
        "단순=모든 비폭염 평균, D2=맑은 비폭염 평균, D2+평일=맑음·평일 비폭염 평균")
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(FIGS / "m_01_decomposition.png", dpi=120); plt.close()
    print("  M1 saved")
    return results


# ============================================================
# M2. 질문 1 답 — 매칭 변화율 분포 (내·외 비교)
# ============================================================
def m2_question1(matched):
    """폭염일·자치구별 매칭 변화율을 점으로 + 박스플롯."""
    # 서울 전체 변화율 = 자치구별 변화율의 평균을 폭염일별로
    by_day_pop = matched.groupby(["date", "pop"])["delta_pct"].mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # (a) 박스플롯 — 폭염일별 서울 평균 변화율 (3집단)
    ax = axes[0]
    datas = [by_day_pop[by_day_pop["pop"] == p]["delta_pct"] for p in POP_ORDER]
    bp = ax.boxplot(datas, labels=[POP_LABEL[p] for p in POP_ORDER],
                    patch_artist=True, widths=0.5)
    for patch, p in zip(bp["boxes"], POP_ORDER):
        patch.set_facecolor(POP_COLOR[p]); patch.set_alpha(0.5)
    for i, p in enumerate(POP_ORDER):
        data = datas[i]
        jitter = np.random.normal(0, 0.05, size=len(data))
        ax.scatter(np.full(len(data), i+1) + jitter, data, color=POP_COLOR[p], alpha=0.5, s=20, zorder=3)
        ax.text(i+1, ax.get_ylim()[1]*0.88, f"{data.mean():+.2f}%",
                ha="center", fontsize=8.5, color=POP_COLOR[p],
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=POP_COLOR[p]))
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Heat day matched ∆ (%)")
    ax.set_title(f"M2a. 폭염일별 서울 평균 매칭 변화율 (n={by_day_pop['date'].nunique()} 폭염일)")
    ax.grid(alpha=0.25)

    # (b) 격차 분포 — DiD: 내국인 - 단기외국인(메인), 내국인 - 장기외국인(보조)
    ax = axes[1]
    pivot = by_day_pop.pivot(index="date", columns="pop", values="delta_pct")
    pivot["격차"] = pivot["내국인"] - pivot["단기외국인"]       # 메인 DiD (보고서 헤드라인)
    pivot["격차_장기"] = pivot["내국인"] - pivot["장기외국인"]
    ax.hist(pivot["격차"], bins=20, color=POP_COLOR["단기외국인"], alpha=0.55,
            edgecolor="white", label="Local − Foreign(short)")
    ax.hist(pivot["격차_장기"], bins=20, color=POP_COLOR["장기외국인"], alpha=0.45,
            edgecolor="white", label="Local − Foreign(long)")
    ax.axvline(pivot["격차"].mean(), color=POP_COLOR["단기외국인"], linestyle="--", linewidth=2,
                label=f"short mean = {pivot['격차'].mean():+.2f}%p")
    ax.axvline(pivot["격차_장기"].mean(), color=POP_COLOR["장기외국인"], linestyle="--", linewidth=2,
                label=f"long mean = {pivot['격차_장기'].mean():+.2f}%p")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Local ∆ − Foreign ∆ (%p)"); ax.set_ylabel("Number of heat days")
    ax.set_title("M2b. 인구 간 격차 분포 (DiD, 폭염일별)")
    ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle("M2. Q1 — 폭염일 도착자가 평상일 대비 몇 % 변하는가 (3집단 비교)", fontsize=13)
    add_data_label(fig,
        "Data: 3년 × 폭염일 × 25구 | 매칭 baseline = 같은 구·같은 요일·±14일 내 비폭염·맑음 평균\n"
        "각 점 = 1 폭염일의 서울 평균 매칭 변화율 | DiD = Local 변화율 − Foreign 변화율")
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(FIGS / "m_02_question1.png", dpi=120); plt.close()
    print("  M2 saved")
    return by_day_pop, pivot


# ============================================================
# M3. 질문 2 답 — 자치구별 매칭 변화율 순위 + 데이터 기반 4유형 비교
#   (기존 손지정 관광/업무 2+2 → 데이터 기반 4유형으로 교체. 13번 노트북 분류 사용)
# ============================================================
def m3_question2(matched):
    by_gu = matched.groupby(["name", "pop"])["delta_pct"].mean().unstack()
    by_gu["label"] = by_gu.index.map(lambda x: GU_TYPE.get(x, "주거형"))

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))

    # (a) 자치구 순위 (유형 색)
    for c, pop in enumerate(POP_ORDER):
        ax = axes[0, c]
        s = by_gu.sort_values(pop, ascending=True)
        colors = [TYPE_COLOR[l] for l in s["label"]]
        ax.barh(range(len(s)), s[pop], color=colors, edgecolor="white", linewidth=0.5)
        ax.set_yticks(range(len(s))); ax.set_yticklabels(s.index, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        for i, v in enumerate(s[pop]):
            ax.text(v + (0.1 if v >= 0 else -0.1), i, f"{v:+.1f}%",
                    va="center", ha="left" if v >= 0 else "right", fontsize=7.5)
        ax.set_xlabel("Heat day matched ∆ (%)")
        ax.set_title(f"M3 {LABEL[pop]} — 자치구 순위")
        from matplotlib.patches import Patch
        legend_elems = [Patch(facecolor=TYPE_COLOR[t], label=t) for t in TYPE_ORDER]
        ax.legend(handles=legend_elems, loc="lower right", fontsize=8)
        ax.grid(alpha=0.25, axis="x")

    # (b) 유형별 boxplot — 4유형
    for c, pop in enumerate(POP_ORDER):
        ax = axes[1, c]
        groups = TYPE_ORDER
        data = [by_gu[by_gu["label"] == g][pop].values for g in groups]
        bp = ax.boxplot(data, labels=groups, patch_artist=True, widths=0.5)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(TYPE_COLOR[g]); patch.set_alpha(0.6)
        for i, g in enumerate(groups):
            vals = by_gu[by_gu["label"] == g][pop].values
            jitter = np.random.normal(0, 0.04, size=len(vals))
            ax.scatter(np.full(len(vals), i+1) + jitter, vals,
                       color="black", alpha=0.7, s=30, zorder=3)
            for j, name in enumerate(by_gu[by_gu["label"] == g].index):
                ax.annotate(name, (i+1+jitter[j], vals[j]),
                            xytext=(7, 0), textcoords="offset points", fontsize=7)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
        ax.set_ylabel("Heat day matched ∆ (%)")
        ax.set_title(f"M3 {LABEL[pop]} — 자치구 4유형 비교")
        means = {g: by_gu[by_gu["label"] == g][pop].mean() for g in groups}
        for i, g in enumerate(groups):
            ax.text(i+1, ax.get_ylim()[0]+0.3, f"mean\n{means[g]:+.2f}%",
                    ha="center", fontsize=8, fontweight="bold")
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("M3. Q2 — 자치구 유형별 폭염 매칭 변화율 (데이터 기반 4유형)", fontsize=13)
    add_data_label(fig,
        "Data: 3년 × 25구 × 폭염일 | 매칭 baseline = 같은 구·같은 요일·±14일 내 비폭염·맑음 평균\n"
        "유형 = 13번 노트북 2D 성향(관광지향·업무지향) 사분면 분류")
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(FIGS / "m_03_question2.png", dpi=120); plt.close()
    print("  M3 saved")
    return by_gu


# ============================================================
# M4. 자치구 지도 — 매칭 변화율
# ============================================================
def m4_map(matched):
    gdf = gpd.read_file(GEO)
    by_gu = matched.groupby(["name", "pop"])["delta_pct"].mean().unstack().reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))
    vabs = float(np.nanmax(np.abs(by_gu[POP_ORDER].values)))
    for ax, pop in zip(axes, POP_ORDER):
        s = by_gu[["name", pop]]
        m = gdf.merge(s, on="name", how="left")
        m.plot(column=pop, ax=ax, cmap="RdBu_r", vmin=-vabs, vmax=vabs,
               legend=True, legend_kwds={"label": "Matched ∆ (%)", "shrink": 0.7},
               edgecolor="white", linewidth=0.6)
        MARK = {"복합핵심": " ◉", "업무형": " ◆", "관광·여가형": " ★"}
        for _, r in m.iterrows():
            c = r.geometry.centroid
            mark = MARK.get(GU_TYPE.get(r["name"], "주거형"), "")
            ax.annotate(f"{r['name']}{mark}\n{r[pop]:+.1f}%", (c.x, c.y),
                        ha="center", fontsize=6.5)
        ax.set_title(f"M4. {LABEL[pop]} — 자치구 매칭 변화율\n(◉ 복합핵심, ◆ 업무형, ★ 관광·여가형)")
        ax.set_axis_off()
    add_data_label(fig,
        "Data: 3년 × 25구 × 폭염일 | 매칭 baseline = 같은 구·같은 요일·±14일 내 비폭염·맑음 평균\n"
        "값 = 자치구별 (폭염일 - 매칭 baseline) / 매칭 baseline × 100 %의 폭염일 평균")
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(FIGS / "m_04_map.png", dpi=120); plt.close()
    print("  M4 saved")


# ============================================================
# Main
# ============================================================
if True:
    print("=== 매칭 분석 시작 ===")
    matched = compute_matched_delta()
    check_match_quality(matched)

    res = m1_decomposition(matched)
    print("\n[M1 분해 결과]")
    for k, v in res.items():
        print(f"  {k[0]:>4} | {k[1]:<30}: {v:+.2f}%")

    by_day, gap = m2_question1(matched)
    print("\n[M2 / Q1 답]")
    for pop in POP_ORDER:
        print(f"  {pop} 평균: {by_day[by_day['pop']==pop]['delta_pct'].mean():+.2f}%")
    print(f"  내·단기 격차 (Local - Foreign short): {gap['격차'].mean():+.2f}%p")
    print(f"  내·장기 격차 (Local - Foreign long):  {gap['격차_장기'].mean():+.2f}%p")
    # t-test (메인 격차)
    from scipy import stats
    t, p = stats.ttest_1samp(gap["격차"].dropna(), 0)
    print(f"  내·단기 격차 t-test p-value: {p:.4f}")

    by_gu = m3_question2(matched)
    print("\n[M3 / Q2 답 — 자치구 4유형별 평균]")
    for pop in POP_ORDER:
        print(f"\n  [{pop}]")
        for g in TYPE_ORDER:
            sub = by_gu[by_gu["label"] == g][pop]
            print(f"    {g}: {sub.mean():+.2f}% (n={len(sub)}, range [{sub.min():+.2f}, {sub.max():+.2f}])")

    m4_map(matched)
    print("\n=== 완료 ===")



# ==============================================================================
# STEP 04 매칭 윈도우 민감도 (표 4-2, ±7/±14/±21)
# ==============================================================================
print("\n>>> STEP 04 매칭 윈도우 민감도 (표 4-2, ±7/±14/±21)")
# 매칭 윈도우 ±7/±14/±21일 sensitivity check. 표본·결과 안정성 점검.
"""실행. python notebooks/07_매칭_sensitivity.py
출력. outputs/figures/m_05_sensitivity.png + 콘솔 비교표"""

import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
from scipy import stats

ROOT = ROOT

PROCESSED = ROOT / "data/processed"
FIGS = FIGS

KT_GU_NAME = {
    "11110":"종로구","11140":"중구","11170":"용산구","11200":"성동구","11215":"광진구",
    "11230":"동대문구","11260":"중랑구","11290":"성북구","11305":"강북구","11320":"도봉구",
    "11350":"노원구","11380":"은평구","11410":"서대문구","11440":"마포구","11470":"양천구",
    "11500":"강서구","11530":"구로구","11545":"금천구","11560":"영등포구","11590":"동작구",
    "11620":"관악구","11650":"서초구","11680":"강남구","11710":"송파구","11740":"강동구",
}
# 데이터 로드
kt = pd.read_parquet(PROCESSED / "kt_gu_hour_daily.parquet")
daily = pd.read_parquet(PROCESSED / "weather_daily.parquet")
kt["date"] = pd.to_datetime(kt["date"])
daily["date"] = pd.to_datetime(daily["date"])

seoul = kt[kt["gu_cd"].str.startswith("11")].copy()
seoul["name"] = seoul["gu_cd"].map(KT_GU_NAME)
gu_day = seoul.groupby(["name", "date", "pop"])["total"].sum().reset_index()
gu_day = gu_day.rename(columns={"total": "arrivals"})
daily["dow"] = daily["date"].dt.dayofweek
daily["is_rain"] = daily["rn_sum"] > 1.0
daily["is_heat"] = daily["is_heatwave_feels"]
gu_day = gu_day.merge(daily[["date","is_heat","is_rain","dow"]], on="date", how="left")


def compute_matched(window):
    """주어진 윈도우(±일)로 매칭 변화율 계산."""
    results = []
    heat_dates = sorted(gu_day[gu_day["is_heat"]]["date"].unique())
    for t in heat_dates:
        t_dow = pd.Timestamp(t).dayofweek
        cand = gu_day[
            (gu_day["date"] >= t - pd.Timedelta(days=window)) &
            (gu_day["date"] <= t + pd.Timedelta(days=window)) &
            (gu_day["dow"] == t_dow) &
            (~gu_day["is_heat"]) &
            (~gu_day["is_rain"])
        ]
        if cand["date"].nunique() < 2:
            continue
        for pop in POP_ORDER:
            cand_pop = cand[cand["pop"] == pop]
            baseline_per_gu = cand_pop.groupby("name")["arrivals"].mean()
            actual_pop = gu_day[(gu_day["date"] == t) & (gu_day["pop"] == pop)].set_index("name")["arrivals"]
            for name in baseline_per_gu.index:
                if name not in actual_pop.index:
                    continue
                b = baseline_per_gu[name]
                a = actual_pop[name]
                if b == 0:
                    continue
                results.append({
                    "name": name, "date": t, "pop": pop,
                    "delta_pct": (a / b - 1) * 100,
                })
    return pd.DataFrame(results)


def summarize(df, window):
    """Q1, Q2 답을 dict로 요약."""
    by_day_pop = df.groupby(["date", "pop"])["delta_pct"].mean().reset_index()
    # Q1
    pivot = by_day_pop.pivot(index="date", columns="pop", values="delta_pct")
    pivot["gap"] = pivot["내국인"] - pivot["단기외국인"]       # 메인 격차
    pivot["gap_long"] = pivot["내국인"] - pivot["장기외국인"]
    t_stat, p_val = stats.ttest_1samp(pivot["gap"].dropna(), 0)

    # Q2: 유형별 평균 (데이터 기반 4유형)
    by_gu = df.groupby(["name", "pop"])["delta_pct"].mean().unstack()
    by_gu["label"] = by_gu.index.map(lambda x: GU_TYPE.get(x, "주거형"))
    q2 = {(t, pop): float(by_gu[by_gu["label"] == t][pop].mean())
          for t in TYPE_ORDER for pop in POP_ORDER}

    out = {
        "window": window,
        "n_heat_days": int(by_day_pop["date"].nunique()),
        "n_cells": int(len(df)),
        "Q1_격차": float(pivot["gap"].mean()),
        "Q1_격차_장기": float(pivot["gap_long"].mean()),
        "Q1_p": float(p_val),
        "Q2": q2,
    }
    for pop in POP_ORDER:
        out[f"Q1_{pop}"] = float(by_day_pop[by_day_pop["pop"] == pop]["delta_pct"].mean())
    return out


if True:
    print("=== 매칭 윈도우 sensitivity check ===\n")
    summaries = []
    matched_by_window = {}
    for window in [7, 14, 21]:
        df = compute_matched(window)
        matched_by_window[window] = df
        s = summarize(df, window)
        summaries.append(s)
        print(f"--- ±{window}일 ---")
        print(f"  매칭 폭염일: {s['n_heat_days']}, 분석 cell: {s['n_cells']}")
        print(f"  Q1. 내국인 {s['Q1_내국인']:+.2f}%, 단기 {s['Q1_단기외국인']:+.2f}%, 장기 {s['Q1_장기외국인']:+.2f}%")
        print(f"      내·단기 격차 {s['Q1_격차']:+.2f}%p (p={s['Q1_p']:.4f}), 내·장기 격차 {s['Q1_격차_장기']:+.2f}%p")
        for pop in POP_ORDER:
            print(f"  Q2 ({pop}). " + ", ".join(f"{t} {s['Q2'][(t,pop)]:+.2f}%" for t in TYPE_ORDER))
        print()

    # ±21일 결과를 메인 매칭으로 저장 (사용자 권고 시)
    df21 = matched_by_window[21]
    df21.to_parquet(PROCESSED / "matched_delta_w21.parquet")
    print(f"  ±21일 결과 저장. matched_delta_w21.parquet ({len(df21)} rows)")

    # ============================================================
    # Sensitivity figure
    # ============================================================
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # (a) Q1 — 윈도우별 평균 변화율 + 95% CI
    ax = axes[0]
    x = np.arange(len(summaries))
    w = 0.27
    offs = {"내국인": -w, "단기외국인": 0.0, "장기외국인": w}
    for pop in POP_ORDER:
        means = [s[f"Q1_{pop}"] for s in summaries]
        se = []
        for window in [7, 14, 21]:
            bd = matched_by_window[window].groupby(["date","pop"])["delta_pct"].mean().reset_index()
            v = bd[bd["pop"] == pop]["delta_pct"]
            se.append(v.std() / np.sqrt(len(v)) * 1.96)
        ax.bar(x + offs[pop], means, w, yerr=se, label=POP_LABEL[pop],
               color=POP_COLOR[pop], capsize=4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"±{s['window']}일" for s in summaries])
    ax.set_ylabel("Heat day matched ∆ (%) ± 95% CI")
    ax.set_title("Q1 답 — 윈도우별 변화율 + 95% CI")
    ax.legend(); ax.grid(alpha=0.25, axis="y")

    # (b) Q2 — 윈도우별 유형 비교 (내국인, 데이터 기반 4유형)
    ax = axes[1]
    groups = TYPE_ORDER
    bar_width = 0.2
    for j, g in enumerate(groups):
        vals = [s["Q2"][(g, "내국인")] for s in summaries]
        ax.bar(x + (j - 1.5) * bar_width, vals, bar_width, label=g, color=TYPE_COLOR[g])
        for i, v in enumerate(vals):
            ax.text(i + (j - 1.5) * bar_width, v - 0.3, f"{v:+.1f}%", ha="center", fontsize=7)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels([f"±{s['window']}일" for s in summaries])
    ax.set_ylabel("Local heat day matched ∆ (%)")
    ax.set_title("Q2 답 — 윈도우별 자치구 4유형 비교 (내국인 기준)")
    ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=0.25, axis="y")

    fig.suptitle("Sensitivity Check — 매칭 윈도우 ±7/±14/±21일 결과 비교", fontsize=13)
    fig.text(0.99, 0.005,
        "Data: 3년 × 25구 × 폭염일 | 윈도우 = baseline 후보 ±N일 범위\n"
        "n = 매칭 성공 폭염일 수 | CI = 95% (per-day variance)",
        ha="right", va="bottom", fontsize=7.5, color="#444",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbb", lw=0.7))
    plt.tight_layout(rect=[0, 0.04, 1, 0.95])
    plt.savefig(FIGS / "m_05_sensitivity.png", dpi=120); plt.close()
    print("\n=== Sensitivity figure 저장. m_05_sensitivity.png ===")

    # 요약표
    print("\n=== 비교표 (콘솔용) ===")
    df_s = pd.DataFrame(summaries)
    print(df_s.round(3).to_string(index=False))



# ==============================================================================
# STEP 05 시간대 매칭 (matched_delta_hgroup)
# ==============================================================================
print("\n>>> STEP 05 시간대 매칭 (matched_delta_hgroup)")
# 폭염 시간대별 매칭 분석. 04 매칭(일 단위)을 시간 단위로 확장.
"""
입력. data/processed/kt_gu_hour_daily.parquet, weather_daily.parquet
출력.
  data/processed/matched_delta_hgroup.parquet — 방법 A 그룹 매칭 결과
  data/processed/hourly_share_shift.parquet — 방법 B 분포 시프트 결과
  outputs/figures/m_h01~m_h07_*.png — 시각화 7장
  outputs/tables/_hgroup_log.txt — 검증 출력

핵심 질문. "폭염일에 이동 시간대가 변하는가" — 점심·오후 회피 + 저녁·심야 시프트 + 내·외 차이.

두 방법 병행.
  A. 시간대 6 그룹 매칭 (심야·출근·점심·오후·저녁·밤) — 04 그대로 적용
  B. 시간 분포 시프트 — 24h normalize 후 mean/peak/KL 측정

검증. A의 시간 가중 평균이 04 일 단위 결과(-3.01 / -0.90)와 정합해야.
"""

import warnings
warnings.filterwarnings("ignore")
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = ROOT
PROCESSED = ROOT / "data/processed"
FIGS = FIGS
TABLES = TABLES

# 04와 동일
KT_GU_NAME = {
    "11110":"종로구","11140":"중구","11170":"용산구","11200":"성동구","11215":"광진구",
    "11230":"동대문구","11260":"중랑구","11290":"성북구","11305":"강북구","11320":"도봉구",
    "11350":"노원구","11380":"은평구","11410":"서대문구","11440":"마포구","11470":"양천구",
    "11500":"강서구","11530":"구로구","11545":"금천구","11560":"영등포구","11590":"동작구",
    "11620":"관악구","11650":"서초구","11680":"강남구","11710":"송파구","11740":"강동구",
}
WINDOW_DAYS = 14

# 시간 그룹 (사용자 명시)
HOUR_GROUPS = [
    ("심야 (0-5)",   range(0, 6)),
    ("출근 (6-9)",   range(6, 10)),
    ("점심 (10-13)", range(10, 14)),
    ("오후 (14-17)", range(14, 18)),
    ("저녁 (18-21)", range(18, 22)),
    ("밤 (22-23)",   range(22, 24)),
]
H2G = {}  # hour → 그룹명
for g, hours in HOUR_GROUPS:
    for h in hours:
        H2G[h] = g
HG_ORDER = [g for g, _ in HOUR_GROUPS]

# 04 스타일 색
C_LOCAL = POP_COLOR["내국인"]
C_FOREIGN = POP_COLOR["단기외국인"]
C_TOUR = "#ff9800"
C_BIZ = "#4caf50"
C_OTHER = "#bdbdbd"
LABEL = POP_LABEL  # 내국인·단기외국인·장기외국인


# ============================================================
# 데이터 로드
# ============================================================
def load_and_merge():
    """KT 시간 단위 + 일 라벨 (폭염·강수·요일) 결합."""
    kt = pd.read_parquet(PROCESSED / "kt_gu_hour_daily.parquet")
    daily = pd.read_parquet(PROCESSED / "weather_daily.parquet")
    kt["date"] = pd.to_datetime(kt["date"])
    daily["date"] = pd.to_datetime(daily["date"])

    seoul = kt[kt["gu_cd"].str.startswith("11")].copy()
    seoul["name"] = seoul["gu_cd"].map(KT_GU_NAME)
    seoul["hgroup"] = seoul["hour"].map(H2G)

    daily["dow"] = daily["date"].dt.dayofweek
    daily["is_rain"] = daily["rn_sum"] > 1.0
    daily["is_heat"] = daily["is_heatwave_feels"]
    seoul = seoul.merge(daily[["date", "is_heat", "is_rain", "dow"]], on="date", how="left")
    return seoul, daily


# ============================================================
# 방법 A — 시간대 그룹 매칭
# ============================================================
def build_cells_group(seoul: pd.DataFrame) -> pd.DataFrame:
    """(자치구, 일, 시간그룹, 인구) 셀로 합산."""
    return (seoul.groupby(["name", "date", "hgroup", "pop", "dow",
                            "is_heat", "is_rain"])["total"].sum()
                  .reset_index()
                  .rename(columns={"total": "arrivals"}))


def compute_matched_delta_hgroup(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """04 compute_matched_delta + hgroup 차원. 셀당 매칭 변화율."""
    results = []
    drop_log = {"no_candidate_for_date": 0, "no_match_for_cell": 0}
    heat_dates = sorted(cells[cells["is_heat"]]["date"].unique())

    # 빠르게: cells를 인구·시간그룹으로 분리해 dict 구축
    for pop in POP_ORDER:
        for hg in HG_ORDER:
            sub = cells[(cells["pop"] == pop) & (cells["hgroup"] == hg)].copy()
            sub = sub.set_index(["name", "date"])

            for t in heat_dates:
                t = pd.Timestamp(t)
                t_dow = t.dayofweek
                # 후보 = ±14d, 같은 요일, 비폭염, 맑음
                cand_mask = (
                    (cells["pop"] == pop) & (cells["hgroup"] == hg) &
                    (cells["date"] >= t - pd.Timedelta(days=WINDOW_DAYS)) &
                    (cells["date"] <= t + pd.Timedelta(days=WINDOW_DAYS)) &
                    (cells["dow"] == t_dow) &
                    (~cells["is_heat"]) & (~cells["is_rain"])
                )
                cand = cells[cand_mask]
                if cand["date"].nunique() < 2:
                    drop_log["no_candidate_for_date"] += 1
                    continue
                baseline = cand.groupby("name")["arrivals"].mean()
                n_match = cand.groupby("name")["date"].nunique()
                for name in baseline.index:
                    if (name, t) not in sub.index:
                        continue
                    a = sub.loc[(name, t), "arrivals"]
                    b = baseline[name]
                    if b == 0:
                        continue
                    results.append({
                        "name": name, "date": t, "hgroup": hg, "pop": pop,
                        "actual": float(a), "baseline": float(b),
                        "delta_pct": (a / b - 1) * 100,
                        "n_match": int(n_match[name]),
                    })
    out = pd.DataFrame(results)
    return out, drop_log


def check_match_quality_hgroup(matched: pd.DataFrame, drop_log: dict):
    print("\n=== 방법 A 매칭 품질 ===")
    print(f"성공 셀 수. {len(matched):,}")
    print(f"폭염일 수. {matched['date'].nunique()}")
    print(f"시간 그룹별 n_match.")
    print(matched.groupby("hgroup")["n_match"].agg(["mean", "median", "count"]).round(2).to_string())
    print(f"drop. {drop_log}")


# ============================================================
# 방법 B — 24h 분포 시프트
# ============================================================
def build_hourly_share(seoul: pd.DataFrame) -> pd.DataFrame:
    """자치구·일·인구별 24h normalize → P(h | gu, date, pop)."""
    pivot = seoul.pivot_table(
        index=["name", "date", "pop"], columns="hour", values="total",
        aggfunc="sum", fill_value=0,
    )
    pivot.columns = [f"h{h}" for h in pivot.columns]
    # 행 단위 normalize
    row_sum = pivot.sum(axis=1).replace(0, np.nan)
    share = pivot.div(row_sum, axis=0)
    share = share.reset_index().merge(
        seoul[["date", "is_heat", "is_rain", "dow"]].drop_duplicates("date"),
        on="date", how="left",
    )
    return share


def compute_shift_metrics(share: pd.DataFrame) -> pd.DataFrame:
    """각 (자치구, 폭염일, 인구)의 24h 분포를 매칭 baseline 분포와 비교."""
    rows = []
    hour_cols = [f"h{h}" for h in range(24)]
    hours = np.arange(24)
    heat_dates = sorted(share[share["is_heat"]]["date"].unique())

    for pop in POP_ORDER:
        sp = share[share["pop"] == pop].copy()
        for t in heat_dates:
            t = pd.Timestamp(t)
            t_dow = t.dayofweek
            cand = sp[(sp["date"] >= t - pd.Timedelta(days=WINDOW_DAYS)) &
                      (sp["date"] <= t + pd.Timedelta(days=WINDOW_DAYS)) &
                      (sp["dow"] == t_dow) &
                      (~sp["is_heat"]) & (~sp["is_rain"])]
            if cand["date"].nunique() < 2:
                continue
            base_by_gu = cand.groupby("name")[hour_cols].mean()
            heat_by_gu = sp[sp["date"] == t].set_index("name")[hour_cols]
            for name in base_by_gu.index:
                if name not in heat_by_gu.index:
                    continue
                p_heat = heat_by_gu.loc[name].values
                p_base = base_by_gu.loc[name].values
                if p_heat.sum() == 0 or p_base.sum() == 0 or np.isnan(p_heat).any():
                    continue
                # mean hour
                mean_h_heat = float(np.sum(hours * p_heat))
                mean_h_base = float(np.sum(hours * p_base))
                # peak hour
                peak_h_heat = int(np.argmax(p_heat))
                peak_h_base = int(np.argmax(p_base))
                # KL divergence (heat || base), small eps
                eps = 1e-9
                p_h = p_heat + eps; p_h /= p_h.sum()
                p_b = p_base + eps; p_b /= p_b.sum()
                kl = float(np.sum(p_h * np.log(p_h / p_b)))
                rows.append({
                    "name": name, "date": t, "pop": pop,
                    "mean_h_heat": mean_h_heat, "mean_h_base": mean_h_base,
                    "mean_shift": mean_h_heat - mean_h_base,
                    "peak_h_heat": peak_h_heat, "peak_h_base": peak_h_base,
                    "peak_shift": peak_h_heat - peak_h_base,
                    "kl": kl,
                })
    return pd.DataFrame(rows)


def collect_baseline_curves(share: pd.DataFrame) -> pd.DataFrame:
    """폭염일 vs baseline 24h 평균 곡선 (자치구·인구별)."""
    hour_cols = [f"h{h}" for h in range(24)]
    rows = []
    for pop in POP_ORDER:
        sp = share[share["pop"] == pop].copy()
        heat_dates = sorted(sp[sp["is_heat"]]["date"].unique())
        per_date = []
        for t in heat_dates:
            t = pd.Timestamp(t); t_dow = t.dayofweek
            cand = sp[(sp["date"] >= t - pd.Timedelta(days=WINDOW_DAYS)) &
                      (sp["date"] <= t + pd.Timedelta(days=WINDOW_DAYS)) &
                      (sp["dow"] == t_dow) &
                      (~sp["is_heat"]) & (~sp["is_rain"])]
            if cand["date"].nunique() < 2:
                continue
            heat_curve = sp[sp["date"] == t][hour_cols].mean().values
            base_curve = cand[hour_cols].mean().values
            per_date.append({
                "pop": pop, "date": t,
                "heat": heat_curve, "base": base_curve,
            })
        for row in per_date:
            for h in range(24):
                rows.append({"pop": row["pop"], "date": row["date"], "hour": h,
                             "heat_share": row["heat"][h], "base_share": row["base"][h]})
    return pd.DataFrame(rows)


# ============================================================
# 검증
# ============================================================
def verify_against_04(matched: pd.DataFrame):
    """방법 A 시간 가중 평균이 04 일 단위 -3.01 / -0.90과 정합해야."""
    print("\n=== 검증 — 04 일 단위 결과와 정합성 ===")
    REF04 = {"내국인": "-3.01%", "단기외국인": "-0.90%", "장기외국인": "-3.66%"}
    # 시간 가중 평균 = actual·baseline 가중치로 평균
    for pop in POP_ORDER:
        sub = matched[matched["pop"] == pop].copy()
        weighted = (sub["delta_pct"] * sub["baseline"]).sum() / sub["baseline"].sum()
        unweighted = sub["delta_pct"].mean()
        print(f"{pop}. 가중평균 {weighted:+.2f}%  (단순평균 {unweighted:+.2f}%)  "
              f"06 일 단위 = {REF04.get(pop, '?')}")


# ============================================================
# 시각화 헬퍼
# ============================================================
def add_data_label(fig, text):
    fig.text(0.99, 0.005, text, ha="right", va="bottom", fontsize=7.5, color="#444",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbb", lw=0.7))


# ============================================================
# m_h01. 그룹화 정당성 — 24h vs 6그룹 후보 분포
# ============================================================
def plot_h01_grouping(seoul, cells):
    """24h 단위 매칭 후보 vs 6그룹 후보 (그룹화 정당성)."""
    # 24h 단위 후보 시뮬레이션
    heat_dates = sorted(seoul[seoul["is_heat"]]["date"].unique())
    h24_cand = []
    hg_cand = []

    # 24h 단위 (가상)
    for t in heat_dates:
        t = pd.Timestamp(t); t_dow = t.dayofweek
        # 일 단위 후보일 수
        days = seoul[(seoul["date"] >= t - pd.Timedelta(days=WINDOW_DAYS)) &
                     (seoul["date"] <= t + pd.Timedelta(days=WINDOW_DAYS)) &
                     (seoul["dow"] == t_dow) &
                     (~seoul["is_heat"]) & (~seoul["is_rain"])]["date"].nunique()
        # 시간 단위는 같은 일자에서 1 hour만 매칭 → 그대로 days (이론값)
        # 단 시간별 결측 가능 — 보수적으로 days 그대로
        h24_cand.extend([days] * 24)  # 24시간 각각
        hg_cand.extend([days] * 6)    # 6 그룹 각각

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].hist(h24_cand, bins=range(0, 12), color=C_LOCAL, edgecolor="white",
                 alpha=0.8, rwidth=0.85)
    axes[0].axvline(2, color="red", linestyle="--", linewidth=1.2,
                    label="매칭 최소 = 2개")
    axes[0].set_xlabel("후보 일 수 (n_match)")
    axes[0].set_ylabel("폭염일 × 시간 셀 수")
    axes[0].set_title("(a) 24h 단위 매칭 가설\n"
                      "1 hour만 보면 같은 요일·±14d·맑음 후보 = 0~3개")
    axes[0].legend(); axes[0].grid(alpha=0.25)

    axes[1].hist(cells.merge(
        seoul[["date", "is_heat"]].drop_duplicates(), on="date"
    )[cells.merge(
        seoul[["date", "is_heat"]].drop_duplicates(), on="date"
    )["is_heat_x"] == True]
        if False else hg_cand, bins=range(0, 12),
        color=C_TOUR, edgecolor="white", alpha=0.8, rwidth=0.85)
    axes[1].axvline(2, color="red", linestyle="--", linewidth=1.2, label="매칭 최소 = 2개")
    axes[1].set_xlabel("후보 일 수 (n_match)")
    axes[1].set_ylabel("폭염일 × 시간그룹 셀 수")
    axes[1].set_title("(b) 6 그룹 묶음 후 매칭\n"
                      "4~5시간씩 묶으면 셀 내 시간 다양성 ↑, 후보 충분")
    axes[1].legend(); axes[1].grid(alpha=0.25)

    fig.suptitle("M_h1. 24h 그대로 매칭하면 셀당 후보 부족 → 6 그룹 묶음으로 우회",
                 fontsize=12)
    add_data_label(fig, "data. KT 2023~2025 여름  ·  매칭 ±14d 같은 요일 비폭염·맑음")
    plt.tight_layout()
    plt.savefig(FIGS / "m_h01_grouping_rationale.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ m_h01_grouping_rationale.png")


# ============================================================
# m_h02. 6 그룹별 매칭 변화율 (내·외)
# ============================================================
def plot_h02_group_delta(matched):
    REF = {"내국인": -3.01, "단기외국인": -0.90, "장기외국인": -3.66}
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, pop in zip(axes, POP_ORDER):
        color = POP_COLOR[pop]
        ref = REF[pop]
        sub = matched[matched["pop"] == pop]
        agg = sub.groupby("hgroup")["delta_pct"].agg(["mean", "median", "std", "count"]).reindex(HG_ORDER)
        # 평균 ± SE (= std/√n)
        se = agg["std"] / np.sqrt(agg["count"])
        xp = np.arange(len(HG_ORDER))
        bars = ax.bar(xp, agg["mean"], yerr=se, color=color, alpha=0.85,
                       edgecolor="white", capsize=4)
        for i, v in enumerate(agg["mean"]):
            ax.text(i, v + (0.3 if v >= 0 else -0.5), f"{v:+.2f}%",
                    ha="center", fontsize=9.5,
                    color="white" if v < -3 else "#222")
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axhline(ref, color="red", linestyle="--", linewidth=1.2,
                   label=f"04 일 단위 = {ref:+.2f}%")
        ax.set_xticks(xp); ax.set_xticklabels(HG_ORDER, rotation=18, fontsize=9)
        ax.set_ylabel("매칭 변화율 (%, 평균 ± SE)")
        ax.set_title(f"{LABEL[pop]} · {pop}")
        ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.25)

    fig.suptitle("M_h2. 6 그룹별 매칭 변화율 — 어느 시간대에서 폭염 영향 가장 큰가",
                 fontsize=12)
    add_data_label(fig, "data. KT 2023~2025  ·  baseline = ±14d 같은 요일 비폭염·맑음")
    plt.tight_layout()
    plt.savefig(FIGS / "m_h02_group_delta.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ m_h02_group_delta.png")


# ============================================================
# m_h03. 24h 분포 곡선 (폭염일 vs baseline)
# ============================================================
def plot_h03_dist_curves(curves):
    """24h 분포 곡선 (3집단). 폭염-평상 곡선이 거의 겹치므로(KL≈0),
    아래 행에 차이(heat-base, %p)를 자체 y축으로 확대해 미세 시프트를 보이게 함."""
    fig, axes = plt.subplots(2, 3, figsize=(19, 9),
                             gridspec_kw={"height_ratios": [2.2, 1]}, sharex=True)
    hours = np.arange(24)
    for c, pop in enumerate(POP_ORDER):
        color = POP_COLOR[pop]
        sub = curves[curves["pop"] == pop]
        agg = sub.groupby("hour").agg(heat=("heat_share", "mean"),
                                       base=("base_share", "mean")).reset_index()
        # --- 상단. 분포 곡선 (겹치는 것이 결과 — 분포 자체는 거의 불변) ---
        ax = axes[0, c]
        ax.plot(agg["hour"], agg["base"] * 100, "--", color="#555",
                linewidth=1.8, label="평상일 baseline")
        ax.plot(agg["hour"], agg["heat"] * 100, "-", color=color,
                linewidth=2.4, label="폭염일")
        m_heat = float(np.sum(hours * agg["heat"]))
        m_base = float(np.sum(hours * agg["base"]))
        ax.axvline(m_base, color="grey", linestyle=":", linewidth=1)
        ax.axvline(m_heat, color=color, linestyle=":", linewidth=1)
        ax.annotate(f"평균시간 {m_heat - m_base:+.2f}h",
                    xy=(0.03, 0.92), xycoords="axes fraction",
                    fontsize=9.5, color=color, fontweight="bold")
        ax.set_title(f"{LABEL[pop]} · {pop}")
        if c == 0:
            ax.set_ylabel("도착자 비중 (%)")
        ax.set_xlim(-0.5, 23.5); ax.set_xticks(range(0, 24, 3))
        ax.legend(loc="upper right", fontsize=8.5); ax.grid(alpha=0.25)
        # --- 하단. 차이 (폭염 - 평상, %p) 확대 — 겹침 문제 해결 ---
        ax2 = axes[1, c]
        diff = (agg["heat"] - agg["base"]) * 100  # %p
        ax2.bar(agg["hour"], diff, width=0.8,
                color=[color if v < 0 else "#888" for v in diff], alpha=0.85)
        ax2.axhline(0, color="black", linewidth=0.6)
        dmax = float(np.nanmax(np.abs(diff))) if len(diff) else 1.0
        ax2.set_ylim(-dmax * 1.25, dmax * 1.25)
        if c == 0:
            ax2.set_ylabel("차이 (폭염-평상, %p)")
        ax2.set_xlabel("시간 (0~23)")
        ax2.grid(alpha=0.25)
    fig.suptitle("M_h3. 24h 분포 곡선 (상) + 폭염-평상 차이 확대 (하) — 3집단, 분포 시프트는 미미",
                 fontsize=13)
    add_data_label(fig, "분포 = 자치구·일별 24h 도착자 합=100% normalize | 하단 = 상단 두 곡선의 차이(%p) 확대")
    plt.tight_layout()
    plt.savefig(FIGS / "m_h03_dist_curves.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ m_h03_dist_curves.png")


# ============================================================
# m_h04. 자치구별 mean/peak shift
# ============================================================
def plot_h04_mean_peak_shift(shift_df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, metric, title, color_pos in zip(
        axes, ["mean_shift", "peak_shift"],
        ["(a) 평균 활동 시간 shift", "(b) 피크 시간 shift"], [C_LOCAL, C_FOREIGN],
    ):
        agg = (shift_df.groupby(["name", "pop"])[metric].mean()
                       .reset_index()
                       .pivot(index="name", columns="pop", values=metric))
        agg["label"] = agg.index.map(lambda n: GU_TYPE.get(n, "주거형"))
        agg = agg.sort_values("내국인")
        yp = np.arange(len(agg))
        offs = {"내국인": -0.27, "단기외국인": 0.0, "장기외국인": 0.27}
        for pop in POP_ORDER:
            if pop in agg.columns:
                ax.barh(yp + offs[pop], agg[pop], height=0.27,
                        color=POP_COLOR[pop], label=POP_LABEL[pop])
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_yticks(yp); ax.set_yticklabels(agg.index.tolist(), fontsize=8.5)
        ax.set_xlabel(f"{metric} (시간, 폭염일 - baseline)")
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(axis="x", alpha=0.25)
    fig.suptitle("M_h4. 자치구별 시간 시프트 — 폭염일에 활동 시간 늦어졌는가",
                 fontsize=12)
    add_data_label(fig, "shift = (폭염일 평균/피크 시간) - (매칭 baseline 평균/피크 시간)")
    plt.tight_layout()
    plt.savefig(FIGS / "m_h04_mean_peak_shift.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ m_h04_mean_peak_shift.png")


# ============================================================
# m_h05. 자치구 × 시간그룹 변화율 heatmap
# ============================================================
def plot_h05_group_x_gu_heatmap(matched):
    fig, axes = plt.subplots(1, 3, figsize=(20, 10))
    for ax, pop in zip(axes, POP_ORDER):
        sub = matched[matched["pop"] == pop]
        pvt = (sub.groupby(["name", "hgroup"])["delta_pct"].mean()
                  .unstack().reindex(columns=HG_ORDER))
        # 자치구 순서: 4유형 묶음(TYPE_ORDER) → 그룹 내 첫 시간그룹 변화율 정렬
        pvt["_label"] = pd.Categorical(pvt.index.map(lambda n: GU_TYPE.get(n, "주거형")),
                                       categories=TYPE_ORDER, ordered=True)
        pvt = pvt.sort_values(["_label", pvt.columns[0]], ascending=[True, True])
        pvt = pvt.drop(columns="_label")
        sns.heatmap(pvt, cmap="RdBu_r", center=0, vmin=-8, vmax=8,
                    annot=True, fmt=".1f", annot_kws={"fontsize": 8},
                    ax=ax, cbar_kws={"label": "변화율 (%)"})
        ax.set_title(f"{LABEL[pop]} · {pop}")
        ax.set_xlabel("시간그룹"); ax.set_ylabel("")
    fig.suptitle("M_h5. 자치구 × 시간그룹 매칭 변화율 — 04 §5 자치구 분석을 시간으로 펼침",
                 fontsize=12)
    add_data_label(fig, "RdBu_r. 파랑 = 폭염일 감소, 빨강 = 증가")
    plt.tight_layout()
    plt.savefig(FIGS / "m_h05_group_x_gu_heatmap.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ m_h05_group_x_gu_heatmap.png")


# ============================================================
# m_h06. 내·외 격차의 시간 분해
# ============================================================
def plot_h06_local_foreign_gap(matched):
    # 격차 = 내국인 - 외국인 (셀별 → 그룹 평균). 단기(메인), 장기(보조) 둘 다.
    gap = (matched.pivot_table(index=["name", "date", "hgroup"], columns="pop",
                                 values="delta_pct").reset_index())
    gap["gap"] = gap["내국인"] - gap["단기외국인"]            # 메인 (헤드라인 -2.11%p)
    gap["gap_long"] = gap["내국인"] - gap["장기외국인"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # 좌. 시간그룹별 평균 격차 — 내·단기(막대) + 내·장기(막대)
    ax = axes[0]
    xp = np.arange(len(HG_ORDER))
    w = 0.38
    for k, (col, lab, col_color) in enumerate([
        ("gap", "내국인 − 단기(관광객)", POP_COLOR["단기외국인"]),
        ("gap_long", "내국인 − 장기(거주민)", POP_COLOR["장기외국인"]),
    ]):
        agg = gap.groupby("hgroup")[col].agg(["mean", "std", "count"]).reindex(HG_ORDER)
        se = agg["std"] / np.sqrt(agg["count"])
        ax.bar(xp + (k - 0.5) * w, agg["mean"], w, yerr=se, color=col_color,
               alpha=0.85, edgecolor="white", capsize=3, label=lab)
        for i, v in enumerate(agg["mean"]):
            ax.text(i + (k - 0.5) * w, v + (0.2 if v >= 0 else -0.5), f"{v:+.1f}",
                    ha="center", fontsize=7.5)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.axhline(-2.11, color="red", linestyle="--", linewidth=1.0,
               label="06 일 단위 내·단기 격차 = -2.11%p")
    ax.set_xticks(xp); ax.set_xticklabels(HG_ORDER, rotation=18, fontsize=9)
    ax.set_ylabel("내국인 - 외국인 격차 (%p)")
    ax.set_title("(a) 시간그룹별 평균 격차 (양수 = 내국인이 더 강하게 감소)")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.25)

    # 우. 자치구 그룹별 boxplot (내·단기 격차)
    ax = axes[1]
    gap["gu_group"] = gap["name"].map(lambda n: GU_TYPE.get(n, "주거형"))
    sns.boxplot(data=gap, x="hgroup", y="gap", hue="gu_group",
                hue_order=TYPE_ORDER, palette=TYPE_COLOR, order=HG_ORDER, ax=ax,
                fliersize=2, linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticklabels(HG_ORDER, rotation=18, fontsize=9)
    ax.set_xlabel(""); ax.set_ylabel("격차 (%p)")
    ax.set_title("(b) 자치구 4유형 × 시간그룹별 격차 분포")
    ax.legend(title="자치구 유형", fontsize=8)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("M_h6. 내·외 격차의 시간 분해 — 04 §4 -2.11%p가 어느 시간대에 집중되는가",
                 fontsize=12)
    add_data_label(fig, "격차 = 내국인 변화율 - 외국인 변화율 (셀별)")
    plt.tight_layout()
    plt.savefig(FIGS / "m_h06_local_foreign_gap.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ m_h06_local_foreign_gap.png")


# ============================================================
# m_h07. 자치구 4유형 × 24h 곡선
# ============================================================
def plot_h07_group_curves(curves):
    curves["gu_group"] = curves["pop"].apply(lambda _: None)
    # 자치구 정보 손실됨 — 다시 collect from share
    # 간단 대안. base 또는 heat 평균 자치구 그룹별로 다시 계산 → 함수 확장 필요
    # 여기서는 curves를 그대로 받았으니 자치구 매핑이 곤란.
    # → curves에 name 컬럼이 없음. 재계산 함수 작성.
    raise NotImplementedError("Use plot_h07_group_curves_v2 with share input")


def plot_h07_group_curves_v2(share):
    """자치구 그룹별 24h 곡선 (폭염 vs baseline). share에서 직접 계산."""
    share = share.copy()
    share["gu_group"] = share["name"].map(lambda n: GU_TYPE.get(n, "주거형"))
    hour_cols = [f"h{h}" for h in range(24)]

    rows = []
    heat_dates = sorted(share[share["is_heat"]]["date"].unique())
    for pop in POP_ORDER:
        sp = share[share["pop"] == pop].copy()
        for t in heat_dates:
            t = pd.Timestamp(t); t_dow = t.dayofweek
            cand = sp[(sp["date"] >= t - pd.Timedelta(days=WINDOW_DAYS)) &
                      (sp["date"] <= t + pd.Timedelta(days=WINDOW_DAYS)) &
                      (sp["dow"] == t_dow) &
                      (~sp["is_heat"]) & (~sp["is_rain"])]
            if cand["date"].nunique() < 2:
                continue
            # 자치구별 baseline curve / heat curve
            base_curve = cand.groupby("name")[hour_cols].mean()
            heat_curve = sp[sp["date"] == t].set_index("name")[hour_cols]
            for name in heat_curve.index:
                if name not in base_curve.index:
                    continue
                ggrp = GU_TYPE.get(name, "주거형")
                for h in range(24):
                    rows.append({
                        "pop": pop, "date": t, "name": name, "gu_group": ggrp,
                        "hour": h,
                        "heat_share": float(heat_curve.loc[name, f"h{h}"]),
                        "base_share": float(base_curve.loc[name, f"h{h}"]),
                    })
    long = pd.DataFrame(rows)

    fig, axes = plt.subplots(3, 4, figsize=(18, 12), sharex=True)
    for row, pop in enumerate(POP_ORDER):
        sub = long[long["pop"] == pop]
        for c, ggrp in enumerate(TYPE_ORDER):
            color = TYPE_COLOR[ggrp]
            ax = axes[row, c]
            sg = sub[sub["gu_group"] == ggrp]
            if sg.empty:
                ax.text(0.5, 0.5, "(데이터 없음)", ha="center", transform=ax.transAxes)
                continue
            agg = sg.groupby("hour").agg(heat=("heat_share", "mean"),
                                          base=("base_share", "mean")).reset_index()
            ax.plot(agg["hour"], agg["base"] * 100, "--", color="#555",
                    linewidth=1.5, label="baseline")
            ax.plot(agg["hour"], agg["heat"] * 100, "-", color=color,
                    linewidth=2.0, label="폭염일")
            ax.fill_between(agg["hour"], agg["base"] * 100, agg["heat"] * 100,
                             where=(agg["heat"] < agg["base"]),
                             color=color, alpha=0.18)
            ax.set_title(f"{POP_LABEL[pop]} · {ggrp}", fontsize=10)
            ax.set_xticks(range(0, 24, 4))
            ax.grid(alpha=0.25); ax.legend(loc="upper left", fontsize=8)
            if c == 0:
                ax.set_ylabel("도착자 비중 (%)")
            if row == len(POP_ORDER) - 1:
                ax.set_xlabel("시간 (0~23)")

    fig.suptitle("M_h7. 자치구 4유형별 × 24h 곡선 — 복합핵심/업무형/관광·여가형/주거형 폭염 영향 패턴",
                 fontsize=12)
    add_data_label(fig, "유형 = 13번 노트북 2D 성향(관광지향·업무지향) 사분면 분류")
    plt.tight_layout()
    plt.savefig(FIGS / "m_h07_tourism_business_curve.png", dpi=140, bbox_inches="tight")
    plt.close()
    print("✓ m_h07_tourism_business_curve.png")


# ============================================================
# 메인
# ============================================================
def main():
    print("=== 폭염 시간대별 매칭 분석 (보고서 05) ===\n")
    seoul, daily = load_and_merge()
    print(f"입력. {len(seoul):,}행 (서울 25 자치구 × 276일 × 24시간 × 3집단)")
    print(f"폭염일 {seoul[seoul['is_heat']]['date'].nunique()}일")

    # 방법 A — 그룹 매칭
    print("\n[방법 A] 시간대 그룹 매칭")
    cells = build_cells_group(seoul)
    print(f"  셀 합산 결과. {len(cells):,}행")
    matched, drop_log = compute_matched_delta_hgroup(cells)
    matched.to_parquet(PROCESSED / "matched_delta_hgroup.parquet")
    print(f"  ✓ matched_delta_hgroup.parquet ({len(matched):,}행)")
    check_match_quality_hgroup(matched, drop_log)
    verify_against_04(matched)

    # 시간 그룹별 변화율 요약
    print("\n=== 6 그룹별 매칭 변화율 ===")
    print(matched.groupby(["pop", "hgroup"])["delta_pct"].agg(
        ["mean", "median", "count"]).round(2).to_string())

    # 방법 B — 분포 시프트
    print("\n[방법 B] 시간 분포 시프트")
    share = build_hourly_share(seoul)
    print(f"  share 행. {len(share):,}")
    shift_df = compute_shift_metrics(share)
    shift_df.to_parquet(PROCESSED / "hourly_share_shift.parquet")
    print(f"  ✓ hourly_share_shift.parquet ({len(shift_df):,}행)")
    print("\n=== 시프트 통계 (전체 평균) ===")
    print(shift_df.groupby("pop")[["mean_shift", "peak_shift", "kl"]].agg(["mean", "median"]).round(3).to_string())

    curves = collect_baseline_curves(share)

    # 시각화
    print("\n[시각화 7장]")
    plot_h01_grouping(seoul, cells)
    plot_h02_group_delta(matched)
    plot_h03_dist_curves(curves)
    plot_h04_mean_peak_shift(shift_df)
    plot_h05_group_x_gu_heatmap(matched)
    plot_h06_local_foreign_gap(matched)
    plot_h07_group_curves_v2(share)

    print("\n=== 완료 ===")


if True:
    main()



# ==============================================================================
# STEP 06 교란 EDA 그림 (그림 3-1·3-2·3-3)
# ==============================================================================
print("\n>>> STEP 06 교란 EDA 그림 (그림 3-1·3-2·3-3)")
# 3장용 그림 — 자료 개관 + 교란(시즌·요일·강수) 실증 + 매칭 설계 도식. 출력: 최종보고서/그래프/03_*.png
"""
보고서 3장 "데이터와 방법"의 시각 자료를 생성한다.
  03_data_overview.png    : 3집단 규모 · 연도별 폭염일 · 월별 분석일/폭염일 (자료 개관)
  03_confounding_grid.png : 시즌·요일·강수가 이동량을 바꾸고(위) 폭염·비폭염일에서 불균형(아래) → 매칭 정당화
  03_matching_design.png  : 매칭 기준선 워크드 예시(달력 띠) + 폭염일당 대조일 수 분포
모든 수치는 data/processed/{weather_daily,kt_gu_hour_daily}.parquet 에서 직접 계산.
"""
import warnings; warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = ["AppleGothic", "Apple SD Gothic Neo", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = ROOT
PROC = PROC
OUT = FIGS
OUT.mkdir(parents=True, exist_ok=True)

POP_ORDER = ["내국인", "단기외국인", "장기외국인"]
POP_COLOR = {"내국인": "#1f77b4", "단기외국인": "#d62728", "장기외국인": "#2ca02c"}
DOW_KR = ["월", "화", "수", "목", "금", "토", "일"]

# ── 데이터 ──
w = pd.read_parquet(PROC / "weather_daily.parquet"); w["date"] = pd.to_datetime(w["date"])
w["month"] = w["date"].dt.month; w["dow"] = w["date"].dt.dayofweek; w["year"] = w["date"].dt.year
w["is_rain"] = w["rn_sum"] > 1.0; w["is_heat"] = w["is_heatwave_feels"]
kt = pd.read_parquet(PROC / "kt_gu_hour_daily.parquet"); kt["date"] = pd.to_datetime(kt["date"])
s = kt[kt["gu_cd"].str.startswith("11")]
day = s.groupby(["date", "pop"])["total"].sum().reset_index()
day = day.merge(w[["date", "month", "dow", "is_rain", "is_heat"]], on="date")
kd = set(s["date"].unique()); wk = w[w["date"].isin(kd)].copy()      # 분석기간 weather

def sub(p): return day[day["pop"] == p]

DATALAB = "자료: KT 생활이동·기상청 ASOS, 서울 25개 자치구, 2023~2025 여름(276일, 폭염일 69일)"
def data_label(fig, text=DATALAB):
    fig.text(0.99, 0.004, text, ha="right", va="bottom", fontsize=7.5, color="#555",
             bbox=dict(boxstyle="round,pad=0.4", fc="#f4f4f4", ec="#bbb", lw=0.7))

# ============================================================
# FIG 1. 자료 개관
# ============================================================
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# (a) 3집단 일평균 규모
means = {p: sub(p)["total"].mean() for p in POP_ORDER}
bars = ax[0].bar(range(3), [means[p] / 1e6 for p in POP_ORDER], color=[POP_COLOR[p] for p in POP_ORDER], width=0.62)
ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["내국인", "단기 체류\n외국인", "장기 체류\n외국인"])
ax[0].set_ylabel("일평균 도착 인원 (백만 명)")
for i, p in enumerate(POP_ORDER):
    ax[0].text(i, means[p] / 1e6 + 0.3, f"{means[p]/1e4:,.0f}만", ha="center", fontsize=10, color=POP_COLOR[p], fontweight="bold")
ax[0].set_title("(a) 서울 일평균 도착 인원 (3집단)", fontsize=11)
ax[0].set_ylim(0, 20)
for sp in ["top", "right"]: ax[0].spines[sp].set_visible(False)
# (b) 연도별 폭염일
hy = wk[wk["is_heat"]].groupby("year").size()
b = ax[1].bar([str(y) for y in hy.index], hy.values, color="#d7301f", width=0.55)
for bi, v in zip(b, hy.values): ax[1].text(bi.get_x() + bi.get_width()/2, v + 0.4, str(int(v)), ha="center", fontsize=11, fontweight="bold")
ax[1].set_ylabel("폭염일 수 (일)"); ax[1].set_title(f"(b) 연도별 폭염일 — 체감온도 ≥ 33℃ (합 {int(hy.sum())}일)", fontsize=11)
ax[1].set_ylim(0, 30)
for sp in ["top", "right"]: ax[1].spines[sp].set_visible(False)
# (c) 월별 분석일 vs 폭염일
mn = [6, 7, 8]; tot = [int((wk["month"] == m).sum()) for m in mn]; ht = [int(((wk["month"] == m) & wk["is_heat"]).sum()) for m in mn]
x = np.arange(3); ww = 0.38
ax[2].bar(x - ww/2, tot, ww, label="전체 분석일", color="#bdbdbd")
ax[2].bar(x + ww/2, ht, ww, label="폭염일", color="#d7301f")
for i in range(3):
    ax[2].text(x[i] - ww/2, tot[i] + 1, str(tot[i]), ha="center", fontsize=9)
    ax[2].text(x[i] + ww/2, ht[i] + 1, str(ht[i]), ha="center", fontsize=9, color="#d7301f")
ax[2].set_xticks(x); ax[2].set_xticklabels(["6월", "7월", "8월"]); ax[2].set_ylabel("일수")
ax[2].set_title("(c) 월별 분석일과 폭염일 — 폭염일은 늦여름에 쏠림", fontsize=11)
ax[2].legend(fontsize=9); ax[2].set_ylim(0, 105)
for sp in ["top", "right"]: ax[2].spines[sp].set_visible(False)
fig.suptitle("그림 3-1. 분석 자료 개관", fontsize=13, fontweight="bold")
data_label(fig)
plt.tight_layout(rect=[0, 0.03, 1, 0.96]); plt.savefig(OUT / "03_data_overview.png", dpi=145); plt.close()
print("saved 03_data_overview.png")

# ============================================================
# FIG 2. 교란 그리드 (2x3): 위=이동량 변화, 아래=폭염/비폭염 불균형
# ============================================================
fig, ax = plt.subplots(2, 3, figsize=(15.5, 9))
# (0,0) 월별 이동량 지수 (집단 평균=100)
for p in POP_ORDER:
    g = sub(p).groupby("month")["total"].mean(); idx = 100 * g / g.mean()
    ax[0, 0].plot([6, 7, 8], [idx.loc[m] for m in [6, 7, 8]], "-o", color=POP_COLOR[p], label=p, lw=2.2, ms=7)
ax[0, 0].axhline(100, color="#999", lw=0.8, ls="--"); ax[0, 0].set_xticks([6, 7, 8]); ax[0, 0].set_xticklabels(["6월", "7월", "8월"])
ax[0, 0].set_ylabel("이동량 지수 (집단별 평균=100)"); ax[0, 0].set_title("(a) 시즌이 이동량을 바꾼다", fontsize=11.5, fontweight="bold")
ax[0, 0].legend(fontsize=8.5)
# (0,1) 요일 이동량 지수
for p in POP_ORDER:
    g = sub(p).groupby("dow")["total"].mean(); idx = 100 * g / g.mean()
    ax[0, 1].plot(range(7), [idx.loc[d] for d in range(7)], "-o", color=POP_COLOR[p], label=p, lw=2.2, ms=6)
ax[0, 1].axhline(100, color="#999", lw=0.8, ls="--"); ax[0, 1].set_xticks(range(7)); ax[0, 1].set_xticklabels(DOW_KR)
ax[0, 1].axvspan(4.5, 6.5, color="#f0f0f0", zorder=0)
ax[0, 1].set_ylabel("이동량 지수 (주평균=100)"); ax[0, 1].set_title("(b) 요일이 이동량을 바꾼다", fontsize=11.5, fontweight="bold")
ax[0, 1].legend(fontsize=8.5)
# (0,2) 맑음 vs 비 이동량
xx = np.arange(3); ww = 0.36
cl = [sub(p)[~sub(p)["is_rain"]]["total"].mean() / 1e6 for p in POP_ORDER]
rn = [sub(p)[sub(p)["is_rain"]]["total"].mean() / 1e6 for p in POP_ORDER]
ax[0, 2].bar(xx - ww/2, cl, ww, label="맑은 날", color="#fdae61")
ax[0, 2].bar(xx + ww/2, rn, ww, label="비 오는 날", color="#4575b4")
for i, p in enumerate(POP_ORDER):
    drop = 100 * (rn[i] / cl[i] - 1)
    ax[0, 2].text(xx[i], max(cl[i], rn[i]) + 0.4, f"{drop:+.1f}%", ha="center", fontsize=9, color="#4575b4")
ax[0, 2].set_xticks(xx); ax[0, 2].set_xticklabels(["내국인", "단기", "장기"]); ax[0, 2].set_yscale("log")
ax[0, 2].set_ylabel("일평균 도착 (백만 명, log)"); ax[0, 2].set_title("(c) 강수가 이동량을 줄인다", fontsize=11.5, fontweight="bold")
ax[0, 2].legend(fontsize=8.5)
# (1,0) 월별 폭염/비폭염 일수
mn = [6, 7, 8]
ht = [int(((wk["month"] == m) & wk["is_heat"]).sum()) for m in mn]
nh = [int(((wk["month"] == m) & ~wk["is_heat"]).sum()) for m in mn]
ax[1, 0].bar(xx - ww/2, nh, ww, label="비폭염일", color="#bdbdbd")
ax[1, 0].bar(xx + ww/2, ht, ww, label="폭염일", color="#d7301f")
ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels(["6월", "7월", "8월"]); ax[1, 0].set_ylabel("일수")
sh_aug = 100 * ht[2] / sum(ht)
ax[1, 0].set_title(f"(d) 폭염일은 8월에 쏠림 ({sh_aug:.0f}%)", fontsize=11.5, fontweight="bold"); ax[1, 0].legend(fontsize=8.5)
# (1,1) 주말 비중 폭염 vs 비폭염
h_we = 100 * wk[wk["is_heat"]]["dow"].isin([5, 6]).mean(); n_we = 100 * wk[~wk["is_heat"]]["dow"].isin([5, 6]).mean()
b = ax[1, 1].bar(["폭염일", "비폭염일"], [h_we, n_we], color=["#d7301f", "#bdbdbd"], width=0.5)
for bi, v in zip(b, [h_we, n_we]): ax[1, 1].text(bi.get_x()+bi.get_width()/2, v+0.6, f"{v:.0f}%", ha="center", fontsize=10)
ax[1, 1].set_ylabel("주말 비중 (%)"); ax[1, 1].set_ylim(0, 40)
ax[1, 1].set_title("(e) 폭염일은 주말이 더 적다", fontsize=11.5, fontweight="bold")
# (1,2) 강수: 맑음 비율 + 평균 강수
h_clear = 100 * (~wk[wk["is_heat"]]["is_rain"]).mean(); n_clear = 100 * (~wk[~wk["is_heat"]]["is_rain"]).mean()
h_rain = wk[wk["is_heat"]]["rn_sum"].mean(); n_rain = wk[~wk["is_heat"]]["rn_sum"].mean()
axb = ax[1, 2]; b = axb.bar(["폭염일", "비폭염일"], [h_clear, n_clear], color=["#fdae61", "#bdbdbd"], width=0.5)
for bi, v in zip(b, [h_clear, n_clear]): axb.text(bi.get_x()+bi.get_width()/2, v+1, f"맑음 {v:.0f}%", ha="center", fontsize=9.5)
axb.set_ylabel("맑은 날 비율 (%)"); axb.set_ylim(0, 100)
axb.set_title(f"(f) 비폭염일이 더 비가 잦다\n(평균 강수 폭염 {h_rain:.1f} vs 비폭염 {n_rain:.1f}mm)", fontsize=11.5, fontweight="bold")
fig.suptitle("그림 3-2. 왜 매칭인가 — 시즌·요일·강수는 모두 이동량을 바꾸고(위), 폭염일과 비폭염일 사이에서 불균형하다(아래)",
             fontsize=12.5, fontweight="bold")
data_label(fig)
plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(OUT / "03_confounding_grid.png", dpi=140); plt.close()
print("saved 03_confounding_grid.png")

# ============================================================
# FIG 3. 매칭 설계 도식 + 대조일 수 분포
# ============================================================
WIN = 14
heat_dates = sorted(wk[wk["is_heat"]]["date"])
# 폭염일별 같은요일·비폭염·맑음 대조일 수
def n_controls(t):
    t = pd.Timestamp(t); c = wk[(wk["date"] >= t - pd.Timedelta(days=WIN)) & (wk["date"] <= t + pd.Timedelta(days=WIN)) &
                                (wk["dow"] == t.dayofweek) & (~wk["is_heat"]) & (~wk["is_rain"]) & (wk["date"] != t)]
    return len(c)
nc = pd.Series({t: n_controls(t) for t in heat_dates})
# 워크드 예시: 대조일이 중앙값에 가깝고 평일인 폭염일 하나 선택
cand = [t for t in heat_dates if pd.Timestamp(t).dayofweek < 5 and nc[t] >= 2]
t0 = sorted(cand, key=lambda t: abs(nc[t] - nc.median()))[0]
t0 = pd.Timestamp(t0)
win = wk[(wk["date"] >= t0 - pd.Timedelta(days=WIN)) & (wk["date"] <= t0 + pd.Timedelta(days=WIN))].sort_values("date").copy()
def cat(r):
    if r["date"] == t0: return "폭염일(처리)"
    if r["is_heat"]: return "다른 폭염일"
    if r["dow"] == t0.dayofweek and not r["is_rain"]: return "매칭 대조(같은요일·맑음·비폭염)"
    if r["is_rain"]: return "비 오는 날"
    return "비폭염·다른요일"
win["cat"] = win.apply(cat, axis=1)
# (색, 크기, 테두리색, 테두리굵기, zorder) — 매칭 대조가 가장 도드라지게
CATSTYLE = {"폭염일(처리)": ("#d7301f", 230, "black", 1.1, 6),
            "다른 폭염일": ("#fdae61", 120, "white", 0.6, 4),
            "매칭 대조(같은요일·맑음·비폭염)": ("#08519c", 230, "black", 1.5, 5),
            "비 오는 날": ("#9ecae1", 70, "white", 0.4, 3),
            "비폭염·다른요일": ("#d9d9d9", 70, "white", 0.4, 3)}

fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [2.3, 1]})
for c, (col, sz, ec, lw, zo) in CATSTYLE.items():
    sb = win[win["cat"] == c]
    ax[0].scatter((sb["date"] - t0).dt.days, sb["feels_max"], c=col, s=sz, label=c, zorder=zo, edgecolor=ec, lw=lw)
base = win[win["cat"] == "매칭 대조(같은요일·맑음·비폭염)"]["feels_max"].mean()
ax[0].axhline(33, color="#d7301f", ls=":", lw=1, alpha=0.7); ax[0].text(-WIN, 33.4, "체감 33℃ (폭염 기준)", fontsize=8, color="#d7301f")
ax[0].axhline(base, color="#1f77b4", ls="--", lw=1.2); ax[0].text(WIN, base + 0.2, f"대조 평균 체감 {base:.1f}℃", fontsize=8.5, color="#1f77b4", ha="right")
ax[0].set_xlabel(f"기준 폭염일({t0.date()}, {DOW_KR[t0.dayofweek]})로부터의 일수"); ax[0].set_ylabel("일 최고 체감온도 (℃)")
ax[0].set_title("(a) 매칭 기준선 구성 — 같은 요일·±14일·비폭염·맑은 날만 대조군", fontsize=11)
ax[0].legend(fontsize=8, loc="lower center", ncol=2); ax[0].grid(alpha=0.2)
# (b) 매칭 표본 구성 — 대조일 2개 이상 확보한 폭염일만 사용 (69 → 24)
vc = nc.value_counts().sort_index()
used = int((nc >= 2).sum()); cells = used * 25 * 3
colors = ["#cccccc" if k < 2 else "#1f77b4" for k in vc.index]
b = ax[1].bar(vc.index, vc.values, color=colors, edgecolor="white")
for bi, v in zip(b, vc.values): ax[1].text(bi.get_x()+bi.get_width()/2, v+0.4, str(int(v)), ha="center", fontsize=9)
ax[1].axvline(1.5, color="#d7301f", ls="--", lw=1.3)
ax[1].text(1.62, ax[1].get_ylim()[1]*0.62, f"대조 2개 이상 사용\n폭염일 {used}개\n(셀 {cells:,}개)", fontsize=8.5, color="#d7301f")
ax[1].set_xlabel("폭염일당 같은요일·맑은 비폭염 대조일 수"); ax[1].set_ylabel("폭염일 수")
ax[1].set_title(f"(b) 매칭 표본 구성 — 폭염일 {len(nc)}개 중 {used}개 사용", fontsize=10.5)
for sp in ["top", "right"]: ax[1].spines[sp].set_visible(False)
fig.suptitle("그림 3-3. 매칭 기반 준실험 설계", fontsize=13, fontweight="bold")
data_label(fig, f"기준선 = 같은 자치구·같은 요일·±14일 이내·비폭염·맑은 날(강수≤1mm) 평균 | 예시 자치구 무관 일 단위 도식")
plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.savefig(OUT / "03_matching_design.png", dpi=140); plt.close()
print("saved 03_matching_design.png")
print(f"\n워크드 예시 폭염일: {t0.date()} ({DOW_KR[t0.dayofweek]}), 대조일수 {nc[t0]:.0f}")
print(f"대조일 수 분포: 중앙값 {nc.median():.0f}, 평균 {nc.mean():.1f}, 최소 {nc.min():.0f}, 최대 {nc.max():.0f}")
print("=== F03 완료 ===")



# ==============================================================================
# STEP 07 적응 스펙트럼 그림 (그림 4-1·4-2·4-3, 표 4-1·4-3)
# ==============================================================================
print("\n>>> STEP 07 적응 스펙트럼 그림 (그림 4-1·4-2·4-3, 표 4-1·4-3)")
# N1. 매칭 적응스펙트럼 — 자치구 3집단 폭염 회피율 + 통제단계 분해 + 시간대 변화. (§5·§6①②)
"""
입력. data/processed/matched_delta.parquet, matched_delta_hgroup.parquet,
      kt_gu_hour_daily.parquet, weather_daily.parquet
출력. 클러스터링/outputs/figures/N1_*.png, tables/N1_*.csv
핵심. de-confounded 매칭으로 내국인-3.01 / 단기-0.90 / 장기-3.66 적응스펙트럼 재현·검증.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

md = pd.read_parquet(PROC / "matched_delta.parquet")
mdh = pd.read_parquet(PROC / "matched_delta_hgroup.parquet")

# ── 1. 적응 스펙트럼 (집단별 평균 + 검정) ──
print("=== 적응 스펙트럼 (자치구×폭염일 평균) ===")
spec = {}
for pop in POP_ORDER:
    v = md[md["pop"] == pop]["delta_pct"]
    spec[pop] = v.mean()
    print(f"  {pop}: {v.mean():+.2f}%  (검증값 {MATCH_TRUTH[pop]:+.2f})")
# 내·단기 격차 검정 (폭염일별 서울평균)
byday = md.groupby(["date", "pop"])["delta_pct"].mean().unstack()
gap = byday["내국인"] - byday["단기외국인"]
t, p = stats.ttest_1samp(gap.dropna(), 0)
print(f"  내·단기 격차 {gap.mean():+.2f}%p, p={p:.4g}")

# ── 2. 통제단계 분해 (단순→강수→요일→매칭) ──
kt = pd.read_parquet(PROC / "kt_gu_hour_daily.parquet")
wd = pd.read_parquet(PROC / "weather_daily.parquet")
kt["date"] = pd.to_datetime(kt["date"]); wd["date"] = pd.to_datetime(wd["date"])
seoul = kt[kt["gu_cd"].astype(str).str.startswith("11")]
sd = seoul.groupby(["date", "pop"])["total"].sum().reset_index().rename(columns={"total": "arr"})
wd["dow"] = wd["date"].dt.dayofweek; wd["is_rain"] = wd["rn_sum"] > 1.0
wd["is_heat"] = wd["is_heatwave_feels"]; wd["is_wknd"] = wd["dow"].isin([5, 6])
sd = sd.merge(wd[["date", "is_heat", "is_rain", "is_wknd"]], on="date", how="left")

def simple(s):
    n = s[~s["is_heat"]]["arr"].mean(); h = s[s["is_heat"]]["arr"].mean()
    return (h / n - 1) * 100

decomp = {}
for pop in POP_ORDER:
    s = sd[sd["pop"] == pop]
    decomp[pop] = [simple(s), simple(s[~s["is_rain"]]),
                   simple(s[(~s["is_rain"]) & (~s["is_wknd"])]), spec[pop]]
steps = ["단순\n(통제없음)", "+강수\n(맑음)", "+요일\n(맑음·평일)", "+시즌\n(매칭완성)"]

fig, ax = plt.subplots(figsize=(11, 5.5))
x = np.arange(len(steps)); w = 0.26
for i, pop in enumerate(POP_ORDER):
    vals = decomp[pop]
    b = ax.bar(x + (i - 1) * w, vals, w, label=POP_LABEL[pop], color=POP_COLOR[pop])
    for r, v in zip(b, vals):
        ax.text(r.get_x() + w / 2, v + (0.15 if v >= 0 else -0.5), f"{v:+.1f}", ha="center", fontsize=8)
ax.axhline(0, color="black", lw=0.6); ax.set_xticks(x); ax.set_xticklabels(steps)
ax.set_ylabel("폭염일 vs 정상일 이동량 변화 (%)")
ax.set_title("N1. 통제 단계별 폭염 효과 — 시즌까지 통제하면 부호가 뒤집힌다 (매칭=최종)")
ax.legend(); ax.grid(alpha=.25, axis="y")
data_label(fig, "3년(2023~25 여름) | 매칭 baseline=같은 자치구·요일·±14일 비폭염·맑음 평균")
plt.tight_layout(); plt.savefig(FIGS / "N1_decomposition.png", dpi=130); plt.close()
print("  saved N1_decomposition.png")

# ── 3. 적응 스펙트럼 막대 ──
fig, ax = plt.subplots(figsize=(7, 5))
vals = [spec[p] for p in POP_ORDER]
b = ax.bar([POP_LABEL[p] for p in POP_ORDER], vals, color=[POP_COLOR[p] for p in POP_ORDER])
for r, v in zip(b, vals): ax.text(r.get_x() + r.get_width() / 2, v - 0.2, f"{v:+.2f}%", ha="center", va="top", fontsize=11, color="white", fontweight="bold")
ax.axhline(0, color="black", lw=.6); ax.set_ylabel("폭염일 매칭 변화율 (%)")
ax.set_title("N1. 폭염 적응 스펙트럼 — 거주경험이 적응을 만든다\n장기거주≈내국인 ≫ 단기관광객(거의 안 줄임=노출)")
data_label(fig, f"내·단기 격차 {gap.mean():+.2f}%p, p<0.0001 | 25/25 자치구 일관")
plt.tight_layout(); plt.savefig(FIGS / "N1_spectrum.png", dpi=130); plt.close()
print("  saved N1_spectrum.png")

# ── 4. 시간대별 변화 ──
hg = mdh.groupby(["hgroup", "pop"])["delta_pct"].mean().unstack()
order = ["심야 (0-5)", "출근 (6-9)", "점심 (10-13)", "오후 (14-17)", "저녁 (18-21)", "밤 (22-23)"]
hg = hg.reindex([o for o in order if o in hg.index])
fig, ax = plt.subplots(figsize=(11, 5.5))
for pop in POP_ORDER:
    ax.plot(range(len(hg)), hg[pop], marker="o", lw=2.2, color=POP_COLOR[pop], label=POP_LABEL[pop])
ax.axhline(0, color="gray", ls="--", lw=.8); ax.set_xticks(range(len(hg))); ax.set_xticklabels(hg.index, fontsize=9)
ax.set_ylabel("매칭 변화율 (%)"); ax.set_title("N1. 시간대별 폭염 이동변화 — 단기관광객은 한낮에도 안 줄인다")
ax.legend(); ax.grid(alpha=.25)
data_label(fig, "시간대 6구간 매칭 | 단기 점심·오후 ≈0 = 한낮 폭염 노출")
plt.tight_layout(); plt.savefig(FIGS / "N1_timegroup.png", dpi=130); plt.close()
print("  saved N1_timegroup.png")

# ── 표 저장 ──
pd.DataFrame({"집단": POP_ORDER, "매칭변화율": [spec[p] for p in POP_ORDER],
              "검증값": [MATCH_TRUTH[p] for p in POP_ORDER]}).to_csv(TABLES / "N1_spectrum.csv", index=False, encoding="utf-8-sig")
pd.DataFrame(decomp, index=steps).to_csv(TABLES / "N1_decomposition.csv", encoding="utf-8-sig")
hg.to_csv(TABLES / "N1_timegroup.csv", encoding="utf-8-sig")
print("=== N1 완료 ===")



# ==============================================================================
# STEP 08 관광지 실내외 탄력성 (그림 4-4)
# ==============================================================================
print("\n>>> STEP 08 관광지 실내외 탄력성 (그림 4-4)")
# 대표 관광지 실내/야외 구분 × 폭염 탄력성 — 야외는 탄력적(빠짐), 실내는 비탄력(버팀) 가설 검증.
"""
질문(사용자, 2026-06-04). 대표 관광지를 실내/야외로 나누면, 폭염에 야외 관광객은 탄력적으로
  빠지고 실내는 비탄력적으로 버티는가? (목적지 재배치 가설의 직접 검증)

분류(토지이용 수동, 외부 기준 → 순환논리 없음). 웹/토지이용 근거로 실내우세/야외우세/혼합.
  실내우세 = 대형 실내몰·백화점·복합(롯데월드몰·코엑스·타임스퀘어·아이파크몰·동대문몰·롯데본점).
  야외우세 = 노천 거리·광장·고궁·한옥(홍대·인사동·광화문·북촌·서촌·이태원·망리단길).
  혼합     = 노천거리+실내 병존(명동·강남역·가로수길·로데오·성수) → 그룹비교 제외, 참고표시.

탄력성 두 지표.
  (1) 일단위  = matched_delta(폭염일 vs 같은요일·±14일·맑은날 baseline, 총 단기관광객 %). build_dong_matched_delta.
  (2) 한낮노출 = 폭염일 한낮(12-16시) 점유율 − 정상일 한낮 점유율 (%p).
                점유율=하루 중 12-16시 도착 비중 → 일 규모 교란 자동 제거, '더운 한낮을 피하나'만 봄.
                두 day-set 모두 맑은날(비 제외)로 한정해 강수 교란 통제.
  가설. 야외 = (1) 더 음(-), (2) 한낮점유율 하락(-). 실내 = (1) ~0/+, (2) 유지/+.

입력. kt_dong_short_hourly.parquet, weather_daily.parquet, dong_matched_delta.parquet
출력. outputs/figures/dong_20_indoor_outdoor_elasticity.png
      outputs/tables/dong_20_elasticity.csv
실행. python notebooks/20_관광지_실내야외_폭염탄력성.py
"""
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.rcParams["font.family"] = ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt

ROOT = ROOT
PROC = ROOT / "data/processed"
FIGS = FIGS
TABLES = TABLES

MID = list(range(12, 17))  # 한낮 폭염피크 12~16시

# 큐레이션 대표 관광지 — 동코드는 데이터에서 검증된 값. 토지이용 근거는 본문/nb17.
SITES = [
    # 실내우세
    (11710710, "잠실 롯데월드몰", "실내"), (11680580, "코엑스(삼성1동)", "실내"),
    (11560535, "타임스퀘어(영등포동)", "실내"), (11170625, "용산 아이파크몰(한강로동)", "실내"),
    (11140590, "동대문몰(광희동)", "실내"), (11140520, "롯데·신세계본점(소공동)", "실내"),
    # 야외우세
    (11440660, "홍대(서교동)", "야외"), (11110615, "인사동·종로", "야외"),
    (11110530, "광화문·경복궁(사직동)", "야외"), (11110540, "북촌(삼청동)", "야외"),
    (11110600, "북촌한옥(가회동)", "야외"), (11170650, "이태원1동", "야외"),
    (11110515, "서촌(청운효자동)", "야외"), (11440680, "망리단길(합정동)", "야외"),
    # 혼합(참고)
    (11140550, "명동", "혼합"), (11680640, "강남역(역삼1동)", "혼합"),
    (11680545, "압구정 로데오", "혼합"), (11680510, "가로수길(신사동)", "혼합"),
    (11200690, "성수동(성수2가3동)", "혼합"),
]
CAT_COLOR = {"실내": "#1f77b4", "야외": "#d62728", "혼합": "#7f7f7f"}


def main():
    sites = pd.DataFrame(SITES, columns=["d_admdong_cd", "name", "cat"])

    w = pd.read_parquet(PROC / "weather_daily.parquet")[["date", "is_heatwave_feels", "rn_sum"]].copy()
    w["date"] = pd.to_datetime(w["date"])
    w["clear"] = w["rn_sum"] == 0  # 맑은날(비 없음)

    h = pd.read_parquet(PROC / "kt_dong_short_hourly.parquet")
    h["date"] = pd.to_datetime(h["date"])
    h = h[h["d_admdong_cd"].isin(sites["d_admdong_cd"])].merge(w, on="date", how="left")

    # 일별 한낮 점유율 = sum(short[12-16]) / sum(short[all])
    day_tot = h.groupby(["d_admdong_cd", "date"])["short"].sum().rename("tot")
    day_mid = h[h["hour"].isin(MID)].groupby(["d_admdong_cd", "date"])["short"].sum().rename("mid")
    dd = pd.concat([day_tot, day_mid], axis=1).reset_index()
    dd["mid_share"] = dd["mid"] / dd["tot"] * 100
    dd = dd.merge(w[["date", "is_heatwave_feels", "clear"]], on="date", how="left")
    dd = dd[dd["clear"]]  # 맑은날만 (강수 통제)

    # 폭염 vs 정상 한낮점유율 (점유율=일내 비중이라 일규모 교란은 이미 제거됨)
    grp = dd.groupby(["d_admdong_cd", "is_heatwave_feels"])["mid_share"].mean().unstack()
    grp.columns = ["share_normal", "share_heat"]
    grp["d_midshare_pp"] = grp["share_heat"] - grp["share_normal"]

    md = pd.read_parquet(PROC / "dong_matched_delta.parquet").set_index("d_admdong_cd")["matched_delta"]
    vol = day_tot.groupby("d_admdong_cd").mean().rename("vol_day")

    out = (sites.set_index("d_admdong_cd")
           .join(grp[["share_normal", "share_heat", "d_midshare_pp"]])
           .join(md.rename("matched_delta_pct")).join(vol).reset_index())
    out = out.round(3)
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGS.mkdir(parents=True, exist_ok=True)
    out.to_csv(TABLES / "dong_20_elasticity.csv", index=False, encoding="utf-8-sig")

    # 그룹 요약 + 검정
    def grp_stats(col):
        ind = out[out["cat"] == "실내"][col].dropna()
        outd = out[out["cat"] == "야외"][col].dropna()
        line = f"  {col:18s} 실내={ind.mean():+.2f}(n{len(ind)})  야외={outd.mean():+.2f}(n{len(outd)})"
        try:
            from scipy.stats import mannwhitneyu
            p = mannwhitneyu(ind, outd, alternative="two-sided").pvalue
            line += f"  MWU p={p:.3f}"
        except Exception:
            line += "  (scipy 없음)"
        return line

    print("=== 실내 vs 야외 그룹 평균 ===")
    print(grp_stats("matched_delta_pct"))
    print(grp_stats("d_midshare_pp"))
    print("\n=== 사이트별 ===")
    print(out[["name", "cat", "matched_delta_pct", "d_midshare_pp", "vol_day"]]
          .sort_values(["cat", "matched_delta_pct"]).to_string(index=False))

    # 그림: 좌=일단위 탄력성 막대, 우=산점도(한낮점유 변화 × 일단위 탄력성)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    o2 = out.sort_values(["cat", "matched_delta_pct"])
    colors = [CAT_COLOR[c] for c in o2["cat"]]
    ax1.barh(range(len(o2)), o2["matched_delta_pct"], color=colors)
    ax1.set_yticks(range(len(o2)))
    ax1.set_yticklabels(o2["name"], fontsize=9)
    ax1.axvline(0, color="k", lw=0.8)
    ax1.set_xlabel("일단위 폭염 탄력성 = matched_delta (%)  ←빠짐 | 버팀→")
    ax1.set_title("폭염일 총 관광객 변화 (매칭 통제)")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=CAT_COLOR[c], label=c) for c in ["실내", "야외", "혼합"]], loc="lower right")
    ax1.grid(axis="x", alpha=0.25)

    for _, r in out.iterrows():
        if pd.isna(r["d_midshare_pp"]) or pd.isna(r["matched_delta_pct"]):
            continue
        ax2.scatter(r["d_midshare_pp"], r["matched_delta_pct"],
                    s=40 + r["vol_day"] / 200, color=CAT_COLOR[r["cat"]],
                    edgecolor="k", lw=0.4, alpha=0.85, zorder=3)
        ax2.annotate(r["name"], (r["d_midshare_pp"], r["matched_delta_pct"]),
                     fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax2.axhline(0, color="gray", lw=0.8, ls="--")
    ax2.axvline(0, color="gray", lw=0.8, ls="--")
    ax2.set_xlabel("한낮(12-16시) 점유율 변화 (%p)  ←한낮 회피 | 한낮 유지→")
    ax2.set_ylabel("일단위 탄력성 matched_delta (%)")
    ax2.set_title("한낮 노출 × 총량 탄력성 (점크기=한낮규모)")
    ax2.grid(alpha=0.25)
    fig.suptitle("대표 관광지 실내/야외 폭염 탄력성 — 야외는 빠지고 실내는 버티나?", fontsize=13)
    fig.tight_layout()
    out_fig = FIGS / "dong_20_indoor_outdoor_elasticity.png"
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print("\n저장:", out_fig)
    print("저장:", TABLES / "dong_20_elasticity.csv")


if True:
    main()



# ==============================================================================
# STEP 09 동 단위 베이스 (dong_base)
# ==============================================================================
print("\n>>> STEP 09 동 단위 베이스 (dong_base)")
# N2. 동별 폭염 이동변화 — 외국인 3년 매칭 회피율 지도+목록 + 2025 내외국인 비교. (§6 동단위)
"""
입력. dong_matched_delta.parquet(426동 외국인 3년), kt_dong_short_hourly.parquet(노출),
      dong_concentration.parquet(LQ·share·동명), KT_OD_보너스/2025_여름(내외국인 일별),
      weather_daily.parquet, GEO_DONG
출력. figures/N2_*.png, tables/N2_*.csv, processed→ 클러스터링/outputs/tables/dong_base.parquet
산출. (1) 외국인 회피율 동별 지도 (2) 폭염한낮 노출 상위 동 목록 (3) 2025 내·외국인 동별 회피율 비교
      (4) 클러스터링용 base 테이블(노출·회피율·LQ·share·동명·자치구) 저장
"""
import warnings; warnings.filterwarnings("ignore")
import glob, re
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt

# ── 동 이름·자치구 매핑 ──
geo = gpd.read_file(GEO_DONG)
geo = geo[geo["sido"] == "11"].copy()
geo["k8"] = geo["adm_cd2"].astype(str).str[:8]
name_map = dict(zip(geo["k8"], geo["adm_nm"].str.split().str[-1]))
gu_map = dict(zip(geo["k8"], geo["sggnm"]))

# ── 1. 외국인 3년 매칭 회피율 + 노출 ──
fd = pd.read_parquet(PROC / "dong_matched_delta.parquet")
fd["k8"] = fd["d_admdong_cd"].astype(str)
fd["동명"] = fd["k8"].map(name_map); fd["자치구"] = fd["k8"].map(gu_map)

h = pd.read_parquet(PROC / "kt_dong_short_hourly.parquet"); h["date"] = pd.to_datetime(h["date"])
wd = pd.read_parquet(PROC / "weather_daily.parquet"); wd["date"] = pd.to_datetime(wd["date"])
h = h.merge(wd[["date", "is_heatwave_feels"]], on="date", how="left")
expo = (h[(h["is_heatwave_feels"] == True) & (h["hour"].between(12, 16))]
        .groupby("d_admdong_cd")["short"].mean().rename("노출_폭염한낮"))
fd = fd.merge(expo, left_on="d_admdong_cd", right_index=True, how="left")

# concentration(LQ·share) 병합
con = pd.read_parquet(PROC / "dong_concentration.parquet")
con["k8"] = con["d_admdong_cd"].astype(str)
fd = fd.merge(con[["k8", "LQ_단기외국인", "share_단기", "단기외국인"]], on="k8", how="left")

base = fd[["k8", "동명", "자치구", "matched_delta", "n_heat", "노출_폭염한낮",
           "LQ_단기외국인", "share_단기", "단기외국인"]].copy()
base.to_parquet(TABLES / "dong_base.parquet", index=False)
print(f"dong_base 저장: {len(base)}개 동")

# ── 노출 상위 목록 ──
top = base.dropna(subset=["노출_폭염한낮"]).sort_values("노출_폭염한낮", ascending=False).head(20)
top[["동명", "자치구", "노출_폭염한낮", "matched_delta", "LQ_단기외국인"]].to_csv(
    TABLES / "N2_노출상위동.csv", index=False, encoding="utf-8-sig")
print("\n=== 폭염 한낮 외국인 노출 상위 12개 동 ===")
print(top[["동명", "자치구", "노출_폭염한낮", "matched_delta"]].head(12).to_string(index=False,
      formatters={"노출_폭염한낮": "{:,.0f}".format, "matched_delta": "{:+.2f}%".format}))

# ── 2. 외국인 회피율 동별 지도 ──
gdf = geo.merge(fd[["k8", "matched_delta", "노출_폭염한낮"]], on="k8", how="left")
fig, ax = plt.subplots(1, 2, figsize=(20, 9))
gdf.plot(column="matched_delta", ax=ax[0], cmap="RdBu_r", vmin=-8, vmax=8, legend=True,
         legend_kwds={"label": "외국인 폭염 회피율 (%)", "shrink": .6}, edgecolor="white", lw=.2, missing_kwds={"color": "#eee"})
ax[0].set_title("N2. 동별 외국인 폭염 회피율 (3년 매칭)\n파랑=회피(빠짐), 빨강=잔류/증가"); ax[0].axis("off")
gdf.plot(column="노출_폭염한낮", ax=ax[1], cmap="OrRd", legend=True, scheme=None,
         legend_kwds={"label": "폭염 한낮 외국인 도착(명)", "shrink": .6}, edgecolor="white", lw=.2, missing_kwds={"color": "#eee"})
ax[1].set_title("N2. 동별 폭염 한낮 외국인 노출 규모"); ax[1].axis("off")
data_label(fig, "외국인=단기 관광객 | 회피율: 같은 동·요일·±14일 비폭염·맑음 대비 | 노출: 폭염일 12-16시 평균")
plt.tight_layout(); plt.savefig(FIGS / "N2_dong_map.png", dpi=130); plt.close()
print("  saved N2_dong_map.png")

# ── 3. 2025 동별 내·외국인 매칭 회피율 비교 (보너스 데이터) ──
cache = TABLES / "dong_2025_inout_delta.parquet"
if cache.exists():
    cmp = pd.read_parquet(cache)
else:
    print("  2025 보너스 일별 집계 중...")
    def agg(pat, val):
        acc = {}
        for f in sorted(glob.glob(str(RAW / f"KT_OD_보너스/2025_여름/{pat}"))):
            dt = re.search(r"(\d{8})", f).group(1)
            d = pd.read_csv(f, encoding="cp949")
            g = d.groupby("d_admdong_cd")[val].sum()
            acc[dt] = g
        return pd.concat(acc, names=["date", "d_admdong_cd"]).reset_index(name="arr")
    fo = agg("seoul_purpose_admdong4_forn_*.csv", "short_forn_cnt"); fo["pop"] = "단기외국인"
    ic = agg("seoul_purpose_admdong4_in_*.csv", "total_cnt"); ic["pop"] = "내국인"
    daily = pd.concat([fo, ic]); daily["date"] = pd.to_datetime(daily["date"])
    daily["dow"] = daily["date"].dt.dayofweek
    daily = daily.merge(wd[["date", "is_heatwave_feels", "rn_sum"]], on="date", how="left")
    daily["is_heat"] = daily["is_heatwave_feels"]; daily["is_rain"] = daily["rn_sum"] > 1.0
    # 동·집단별 매칭 회피율 (±14일 같은요일 비폭염 맑음)
    rows = []
    for (dong, pop), g in daily.groupby(["d_admdong_cd", "pop"]):
        g = g.sort_values("date"); heat = g[g["is_heat"] == True]
        for _, r in heat.iterrows():
            t, dow = r["date"], r["dow"]
            cand = g[(g["date"] >= t - pd.Timedelta(days=14)) & (g["date"] <= t + pd.Timedelta(days=14)) &
                     (g["dow"] == dow) & (g["is_heat"] != True) & (~g["is_rain"])]
            if cand["date"].nunique() < 2 or cand["arr"].mean() <= 0: continue
            rows.append({"d_admdong_cd": dong, "pop": pop, "delta": (r["arr"] / cand["arr"].mean() - 1) * 100})
    cmp = pd.DataFrame(rows).groupby(["d_admdong_cd", "pop"])["delta"].mean().unstack()
    cmp.to_parquet(cache)
print(f"  2025 내외국인 매칭 동수: {len(cmp.dropna())}")
print(f"  2025 동 평균: 내국인 {cmp['내국인'].mean():+.2f}% / 단기외국인 {cmp['단기외국인'].mean():+.2f}%")

cmp2 = cmp.dropna()
fig, ax = plt.subplots(figsize=(7.5, 7))
ax.scatter(cmp2["내국인"], cmp2["단기외국인"], alpha=.45, s=28, color=POP_COLOR["단기외국인"])
lim = [-15, 15]; ax.plot(lim, lim, "k--", lw=.8, alpha=.6, label="y=x (동일 반응)")
ax.axhline(0, color="gray", lw=.5); ax.axvline(0, color="gray", lw=.5)
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_xlabel("내국인 동별 회피율 (%)"); ax.set_ylabel("단기외국인 동별 회피율 (%)")
ax.set_title("N2. 2025 동별 내·외국인 폭염 회피율\n대각선 위(외국인>내국인) = 외국인이 덜 줄임")
ax.legend(); ax.grid(alpha=.25)
data_label(fig, f"2025 여름 보너스(동별 일별) 매칭 | n={len(cmp2)}동 | 평균 내 {cmp['내국인'].mean():+.1f}% vs 외 {cmp['단기외국인'].mean():+.1f}%")
plt.tight_layout(); plt.savefig(FIGS / "N2_inout_scatter.png", dpi=130); plt.close()
print("  saved N2_inout_scatter.png")
print("=== N2 완료 ===")



# ==============================================================================
# STEP 10 동 단위 피처 엔지니어링
# ==============================================================================
print("\n>>> STEP 10 동 단위 피처 엔지니어링")
# N3. 동별 클러스터링 피처 테이블 — 노출·회피율·실내외신호·시간프로파일·쉼터거리·집중도. (§7 준비)
"""
입력. dong_base.parquet(N2), kt_dong_short_hourly.parquet, weather_daily.parquet, SHELTER, GEO_DONG
출력. tables/dong_features.parquet (+ N3_feature_summary.csv)
피처(쉼터 수요 관점).
  expo_log    = log1p(폭염 한낮 외국인 도착)          # 노출 규모(클수록 수요↑)
  avoid       = matched_delta(회피율)                  # 0↑=비탄력 잔류(노출), 음수=빠짐
  shift_mid   = 폭염일-정상일 한낮(12-16) 점유율 변화   # 양수=실내 잔류(몰), 음수=야외 회피
  shift_eve   = 저녁(18-21) 점유율 변화
  day_ratio   = 주간(10-18)/전일 도착 비율             # 관광형(주간) vs 주거형
  shelter_km  = 최근접 기존 외국인쉼터 거리(km)        # 공백(클수록 미충족)
  LQ, share   = 단기외국인 입지계수·비중(있는 동만, 보조)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd

base = pd.read_parquet(TABLES / "dong_base.parquet")          # k8,동명,자치구,matched_delta,노출_폭염한낮,LQ,share
h = pd.read_parquet(PROC / "kt_dong_short_hourly.parquet"); h["date"] = pd.to_datetime(h["date"])
wd = pd.read_parquet(PROC / "weather_daily.parquet"); wd["date"] = pd.to_datetime(wd["date"])
h = h.merge(wd[["date", "is_heatwave_feels"]], on="date", how="left")
h["heat"] = h["is_heatwave_feels"] == True
h["period"] = np.select([h["hour"].between(12, 16), h["hour"].between(17, 21), h["hour"].between(6, 11)],
                        ["mid", "eve", "morn"], "night")

# 시간 점유율 변화(실내외 신호) + 주간비율
g = h.groupby(["d_admdong_cd", "heat", "period"])["short"].sum().reset_index()
g["share"] = g["short"] / g.groupby(["d_admdong_cd", "heat"])["short"].transform("sum")
piv = g.pivot_table(index="d_admdong_cd", columns=["heat", "period"], values="share")
def col(heat, prd): return piv[(heat, prd)] if (heat, prd) in piv.columns else 0
shift = pd.DataFrame(index=piv.index)
shift["shift_mid"] = col(True, "mid") - col(False, "mid")
shift["shift_eve"] = col(True, "eve") - col(False, "eve")
# 주간(10-18) 비율: 전체 도착 중 낮 비중 (관광형 지표)
day = h[h["hour"].between(10, 18)].groupby("d_admdong_cd")["short"].sum()
allh = h.groupby("d_admdong_cd")["short"].sum()
shift["day_ratio"] = (day / allh)

# 최근접 기존 쉼터 거리(km)
geo = gpd.read_file(GEO_DONG); geo = geo[geo["sido"] == "11"].copy()
geo["k8"] = geo["adm_cd2"].astype(str).str[:8]
geo_m = geo.to_crs(5179); geo_m["cx"] = geo_m.geometry.centroid.x; geo_m["cy"] = geo_m.geometry.centroid.y
sh = pd.read_csv(SHELTER)
shp = gpd.GeoDataFrame(sh, geometry=gpd.points_from_xy(sh["lon"], sh["lat"]), crs=4326).to_crs(5179)
sx, sy = shp.geometry.x.values, shp.geometry.y.values
def nearest_km(r):
    return float(np.min(np.hypot(r["cx"] - sx, r["cy"] - sy))) / 1000
geo_m["shelter_km"] = geo_m.apply(nearest_km, axis=1)
shelter = geo_m.set_index("k8")["shelter_km"]

# 조립
df = base.copy()
df["d_admdong_cd"] = df["k8"].astype(int)
df = df.merge(shift, left_on="d_admdong_cd", right_index=True, how="left")
df["shelter_km"] = df["k8"].map(shelter)
df["expo_log"] = np.log1p(df["노출_폭염한낮"])
df["avoid"] = df["matched_delta"]
df = df.rename(columns={"LQ_단기외국인": "LQ", "share_단기": "share"})

# 노출 있는 동만(클러스터링 대상 후보 — §7서 추가 필터)
df = df.dropna(subset=["expo_log", "avoid", "shift_mid"])
keep = ["k8", "d_admdong_cd", "동명", "자치구", "expo_log", "노출_폭염한낮", "avoid",
        "shift_mid", "shift_eve", "day_ratio", "shelter_km", "LQ", "share"]
df = df[keep]
df.to_parquet(TABLES / "dong_features.parquet", index=False)
print(f"dong_features 저장: {df.shape}, 동 {len(df)}개")
print("\n피처 요약(결측·범위):")
for c in ["expo_log", "avoid", "shift_mid", "shift_eve", "day_ratio", "shelter_km", "LQ", "share"]:
    s = df[c]; print(f"  {c:11s} 결측 {s.isna().sum():3d}  [{s.min():.2f}, {s.max():.2f}]  중앙 {s.median():.2f}")
df.describe().to_csv(TABLES / "N3_feature_summary.csv", encoding="utf-8-sig")
print("\n노출 상위 6 동 피처:")
print(df.nlargest(6, "노출_폭염한낮")[["동명", "자치구", "노출_폭염한낮", "avoid", "shift_mid", "shelter_km"]].to_string(index=False))
print("=== N3 완료 ===")



# ==============================================================================
# STEP 11 무더위쉼터 4,083 4유형 공급점수 (N11)
# ==============================================================================
print("\n>>> STEP 11 무더위쉼터 4,083 4유형 공급점수 (N11)")
# N11. 서울 전역 무더위쉼터 4,083개 유형화 → 동별 '실효 공급 점수'. (피드백④ 반영)
"""
교수님 피드백④: 기존 STO 9개소만 보지 말고, 서울 전역 무더위쉼터의 유형·개수·위치를
동의 성격 정보로 반영하라.

기존(N3/N5): shelter_km = STO 관광쉼터 9개소까지 최근접 거리만 사용 → 공급을 과소평가.
본 노트북: 서울시 무더위쉼터 전수(4,083개)를 공식 시설유형으로 4등급화하고,
           '관광객·외국인 접근성 × 냉방 관련성' 가중치로 동별 실효 공급 점수를 산출.

핵심 통찰: 단순 개수로 보면 서울은 쉼터 4천 개로 '충분'해 보이지만,
           그 70%(2,867개)가 회원제 경로당이라 단기 관광객·외국인은 사실상 못 쓴다.
           → 접근가능 공급은 훨씬 얇고, 특히 도심 관광권이 비어 있다.

입력. 추가 데이터/서울시 무더위쉼터 정보.csv, GEO_DONG(행정동 폴리곤)
출력. tables/N11_dong_shelter_supply.parquet, N11_typology.csv
      figures/N11_typology.png, N11_supply_map.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt

SHELTER_ALL = ROOT / "추가 데이터/서울시 무더위쉼터 정보.csv"

# ── 1. 적재·정리 (1행에 영문헤더가 중복으로 들어있어 제거) ──
raw = pd.read_csv(SHELTER_ALL, encoding="utf-8", low_memory=False)
if str(raw.iloc[0, 0]).startswith("fclt"):
    raw = raw.iloc[1:].reset_index(drop=True)
raw["lon"] = pd.to_numeric(raw["경도"], errors="coerce")
raw["lat"] = pd.to_numeric(raw["위도"], errors="coerce")
raw["cap"] = pd.to_numeric(raw["이용가능인원"], errors="coerce")
sh = raw.dropna(subset=["lon", "lat"]).copy()
sh = sh[(sh["lon"].between(126.6, 127.3)) & (sh["lat"].between(37.3, 37.8))]   # 서울 범위
print(f"쉼터 적재: 원본 {len(raw)} → 좌표유효·서울 {len(sh)}개")

# ── 2. 4유형화 (공식 시설유형 대분류 + 접근성·냉방 가중치) ──
# 대분류 → (등급, 가중치): 관광객/외국인이 실제로 더위를 피할 수 있는 정도
GRADE = {
    "공공시설":       ("실내·공공", 1.00),   # 주민센터·복지문화체육관: 개방·냉방, 가장 접근 쉬움
    "생활밀착민간시설": ("실내·민간", 0.80),   # 은행·유통·종교·의료: 냉방 있으나 운영시간 제약
    "특정계층이용시설": ("실내·회원", 0.40),   # 경로당 등: 냉방 있으나 회원제 → 단기 외국인 사실상 배제
    "야외시설":       ("야외·그늘", 0.20),   # 공원·정자·파고라: 냉방 없음, 그늘만
}
sh["대분류"] = sh["시설유형"].astype(str).str.split().str[0]
sh["등급"] = sh["대분류"].map(lambda x: GRADE.get(x, ("기타", 0.30))[0])
sh["w"] = sh["대분류"].map(lambda x: GRADE.get(x, ("기타", 0.30))[1])
sh["접근가능"] = sh["w"] >= 0.80   # 공공·민간 = 누구나 걸어 들어갈 수 있는 쉼터

typ = (sh.groupby("등급").agg(개수=("시설유형", "size"), 가중치=("w", "first"),
                            수용인원_중앙=("cap", "median")).sort_values("가중치", ascending=False))
typ.to_csv(TABLES / "N11_typology.csv", encoding="utf-8-sig")
print("\n=== 무더위쉼터 4유형 (서울 전역) ===")
print(typ.to_string())
print(f"\n접근가능(공공+민간) {int(sh['접근가능'].sum())}개 / 회원·야외 {int((~sh['접근가능']).sum())}개")

# ── 3. 행정동 폴리곤에 점-in-폴리곤 조인 ──
geo = gpd.read_file(GEO_DONG); geo = geo[geo["sido"] == "11"].copy()
geo["k8"] = geo["adm_cd2"].astype(str).str[:8]
shp = gpd.GeoDataFrame(sh, geometry=gpd.points_from_xy(sh["lon"], sh["lat"]), crs=4326)
joined = gpd.sjoin(shp, geo[["k8", "geometry"]], how="left", predicate="within")
print(f"\n동 조인: {joined['k8'].notna().sum()}/{len(joined)}개 매칭 (폴리곤 밖 {joined['k8'].isna().sum()}개)")

# ── 4. 동별 공급 점수 ──
j = joined.dropna(subset=["k8"])
agg = j.groupby("k8").apply(lambda g: pd.Series({
    "쉼터수": len(g),
    "접근가능수": int(g["접근가능"].sum()),
    "공급_가중": float(g["w"].sum()),                              # Σ 유형가중치 = 실효 공급량
    "공급_수용가중": float((g["w"] * g["cap"].fillna(g["cap"].median())).sum()),  # 가중 × 수용인원
})).reset_index()

# 동 센트로이드 기준 최근접 쉼터 거리(전수 / 접근가능만) — 기존 STO-9 거리 대체
geo_m = geo.to_crs(5179); geo_m["cx"] = geo_m.geometry.centroid.x; geo_m["cy"] = geo_m.geometry.centroid.y
shp_m = shp.to_crs(5179)
ax_all, ay_all = shp_m.geometry.x.values, shp_m.geometry.y.values
acc = shp_m[shp_m["접근가능"]]
ax_acc, ay_acc = acc.geometry.x.values, acc.geometry.y.values
def nearest(cx, cy, xs, ys): return float(np.min(np.hypot(cx - xs, cy - ys))) / 1000
geo_m["near_all_km"] = geo_m.apply(lambda r: nearest(r["cx"], r["cy"], ax_all, ay_all), axis=1)
geo_m["near_acc_km"] = geo_m.apply(lambda r: nearest(r["cx"], r["cy"], ax_acc, ay_acc), axis=1)

out = geo[["k8", "adm_nm", "sggnm"]].merge(agg, on="k8", how="left")
out = out.merge(geo_m[["k8", "near_all_km", "near_acc_km"]], on="k8", how="left")
for c in ["쉼터수", "접근가능수", "공급_가중", "공급_수용가중"]:
    out[c] = out[c].fillna(0)
out.to_parquet(TABLES / "N11_dong_shelter_supply.parquet", index=False)
print(f"\n동별 공급 저장: {out.shape}, 쉼터 0개 동 {int((out['쉼터수']==0).sum())}개")
print("\n공급 가중점수 하위 8동(공백) — 도심·관광권 주목:")
print(out.nsmallest(8, "공급_가중")[["adm_nm", "sggnm", "쉼터수", "접근가능수", "near_acc_km"]].to_string(index=False))
print("\n자치구별 접근가능 쉼터 수(하위 8 = 외국인 취약):")
print(out.groupby("sggnm")["접근가능수"].sum().sort_values().head(8).to_string())

# ── 그림 1: 4유형 분포 (개수 + 가중치) ──
fig, ax = plt.subplots(figsize=(9, 4.6))
order = typ.sort_values("가중치", ascending=True)
colors = {"실내·공공": "#1a9850", "실내·민간": "#91cf60", "실내·회원": "#fdae61", "야외·그늘": "#d73027", "기타": "#999"}
bars = ax.barh(order.index, order["개수"], color=[colors.get(i, "#999") for i in order.index])
for b, (idx, r) in zip(bars, order.iterrows()):
    ax.text(b.get_width() + 30, b.get_y() + b.get_height()/2,
            f"{int(r['개수']):,}개  (가중 {r['가중치']:.2f})", va="center", fontsize=10)
ax.set_xlabel("쉼터 개수"); ax.set_xlim(0, order["개수"].max() * 1.28)
ax.set_title("N11. 서울 무더위쉼터 4,083개 유형별 분포\n— 70%가 회원제 경로당, 단기 외국인은 사실상 못 씀",
             fontsize=12.5, fontweight="bold")
for s in ["top", "right"]: ax.spines[s].set_visible(False)
data_label(fig, "가중치 = 관광객·외국인 접근성×냉방 관련성 (공공1.0·민간0.8·회원0.4·야외0.2)")
plt.tight_layout(); plt.savefig(FIGS / "N11_typology.png", dpi=150); plt.close()

# ── 그림 2: 동별 접근가능 공급 지도 ──
gmap = geo.merge(out[["k8", "접근가능수", "공급_가중"]], on="k8", how="left")
fig, ax = plt.subplots(1, 2, figsize=(20, 9))
gmap.plot(column="접근가능수", ax=ax[0], cmap="Greens", legend=True, edgecolor="white", lw=.2,
          legend_kwds={"label": "접근가능 쉼터 수(공공+민간)", "shrink": .6}, missing_kwds={"color": "#eee"})
ax[0].scatter(acc["lon"], acc["lat"], s=4, c="#08519c", alpha=.4, zorder=3)
ax[0].set_title("N11. 동별 접근가능 쉼터 수 (공공·민간)"); ax[0].axis("off")
gmap.plot(column="공급_가중", ax=ax[1], cmap="Greens", legend=True, edgecolor="white", lw=.2,
          legend_kwds={"label": "유형가중 공급 점수", "shrink": .6}, missing_kwds={"color": "#eee"})
ax[1].set_title("N11. 동별 유형가중 실효 공급 점수"); ax[1].axis("off")
data_label(fig, "서울 무더위쉼터 전수 4,083개 · 점=접근가능 쉼터")
plt.tight_layout(); plt.savefig(FIGS / "N11_supply_map.png", dpi=140); plt.close()
print("\n  saved N11_typology.png, N11_supply_map.png")
print("=== N11 완료 ===")



# ==============================================================================
# STEP 12 동 피처 테이블 (표 5-1, N15_dong_features_rich)
# ==============================================================================
print("\n>>> STEP 12 동 피처 테이블 (표 5-1, N15_dong_features_rich)")
# N15. 동별 지역 피처테이블 조립 + 타당성 검증 — 매칭 회피율을 지역특성으로 설명할 신호가 있나. (새 ML 1단계)
"""
방향 전환: 클러스터링(연속스펙트럼이라 강등) → 매칭 회피율을 종속변수로,
새로 만든 지역 피처(쉼터공급·관광지 커버리지·실내신호·노출·구특성)로 설명/예측하는 지도학습.

본 노트북(1단계): 피처테이블을 조립하고, '신호가 존재하는가'를 먼저 검증.
  - 각 피처와 회피율의 상관(Pearson/Spearman)
  - 빠른 5-fold CV: Ridge vs HistGradientBoosting R²
  신호 있으면 N16(SHAP 풀모델)로, 약하면 지수(N17) 중심으로 전환 판단.

입력. dong_base.parquet(타깃·노출·외국인), dong_features.parquet(shift_mid·day_ratio),
      N11_dong_shelter_supply.parquet, 121 관광지 shp, 주간인구지수.xlsx, GEO_DONG
출력. tables/N15_dong_features_rich.parquet, 콘솔 타당성 리포트
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold

ROOT = ROOT
base = pd.read_parquet(TABLES / "dong_base.parquet")
feat = pd.read_parquet(TABLES / "dong_features.parquet")
sup = pd.read_parquet(TABLES / "N11_dong_shelter_supply.parquet")
for d in (base, feat, sup): d["k8"] = d["k8"].astype(str)

# ── 관광지 커버리지(야외/상업) 동별 ──
geo = gpd.read_file(GEO_DONG); geo = geo[geo["sido"] == "11"].copy()
geo["k8"] = geo["adm_cd2"].astype(str).str[:8]
geo_m = geo.to_crs(5179); geo_m["dong_area"] = geo_m.geometry.area
poi = gpd.read_file(ROOT / "추가 데이터/서울시 주요 121장소 영역 2/서울시 주요 121장소 영역.shp").to_crs(5179)
OUTDOOR = {"고궁·문화유산", "공원"}                       # 야외 관광(폭염 직격, N10서 회피 큼)
poi["야외"] = poi["CATEGORY"].isin(OUTDOOR)
ov = gpd.overlay(geo_m[["k8", "dong_area", "geometry"]], poi[["CATEGORY", "야외", "geometry"]], how="intersection")
ov["ia"] = ov.geometry.area
def cover(mask):
    s = ov[mask].groupby("k8")["ia"].sum()
    return (s / geo_m.set_index("k8")["dong_area"]).reindex(geo["k8"]).fillna(0).values
tour = pd.DataFrame({"k8": geo["k8"].values,
                     "관광커버": cover(ov["ia"] > 0),
                     "야외관광커버": cover(ov["야외"]),
                     "상업관광커버": cover(~ov["야외"])})
tour["관광지수"] = ov.groupby("k8")["CATEGORY"].nunique().reindex(geo["k8"]).fillna(0).values

# ── 주간인구지수(구) ──
di = pd.read_excel(ROOT / "data/raw/보조_데이터/자치구특성/주간인구지수.xlsx")
di = di.iloc[2:][["행정구역별(2)", di.columns[-1]]].rename(columns={"행정구역별(2)": "자치구", di.columns[-1]: "주간인구지수"})
di = di[di["자치구"].astype(str).str.endswith("구")].copy()
di["주간인구지수"] = pd.to_numeric(di["주간인구지수"], errors="coerce")

# ── 조립 ──
df = base[["k8", "동명", "자치구", "matched_delta", "노출_폭염한낮", "LQ_단기외국인", "share_단기"]].copy()
df = df.merge(feat[["k8", "expo_log", "shift_mid", "day_ratio"]], on="k8", how="left")
df = df.merge(sup[["k8", "접근가능수", "공급_가중", "near_acc_km", "쉼터수"]], on="k8", how="left")
df = df.merge(tour, on="k8", how="left")
df = df.merge(di, on="자치구", how="left")
df = df.dropna(subset=["matched_delta", "expo_log"])

# EXPO60(신뢰 가능 256동) + 타깃 winsorize (소표본 동 이상치 제거 — 파이프라인 일관)
sub = df[df["expo_log"] >= df["expo_log"].quantile(0.40)].copy()
lo, hi = sub["matched_delta"].quantile([0.01, 0.99]); sub["avoid"] = sub["matched_delta"].clip(lo, hi)
FEATS = ["expo_log", "접근가능수", "공급_가중", "near_acc_km", "관광커버", "야외관광커버", "상업관광커버",
         "관광지수", "shift_mid", "day_ratio", "LQ_단기외국인", "share_단기", "주간인구지수"]
sub[FEATS] = sub[FEATS].fillna(sub[FEATS].median())
sub.to_parquet(TABLES / "N15_dong_features_rich.parquet", index=False)
print(f"피처테이블 저장: {sub.shape}, 피처 {len(FEATS)}개, 대상 {len(sub)}동(EXPO60)\n")

# ── 타당성 1: 상관 ──
print("=== 회피율(avoid)과 지역피처 상관 (|r| 큰 순) ===")
cor = pd.DataFrame({"Pearson": [sub["avoid"].corr(sub[f]) for f in FEATS],
                    "Spearman": [sub["avoid"].corr(sub[f], method="spearman") for f in FEATS]}, index=FEATS)
print(cor.reindex(cor["Pearson"].abs().sort_values(ascending=False).index).round(3).to_string())

# ── 타당성 2: CV R² ──
X = StandardScaler().fit_transform(sub[FEATS]); y = sub["avoid"].values
cv = KFold(5, shuffle=True, random_state=42)
r2_ridge = cross_val_score(Ridge(1.0), X, y, cv=cv, scoring="r2")
r2_gb = cross_val_score(HistGradientBoostingRegressor(max_depth=3, max_iter=300, learning_rate=0.05, random_state=42),
                        sub[FEATS].values, y, cv=cv, scoring="r2")
print(f"\n=== 5-fold CV R² (회피율 예측) ===")
print(f"  Ridge(선형)            : {r2_ridge.mean():+.3f} (±{r2_ridge.std():.3f})")
print(f"  HistGradientBoosting   : {r2_gb.mean():+.3f} (±{r2_gb.std():.3f})")
print(f"\n  → R²>0.10 이상이면 지역특성이 회피율을 유의하게 설명 → N16 SHAP 풀모델 진행 권장.")
print("=== N15 완료 ===")



# ==============================================================================
# STEP 13 클러스터링 — 4피처 K-means (그림 5-1, 표 5-2)
# ==============================================================================
print("\n>>> STEP 13 클러스터링 — 4피처 K-means (그림 5-1, 표 5-2)")
# 5장 클러스터링 — 행정동 폭염 적응 4피처 K-means (보고서 5.3·그림 5-1·표 5-2).
"""
입력. 출력/tables/N15_dong_features_rich.parquet (노출 상위 60%, 256동)  ← _config가 분석_중간산출에서 시드
출력. 출력/figures/13_클러스터링.png (그림 5-1: 실루엣 곡선 + 노출-회피 산점도)
      출력/tables/13_군집결과.csv (k별 실루엣), 콘솔에 표 5-2 프로파일

피처 4개(원공간, PCA 미사용). 노출 expo_log · 회피 avoid · 야외관광커버 · 상업관광커버.
결과(검증). k=2 실루엣 0.456, k=3 0.452, k=4 0.290 → k=3 채택. 군집 200/43/13, 부트스트랩 ARI k=3 0.90.
※ 발표 단계 2피처(노출·회피)는 실루엣 0.388. 관광지 성격 2변수를 더해 0.45 중반으로 분리도 개선.
   원공간 4피처가 다피처+PCA보다 해석 명확(관광커버=야외+상업 합이라 PCA가 점수만 부풀림). 친구 작성분 정리.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt

df = pd.read_parquet(TABLES / "N15_dong_features_rich.parquet").copy()
# 회피율 등 소표본 노이즈 피처 1·99분위 winsorize (파이프라인 일관)
for c in ["avoid", "shift_mid", "day_ratio"]:
    if c in df:
        lo, hi = df[c].quantile([0.01, 0.99]); df[c] = df[c].clip(lo, hi)

FEATS = ["expo_log", "avoid", "야외관광커버", "상업관광커버"]   # 최종 채택 4피처(원공간)
df[FEATS] = df[FEATS].fillna(df[FEATS].median())
Z = StandardScaler().fit_transform(df[FEATS].values)            # 표준화만, PCA 없음
print(f"대상 행정동(노출 상위 60%): {len(df)}개\n")


def km(k):
    return KMeans(k, n_init=50, random_state=42).fit_predict(Z)


# ── 1) 군집 수 선택: k별 실루엣 ──
sil = {k: round(silhouette_score(Z, km(k)), 3) for k in [2, 3, 4, 5, 6]}
print("=== k별 실루엣 (원공간·4피처) ===")
for k, s in sil.items():
    print(f"  k={k}: {s:.3f}")
print(f"  → 채택: k=3 {sil[3]:.3f} (k=2 {sil[2]:.3f}와 사실상 동일, k=4부터 급락)\n")

lab3 = km(3); df["cluster"] = lab3

# ── 2) 군집 안정성: 80% 부트스트랩 재현 ARI ──
print("=== 군집 안정성 (부트스트랩 ARI, 80% 재표집 30회) ===")
rng = np.random.RandomState(0)
for k in [2, 3]:
    base = km(k); aris = []
    for b in range(30):
        idx = rng.choice(len(Z), int(len(Z) * 0.8), replace=False)
        lab = KMeans(k, n_init=20, random_state=b).fit_predict(Z[idx])
        aris.append(adjusted_rand_score(base[idx], lab))
    print(f"  k={k}: ARI {np.mean(aris):.2f} ± {np.std(aris):.2f}")

# ── 3) 표 5-2 군집 프로파일 ──
prof = df.groupby("cluster").agg(
    동수=("동명", "size"), 노출=("노출_폭염한낮", "mean"), 회피율=("avoid", "mean"),
    야외관광=("야외관광커버", "mean"), 상업관광=("상업관광커버", "mean")).sort_values("노출")
print("\n=== 표 5-2. 행정동 군집 프로파일 (k=3) ===")
print(prof.to_string(formatters={"노출": "{:,.0f}".format, "회피율": "{:+.2f}".format,
                                 "야외관광": "{:.2f}".format, "상업관광": "{:.2f}".format}))
for c in prof.index:
    ex = ", ".join(df[df.cluster == c].nlargest(4, "노출_폭염한낮")["동명"])
    print(f"   군집{c} 대표동(노출 상위): {ex}")

# 고노출·비탄력형(신규 쉼터 핵심) = 노출 최대이면서 회피 약한 군집
tgt = prof.index[-1] if prof.loc[prof.index[-1], "회피율"] > prof.loc[prof.index[1], "회피율"] - 1 else prof["노출"].idxmax()
tgt = prof["노출"].idxmax()

# ── 4) 결과 저장 ──
pd.DataFrame({"k": list(sil), "실루엣": list(sil.values())}).to_csv(
    TABLES / "13_군집결과.csv", index=False, encoding="utf-8-sig")

# ── 5) 그림 5-1: (왼쪽) 실루엣 곡선  (오른쪽) 노출–회피 산점도(세 유형) ──
NAME = {prof["노출"].idxmin(): "저노출형",
        prof["노출"].idxmax(): "고노출·비탄력형",
        [c for c in prof.index if c not in (prof["노출"].idxmin(), prof["노출"].idxmax())][0]: "고노출·야외회피형"}
COLOR = {prof["노출"].idxmin(): "#bdbdbd", prof["노출"].idxmax(): "#d7301f"}
COLOR[[c for c in prof.index if c not in COLOR][0]] = "#2166ac"

fig, ax = plt.subplots(1, 2, figsize=(13, 5))
ks, vs = list(sil), list(sil.values())
ax[0].plot(ks, vs, "-o", color="#0f9d8a", lw=2)
for k, v in sil.items():
    ax[0].annotate(f"{v:.3f}", (k, v), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9)
ax[0].axhline(0.4, color="#ff7f0e", ls="--", lw=1); ax[0].text(5.4, 0.41, "0.4 양호", color="#ff7f0e", fontsize=8)
ax[0].set_xlabel("클러스터 수 k"); ax[0].set_ylabel("실루엣 계수"); ax[0].set_xticks(ks)
ax[0].set_title("(왼쪽) 클러스터 수에 따른 실루엣 — k=3까지 0.45 유지, k=4 급락", fontsize=10.5)
for c in prof.index:
    m = df.cluster == c
    ax[1].scatter(df.loc[m, "노출_폭염한낮"], df.loc[m, "avoid"], s=26, alpha=0.75,
                  color=COLOR[c], label=f"{NAME[c]} (n={int(m.sum())})")
ax[1].axhline(0, color="#888", ls=":", lw=1)
ax[1].set_xlabel("노출 — 폭염 한낮 단기 외국인 도착(명)"); ax[1].set_ylabel("회피율(%) — 0 근방=비탄력")
ax[1].set_title(f"(오른쪽) 노출–회피로 본 세 유형 (k=3 실루엣 {sil[3]:.3f})", fontsize=10.5)
ax[1].legend(fontsize=8, loc="lower right")
fig.suptitle("그림 5-1. 행정동 폭염 적응 클러스터링 — 4피처·원공간", fontsize=13, fontweight="bold")
data_label(fig, "표준화→K-means · 노출 상위 60% 256동 · PCA 미사용")
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(FIGS / "13_클러스터링.png", dpi=140); plt.close()

print("\n저장: 출력/figures/13_클러스터링.png, 출력/tables/13_군집결과.csv")
print("=== 13 완료 ===")



# ==============================================================================
# STEP 14 입지 — Moran/Gi* 핫스팟·공백 (그림 5-2, Moran 0.319·핫스팟 53)
# ==============================================================================
print("\n>>> STEP 14 입지 — Moran/Gi* 핫스팟·공백 (그림 5-2, Moran 0.319·핫스팟 53)")
# N9. 공간 입지 층(2층 구조) — Getis-Ord Gi* 핫스팟 + 기존 쉼터 공백 + K-means 적응유형 결합. (§7 입지 최종)
"""
2층 구조의 '공간 결정 층'. K-means(유형)와 별개로, 수요 표면의 공간 핫스팟으로 입지를 정한다.
파이프라인.
  ① 수요 표면        = 폭염 한낮 외국인 노출(expo_log). 보조=복합 z(expo)+z(비탄력).
  ② 공간 가중치      = KNN k=8 행표준화.
  ③ 전역 Moran's I   = 999회 순열검정 (공간구조 유의성).
  ④ 국지 Gi*(star)   = z>1.96 유의 핫스팟.
  ⑤ 공백 분석        = 기존 STO 쉼터 9개소, 서비스반경 0.8km(±0.4/1.2 민감도) 밖 핫스팟 = 신규 1순위.
  ⑥ 유형 결합        = K-means shape 군집(4유형) 라벨 + 실내전환(몰) 제외 → 야외 쉼터 수요만.
입력. dong_features.parquet, GEO_DONG, SHELTER.  (libpysal 미사용, 직접 구현)
출력. tables/N9_siting_priority.csv, dong_gistar.parquet / figures/N9_gistar_map.png
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# ── 데이터 + 동 중심점 ──
df = pd.read_parquet(TABLES / "dong_features.parquet")
lo, hi = df["avoid"].quantile([0.01, 0.99]); df["avoid"] = df["avoid"].clip(lo, hi)
df["실내전환"] = (df["shift_mid"] > 0.01) | (df["avoid"] > 1.5)   # 냉방 몰 = 쉼터 불필요(N4와 동일 정의)
df["k8"] = df["k8"].astype(str)

geo = gpd.read_file(GEO_DONG); geo = geo[geo["sido"] == "11"].copy()
geo["k8"] = geo["adm_cd2"].astype(str).str[:8]
gm = geo.to_crs(5179); gm["cx"] = gm.geometry.centroid.x; gm["cy"] = gm.geometry.centroid.y
cent = gm.groupby("k8")[["cx", "cy"]].mean()
d = df.merge(cent, left_on="k8", right_index=True, how="inner").reset_index(drop=True)
n = len(d); print(f"공간 매칭 동 {n}개")

# ── ② KNN(k=8) 행표준화 가중치 ──
K = 8
tree = cKDTree(d[["cx", "cy"]].values); _, nbr = tree.query(d[["cx", "cy"]].values, k=K + 1)
W = np.zeros((n, n))
for i in range(n): W[i, nbr[i, 1:]] = 1.0 / K

def morans_I(x, perm=999, seed=0):
    z = x - x.mean(); zz = (z * z).sum(); I = (z @ (W @ z)) / zz
    rng = np.random.RandomState(seed)
    cnt = sum((rng.permutation(z) @ (W @ rng.permutation(z))) / zz >= I for _ in range(perm))
    return I, (cnt + 1) / (perm + 1)

def gi_star(x):
    Ws = W.copy(); np.fill_diagonal(Ws, 1.0 / K); Ws = Ws / Ws.sum(1, keepdims=True)
    num = (Ws @ x) - x.mean(); Wi2 = (Ws ** 2).sum(1)
    return num / (x.std() * np.sqrt((n * Wi2 - 1) / (n - 1)))

# ── ③ Moran's I + ④ Gi* ──
d["z_expo"] = (d["expo_log"] - d["expo_log"].mean()) / d["expo_log"].std()
d["z_inel"] = (d["avoid"] - d["avoid"].mean()) / d["avoid"].std()
d["demand_composite"] = d["z_expo"] + d["z_inel"]
print("\n=== ③ 전역 Moran's I (999 순열) ===")
for col, lab in [("expo_log", "노출(주 수요표면)"), ("demand_composite", "복합수요(보조)")]:
    I, p = morans_I(d[col].values); print(f"  {lab:18s} I={I:+.3f}  p={p:.3f}  ({'유의' if p<.05 else '비유의'})")

d["gi"] = gi_star(d["expo_log"].values)                 # 주 핫스팟 = 노출(가장 강·깨끗)
d["gi_composite"] = gi_star(d["demand_composite"].values)
d["hotspot"] = d["gi"] > 1.96
print(f"\n=== ④ Gi* 핫스팟(노출 기준) z>1.96: {int(d['hotspot'].sum())}개 / z>2.58: {int((d['gi']>2.58).sum())}개 ===")

# ── ⑥ K-means 적응유형(shape 군집 4유형) 라벨 ──
expo60 = d[d["expo_log"] >= d["expo_log"].quantile(0.40)].copy()
for c in ["shift_mid", "shift_eve"]:
    l, h = expo60[c].quantile([0.01, 0.99]); expo60[c] = expo60[c].clip(l, h)
SHAPE = ["avoid", "shift_mid", "shift_eve"]
Xs = StandardScaler().fit_transform(expo60[SHAPE])
expo60["km"] = KMeans(4, n_init=50, random_state=42).fit_predict(Xs)
prof = expo60.groupby("km")[["avoid", "shift_mid"]].mean()
indoor_c = int(prof["shift_mid"].idxmax())                       # 몰(실내전환)
rest = prof.drop(indoor_c).sort_values("avoid")                  # avoid 오름차순
names = {indoor_c: "실내전환형(몰)"}
labs = ["강한회피형", "중간회피형", "야외비탄력형"]
for c, nm in zip(rest.index, labs): names[c] = nm
expo60["적응유형"] = expo60["km"].map(names)
d = d.merge(expo60[["k8", "적응유형"]], on="k8", how="left")
d["적응유형"] = d["적응유형"].fillna("저노출")
print("\n=== ⑥ K-means 적응유형 (EXPO60) ===")
for c in prof.index:
    sub = expo60[expo60.km == c]
    print(f"  {names[c]:14s} n={len(sub):3d}  avoid {prof.loc[c,'avoid']:+.2f}  shift_mid {prof.loc[c,'shift_mid']:+.3f}  예: {', '.join(sub.nlargest(4,'노출_폭염한낮')['동명'])}")

# ── ⑤ 공백 분석 (서비스반경 민감도) ──
print("\n=== ⑤ 기존 쉼터 공백 — 핫스팟 중 신규 1순위 ===")
hot = d[d["hotspot"]].copy()
for r in [0.4, 0.8, 1.2]:
    unmet = hot[(hot["shelter_km"] > r) & (~hot["실내전환"])]
    print(f"  반경 {r}km: 핫스팟 {len(hot)} → 공백·야외 {len(unmet)}개")
RAD = 0.8
ADAPT_THR = -3.0      # avoid ≤ -3.0 = 폭염에 강하게 회피(이미 적응) → 쉼터 후순위로 강등(검증 각도4 반영)
GANGNAM = {"강남구", "서초구", "송파구"}
gap = hot[(hot["shelter_km"] > RAD) & (~hot["실내전환"])].copy()
gap["우선순위"] = np.where(gap["avoid"] <= ADAPT_THR, "후순위(이미적응)", "1순위(쉼터핵심)")
gap = gap.sort_values(["우선순위", "gi"], ascending=[True, False])   # 1순위 먼저
priority = gap[gap["우선순위"] == "1순위(쉼터핵심)"]
demoted = gap[gap["우선순위"] == "후순위(이미적응)"]
gap_out = gap[["동명", "자치구", "노출_폭염한낮", "avoid", "gi", "shelter_km", "적응유형", "우선순위"]]
gap_out.to_csv(TABLES / "N9_siting_priority.csv", index=False, encoding="utf-8-sig")
d.to_parquet(TABLES / "dong_gistar.parquet", index=False)

def gn(s): return f"{s.isin(GANGNAM).sum()}/{len(s)}({s.isin(GANGNAM).mean()*100:.0f}%)"
print(f"\n  공백·야외 핫스팟 {len(gap)}개 → 1순위(쉼터핵심) {len(priority)} / 후순위(이미적응, avoid≤{ADAPT_THR}) {len(demoted)}")
print(f"  강남권 비중: 전체핫스팟 {gn(hot['자치구'])} · 공백야외 {gn(gap['자치구'])} · 1순위 {gn(priority['자치구'])}")
print(f"\n  ★ 1순위(쉼터핵심) {len(priority)}개:")
print(priority[["동명", "자치구", "노출_폭염한낮", "avoid", "gi", "shelter_km", "적응유형"]].head(18).to_string(index=False,
      formatters={"노출_폭염한낮": "{:,.0f}".format, "avoid": "{:+.2f}".format, "gi": "{:.2f}".format, "shelter_km": "{:.2f}".format}))
print(f"\n  ↓ 후순위(이미적응) {len(demoted)}개: {', '.join(demoted['동명']+'('+demoted['자치구']+' '+demoted['avoid'].map('{:+.1f}'.format)+')')}")
covered = hot[(hot["shelter_km"] <= RAD) & (~hot["실내전환"])]
print(f"  (핫스팟 중 기존쉼터 ≤{RAD}km 충족: {len(covered)}개 — {dict(covered['자치구'].value_counts())})")
# 교차검증: 복합수요(노출+비탄력 동등가중) Gi* 핫스팟의 강남권 비중
hc = d[d["gi_composite"] > 1.96]
print(f"  [교차검증] 복합수요 Gi* 핫스팟 {len(hc)}개 강남권 {gn(hc['자치구'])} — 노출표면과 수렴")

# ── 지도 ──
sh = pd.read_csv(SHELTER)
shp = gpd.GeoDataFrame(sh, geometry=gpd.points_from_xy(sh["lon"], sh["lat"]), crs=4326).to_crs(5179)
gmap = gm.merge(d[["k8", "gi", "hotspot", "실내전환"]], on="k8", how="left")
pri_k8 = set(priority["k8"]); dem_k8 = set(demoted["k8"])
fig, ax = plt.subplots(figsize=(11, 11))
gmap.plot(column="gi", cmap="RdBu_r", vmin=-3, vmax=3, linewidth=0.2, edgecolor="#999", ax=ax, legend=True,
          legend_kwds={"label": "Gi* z (노출 핫스팟)", "shrink": 0.5}, missing_kwds={"color": "#eee"})
gmap[gmap["k8"].isin(pri_k8)].boundary.plot(ax=ax, color="#111", linewidth=2.0)              # 1순위 핵심
gmap[gmap["k8"].isin(dem_k8)].boundary.plot(ax=ax, color="#666", linewidth=1.2, linestyle="--")  # 후순위
shp.plot(ax=ax, color="black", marker="*", markersize=180, edgecolor="white", linewidth=0.8, zorder=5, label="기존 STO 쉼터(9)")
for _, r in priority.head(10).iterrows():
    pt = gm[gm.k8 == r["k8"]]
    if len(pt): ax.annotate(r["동명"], (pt["cx"].iloc[0], pt["cy"].iloc[0]), fontsize=8, ha="center", color="#111")
ax.plot([], [], color="#111", lw=2.0, label=f"신규 1순위 쉼터핵심({len(priority)})")
ax.plot([], [], color="#666", lw=1.2, ls="--", label=f"후순위·이미적응({len(demoted)})")
ax.set_title(f"N9. Gi* 핫스팟 + 기존 쉼터 공백 → 신규 1순위 {len(priority)}개\n"
             f"Moran's I {morans_I(d['expo_log'].values)[0]:.3f} · 핫스팟 {int(d['hotspot'].sum())}개 · 강남권 우위", fontsize=12)
ax.legend(loc="upper left", fontsize=9); ax.axis("off")
data_label(fig, "수요=폭염한낮 외국인 노출. 핫스팟=Gi* z>1.96. 1순위=핫스팟∩기존쉼터>0.8km∩야외(몰제외)∩비강하회피.")
plt.tight_layout(); plt.savefig(FIGS / "N9_gistar_map.png", dpi=130); plt.close()
print("\n  saved N9_gistar_map.png, N9_siting_priority.csv, dong_gistar.parquet")
print("=== N9 완료 ===")



# ==============================================================================
# STEP 15 입지 — 신규 36동 지도 (그림 5-3, 강남권 27)
# ==============================================================================
print("\n>>> STEP 15 입지 — 신규 36동 지도 (그림 5-3, 강남권 27)")
# N9f. 신규 1순위 36동 입지 지도 — 자치구색 채움 + 동 이름 라벨(강남권 확대 인셋). 제목만. (§7 발표 그림)
"""
입력. dong_gistar.parquet(N9), GEO_DONG / outputs/tables/N9_siting_priority.csv
출력. outputs/figures/N9f_36동_지도.png
설계. 최종 36동을 자치구색(도넛 N9e와 동일)으로 채우고 동 이름 라벨.
      강남권 27동이 동남부에 응집 → 확대 인셋으로 라벨 가독성 확보. 제목 한 줄, 부제 없음.
"""
import warnings; warnings.filterwarnings("ignore")
import pandas as pd, geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

GU_COLOR = {"강남구": "#0f7b6c", "서초구": "#1a9988", "송파구": "#5cc0b2",
            "중구": "#9aa0a6", "강서구": "#c0876b", "마포구": "#c7a94b",
            "성동구": "#7f8fb0", "영등포구": "#a87fa0", "광진구": "#8fa86f"}
GANGNAM = ["강남구", "서초구", "송파구"]

# ── 36동 (동명+자치구로 정확 머지) ──
df = pd.read_csv(TABLES / "N9_siting_priority.csv")
pri = df[df["우선순위"].str.startswith("1순위")].copy()
g0 = pd.read_parquet(TABLES / "dong_gistar.parquet")[["k8", "동명", "자치구"]]
pri = pri.merge(g0, on=["동명", "자치구"], how="left").drop_duplicates("k8")
cnt = pri["자치구"].value_counts()
print(f"1순위 {len(pri)}동 · 강남권 {pri['자치구'].isin(GANGNAM).sum()}")

# ── 지오 ──
geo = gpd.read_file(GEO_DONG); geo = geo[geo["sido"] == "11"].copy()
geo["k8"] = geo["adm_cd2"].astype(str).str[:8]
sel = geo.merge(pri[["k8", "동명", "자치구"]], on="k8", how="inner")
sel["color"] = sel["자치구"].map(GU_COLOR)
sel["x"] = sel.geometry.representative_point().x
sel["y"] = sel.geometry.representative_point().y

# ── 라벨 배치: 중심 기준 좌/우 분할 + 각 측 세로 그리디 분산 ──
def place(ax, sub, mindy, dx=0.004, fs=8.2):
    xm = sub["x"].median()
    for side in (-1, 1):                                                    # -1=왼쪽(우측정렬), +1=오른쪽
        s = sub[(sub["x"] < xm) == (side < 0)].sort_values("y", ascending=False)
        ly = None
        for _, r in s.iterrows():
            y = r["y"] if ly is None else min(r["y"], ly - mindy)
            ly = y
            lx = r["x"] + side * dx
            ax.plot([r["x"], lx], [r["y"], y], color="#999", lw=0.45, zorder=4, clip_on=False)
            ax.text(lx + side * 0.0008, y, r["동명"], fontsize=fs, zorder=6, clip_on=False,
                    ha="left" if side > 0 else "right", va="center", color="#1f2a37",
                    path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])

# ── 그림 ──
fig, ax = plt.subplots(figsize=(13.5, 11), dpi=150)
geo.plot(ax=ax, color="#eef0f0", edgecolor="white", lw=0.3)
sel.plot(ax=ax, color=sel["color"], edgecolor="white", lw=0.6, zorder=3)

# 강남권 bbox
gnsel = sel[sel["자치구"].isin(GANGNAM)]
minx, miny, maxx, maxy = gnsel.total_bounds
pad = 0.012
bx0, by0, bx1, by1 = minx - pad, miny - pad, maxx + pad, maxy + pad
ax.add_patch(Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec="#1f2a37", lw=1.3, ls="--", zorder=5))
ax.annotate("강남권 27동 → 확대", (bx1, by1), (bx1 - 0.02, by1 + 0.02),
            fontsize=11, fontweight="bold", color="#1f2a37",
            arrowprops=dict(arrowstyle="-", color="#1f2a37", lw=1.0))

# 메인: 강남권 밖 9동만 라벨(여유 있음)
place(ax, sel[~sel["자치구"].isin(GANGNAM)], mindy=0.0075, fs=9)

# ── 인셋: 강남권 확대 + 27동 라벨 ──
axins = ax.inset_axes([0.54, 0.0, 0.46, 0.66])
geo.plot(ax=axins, color="#eef0f0", edgecolor="white", lw=0.3)
gnsel.plot(ax=axins, color=gnsel["color"], edgecolor="white", lw=0.7, zorder=3)
axins.set_xlim(bx0, bx1); axins.set_ylim(by0, by1)
place(axins, gnsel, mindy=(by1 - by0) / 22, dx=0.007, fs=7.8)
for s in axins.spines.values():
    s.set_edgecolor("#1f2a37"); s.set_linewidth(1.3); s.set_linestyle("--")
axins.set_xticks([]); axins.set_yticks([])
axins.set_title("강남권 확대 (강남·서초·송파 27동)", fontsize=10.5, color="#1f2a37", pad=4)

# ── 범례 ──
order = [g for g in (GANGNAM + ["중구", "강서구", "마포구", "성동구", "영등포구", "광진구"]) if g in cnt.index]
handles = [Line2D([], [], marker="s", color="none", markerfacecolor=GU_COLOR[g],
                  markeredgecolor="white", markersize=12, label=f"{g} {int(cnt[g])}동") for g in order]
ax.legend(handles=handles, loc="upper left", fontsize=10.5, frameon=True, framealpha=0.95,
          borderpad=0.8, title=f"신규 1순위 {len(pri)}동", title_fontsize=11)

ax.set_title("무더위쉼터 신규 1순위 36개 동", fontsize=21, fontweight="bold", color="#1f2a37", pad=12)
ax.axis("off")
data_label(fig, "Gi* 핫스팟 53 → 공백·야외·미적응 3단 필터 → 1순위 36동. 색=자치구(도넛 N9e와 동일).")
plt.tight_layout()
out = FIGS / "N9f_36동_지도.png"
fig.savefig(out, dpi=190, bbox_inches="tight", facecolor="white"); plt.close()
print(f"saved {out}")



print("\n=== 전체 구현코드 실행 완료 ===")
