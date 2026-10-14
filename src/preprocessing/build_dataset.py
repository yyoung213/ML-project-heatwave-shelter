"""전처리 파이프라인 진입점.

    python -m src.preprocessing.build_dataset

산출:
    processed/analysis_dataset.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from ..config import PROCESSED_DIR
from .loaders import load_lifepop
from .weather import add_derived, load_asos
from .panel import build_panel

log = logging.getLogger(__name__)


def _months_for_summers(years: list[int]) -> list[str]:
    return [f"{y}{m:02d}" for y in years for m in (6, 7, 8)]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2023, 2024, 2025])
    ap.add_argument("--out", type=str, default="analysis_dataset.parquet")
    args = ap.parse_args()

    months = _months_for_summers(args.years)
    log.info("target months: %s", months)

    lifepop = load_lifepop(months)
    log.info("lifepop: %d rows", len(lifepop))

    weather = add_derived(load_asos())
    log.info("weather: %d rows", len(weather))

    panel = build_panel(lifepop, weather)
    log.info("panel: %d rows  %d cols", len(panel), panel.shape[1])

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / args.out
    panel.to_parquet(out_path, index=False)
    log.info("wrote %s (%.1f MB)", out_path, out_path.stat().st_size / 1e6)

    # 빠른 무결성 점검
    print("\n=== 점검 ===")
    print(f"행: {len(panel):,}")
    print(f"동: {panel['dong'].nunique()}")
    print(f"일자 범위: {panel['dt'].min()} ~ {panel['dt'].max()}")
    print(f"일자 수: {panel['dt'].nunique()}")
    print("\n핵심 컬럼 결측률:")
    for c in ("foreign_total", "domestic", "base_foreign", "base_domestic",
              "ratio_foreign", "ratio_domestic", "gap", "ta", "apt"):
        miss = panel[c].isna().mean() * 100
        print(f"  {c:<18} {miss:5.2f}%")
    print("\ngap 분포 (전체):")
    print(panel["gap"].describe([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(3))
    print("\n폭염일(heatwave_apt=1) 평균 gap vs 평상일:")
    print(panel.groupby("heatwave_apt")["gap"].describe()[["count", "mean", "50%"]].round(3))


if __name__ == "__main__":
    main()
