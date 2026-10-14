"""기상청 ASOS 시간자료 (B-1) 수집기.

공공데이터포털: 기상청_지상(종관, ASOS) 시간자료 조회서비스 (publicDataPk=15057210)
엔드포인트: getWthrDataList

응답 필드(시간자료):
    tm   관측시각 YYYY-MM-DD HH:MM
    stnId 지점번호 (서울=108)
    ta   기온(℃)
    rn   강수량(mm)
    hm   습도(%)
    ws   풍속(m/s)
    ...
빈 값은 ""(공문자) 로 옴 — 결측 처리 필요.

사용:
    python -m src.collectors.kma_asos 20240701 20240731 --stn 108
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

from ..api_client import ApiError, get_json
from ..config import RAW_DIR, require_key

BASE_URL = "http://apis.data.go.kr/1360000/AsosHourlyInfoService/getWthrDataList"
OUT_DIR = RAW_DIR / "weather" / "asos_hourly"

log = logging.getLogger(__name__)


def _month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    cur = start.replace(day=1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        month_end = nxt - timedelta(days=1)
        yield max(cur, start), min(month_end, end)
        cur = nxt


def fetch_range(stn_id: int, from_date: date, to_date: date, page_size: int = 999) -> list[dict]:
    """주어진 기간(YYYYMMDD) 시간자료를 모두 수집. 페이지네이션 자동."""
    service_key = require_key("DATA_GO_KR_KEY")
    items: list[dict] = []
    page = 1
    while True:
        params = {
            "serviceKey": service_key,
            "pageNo": page,
            "numOfRows": page_size,
            "dataType": "JSON",
            "dataCd": "ASOS",
            "dateCd": "HR",
            "startDt": from_date.strftime("%Y%m%d"),
            "startHh": "00",
            "endDt": to_date.strftime("%Y%m%d"),
            "endHh": "23",
            "stnIds": str(stn_id),
        }
        data = get_json(BASE_URL, params)
        body = (data.get("response") or {}).get("body") or {}
        header = (data.get("response") or {}).get("header") or {}
        result_code = header.get("resultCode")
        if result_code not in ("00", "0", None):
            raise ApiError(f"resultCode={result_code} msg={header.get('resultMsg')}")

        page_items_container = body.get("items") or {}
        page_items = page_items_container.get("item") if isinstance(page_items_container, dict) else []
        if isinstance(page_items, dict):
            page_items = [page_items]
        page_items = page_items or []
        items.extend(page_items)

        total = int(body.get("totalCount") or 0)
        if page * page_size >= total or not page_items:
            log.info("fetched stn=%d %s..%s total=%d pages=%d", stn_id, from_date, to_date, total, page)
            break
        page += 1
    return items


def collect(stn_id: int, from_date: date, to_date: date) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for chunk_from, chunk_to in _month_chunks(from_date, to_date):
        items = fetch_range(stn_id, chunk_from, chunk_to)
        out_path = OUT_DIR / f"stn{stn_id}_{chunk_from.strftime('%Y%m')}.json"
        out_path.write_text(
            json.dumps(
                {
                    "stnId": stn_id,
                    "startDt": chunk_from.isoformat(),
                    "endDt": chunk_to.isoformat(),
                    "count": len(items),
                    "items": items,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("wrote %s (%d rows)", out_path.name, len(items))
    return OUT_DIR


def _parse_yyyymmdd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="ASOS 시간자료(B-1) 수집기")
    ap.add_argument("from_date", help="시작일 YYYYMMDD")
    ap.add_argument("to_date", help="종료일 YYYYMMDD")
    ap.add_argument("--stn", type=int, default=108, help="지점번호 (기본 108=서울)")
    args = ap.parse_args()
    collect(args.stn, _parse_yyyymmdd(args.from_date), _parse_yyyymmdd(args.to_date))


if __name__ == "__main__":
    main()
