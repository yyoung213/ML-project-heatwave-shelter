"""기상청 기상특보 조회서비스 (B-2) 수집기.

공공데이터포털: 기상청_기상특보 조회서비스 (publicDataPk=15000415)
엔드포인트: getWthrWrnList  — 발효 기상특보 목록 (날짜범위 지원)

응답 필드(확인된 것): title, stnId, tmSeq, tmFc(YYYYMMDDHHmm)
※ title 에 특보 종류(폭염주의보·폭염경보 등) 포함되는지는 1차 호출로 검증해야 함.

사용:
    python -m src.collectors.kma_warnings 20240701 20240731
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

BASE_URL = "http://apis.data.go.kr/1360000/WthrWrnInfoService/getWthrWrnList"
OUT_DIR = RAW_DIR / "weather" / "warnings"

log = logging.getLogger(__name__)


def _month_chunks(start: date, end: date) -> Iterator[tuple[date, date]]:
    """월 단위로 쪼개서 호출. (응답량 폭주 방지 + 페이지네이션 단순화)"""
    cur = start.replace(day=1)
    while cur <= end:
        # 다음달 1일 - 1일 = 이번달 말일
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        month_end = nxt - timedelta(days=1)
        yield max(cur, start), min(month_end, end)
        cur = nxt


def fetch_range(from_date: date, to_date: date, page_size: int = 100) -> list[dict]:
    """지정 기간(YYYYMMDD)의 기상특보 항목을 모두 가져옴 (페이지네이션 자동)."""
    service_key = require_key("DATA_GO_KR_KEY")
    items: list[dict] = []
    page = 1
    while True:
        params = {
            "serviceKey": service_key,
            "pageNo": page,
            "numOfRows": page_size,
            "dataType": "JSON",
            "fromTmFc": from_date.strftime("%Y%m%d"),
            "toTmFc": to_date.strftime("%Y%m%d"),
        }
        data = get_json(BASE_URL, params)
        body = (data.get("response") or {}).get("body") or {}
        header = (data.get("response") or {}).get("header") or {}
        result_code = header.get("resultCode")
        if result_code not in ("00", "0", None):
            raise ApiError(f"resultCode={result_code} msg={header.get('resultMsg')}")

        page_items_container = body.get("items") or {}
        page_items = page_items_container.get("item") if isinstance(page_items_container, dict) else []
        if isinstance(page_items, dict):  # 단건이면 dict 로 오기도 함
            page_items = [page_items]
        page_items = page_items or []
        items.extend(page_items)

        total = int(body.get("totalCount") or 0)
        if page * page_size >= total or not page_items:
            log.info(
                "fetched %s..%s  total=%d  pages=%d",
                from_date, to_date, total, page,
            )
            break
        page += 1
    return items


def collect(from_date: date, to_date: date) -> Path:
    """월 단위로 잘라 받아 JSON 으로 저장. 반환값은 출력 디렉터리."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for chunk_from, chunk_to in _month_chunks(from_date, to_date):
        items = fetch_range(chunk_from, chunk_to)
        out_path = OUT_DIR / f"{chunk_from.strftime('%Y%m')}.json"
        out_path.write_text(
            json.dumps(
                {
                    "fromTmFc": chunk_from.isoformat(),
                    "toTmFc": chunk_to.isoformat(),
                    "count": len(items),
                    "items": items,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        log.info("wrote %s (%d items)", out_path.name, len(items))
    return OUT_DIR


def _parse_yyyymmdd(s: str) -> date:
    return date(int(s[:4]), int(s[4:6]), int(s[6:8]))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="기상특보(B-2) 수집기")
    ap.add_argument("from_date", help="시작일 YYYYMMDD")
    ap.add_argument("to_date", help="종료일 YYYYMMDD")
    args = ap.parse_args()
    collect(_parse_yyyymmdd(args.from_date), _parse_yyyymmdd(args.to_date))


if __name__ == "__main__":
    main()
