"""공공 API 공통 GET 헬퍼.

- tenacity 기반 재시도 (네트워크·5xx)
- 파일 캐싱: 동일 URL+파라미터 재호출 시 디스크 캐시 반환 (호출 한도 보호)
- 응답은 JSON 가정. XML 응답은 호출처에서 dataType=JSON 강제.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from .config import PROJECT_ROOT

CACHE_DIR = PROJECT_ROOT / "raw" / "_http_cache"
log = logging.getLogger(__name__)


class ApiError(RuntimeError):
    """공공 API 가 비정상 응답코드를 돌려준 경우."""


def _cache_key(url: str, params: dict[str, Any]) -> str:
    # 서비스키는 캐시 키에서 제외해야 키 교체 시 캐시가 무효화되지 않음
    safe_params = {k: v for k, v in sorted(params.items()) if k != "serviceKey"}
    payload = json.dumps({"url": url, "params": safe_params}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@retry(
    retry=retry_if_exception_type((requests.RequestException, ApiError)),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
def _http_get(url: str, params: dict[str, Any], timeout: int) -> requests.Response:
    resp = requests.get(url, params=params, timeout=timeout)
    if resp.status_code >= 500:
        raise ApiError(f"{resp.status_code} {resp.text[:200]}")
    resp.raise_for_status()
    return resp


def get_json(
    url: str,
    params: dict[str, Any],
    *,
    timeout: int = 20,
    use_cache: bool = True,
) -> dict[str, Any]:
    """JSON 응답 GET. 캐시 히트면 디스크에서 즉시 반환."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / f"{_cache_key(url, params)}.json"

    if use_cache and cache_path.exists():
        log.debug("cache hit: %s", cache_path.name)
        return json.loads(cache_path.read_text(encoding="utf-8"))

    resp = _http_get(url, params, timeout)
    try:
        data = resp.json()
    except ValueError as e:
        raise ApiError(f"JSON 파싱 실패: {resp.text[:300]}") from e

    cache_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data
