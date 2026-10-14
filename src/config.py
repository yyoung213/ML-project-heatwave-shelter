"""공통 설정: 프로젝트 경로 상수 + .env 인증키 로딩."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "raw"
PROCESSED_DIR = PROJECT_ROOT / "processed"

load_dotenv(PROJECT_ROOT / ".env")


def require_key(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(
            f"환경변수 {name} 가 비어 있습니다. "
            f"{PROJECT_ROOT / '.env'} 에 키를 채우세요. "
            f"발급 안내: docs/api_keys.md"
        )
    return value
