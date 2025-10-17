# llm_plan/controller/async_llm.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional  # ← ★ Optional を先に読み込む
import re
from urllib.parse import urlparse, urlunparse
import httpx
from openai import AsyncOpenAI


def _normalize_base_url(base_url: Optional[str], host: Optional[str], port: Optional[int], scheme: str) -> str:
    # 0.0.0.0 は接続不可なので 127.0.0.1 へ
    if host == "0.0.0.0":
        host = "127.0.0.1"

    # base_url 未指定なら host/port から組み立て
    if not base_url and host:
        base_url = f"{scheme}://{host}{f':{port}' if port else ''}/v1"

    # まだ無ければ最後の砦（ローカル既定）
    base_url = base_url or "http://127.0.0.1:8000/v1"

    # スキーム無ければ http を付与
    if not re.match(r"^https?://", base_url):
        base_url = "http://" + base_url.lstrip("/")

    # URL を解析して補正
    p = urlparse(base_url)

    # ホスト補正（0.0.0.0 → 127.0.0.1）
    hostname = p.hostname or "127.0.0.1"
    if hostname == "0.0.0.0":
        hostname = "127.0.0.1"

    # ポート補正：localhost/127.0.0.1 で未指定なら 8000
    final_port = p.port
    if final_port is None and hostname in ("localhost", "127.0.0.1") and (p.scheme or "http") == "http":
        final_port = 8000

    # パス補正：末尾に /v1 を付ける
    path = p.path or ""
    if not path.endswith("/v1") and not path.endswith("/v1/"):
        path = path.rstrip("/") + "/v1"

    netloc = f"{hostname}:{final_port}" if final_port else hostname
    fixed = p._replace(scheme=(p.scheme or "http"), netloc=netloc, path=path)
    return urlunparse(fixed)



