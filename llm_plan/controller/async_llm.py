# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Any, Dict, Optional
import re
from urllib.parse import urlparse, urlunparse
import httpx
from openai import AsyncOpenAI


def _normalize_base_url(base_url: Optional[str], host: Optional[str], port: Optional[int], scheme: str) -> str:
    """host/port/base_url を正規化して、必ず /v1 まで含む URL を返す。"""
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

    # ポート補正：localhost/127.0.0.1 で未指定なら 8000（httpの既定80回避）
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


class AsyncChatLLM:
    """
    OpenAI SDK (vLLM互換API) ラッパ

    - host/port/scheme → base_url に正規化（必ず /v1 を含む）
    - api_key 未指定なら 'EMPTY'（vLLM既定）
    - SDK初期化引数/作成引数はホワイトリストで安全にフィルタ
    - structured_schema を extra_body.guided_json に吸収（互換）
    """

    def __init__(self, kwargs: Dict[str, Any]):
        # model は SDK 初期化引数ではないので退避
        self.model: Optional[str] = kwargs.pop("model", None)

        # host/port/scheme → base_url 正規化
        base_url: Optional[str] = kwargs.get("base_url")
        host: Optional[str] = kwargs.pop("host", None)
        port: Optional[int] = kwargs.pop("port", None)
        scheme: str = kwargs.pop("scheme", "http")
        base_url = _normalize_base_url(base_url, host, port, scheme)
        kwargs["base_url"] = base_url

        # vLLM 用のデフォルト API キー
        kwargs.setdefault("api_key", "EMPTY")

        # SDK __init__ に渡せるキーだけを許可
        allowed_init = {
            "api_key", "organization", "project",
            "base_url", "timeout", "max_retries",
            "default_headers", "default_query", "http_client",
        }
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed_init}

        # プロキシ無効化＆タイムアウト設定
        client_kwargs.setdefault("http_client", httpx.AsyncClient(trust_env=False, timeout=30.0))

        print(f"[AsyncChatLLM] base_url={client_kwargs.get('base_url')}, model={self.model}")
        self.client = AsyncOpenAI(**client_kwargs)

    async def __call__(self, *, messages, **kwargs):
        """
        Chat Completions 互換呼び出し
        """
        model = kwargs.pop("model", self.model)

        # 互換：structured_schema → extra_body.guided_json
        schema = kwargs.pop("structured_schema", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        if schema is not None:
            extra_body["guided_json"] = schema
            extra_body.setdefault("guided_decoding_backend", "lm-format-enforcer")

        # create() に渡せるキーだけ許可
        allowed_create = {
            "temperature", "top_p", "n", "stream", "stop", "max_tokens",
            "presence_penalty", "frequency_penalty", "logit_bias",
            "user", "seed", "tools", "tool_choice", "functions", "function_call",
            "response_format", "logprobs", "top_logprobs",
        }
        filtered = {k: v for k, v in kwargs.items() if k in allowed_create}

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body or None,
            **filtered
        )
