# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional
import re
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

    # パス末尾に /v1 が無ければ付ける
    if not re.search(r"/v1/?$", base_url):
        # 末尾スラッシュ調整
        base_url = base_url.rstrip("/") + "/v1"

    return base_url


class AsyncChatLLM:
    """
    OpenAI SDK (vLLM互換API含む) ラッパ
    - host/port/scheme を base_url に正規化（/v1 を強制）
    - api_key 未指定なら 'EMPTY'
    - SDK初期化引数と create() 引数をホワイトリストで安全化
    - structured_schema -> extra_body.guided_json へ吸収
    """

    def __init__(self, kwargs: Dict[str, Any]):
        # model は SDK 初期化パラメータではない
        self.model: Optional[str] = kwargs.pop("model", None)

        # host/port/scheme を取得（scheme 既定 http）
        base_url: Optional[str] = kwargs.get("base_url")
        host: Optional[str] = kwargs.pop("host", None)
        port: Optional[int] = kwargs.pop("port", None)
        scheme: str = kwargs.pop("scheme", "http")

        # base_url 正規化（/v1 付加・0.0.0.0 補正等）
        base_url = _normalize_base_url(base_url, host, port, scheme)
        kwargs["base_url"] = base_url

        # vLLM 用のデフォルト API キー
        kwargs.setdefault("api_key", "EMPTY")

        # SDK __init__ の許可キーのみ通す
        allowed_init = {
            "api_key", "organization", "project",
            "base_url", "timeout", "max_retries",
            "default_headers", "default_query", "http_client",
        }
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed_init}

        # プロキシ無効 & タイムアウト
        client_kwargs.setdefault("http_client", httpx.AsyncClient(trust_env=False, timeout=30.0))

        print(f"[AsyncChatLLM] base_url={client_kwargs.get('base_url')}, model={self.model}")
        self.client = AsyncOpenAI(**client_kwargs)

    async def __call__(self, *, messages, **kwargs):
        """
        OpenAI Chat Completions 互換呼び出し
        - model: 指定なければ self.model
        - structured_schema: extra_body.guided_json へ吸収（互換）
        - extra_body: guided_json/guided_decoding_backend 等を透過
        """
        model = kwargs.pop("model", self.model)

        # 互換: structured_schema を extra_body に移す
        schema = kwargs.pop("structured_schema", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        if schema is not None:
            extra_body["guided_json"] = schema
            extra_body.setdefault("guided_decoding_backend", "lm-format-enforcer")

        # create() の許可キーのみ通す
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

