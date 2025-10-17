# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


class AsyncChatLLM:
    """
    OpenAI SDK (vLLM互換API含む) をラップする薄いクラス。

    - host/port/scheme を base_url に正規化
    - api_key 未指定なら vLLM 用に 'EMPTY' を採用
    - SDKの __init__ に渡す引数はホワイトリストで制限（未知キーを落とす）
    - 呼び出し時: structured_schema を extra_body.guided_json に変換して透過
    """

    def __init__(self, kwargs: Dict[str, Any]):
        # 1) model は SDK 初期化パラメータではないので退避
        self.model: Optional[str] = kwargs.pop("model", None)

        # 2) host/port/scheme → base_url 正規化
        base_url: Optional[str] = kwargs.get("base_url")
        host: Optional[str] = kwargs.pop("host", None)
        port: Optional[int] = kwargs.pop("port", None)
        scheme: str = kwargs.pop("scheme", "http")
        if not base_url and host:
            base_url = f"{scheme}://{host}{f':{port}' if port else ''}/v1"
            kwargs["base_url"] = base_url

        # 3) vLLM 用のデフォルト API キー
        kwargs.setdefault("api_key", "EMPTY")

        # 4) AsyncOpenAI が受け付けるキーだけを通す（ホワイトリスト）
        allowed = {
            "api_key", "organization", "project",
            "base_url", "timeout", "max_retries",
            "default_headers", "default_query", "http_client",
        }
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        # 5) 初期化
        self.client = AsyncOpenAI(**client_kwargs)

    async def __call__(self, *, messages, **kwargs):
        """
        OpenAI Chat Completions 互換の呼び出し。
        - model: 指定が無ければ self.model を使う
        - structured_schema: もし渡されても extra_body.guided_json に変換して受け入れる
        - extra_body: guided_json / guided_decoding_backend などを透過
        - その他の引数は create() が受け取れるものだけにフィルタ
        """
        model = kwargs.pop("model", self.model)

        # 旧/他経路のための互換: structured_schema を extra_body へ吸収
        schema = kwargs.pop("structured_schema", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        if schema is not None:
            extra_body["guided_json"] = schema
            # backend が未指定ならデフォルト補完（必要に応じて調整）
            extra_body.setdefault("guided_decoding_backend", "lm-format-enforcer")

        # create() が受け付けるパラメータだけ通す（最低限のホワイトリスト）
        allowed_create = {
            "messages",  # 実際には位置引数で渡しているが統一のため残す
            "temperature", "top_p", "n", "stream", "stop", "max_tokens",
            "presence_penalty", "frequency_penalty", "logit_bias",
            "user", "seed",
            # OpenAI SDK v1 の追加オプション類（必要に応じて拡張）
            "tools", "tool_choice", "functions", "function_call",
            "response_format", "logprobs", "top_logprobs",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_create}

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body or None,
            **filtered_kwargs
        )



