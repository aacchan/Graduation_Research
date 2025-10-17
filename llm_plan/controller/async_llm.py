# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional

from openai import AsyncOpenAI


class AsyncChatLLM:
    """
    OpenAI SDK (vLLM互換API含む) をラップする薄いクラス。

    - host/port/scheme を base_url に正規化
    - api_key 未指定なら vLLM 用に 'EMPTY' を採用（OpenAI互換サーバで動作）
    - model は self.model に保持
    - 呼び出し時の extra_body をそのまま透過（guided_json 等）
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

        # 4) AsyncOpenAI は host/port を受け付けないので渡さない
        self.client = AsyncOpenAI(**kwargs)

    async def __call__(self, *, messages, **kwargs):
        """
        OpenAI Chat Completions 互換の呼び出し。
        - model: 指定が無ければ self.model を使う
        - extra_body: guided_json / guided_decoding_backend などを透過
        """
        model = kwargs.pop("model", self.model)
        extra_body = kwargs.pop("extra_body", None)
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body,
            **kwargs
        )

