# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, Optional

from openai import AsyncOpenAI  # ← 非同期クライアント
# 他に必要な import があれば追加

class AsyncChatLLM:
    """
    vLLM / OpenAI 互換サーバ用の非同期 LLM ラッパ。
    kwargs には model, api_key, base_url の他に host/port が来ることがあるので吸収する。
    """

class AsyncChatLLM:
    def __init__(self, kwargs: Dict[str, Any]) -> None:
        self.model: str = kwargs.pop("model", "")
        base_url: Optional[str] = kwargs.pop("base_url", None)
        host: Optional[str] = kwargs.pop("host", None)
        port: Optional[int]  = kwargs.pop("port", None)
        if not base_url:
            if host and port:
                base_url = f"http://{host}:{port}/v1"
            elif host and not port:
                base_url = f"http://{host}/v1"
        if not base_url:
            base_url = "http://localhost:8000/v1"

        api_key: str = kwargs.pop("api_key", "EMPTY")

        # ★ ここで未対応キーを除去（予防）
        UNSUPPORTED_INIT_KEYS = {"version"}  # 必要なら増やす
        for k in list(kwargs.keys()):
            if k in UNSUPPORTED_INIT_KEYS:
                kwargs.pop(k, None)

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.default_kwargs: Dict[str, Any] = kwargs


    async def __call__(self, *, messages, **kwargs):
        """
        Chat Completions を叩く薄いラッパ。
        controller 側から渡された extra_body（guided_json など）もそのまま通す。
        """
        # 呼び出し毎の上書き
        call_kwargs = {**self.default_kwargs, **kwargs}

        # model の二重指定を回避：優先は call_kwargs 側
        model_param = call_kwargs.pop("model", self.model)

        # extra_body が無ければ明示的に渡さない（None を渡すとSDKが嫌がる場合があるため）
        extra_body = call_kwargs.pop("extra_body", None)

        return await self.client.chat.completions.create(
            model=model_param,
            messages=messages,
            **call_kwargs,
            **({"extra_body": extra_body} if extra_body else {}),
        )
