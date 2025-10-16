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
    def __init__(self, kwargs: Dict[str, Any]) -> None:
        # 1) model を退避（OpenAI SDK の __init__ 引数ではない）
        self.model: str = kwargs.pop("model", "")

        # 2) base_url の調整：host/port から組み立てる（渡されていれば優先）
        base_url: Optional[str] = kwargs.pop("base_url", None)
        host: Optional[str] = kwargs.pop("host", None)
        port: Optional[int] = kwargs.pop("port", None)
        if not base_url:
            if host and port:
                base_url = f"http://{host}:{port}/v1"
            elif host and not port:
                base_url = f"http://{host}/v1"
        # fallback
        if not base_url:
            # 環境に応じて既定を調整（ローカル vLLM なら 8000 が多い）
            base_url = "http://localhost:8000/v1"

        # 3) API キー（vLLM ならダミーでも可）
        api_key: str = kwargs.pop("api_key", "EMPTY")

        # 4) ここで SDK を初期化（port は __init__ に渡さない！）
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        # 5) その他の既定パラメータ（温度など）は保持しておき、呼び出し時にマージ
        self.default_kwargs: Dict[str, Any] = kwargs

    async def __call__(self, *, messages, **kwargs):
        """
        Chat Completions を叩く薄いラッパ。
        controller 側から渡された extra_body（guided_json など）もそのまま通す。
        """
        # 呼び出し毎の上書き
        call_kwargs = {**self.default_kwargs, **kwargs}

        # extra_body が無ければ明示的に渡さない（None を渡すとSDKが嫌がる場合があるため）
        extra_body = call_kwargs.pop("extra_body", None)

        return await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **call_kwargs,
            **({"extra_body": extra_body} if extra_body else {}),
        )
