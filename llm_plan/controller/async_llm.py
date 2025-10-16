# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict
from openai import AsyncOpenAI

class AsyncChatLLM:
    """
    OpenAI Async クライアントの薄いラッパ。
    kwargs には model 以外のパラメータ（temperature/top_p など）を保持し、
    呼び出しごとに上書きできる。
    """

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        # ---- 接続情報を整理 ----
        # 受け取りうるキーを吸い上げ（残りは default_kwargs として保持）
        base_url = kwargs.pop("base_url", None)
        api_base = kwargs.pop("api_base", None)  # 互換名が来る場合に備え
        host = kwargs.pop("host", None)
        port = kwargs.pop("port", None)
        api_key = kwargs.pop("api_key", None) or kwargs.pop("OPENAI_API_KEY", "EMPTY")

        # base_url を決定（優先順位: base_url > api_base > host+port）
        if not base_url:
            if api_base:
                base_url = api_base
            elif host and port:
                # host が "http://..." を含むかで前処理
                if str(host).startswith("http://") or str(host).startswith("https://"):
                    base_url = f"{host}:{port}/v1"
                else:
                    base_url = f"http://{host}:{port}/v1"

        # vLLM/OpenAI互換API のクライアント生成
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

        # モデル名は別持ち（呼び出しごとに上書き可）
        self.model = kwargs.pop("model", None)

        # それ以外はデフォルト引数として保持（temperature/top_p/n/extra_body 等）
        self.default_kwargs: Dict[str, Any] = kwargs

    async def __call__(self, *, messages, **kwargs):
        """
        Chat Completions エンドポイントを叩く統一口。
        controller 側から渡す extra_body（guided_json など）もそのまま透過。
        """
        payload: Dict[str, Any] = {**self.default_kwargs, **kwargs}

        # OpenAI SDK が受け付けない / ここで使うべきでないキーを除去
        DENY_KEYS = {
            "version",        # OpenAI SDK の引数には無い
            "host", "port",   # 接続は base_url へ集約済み
            "api_base", "base_url",
            "api_key",        # SDK 初期化で使用済み
            "structured_schema",  # ← これを追加！(犯人)
        }
        for k in list(payload.keys()):
            if k in DENY_KEYS:
                payload.pop(k, None)

        model = payload.pop("model", None) or self.model
        if not model:
            raise ValueError("model is not specified for AsyncChatLLM call.")

        # extra_body は payload に含まれていればそのまま渡る
        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **payload
        )

