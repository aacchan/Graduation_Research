# llm_plan/controller/async_llm.py
from __future__ import annotations
from typing import Any, Dict, Optional
import os

try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

class AsyncChatLLM:
    """
    OpenAI 互換クライアントの薄いラッパ。
    2通りの初期化に対応（後方互換）:

    A) 新方式（推奨）
       AsyncChatLLM(client=OpenAI(...), model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.2, ...)
    B) 旧方式（main.py がこれ）
       AsyncChatLLM(kwargs={"base_url":"...", "api_key":"...", "model":"...", "temperature":0.2, ...})
    """

    def __init__(self, client: Any = None, model: Optional[str] = None, **default_kwargs) -> None:
        # ---- 旧方式: kwargs=... を受け取る後方互換
        if client is None and model is None and "kwargs" in default_kwargs:
            legacy: Dict[str, Any] = default_kwargs.pop("kwargs") or {}

            base_url = (
                legacy.get("base_url")
                or legacy.get("openai_api_base")
                or legacy.get("api_base")
                or os.getenv("OPENAI_BASE_URL")
                or "http://localhost:8000/v1"
            )
            api_key = legacy.get("api_key") or os.getenv("OPENAI_API_KEY") or "EMPTY"
            model = (
                legacy.get("model")
                or legacy.get("openai_model")
                or legacy.get("llm_model")
                or legacy.get("model_name")
            )
            if model is None:
                raise ValueError("AsyncChatLLM(kwargs=...) では kwargs['model'] が必須です。")
            if OpenAI is None:
                raise RuntimeError("openai パッケージが見つかりません。`pip install openai` を実行してください。")

            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model

            # 推論デフォルト（温度など）を集約
            self.default_kwargs: Dict[str, Any] = {}
            for k in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "n"):
                if k in legacy:
                    self.default_kwargs[k] = legacy[k]
            # 追加指定があれば上書き
            self.default_kwargs.update(default_kwargs)
            return

        # ---- 新方式: client と model を明示
        if client is None or model is None:
            raise TypeError("AsyncChatLLM は (client, model) か、legacy の AsyncChatLLM(kwargs=...) で初期化してください。")
        self.client = client
        self.model = model
        self.default_kwargs = default_kwargs

    async def __call__(
        self,
        *,
        messages: list[dict[str, str]],
        structured_schema: Optional[Dict[str, Any]] = None,
        guided_backend: str = "lm-format-enforcer",
        **kwargs: Any,
    ) -> Any:
        # 実行時パラメータをマージ（kwargs が優先）
        params: Dict[str, Any] = {**self.default_kwargs, **kwargs}

        # vLLM Structured Outputs 用の extra_body（既存があれば壊さない）
        extra_body: Dict[str, Any] = dict(params.pop("extra_body", {}) or {})
        if structured_schema is not None:
            extra_body.update({
                "guided_json": structured_schema,
                "guided_decoding_backend": guided_backend,
            })

        return await self.client.chat.completions.create(
            model=params.get("model", self.model),
            messages=messages,
            extra_body=extra_body if extra_body else None,
            **params,
        )



