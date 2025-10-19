# llm_plan/controller/async_llm.py
from __future__ import annotations
from typing import Any, Dict, Optional
import os

try:
    # vLLM/OpenAI 互換クライアント
    from openai import OpenAI
except Exception:  # ランタイム環境で未導入でもエラーにしない
    OpenAI = None  # type: ignore


class AsyncChatLLM:
    """
    OpenAI 互換クライアントを包む最小ラッパ。
    2通りの初期化に対応（後方互換）:

    A) 新方式（推奨）
        AsyncChatLLM(client=OpenAI(...), model="meta-llama/Meta-Llama-3-8B-Instruct", temperature=0.2, top_p=0.9, ...)
    B) 旧方式（今回の main.py がこれ）
        AsyncChatLLM(kwargs={ "base_url": "...", "api_key": "...", "model": "..." , "temperature": 0.2, "top_p": 0.9, ... })

    ※ guided decoding は controller 側から structured_schema を受け取った時だけ extra_body に付与。
    """

    def __init__(self, client: Any = None, model: Optional[str] = None, **default_kwargs) -> None:
        # ------ 後方互換：AsyncChatLLM(kwargs=kwargs) を受け付ける ------
        if client is None and model is None and "kwargs" in default_kwargs:
            legacy: Dict[str, Any] = default_kwargs.pop("kwargs") or {}

            # base_url / api_key / model をレガシー dict から解決
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
                raise ValueError(
                    "AsyncChatLLM(kwargs=...) の legacy 形式では kwargs['model'] が必須です。"
                )
            if OpenAI is None:
                raise RuntimeError(
                    "openai パッケージが見つかりません。`pip install openai` を実行してください。"
                )
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model

            # 推論パラメータをまとめる（温度など）
            self.default_kwargs: Dict[str, Any] = {}
            for k in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "n"):
                if k in legacy:
                    self.default_kwargs[k] = legacy[k]
            # 明示の default_kwargs があれば上書き
            self.default_kwargs.update(default_kwargs)
            return

        # ------ 新方式：client + model を直接受け取る ------
        if client is None or model is None:
            raise TypeError(
                "AsyncChatLLM は (client, model) もしくは legacy の AsyncChatLLM(kwargs=...) のいずれかで初期化してください。"
            )
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
        # 呼び出し時パラメータをマージ（実行時指定が最優先）
        params: Dict[str, Any] = {**self.default_kwargs, **kwargs}

        # extra_body を組み立て（既存があれば壊さない）
        extra_body: Dict[str, Any] = dict(params.pop("extra_body", {}) or {})
        if structured_schema is not None:
            # vLLM の structured outputs は extra_body で渡す
            extra_body.update({
                "guided_json": structured_schema,
                "guided_decoding_backend": guided_backend,
            })

        # OpenAI 互換 API 呼び出し
        return await self.client.chat.completions.create(
            model=params.get("model", self.model),
            messages=messages,
            extra_body=extra_body if extra_body else None,
            **params,
        )
