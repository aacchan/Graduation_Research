# llm_plan/controller/async_llm.py
from __future__ import annotations
from typing import Any, Dict, Optional
from openai import AsyncOpenAI

class AsyncChatLLM:
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

        # ★ ここでログ（client_kwargs を作った後に）
        print(f"[AsyncChatLLM] base_url={client_kwargs.get('base_url')}, model={self.model}")

        # 5) 初期化
        self.client = AsyncOpenAI(**client_kwargs)

    async def __call__(self, *, messages, **kwargs):
        model = kwargs.pop("model", self.model)

        # 互換吸収: structured_schema -> extra_body.guided_json
        schema = kwargs.pop("structured_schema", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        if schema is not None:
            extra_body["guided_json"] = schema
            extra_body.setdefault("guided_decoding_backend", "lm-format-enforcer")

        # create() が受け付けるパラメータだけ通す（最低限のホワイトリスト）
        allowed_create = {
            "temperature", "top_p", "n", "stream", "stop", "max_tokens",
            "presence_penalty", "frequency_penalty", "logit_bias",
            "user", "seed", "tools", "tool_choice", "functions", "function_call",
            "response_format", "logprobs", "top_logprobs",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_create}

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body or None,
            **filtered_kwargs
        )




