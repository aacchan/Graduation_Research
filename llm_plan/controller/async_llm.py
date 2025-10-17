# llm_plan/controller/async_llm.py
from __future__ import annotations
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
import httpx  # ← 追加

class AsyncChatLLM:
    def __init__(self, kwargs: Dict[str, Any]):
        self.model: Optional[str] = kwargs.pop("model", None)

        base_url: Optional[str] = kwargs.get("base_url")
        host: Optional[str] = kwargs.pop("host", None)
        port: Optional[int] = kwargs.pop("port", None)
        scheme: str = kwargs.pop("scheme", "http")

        # ★ クライアント側の 0.0.0.0 は接続不能なので補正
        if host == "0.0.0.0":
            host = "127.0.0.1"

        if not base_url and host:
            base_url = f"{scheme}://{host}{f':{port}' if port else ''}/v1"
            kwargs["base_url"] = base_url

        kwargs.setdefault("api_key", "EMPTY")

        allowed = {
            "api_key","organization","project","base_url","timeout","max_retries",
            "default_headers","default_query","http_client",
        }
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        # ★ プロキシ環境を無視してローカルへ直でつなぐ
        client_kwargs.setdefault("http_client", httpx.AsyncClient(trust_env=False, timeout=30.0))

        print(f"[AsyncChatLLM] base_url={client_kwargs.get('base_url')}, model={self.model}")

        self.client = AsyncOpenAI(**client_kwargs)

    async def __call__(self, *, messages, **kwargs):
        model = kwargs.pop("model", self.model)

        schema = kwargs.pop("structured_schema", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        if schema is not None:
            extra_body["guided_json"] = schema
            extra_body.setdefault("guided_decoding_backend", "lm-format-enforcer")

        allowed_create = {
            "temperature","top_p","n","stream","stop","max_tokens",
            "presence_penalty","frequency_penalty","logit_bias",
            "user","seed","tools","tool_choice","functions","function_call",
            "response_format","logprobs","top_logprobs",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_create}

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body=extra_body or None,
            **filtered_kwargs
        )





