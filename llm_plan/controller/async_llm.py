# --- PATCH BEGIN (async_llm.py) ---
from __future__ import annotations
from typing import Any, Dict, Optional, List
import os

# できるだけ旧版に合わせて Async クライアントを使う
AsyncClientType = None
try:
    # 古い環境互換（ユーザ旧コード）
    from openai import AsyncClient as _AsyncClient  # type: ignore
    AsyncClientType = _AsyncClient
except Exception:
    try:
        # 新しい SDK 互換
        from openai import AsyncOpenAI as _AsyncClient  # type: ignore
        AsyncClientType = _AsyncClient
    except Exception:
        _AsyncClient = None  # type: ignore

class AsyncChatLLM:
    """
    OpenAI 互換クライアントの薄いラッパ（旧: kwargs=..., 新: client+model）。
    guided は structured_schema を受けた時だけ extra_body に注入。
    """

    def __init__(self, client: Any = None, model: Optional[str] = None, **default_kwargs) -> None:
        # ---- 旧方式: kwargs=... を受け取る（完全互換） ----
        if client is None and model is None and "kwargs" in default_kwargs:
            legacy: Dict[str, Any] = default_kwargs.pop("kwargs") or {}
            self.model = legacy.pop("model")

            # 旧版の base_url 組み立てを踏襲
            if self.model in ("gpt-4-1106-preview", "gpt-4o", "gpt-3.5-turbo-1106"):
                base_url = legacy.pop("base_url", os.getenv("OPENAI_BASE_URL", None))
                if base_url:  # あれば使う
                    legacy["base_url"] = base_url
            else:
                base_url = legacy.pop("base_url")
                port = legacy.pop("port")
                version = legacy.pop("version")
                legacy["base_url"] = f"{base_url}:{port}/{version}"

            # Async クライアント必須
            if AsyncClientType is None:
                raise RuntimeError("openai の Async クライアントが見つかりません。`pip install openai` を実行してください。")

            # 旧は AsyncClient(**kwargs) で生成
            self.client = AsyncClientType(**legacy)

            # 旧は推論デフォルトを kwargs に入れていた（温度などは main 側から注入される前提）
            self.default_kwargs: Dict[str, Any] = {}
            self.default_kwargs.update(default_kwargs)
            return

        # ---- 新方式: client と model を明示 ----
        if client is None or model is None:
            raise TypeError("AsyncChatLLM は (client, model) か、legacy の AsyncChatLLM(kwargs=...) で初期化してください。")
        self.client = client
        self.model = model
        self.default_kwargs = default_kwargs

    @property
    def llm_type(self):
        # 旧互換
        return "AsyncClient"

    async def __call__(
        self,
        *,
        messages: List[Dict[str, str]],
        structured_schema: Optional[Dict[str, Any]] = None,
        guided_backend: str = "lm-format-enforcer",
        **kwargs: Any,
    ) -> Any:
        """
        Make an async API call.
        旧互換: Mixtral は ['system','assistant','user',...] に並び替え。
        """
        # 旧互換: Mixtral の並べ替え
        if self.model == "mistralai/Mixtral-8x7B-Instruct-v0.1" and len(messages) >= 2:
            user_message = messages.pop()
            assistant_message = messages.pop()
            assistant_message = dict(assistant_message)
            assistant_message["role"] = "assistant"
            messages.append(user_message)
            messages.append(assistant_message)

        # guided は structured_schema が来た時だけ extra_body に注入
        extra_body: Dict[str, Any] = dict(kwargs.pop("extra_body", {}) or {})
        if structured_schema is not None:
            extra_body.update({
                "guided_json": structured_schema,
                "guided_decoding_backend": guided_backend,
            })

        return await self.client.chat.completions.create(
            messages=messages,
            extra_body=extra_body if extra_body else None,
            **{**self.default_kwargs, **kwargs, "model": kwargs.get("model", self.model)},
        )
# --- PATCH END (async_llm.py) ---




