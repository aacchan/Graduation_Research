from __future__ import annotations
import json
import os
from typing import Any, Dict, List, Optional

# Prefer async client for await compatibility; support both legacy and new SDKs
AsyncClientType = None
try:
    from openai import AsyncClient as _AsyncClient  # legacy SDK
    AsyncClientType = _AsyncClient
except Exception:
    try:
        from openai import AsyncOpenAI as _AsyncClient  # new SDK
        AsyncClientType = _AsyncClient
    except Exception:
        _AsyncClient = None  # type: ignore

class AsyncChatLLM:
    """
    Wrapper for an (Async) Chat Model.
    Supports legacy initialization with kwargs dict (old main.py) and new style with (client, model).
    """

    def __init__(self, kwargs: Optional[Dict[str, Any]] = None, *, client: Any = None, model: Optional[str] = None, **default_kwargs):
        # Legacy path: AsyncChatLLM(kwargs=...)
        if kwargs is not None:
            if AsyncClientType is None:
                raise RuntimeError("openai async client not available. `pip install openai`.")
            if 'model' not in kwargs:
                raise ValueError("kwargs['model'] is required")

            self.model = kwargs['model']

            # Old code composed base_url as f"{base_url}:{port}/{version}" for non-OpenAI backends
            base_url = kwargs.get('base_url')
            if base_url and all(k in kwargs for k in ('port','version')):
                base_url = f"{base_url}:{kwargs['port']}/{kwargs['version']}"

            client_kwargs = dict(kwargs)
            client_kwargs['base_url'] = base_url or client_kwargs.get('base_url')
            for k in ('model','port','version'):
                client_kwargs.pop(k, None)

            self.client = AsyncClientType(**client_kwargs)
            self.default_kwargs: Dict[str, Any] = {}
            self.default_kwargs.update(default_kwargs)
            return

        # New style: explicit (client, model)
        if client is None or model is None:
            raise TypeError("Provide either kwargs=... or (client, model)")
        self.client = client
        self.model = model
        self.default_kwargs = default_kwargs

    @property
    def llm_type(self) -> str:
        # old code returned 'AsyncClient'
        return "AsyncClient"

    async def __call__(self, *, messages: List[Dict[str,str]], structured_schema: Optional[Dict[str, Any]] = None,
                       guided_backend: str = "lm-format-enforcer", **kwargs) -> Any:
        """Make an async API call. Preserves old Mixtral reordering."""
        # Mixtral requires ['system','assistant','user',...] order (old behavior)
        if self.model == "mistralai/Mixtral-8x7B-Instruct-v0.1" and len(messages) >= 2:
            user_message = messages.pop()
            assistant_message = messages.pop()
            assistant_message = dict(assistant_message)
            assistant_message['role'] = 'assistant'
            messages.append(user_message)
            messages.append(assistant_message)

        # Inject guided decoding only when schema provided
        extra_body: Dict[str, Any] = dict(kwargs.pop("extra_body", {}) or {})
        if structured_schema is not None:
            extra_body.update({
                "guided_json": structured_schema,
                "guided_decoding_backend": guided_backend,
            })
        """
        # --- デバッグ出力ここから（__call__ の API呼び出し直前） ---
        print("\n================== LLM CALL DEBUG ==================")
        print("Model:", self.model)
        print("Base URL:", getattr(self.client, "base_url", "unknown"))

        # メッセージは長すぎてターミナルが埋まることがあるので頭だけ表示
        print("Messages:")
        for m in messages:
            role = m.get("role")
            content = (m.get("content") or "")
            preview = content[:500] + ("..." if len(content) > 500 else "")
            print("  ", {"role": role, "content": preview})

        # extra_body は JSON化できない型が紛れることがあるので try/except
        try:
            print("Extra body (for guided decoding):", json.dumps(extra_body, indent=2, ensure_ascii=False)[:4000])
        except Exception as e:
            print("Extra body (raw):", str(extra_body)[:4000], "  <-- json.dumps失敗:", repr(e))

        print("====================================================\n")
        # --- ここまで ---
        """
                           
        return await self.client.chat.completions.create(
            messages=messages,
            extra_body=extra_body,  # 空でも {} を渡す
            **{**self.default_kwargs, **kwargs, "model": kwargs.get("model", self.model)}
        )
