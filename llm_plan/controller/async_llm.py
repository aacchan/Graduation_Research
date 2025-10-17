# llm_plan/controller/async_llm.py （__init__ の該当箇所だけ追記/修正）

from __future__ import annotations
from typing import Any, Dict, Optional
from openai import AsyncOpenAI
import httpx  # 追加

class AsyncChatLLM:
    def __init__(self, kwargs: Dict[str, Any]):
        self.model: Optional[str] = kwargs.pop("model", None)

        base_url: Optional[str] = kwargs.get("base_url")
        host: Optional[str] = kwargs.pop("host", None)
        port: Optional[int] = kwargs.pop("port", None)
        scheme: str = kwargs.pop("scheme", "http")

        # クライアント側 0.0.0.0 は接続不能なので補正
        if host == "0.0.0.0":
            host = "127.0.0.1"

        if not base_url and host:
            base_url = f"{scheme}://{host}{f':{port}' if port else ''}/v1"
            kwargs["base_url"] = base_url

        # ★ ここが無いと base_url が空のままになり得る → 最後の砦として既定
        kwargs.setdefault("base_url", "http://127.0.0.1:8000/v1")

        kwargs.setdefault("api_key", "EMPTY")

        allowed = {
            "api_key","organization","project","base_url","timeout","max_retries",
            "default_headers","default_query","http_client",
        }
        client_kwargs = {k: v for k, v in kwargs.items() if k in allowed}

        # ★ プロキシを無効化してローカル直結（必要に応じてtimeout調整）
        client_kwargs.setdefault("http_client", httpx.AsyncClient(trust_env=False, timeout=30.0))

        print(f"[AsyncChatLLM] base_url={client_kwargs.get('base_url')}, model={self.model}")
        self.client = AsyncOpenAI(**client_kwargs)
