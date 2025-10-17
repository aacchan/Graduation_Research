# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Any, Dict, List
from openai import AsyncClient


class AsyncChatLLM:
    """
    Wrapper for an (Async) Chat Model.
    - OpenAI SDK の AsyncClient を使う（あなたの元コード路線）
    - base_url/port/version を自前で組み立てる
    - SDK に渡せないキー（version/host/port/structured_schema 等）は除去
    """

    def __init__(self, kwargs: Dict[str, Any]) -> None:
        # モデルはここで確定（呼び出し部で上書きも可）
        self.model = kwargs.pop("model")

        # 既存の挙動を維持：gpt系はそのまま（OpenAI本家を想定）
        if self.model in {"gpt-4-1106-preview", "gpt-4o", "gpt-3.5-turbo-1106"}:
            pass
        else:
            # vLLM など OpenAI互換サーバ向けに base_url を合成
            # 入ってくる可能性のあるキーを吸収
            base_url = kwargs.pop("base_url", None)
            api_base = kwargs.pop("api_base", None)  # 互換エイリアス
            host     = kwargs.pop("host", None)
            port     = kwargs.pop("port", None)
            version  = kwargs.pop("version", None) or "v1"

            # 0.0.0.0 は待受用であり、クライアント接続には使えない → 127.0.0.1 に正規化
            if host in ("0.0.0.0", "::", "[::]"):
                host = "127.0.0.1"

            # 最優先: base_url → 次: api_base → それも無ければ host+port で作る
            if not base_url:
                base_url = api_base

            if not base_url and host and port:
                if str(host).startswith("http://") or str(host).startswith("https://"):
                    base_url = f"{host}:{port}/{version}"
                else:
                    base_url = f"http://{host}:{port}/{version}"

            if not base_url:
                raise ValueError("base_url (or host/port) is required for non-OpenAI models.")

            kwargs["base_url"] = base_url

        # AsyncClient を生成（api_key などは kwargs から渡る）
        self.client = AsyncClient(**kwargs)

    @property
    def llm_type(self) -> str:
        return "AsyncClient"

    async def __call__(self, *, messages: List[Dict[str, str]], **kwargs):
        """
        Make an async API call.
        - Mixtral のメッセージ並び替えロジックは元コードを踏襲
        - SDK が受け付けない余分なキーを除去（structured_schema など）
        """
        # Mixtral の並び替え（元コードの挙動を維持）
        if self.model == "mistralai/Mixtral-8x7B-Instruct-v0.1":
            user_message = messages.pop()
            assistant_message = messages.pop()
            assistant_message["role"] = "assistant"
            messages.append(user_message)
            messages.append(assistant_message)

        # OpenAI SDK に渡すべきでないキーを除去（ここが今回の TypeError 対策）
        DENY_KEYS = {
            "version",            # SDK 引数には存在しない
            "host", "port",       # 接続は base_url に集約
            "api_base", "base_url",  # 生成時に消費済み（__init__で処理）
            "api_key",            # 生成時に設定済み
            "structured_schema",  # ← 重要: controller 側で extra_body に詰める
        }
        for k in list(kwargs.keys()):
            if k in DENY_KEYS:
                kwargs.pop(k, None)

        # model は self.model を優先、kwargs 側にあれば上書きも許容
        model = kwargs.pop("model", None) or self.model

        return await self.client.chat.completions.create(
            model=model,
            messages=messages,
            **kwargs,  # temperature/top_p/n/extra_body など
        )
