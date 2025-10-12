# llm_plan/controller/async_llm.py
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
from typing import Optional, Dict, Any, List

from openai import AsyncOpenAI  # ← 公式SDKの非同期クライアントに統一
from omegaconf import OmegaConf

def _build_extra_body(structured_schema: Optional[Dict[str, Any]] = None,
                      backend: str = "lm-format-enforcer") -> Dict[str, Any]:
    """
    vLLM の Structured Outputs 用 追加パラメータを組み立てる。
    guided_json / guided_decoding_backend は vLLM 拡張なので extra に入れる。
    """
    extra: Dict[str, Any] = {}
    if structured_schema is not None:
        extra["guided_json"] = structured_schema
        extra["guided_decoding_backend"] = backend
    return extra

class AsyncChatLLM:
    """
    OpenAI 互換 API（vLLM サーバ想定）を叩く非同期ラッパー。

    - structured_schema=None の場合は従来通り（制約なし）
    - structured_schema が dict の場合は guided_json を付与（構造化出力）
    """
    def __init__(self, kwargs: Dict[str, Any]):
        # `kwargs` には base_url, api_key, model などが入る前提
        self.client = AsyncOpenAI(**{k: v for k, v in kwargs.items() if k != "model"})
        self.model: Optional[str] = kwargs.get("model")  # 参照されるので保持
        self.default_kwargs: Dict[str, Any] = {k: v for k, v in kwargs.items() if k != "model"}

    @property
    def llm_type(self) -> str:
        return "AsyncOpenAI"

    async def __call__(
        self,
        messages: List[Dict[str, str]],
        *,
        structured_schema: Optional[Dict[str, Any]] = None,  # dict/tuple 用スキーマ or None
        guided_backend: str = "lm-format-enforcer",
        **kwargs: Any,
    ):
        """
        非同期で Chat Completions を呼ぶ。structured_schema が与えられたら
        vLLM の guided decoding（構造化出力）を extra_body で有効化する。
        """
        # Mixtralだけメッセージ順の例外処理（既存ロジックを踏襲）
        if self.model == "mistralai/Mixtral-8x7B-Instruct-v0.1" and len(messages) >= 2:
            user_message = messages.pop()
            assistant_message = messages.pop()
            assistant_message["role"] = "assistant"
            messages.append(user_message)
            messages.append(assistant_message)

        # 既定パラメータと呼び出し時パラメータを合成
        call_kwargs: Dict[str, Any] = {**self.default_kwargs, **kwargs}
        # Model は明示優先、なければ保持している self.model を使う
        call_kwargs.setdefault("model", self.model)

        # guided decoding 用の extra（vLLM拡張）をマージ
        extra_body = _build_extra_body(structured_schema, guided_backend)
        if extra_body:
            # 呼び出し元が渡した extra_body を優先（互換重視）
            caller_extra = call_kwargs.get("extra_body", {})
            merged = dict(extra_body)
            merged.update(caller_extra)  # ← 衝突時は呼び出し元の値を採用
            call_kwargs["extra_body"] = merged

        # OpenAI 互換API（vLLM）へ実行
        return await self.client.chat.completions.create(
            messages=messages,
            **call_kwargs,
        )
