# llm_plan/controller/async_llm.py
from __future__ import annotations
from typing import Any, Dict, Optional

class AsyncChatLLM:
    """
    OpenAI互換のクライアントを包む最小ラッパ例。
    - client: openai.OpenAI 互換
    - model:  モデル名（例: "meta-llama/Meta-Llama-3-8B-Instruct"）
    - default_kwargs: temperature / top_p / その他デフォルト
    """
    def __init__(self, client: Any, model: str, **default_kwargs) -> None:
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
        # 呼び出し時パラメータをマージ
        params: Dict[str, Any] = {**self.default_kwargs, **kwargs}
        # extra_body を組み立て（既存があれば壊さない）
        extra_body: Dict[str, Any] = dict(params.pop("extra_body", {}) or {})
        if structured_schema is not None:
            # vLLM の structured outputs は extra_body で渡す
            extra_body.update({
                "guided_json": structured_schema,
                "guided_decoding_backend": guided_backend,
            })
        # OpenAI互換 API 呼び出し
        return await self.client.chat.completions.create(
            model=params.get("model", self.model),
            messages=messages,
            extra_body=extra_body if extra_body else None,
            **params,
        )


