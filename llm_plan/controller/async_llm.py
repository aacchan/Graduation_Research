# llm_plan/controller/async_llm.py の AsyncChatLLM 内（あなたの AsyncClient 版）

class AsyncChatLLM:
    ...
    def __init__(self, kwargs: Dict[str, Any]) -> None:
        self.model = kwargs.pop("model")

        if self.model in {"gpt-4-1106-preview", "gpt-4o", "gpt-3.5-turbo-1106"}:
            pass
        else:
            base_url = kwargs.pop("base_url", None)
            api_base = kwargs.pop("api_base", None)
            host     = kwargs.pop("host", None)
            port     = kwargs.pop("port", None)
            version  = kwargs.pop("version", None) or "v1"

            # 0.0.0.0 / :: はクライアント接続に不可 → 127.0.0.1 に変換
            if host in ("0.0.0.0", "::", "[::]"):
                host = "127.0.0.1"

            if not base_url:
                base_url = api_base
            if not base_url and host and port:
                if str(host).startswith(("http://","https://")):
                    base_url = f"{host}:{port}/{version}"
                else:
                    base_url = f"http://{host}:{port}/{version}"
            if not base_url:
                raise ValueError("base_url (or host/port) is required for non-OpenAI models.")

            # デバッグ：最終的にどこへ繋ぐかを出力
            print(f"[AsyncChatLLM] base_url={base_url}")
            kwargs["base_url"] = base_url

        self.client = AsyncClient(**kwargs)
        self.base_url = kwargs.get("base_url")  # デバッグ用に保持

    async def __call__(self, *, messages: List[Dict[str, str]], **kwargs):
        ...
        try:
            return await self.client.chat.completions.create(
                model=model, messages=messages, **kwargs
            )
        except Exception as e:
            # どこに繋ぎに行って落ちたかが分かるように情報を付与
            raise RuntimeError(f"Chat request failed (base_url={getattr(self,'base_url',None)}): {e}") from e
