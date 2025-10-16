*** a/llm_plan/controller/async_llm.py
--- b/llm_plan/controller/async_llm.py
@@
     async def __call__(self, *, messages, **kwargs):
         """
         Chat Completions エンドポイントを叩く統一口。
         フェーズ2で controller 側から渡す extra_body（guided_json など）もそのまま透過。
         """
-        payload: Dict[str, Any] = {**self.default_kwargs, **kwargs}
+        payload: Dict[str, Any] = {**self.default_kwargs, **kwargs}
+
+        # OpenAI SDK が受け付けない / ここで使うべきでないキーを除去
+        # 例: main.py 側から混入することがある 'version' など
+        DENY_KEYS = {
+            "version",        # ← 本件の犯人
+            "host", "port",   # 接続は base_url へ集約済み
+            "api_base", "base_url",
+            "api_key",        # SDK 初期化で使用済み
+        }
+        for k in list(payload.keys()):
+            if k in DENY_KEYS:
+                payload.pop(k, None)
 
         model = payload.pop("model", None) or self.model
         if not model:
             raise ValueError("model is not specified for AsyncChatLLM call.")
         # extra_body は payload に入っていればそのまま渡る
         return await self.client.chat.completions.create(
             model=model,
             messages=messages,
             **payload
         )
