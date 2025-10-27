# tests/test_force_none_no_guided.py
import types, asyncio
from llm_plan.controller.async_gpt_controller import AsyncGPTController

class DummyLLM:
    def __init__(self):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.last_kwargs = None
    async def __call__(self, *, messages, structured_schema=None, **kwargs):
        # structured_schema が None のまま来ていることを確認
        assert structured_schema is None
        self.last_kwargs = kwargs  # extra_body の有無はラッパ側の責務（実クライアントならここで渡す）
        usage = types.SimpleNamespace(prompt_tokens=10, completion_tokens=5)
        msg = types.SimpleNamespace(content="free text ok")
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(model=self.model, choices=[choice], usage=usage)

async def main():
    ctl = AsyncGPTController(DummyLLM(), "m")
    text = await ctl.run("sys", "hi", 0.2)  # force 省略 = "none"
    print(text)

if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
