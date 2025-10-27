import asyncio
from llm_plan.controller.async_gpt_controller import AsyncGPTController

class DummyLLM:
    def __init__(self):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.last = None
    async def __call__(self, *, messages, **kwargs):
        self.last = kwargs
        class U: prompt_tokens=10; completion_tokens=5
        class Msg: content = "ok"
        class Ch: message = Msg()
        return type("Resp", (), {"choices": [Ch()], "usage": U(), "model": self.model})

async def main():
    ctl = AsyncGPTController(DummyLLM(), "test")
    print(await ctl.run("sys", "hi", 0.1))
    print(await ctl.run("sys", "dict test", 0.1, force="dict"))
    print(await ctl.run("sys", "tuple test", 0.1, force="tuple"))

if __name__ == "__main__":
    asyncio.run(main())
