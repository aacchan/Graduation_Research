# tests/test_guided_entrypoint.py
from types import SimpleNamespace
from llm_plan.structured_schemas import get_schema

class DummyClient:
    def __init__(self):
        self.last_payload = None
    class chat:
        class completions:
            create = None

def inject_openai(monkeypatch, dummy):
    # OpenAI クライアントを差し替え
    import llm_plan.controller.async_llm as m
    def _OpenAI(**_):
        class _C:
            def __init__(self):
                self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
            def _create(self, **kwargs):
                dummy.last_payload = kwargs
                msg = SimpleNamespace(content='{"action":"noop","args":{}}')
                choice = SimpleNamespace(message=msg)
                usage = SimpleNamespace(prompt_tokens=1, completion_tokens=1)
                return SimpleNamespace(model="meta-llama/Meta-Llama-3-8B-Instruct", choices=[choice], usage=usage)
        return _C()
    monkeypatch.setattr(m, "OpenAI", _OpenAI)

def test_force_dict_adds_guided_json(monkeypatch):
    dummy = DummyClient()
    inject_openai(monkeypatch, dummy)

    # AsyncLLM を作る
    import llm_plan.controller.async_llm as m
    llm = m.AsyncLLM(base_url="http://localhost:8000/v1", api_key="EMPTY",
                     model="meta-llama/Meta-Llama-3-8B-Instruct")

    # Controller 側（run → get_response → llm.chat）を通す
    from llm_plan.controller.async_gpt_controller import AsyncGPTController
    ctrl = AsyncGPTController(llm=llm, model_id="test")
    import asyncio
    out = asyncio.get_event_loop().run_until_complete(
        ctrl.run("sys", "user", 0.1, force="dict")
    )

    assert dummy.last_payload is not None
    extra = dummy.last_payload.get("extra_body", {})
    assert "guided_json" in extra  # dict スキーマが付与されている

def test_force_none_does_not_add_extra(monkeypatch):
    dummy = DummyClient()
    inject_openai(monkeypatch, dummy)

    import llm_plan.controller.async_llm as m
    llm = m.AsyncLLM(base_url="http://localhost:8000/v1", api_key="EMPTY",
                     model="meta-llama/Meta-Llama-3-8B-Instruct")

    from llm_plan.controller.async_gpt_controller import AsyncGPTController
    ctrl = AsyncGPTController(llm=llm, model_id="test")
    import asyncio
    out = asyncio.get_event_loop().run_until_complete(
        ctrl.run("sys", "user", 0.1, force="none")
    )

    payload = dummy.last_payload or {}
    assert "extra_body" not in payload or not payload["extra_body"]
