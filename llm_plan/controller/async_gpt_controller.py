# llm_plan/controller/async_gpt_controller.py
from typing import Any, Dict, List, Optional, Union
import asyncio
import re
from llm_plan.structured_schemas import get_schema

def _base_model(name: str) -> str:
    # 末尾が -YYYY-MM-DD のモデル名は家族名に丸める（例: gpt-4o-2024-08-06 → gpt-4o）
    return re.sub(r"-20\d{2}-\d{2}-\d{2}$", "", (name or ""))

MODEL_COST_PER_INPUT = {
    "gpt-4": 3e-05,
    "gpt-4-0613": 3e-05,
    "gpt-4o": 1e-05,
    "gpt-4-0125-preview": 1e-05,
    "gpt-4o-2024-05-13": 5e-06,
    "gpt-3.5-turbo-1106": 1e-06,
    "meta-llama/Llama-3.1-8B-Instruct": 0.0,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.0,
}
MODEL_COST_PER_INPUT.setdefault("gpt-4o-2024-08-06", MODEL_COST_PER_INPUT.get("gpt-4o", 0.0))

MODEL_COST_PER_OUTPUT = {
    "gpt-4": 6e-05,
    "gpt-4-0613": 6e-05,
    "gpt-4o": 3e-05,
    "gpt-4-0125-preview": 3e-05,
    "gpt-4o-2024-05-13": 1.5e-05,
    "gpt-3.5-turbo-1106": 2e-06,
    "meta-llama/Llama-3.1-8B-Instruct": 0.0,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 0.0,
}
MODEL_COST_PER_OUTPUT.setdefault("gpt-4o-2024-08-06", MODEL_COST_PER_OUTPUT.get("gpt-4o", 0.0))

class AsyncGPTController:
    """
    LLM wrapper for async API calls (with optional structured outputs).
    """
    def __init__(self, llm: Any, model_id: str, **model_args) -> None:
        self.llm = llm
        self.model_id = model_id
        self.model_args = model_args
        self.all_responses: List[Dict[str, Any]] = []
        self.total_inference_cost = 0.0
        self.guided_backend: str = "lm-format-enforcer"

    def calc_cost(self, response) -> float:
        raw = getattr(response, "model", None) or getattr(self, "model", None) or ""
        model_name = _base_model(raw)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        return (
            MODEL_COST_PER_INPUT.get(model_name, 0.0) * input_tokens
            + MODEL_COST_PER_OUTPUT.get(model_name, 0.0) * output_tokens
        )

    def get_prompt(self, system_message: str, user_message: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    async def get_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        *,
        force: str = "none",  # "dict" | "tuple" | "none"
    ) -> Any:
        base_args = dict(self.model_args)
        base_args["temperature"] = temperature
        base_args["top_p"] = 0.9
        base_args["model"] = self.llm.model

        schema: Optional[Dict[str, Any]] = get_schema(force if force in ("dict", "tuple") else "none")
        # self.llm は structured_schema=None を無視できる実装（async_llm.py）
        return await self.llm(messages=messages, structured_schema=schema, **base_args)

    async def run(
        self,
        expertise: str,
        message: str,
        temperature: float,
        *,
        force: str = "none",  # "dict" | "tuple" | "none"
    ) -> Union[str, List[str]]:
        messages = self.get_prompt(system_message=expertise, user_message=message)
        response = await self.get_response(messages=messages, temperature=temperature, force=force)

        cost = self.calc_cost(response=response)
        print(f"Cost for running {self.llm.model}: {cost}")

        if len(response.choices) == 1:
            response_str: Union[str, List[str]] = response.choices[0].message.content
        else:
            response_str = [c.message.content for c in response.choices]

        full_response = {"response": response, "response_str": response_str, "cost": cost}
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
        return full_response["response_str"]
