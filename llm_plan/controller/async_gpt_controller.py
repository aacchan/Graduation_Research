from typing import Any, Dict, List, Optional, Union

import asyncio
import re
from llm_plan.structured_schemas import get_schema


def _base_model(name: str) -> str:
    # 末尾が -YYYY-MM-DD のモデル名は家族名に丸める（例: gpt-4o-2024-08-06 → gpt-4o）
    return re.sub(r"-20\d{2}-\d{2}-\d{2}$", "", (name or ""))


# The cost per token for each model input.
MODEL_COST_PER_INPUT = {
    'gpt-4': 3e-05,
    'gpt-4-0613': 3e-05,
    'gpt-4o': 1e-05,
    'gpt-4-0125-preview': 1e-05,    # GPT4 Turbo
    'gpt-4o-2024-05-13': 5e-06,      # GPT4-o
    'gpt-3.5-turbo-1106': 1e-06,
    'meta-llama/Llama-3.1-8B-Instruct': 0.0,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 0.0,
}
MODEL_COST_PER_INPUT.setdefault('gpt-4o-2024-08-06', MODEL_COST_PER_INPUT.get('gpt-4o', 0.0))

# The cost per token for each model output.
MODEL_COST_PER_OUTPUT = {
    'gpt-4': 6e-05,
    'gpt-4-0613': 6e-05,
    'gpt-4o': 3e-05,                 # GPT4 Turbo
    'gpt-4-0125-preview': 3e-05,     # GPT4 Turbo
    'gpt-4o-2024-05-13': 1.5e-05,    # GPT4-o
    'gpt-3.5-turbo-1106': 2e-06,
    'meta-llama/Llama-3.1-8B-Instruct': 0.0,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 0.0,
}
MODEL_COST_PER_OUTPUT.setdefault('gpt-4o-2024-08-06', MODEL_COST_PER_OUTPUT.get('gpt-4o', 0.0))


class AsyncGPTController:
    """
    gpt-4 LLM wrapper for async API calls.
    llm: an instance of AsyncChatLLM,
    model_id: a unique id for the model to use
    model_args: arguments to pass to the api call
    """
    def __init__(
        self,
        llm: Any,
        model_id: str,
        **model_args,
    ) -> None:
        self.llm = llm
        self.model_id = model_id
        self.model_args = model_args
        self.all_responses: List[Dict[str, Any]] = []
        self.total_inference_cost = 0.0
        self.guided_backend: str = "lm-format-enforcer"

    def calc_cost(self, response) -> float:
        """
        Calculates the cost of a response from the openai API.
        """
        raw = getattr(response, 'model', None) or getattr(self, 'model', None) or ''
        model_name = _base_model(raw)
        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = (
            MODEL_COST_PER_INPUT.get(model_name, 0.0) * input_tokens
            + MODEL_COST_PER_OUTPUT.get(model_name, 0.0) * output_tokens
        )
        return cost

    def get_prompt(self, system_message: str, user_message: str) -> List[Dict[str, str]]:
        """Get the (zero shot) prompt for the (chat) model."""
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]

    def _merge_extra_body(self, base: Dict[str, Any], addition: Dict[str, Any]) -> Dict[str, Any]:
        """
        model_args に既存 extra_body がある場合に安全にマージする。
        （ネストは shallow merge。必要なら深いマージに変更可）
        """
        merged = dict(base) if base else {}
        eb0 = dict(merged.get("extra_body", {}))
        eb0.update(addition or {})
        if eb0:
            merged["extra_body"] = eb0
        return merged

    async def get_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        *,
        force: str = "none",  # "dict" | "tuple" | "none"
    ) -> Any:
        base_args = dict(self.model_args)
        base_args['temperature'] = temperature
        base_args['top_p'] = 0.9
        base_args['model'] = self.llm.model

        # force に応じて guided_json を注入（none のときは従来通り）
        schema: Optional[Dict[str, Any]] = get_schema(force if force in ("dict", "tuple") else "none")
        if schema is not None:
            base_args = self._merge_extra_body(
                base_args,
                {"guided_json": schema, "guided_decoding_backend": self.guided_backend}
            )
        return await self.llm(messages=messages, **base_args)

    async def run(
        self,
        expertise: str,
        message: str,
        temperature: float,
        *,
        force: str = "none",  # "dict" | "tuple" | "none"
    ) -> Union[str, List[str]]:
        """
        Runs the Code Agent.

        Args:
            expertise (str): The system message to use.
            message (str): The user message to use.
            temperature (float): Sampling temperature.
            force (str): "dict" / "tuple" / "none" — 出力構造の指定。

        Returns:
            str | List[str]: モデルの返答（n=1 のときは str、n>1 のときは List[str]）
        """
        messages = self.get_prompt(system_message=expertise, user_message=message)
        response = await self.get_response(messages=messages, temperature=temperature, force=force)

        cost = self.calc_cost(response=response)
        print(f"Cost for running {self.llm.model}: {cost}")

        if len(response.choices) == 1:
            response_str: Union[str, List[str]] = response.choices[0].message.content
        else:
            response_str = [c.message.content for c in response.choices]

        full_response = {'response': response, 'response_str': response_str, 'cost': cost}
        self.total_inference_cost += cost
        self.all_responses.append(full_response)
        return full_response['response_str']

    async def batch_prompt_sync(
        self,
        expertise: str,
        messages: List[str],
        temperature: float,
    ) -> List[str]:
        """Handles async API calls for batch prompting."""
        responses = [self.run(expertise, message, temperature) for message in messages]
        return await asyncio.gather(*responses)

    def batch_prompt(
        self,
        expertise: str,
        messages: List[str],
        temperature: float,
    ) -> List[str]:
        """Synchronous wrapper for batch_prompt."""
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Loop is already running.")
        return loop.run_until_complete(self.batch_prompt_sync(expertise, messages, temperature))

    async def async_batch_prompt(self, expertise, messages, temperature=None):
        if temperature is None:
            temperature = self.model_args['temperature']
        responses = [self.run(expertise, message, temperature) for message in messages]
        return await asyncio.gather(*responses)
