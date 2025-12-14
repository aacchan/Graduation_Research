from __future__ import annotations
from transformers import AutoTokenizer
from typing import Any, Dict, List, Optional, Union
import asyncio

from llm_plan.structured_schemas import get_schema


# 料金テーブル（最低限、未知モデルは0円扱い）
MODEL_COST_PER_INPUT: Dict[str, float] = {
    'gpt-4': 3e-05,
    'gpt-4o': 1e-05,
    'gpt-3.5-turbo-1106': 1e-06,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 0.0,
    'meta-llama/Meta-Llama-3-8B-Instruct': 0.0,
}
MODEL_COST_PER_OUTPUT: Dict[str, float] = {
    'gpt-4': 6e-05,
    'gpt-4o': 3e-05,
    'gpt-3.5-turbo-1106': 2e-06,
    'mistralai/Mixtral-8x7B-Instruct-v0.1': 0.0,
    'meta-llama/Meta-Llama-3-8B-Instruct': 0.0,
}


class AsyncGPTController:
    """
    Async chat LLM controller（旧互換のバッチAPI付き）
    - force: "none" | "dict" | "tuple" | "subgoal"
    """
    def __init__(self, llm: Any, model_id: str, **model_args) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            use_fast=True
        )
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True
        )
        print(f"[PROMPT TOKENS] {len(prompt_ids)} | force={force}")
        self.llm = llm
        self.model_id = model_id
        self.model_args = model_args
        self.all_responses: List[Dict[str, Any]] = []
        self.total_inference_cost: float = 0.0

    def calc_cost(self, response) -> float:
        model_name = getattr(response, 'model', '') or ''
        usage = getattr(response, 'usage', None)
        if not usage:
            return 0.0
        input_tokens = getattr(usage, 'prompt_tokens', 0) or 0
        output_tokens = getattr(usage, 'completion_tokens', 0) or 0
        return (
            MODEL_COST_PER_INPUT.get(model_name, 0.0) * input_tokens
            + MODEL_COST_PER_OUTPUT.get(model_name, 0.0) * output_tokens
        )

    def get_prompt(self, system_message: str, user_message: str) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]

    async def get_response(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        *,
        force: str = "none",  # "dict" | "tuple" | "subgoal" | "none"
    ) -> Any:
        # model_args を尊重して最低限だけ上書き
        base_args = dict(self.model_args)
        base_args['temperature'] = temperature
        base_args['model'] = self.llm.model

        # "subgoal" を含め、必要なスキーマを注入（none のときは None）
        schema: Optional[Dict[str, Any]] = get_schema(force)
        return await self.llm(messages=messages, structured_schema=schema, **base_args)

    async def run(
        self,
        expertise: str,
        message: str,
        temperature: float,
        *,
        force: str = "none",
    ) -> Union[str, List[str]]:
        messages = self.get_prompt(system_message=expertise, user_message=message)
        response = await self.get_response(messages=messages, temperature=temperature, force=force)
        cost = self.calc_cost(response)
        model_for_log = self.model_args.get('model', self.llm.model)
        print(f"Cost for running {model_for_log}: {cost}")

        if len(response.choices) == 1:
            response_str: Union[str, List[str]] = response.choices[0].message.content
        else:
            response_str = [c.message.content for c in response.choices]

        full = {'response': response, 'response_str': response_str, 'cost': cost}
        self.total_inference_cost += cost
        self.all_responses.append(full)
        return full['response_str']

    # ===== 旧互換のバッチAPI =====

    async def batch_prompt_sync(
        self,
        expertise: str,
        messages: List[str],
        temperature: float,
        *,
        force: str = "none",
    ) -> List[str]:
        coros = [self.run(expertise, m, temperature, force=force) for m in messages]
        return await asyncio.gather(*coros)

    def batch_prompt(
        self,
        expertise: str,
        messages: List[str],
        temperature: float,
        *,
        force: str = "none",
    ) -> List[str]:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            raise RuntimeError("Loop is already running.")
        return loop.run_until_complete(self.batch_prompt_sync(expertise, messages, temperature, force=force))

    async def async_batch_prompt(
        self,
        expertise: str,
        messages: List[str],
        temperature: Optional[float] = None,
        *,
        force: str = "none",
    ) -> List[str]:
        if temperature is None:
            temperature = self.model_args.get("temperature", 0.2)
        coros = [self.run(expertise, m, temperature, force=force) for m in messages]
        return await asyncio.gather(*coros)



