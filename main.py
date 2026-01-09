import os
import time
import json
import asyncio
import argparse
import datetime
import importlib
import subprocess
from typing import Any, Dict, List, Optional

from llm_plan.agent.agent_config import agent_config
from llm_plan.env.mp_llm_env import MeltingPotLLMEnv
from llm_plan.controller.async_llm import AsyncChatLLM
from llm_plan.controller.async_gpt_controller import AsyncGPTController

import llm_plan.controller.async_gpt_controller as _agc
print("[DEBUG] AGC file:", _agc.__file__)
print("[DEBUG] has AsyncGPTController:", hasattr(_agc, "AsyncGPTController"))
print("[DEBUG] has async_batch_prompt:", hasattr(_agc.AsyncGPTController, "async_batch_prompt"))


def setup_environment(substrate_name, scenario_num):
    sprite_label_path = f'./llm_plan/sprite_labels/{substrate_name}'
    env = MeltingPotLLMEnv(substrate_name, sprite_label_path, scenario_num)
    return env


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _default_log_path(io_log_dir: str, mode: str) -> str:
    # json mode still writes a single .json file; jsonl mode writes .jsonl
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ext = "jsonl" if mode == "jsonl" else "json"
    return os.path.join(io_log_dir, f"io_log_{ts}.{ext}")


def _extract_response_text(resp: Any) -> Optional[str]:
    # OpenAI-like response object
    try:
        choices = getattr(resp, "choices", None)
        if choices and len(choices) > 0:
            msg = getattr(choices[0], "message", None)
            if msg is not None:
                return getattr(msg, "content", None)
            # sometimes it's dict-like
            if isinstance(choices[0], dict):
                return choices[0].get("message", {}).get("content")
    except Exception:
        pass
    # dict-like response
    if isinstance(resp, dict):
        try:
            ch = resp.get("choices")
            if ch and len(ch) > 0:
                m = ch[0].get("message", {})
                return m.get("content")
        except Exception:
            return None
    return None


def _extract_usage(resp: Any) -> Optional[Dict[str, Any]]:
    try:
        usage = getattr(resp, "usage", None)
        if usage is None:
            return None
        # pydantic-like objects often have model_dump
        if hasattr(usage, "model_dump"):
            return usage.model_dump()
        if isinstance(usage, dict):
            return usage
        # fallback for objects with attributes
        out = {}
        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]:
            if hasattr(usage, k):
                out[k] = getattr(usage, k)
        return out or None
    except Exception:
        return None


class IOLoggingLLM:
    """
    Wraps AsyncChatLLM to log prompts/responses to JSONL/JSON without touching controller code.

    Controller calls: await llm(messages=..., structured_schema=schema, **base_args)
    """
    def __init__(self, base_llm: Any, log_path: str, mode: str = "jsonl") -> None:
        self._base = base_llm
        self.log_path = log_path
        self.mode = mode or "jsonl"
        self._lock = asyncio.Lock()

        _ensure_parent_dir(self.log_path)
        if self.mode == "json":
            # Initialize file if missing
            if not os.path.exists(self.log_path):
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    async def __call__(self, *args, **kwargs):
        # Extract input
        messages = kwargs.get("messages")
        structured_schema = kwargs.get("structured_schema")
        # Avoid logging secrets
        safe_kwargs = {k: v for k, v in kwargs.items() if k not in {"api_key"}}

        start = time.time()
        resp = await self._base(*args, **kwargs)
        end = time.time()

        record = {
            "ts": datetime.datetime.now().isoformat(),
            "elapsed_sec": round(end - start, 6),
            "model": getattr(self._base, "model", None) or safe_kwargs.get("model"),
            "messages": messages,
            "structured_schema": structured_schema is not None,
            "request_args": {k: v for k, v in safe_kwargs.items() if k not in {"messages", "structured_schema"}},
            "response_text": _extract_response_text(resp),
            "usage": _extract_usage(resp),
        }

        async with self._lock:
            if self.mode == "jsonl":
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            else:
                # json array (overwrite)
                try:
                    with open(self.log_path, "r", encoding="utf-8") as f:
                        arr = json.load(f)
                    if not isinstance(arr, list):
                        arr = []
                except Exception:
                    arr = []
                arr.append(record)
                with open(self.log_path, "w", encoding="utf-8") as f:
                    json.dump(arr, f, ensure_ascii=False, indent=2)

        return resp


def setup_agent(
    api_key,
    model_id,
    model_settings,
    substrate,
    agent_type,
    llm_type='gpt4',
    *,
    io_log_dir: Optional[str] = None,
    io_log_path: Optional[str] = None,
    io_log_mode: Optional[str] = None,
):
    # ---- resolve log path (optional) ----
    mode = io_log_mode or "jsonl"
    log_path: Optional[str] = None
    if io_log_path:
        log_path = io_log_path
        _ensure_parent_dir(log_path)
    elif io_log_dir:
        os.makedirs(io_log_dir, exist_ok=True)
        log_path = _default_log_path(io_log_dir, mode)

    # ---- create LLM + (optional) wrap with logger ----
    if llm_type == 'gpt4':
        llm = AsyncChatLLM(kwargs={'api_key': api_key, 'model': 'gpt-4o'})
        if log_path:
            llm = IOLoggingLLM(llm, log_path=log_path, mode=mode)
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )
    elif llm_type == 'gpt35':
        llm = AsyncChatLLM(kwargs={'api_key': api_key, 'model': 'gpt-3.5-turbo-1106'})
        if log_path:
            llm = IOLoggingLLM(llm, log_path=log_path, mode=mode)
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )
    elif llm_type == 'llama3':
        kwargs = {
            'api_key': "EMPTY",
            'base_url': "http://127.0.0.1",
            'port': 8000,
            'version': 'v1',
            'model': 'meta-llama/Meta-Llama-3-8B-Instruct'
        }
        llm = AsyncChatLLM(kwargs=kwargs)
        if log_path:
            llm = IOLoggingLLM(llm, log_path=log_path, mode=mode)
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )
    elif llm_type == 'llama2':
        kwargs = {
            'api_key': "EMPTY",
            'base_url': "http://localhost",
            'port': 8000,
            'version': 'v1',
            'model': 'meta-llama/Llama-2-13b-chat-hf'
        }
        llm = AsyncChatLLM(kwargs=kwargs)
        if log_path:
            llm = IOLoggingLLM(llm, log_path=log_path, mode=mode)
        controller = AsyncGPTController(
            llm=llm, model_id=model_id, **model_settings
        )

    elif llm_type == 'mixtral':
        kwargs = {
            'api_key': "EMPTY",
            'base_url': "http://localhost",
            'port': 8000,
            'version': 'v1',
            'model': 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        }
        llm = AsyncChatLLM(kwargs=kwargs)
        if log_path:
            llm = IOLoggingLLM(llm, log_path=log_path, mode=mode)
        controller = AsyncGPTController(
            llm=llm,
            model_id=model_id,
            **model_settings
        )

    agent_config_obj = {'agent_id': model_id}

    agent_class_path = agent_config[substrate][agent_type][llm_type]
    agent_module_path, agent_class_name = agent_class_path.rsplit('.', 1)
    agent_module = importlib.import_module(agent_module_path)
    agent_class = getattr(agent_module, agent_class_name)

    if 'hypothetical_minds' in agent_type or 'hm' in agent_type:
        agent_config_obj['self_improve'] = True

    agent = agent_class(agent_config_obj, controller)
    return agent


async def main_async(substrate_name, scenario_num, agent_type, llm_type, io_log_dir=None, io_log_path=None, io_log_mode=None):
    if llm_type == 'gpt4' or llm_type == 'gpt35':
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("No API key found. Please set the OPENAI_API_KEY environment variable.")
    else:
        api_key = "EMPTY"

    if llm_type == 'gpt4':
        model_settings = {
            "model": "gpt-4o",
            "max_tokens": 4000,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 1,
        }
    elif llm_type == 'gpt35':
        model_settings = {
            "model": "gpt-3.5-turbo-1106",
            "max_tokens": 2000,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 1,
        }
    elif llm_type == 'llama3':
        model_settings = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "max_tokens": 2000,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 1,
        }
    elif llm_type == 'llama2':
        model_settings = {
            "model": "meta-llama/Llama-2-13b-chat-hf",
            "max_tokens": 2000,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 10,
        }
    elif llm_type == 'mixtral':
        model_settings = {
            "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "max_tokens": 4096,
            "temperature": 0.2,
            "top_p": 1.0,
            "n": 10,
        }

    agent = setup_agent(
        api_key,
        model_id=f"player_0",
        model_settings=model_settings,
        substrate=substrate_name,
        agent_type=agent_type,
        llm_type=llm_type,
        io_log_dir=io_log_dir,
        io_log_path=io_log_path,
        io_log_mode=io_log_mode,
    )
    agent.agent_type = agent_type
    agent.llm_type = llm_type

    env = setup_environment(substrate_name, scenario_num)

    run_episode_module = importlib.import_module(f"environments.{substrate_name}")
    run_episode = run_episode_module.run_episode

    frame_folder = await run_episode(env, agent)

    # Save video of the frames
    create_video_script = './create_videos.sh'
    subprocess.call([create_video_script, frame_folder])


def main():
    parser = argparse.ArgumentParser(description='Run the multi-agent environment.')
    parser.add_argument('--substrate', type=str, required=True, help='Substrate name')
    parser.add_argument('--scenario_num', type=int, required=True, help='Scenario number')
    parser.add_argument('--agent_type', type=str, default='hm', help='Agent type')
    parser.add_argument('--llm_type', type=str, default='gpt4', help='LLM Type')
    parser.add_argument('--num_seeds', type=int, default=1, help='Number of seeds')

    # Accept both --io_log_dir and --io-log-dir (same for path/mode)
    parser.add_argument('--io_log_dir', '--io-log-dir', dest='io_log_dir', type=str, default=None,
                        help='Directory to write IO logs (prompts/responses) as JSON/JSONL.')
    parser.add_argument('--io_log_path', '--io-log-path', dest='io_log_path', type=str, default=None,
                        help='Explicit IO log file path. If set, overrides --io_log_dir.')
    parser.add_argument('--io_log_mode', '--io-log-mode', dest='io_log_mode', type=str, default='jsonl', choices=['jsonl', 'json'],
                        help='Log format. jsonl (append) or json (overwrite).')

    args = parser.parse_args()

    substrate_dict = {
        'cc': 'collaborative_cooking__asymmetric',
        'rws': 'running_with_scissors_in_the_matrix__repeated',
        'pd': 'prisoners_dilemma_in_the_matrix__repeated',
        'rws_arena': 'running_with_scissors_in_the_matrix__arena',
    }
    substrate_name = substrate_dict[args.substrate]

    loop = asyncio.get_event_loop()
    for seed in range(args.num_seeds):
        loop.run_until_complete(
            main_async(
                substrate_name,
                args.scenario_num,
                args.agent_type,
                args.llm_type,
                args.io_log_dir,
                args.io_log_path,
                args.io_log_mode,
            )
        )


if __name__ == "__main__":
    main()
