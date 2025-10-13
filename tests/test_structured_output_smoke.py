# -*- coding: utf-8 -*-
"""
構造強制化（guided decoding）が機能しているかのスモークテスト。
- デフォルト: モックLLMで dict/tuple の guided_json 注入とパースを検証
- --live: vLLM 実機 (OpenAI互換) に対して同じ検証を実行

使い方:
  # モックでの検証
  python -m tests.test_structured_output_smoke

  # 実機（localhost:8000 の vLLM サーバ）に対して検証
  python -m tests.test_structured_output_smoke --live

  # サーバURLやモデルを変える場合
  python -m tests.test_structured_output_smoke --live \
      --base-url http://localhost:8000/v1 \
      --model meta-llama/Meta-Llama-3-8B-Instruct
"""
from __future__ import annotations
import argparse
import asyncio
import json
import sys
from typing import Any, Dict, List, Tuple, Union

# パス通し（リポ直下で -m 実行なら不要だが、保険で追加）
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_plan.controller.async_gpt_controller import AsyncGPTController
from llm_plan.structured_parsers import parse_as_dict, parse_as_tuple2, ParseError
from llm_plan.structured_schemas import get_schema


# ----------------------------
# モックLLM（OpenAI互換レスポンス形だけ真似る）
# ----------------------------
class DummyLLM:
    def __init__(self, payload: str):
        self.model = "meta-llama/Llama-3.1-8B-Instruct"
        self.payload = payload
        self.last_kwargs = None

    async def __call__(self, *, messages, **kwargs):
        # guided_json が extra_body で注入されているか後で見る
        self.last_kwargs = kwargs

        class U:  # usage
            prompt_tokens = 10
            completion_tokens = 5

        class Msg:
            def __init__(self, content: str):
                self.content = content

        class Choice:
            def __init__(self, msg: Msg):
                self.message = msg

        return type("Resp", (), {
            "choices": [Choice(Msg(self.payload))],
            "usage": U(),
            "model": self.model
        })


# ----------------------------
# 実機 vLLM クライアント（OpenAI SDK）
# ----------------------------
class LiveLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        self.model = model
        try:
            from openai import OpenAI
        except Exception as e:
            raise RuntimeError("openai パッケージが必要です: pip install openai") from e
        self.client = OpenAI(base_url=base_url, api_key=api_key)

    async def __call__(self, *, messages, **kwargs):
        # openai-python は同期APIなので、スレッドで回避しても良いが、
        # テストは短いので直接呼んでOK（async関数内でも問題ない）
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            **kwargs
        )


# ----------------------------
# スキーマ妥当性の軽いチェック
# ----------------------------
def _is_valid_step_dict(d: Dict[str, Any]) -> bool:
    return isinstance(d, dict) and "action" in d and "args" in d and isinstance(d["args"], dict)

def _is_valid_tuple2(arr: Tuple[str, int]) -> bool:
    return isinstance(arr, tuple) and len(arr) == 2 and isinstance(arr[0], str) and isinstance(arr[1], int)


# ----------------------------
# テスト本体
# ----------------------------
async def run_mock_tests() -> None:
    print("=== MOCK MODE ===")
    # 1) dict 強制
    ctl1 = AsyncGPTController(llm=DummyLLM('{"action":"search","args":{"q":"rps"}}'), model_id="mock")
    raw1 = await ctl1.run("You are an agent.", "次の一手を action/args で返して", 0.1, force="dict")
    # guided_json が注入されたか
    eb1 = ctl1.llm.last_kwargs.get("extra_body", {})
    assert "guided_json" in eb1, "guided_json が extra_body に見つかりません"
    # JSON として parse
    d = parse_as_dict(raw1)
    assert _is_valid_step_dict(d), f"dict 構造が不正: {d!r}"
    print("[PASS] dict guided & parse")

    # 2) tuple 強制
    ctl2 = AsyncGPTController(llm=DummyLLM('["move", 2]'), model_id="mock")
    raw2 = await ctl2.run("You are an agent.", "['action', priority(int)] を返して", 0.1, force="tuple")
    eb2 = ctl2.llm.last_kwargs.get("extra_body", {})
    assert "guided_json" in eb2, "guided_json が extra_body に見つかりません"
    t = parse_as_tuple2(raw2)
    assert _is_valid_tuple2(t), f"tuple 構造が不正: {t!r}"
    print("[PASS] tuple guided & parse")

    # 3) none（従来動作）: guided_json が付かない
    ctl3 = AsyncGPTController(llm=DummyLLM('自由文でもOK'), model_id="mock")
    raw3 = await ctl3.run("You are an agent.", "自由に返して", 0.1, force="none")
    eb3 = ctl3.llm.last_kwargs.get("extra_body", {})
    assert "guided_json" not in eb3, "force=none で guided_json が付いています"
    assert isinstance(raw3, str), "raw3 は str のはず"
    print("[PASS] none mode (no guided_json)")


async def run_live_tests(base_url: str, model: str) -> None:
    print("=== LIVE MODE ===")
    ctl = AsyncGPTController(llm=LiveLLM(base_url=base_url, model=model), model_id="live")

    # 1) dict 強制
    raw1 = await ctl.run(
        "You are an agent.",
        "次の一手を 'action' と 'args' の JSON object で返して",
        0.1,
        force="dict",
    )
    try:
        d = parse_as_dict(raw1)
    except ParseError as e:
        raise AssertionError(f"dict parse 失敗: {e}\nRAW={raw1!r}")
    assert _is_valid_step_dict(d), f"dict 構造が不正: {d!r}"
    print("[PASS] live dict guided & parse")

    # 2) tuple 強制
    raw2 = await ctl.run(
        "You are an agent.",
        "['action'(string), priority(int)] の配列だけを返して",
        0.1,
        force="tuple",
    )
    try:
        t = parse_as_tuple2(raw2)
    except ParseError as e:
        raise AssertionError(f"tuple parse 失敗: {e}\nRAW={raw2!r}")
    assert _is_valid_tuple2(t), f"tuple 構造が不正: {t!r}"
    print("[PASS] live tuple guided & parse")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true", help="vLLM 実機で検証する")
    ap.add_argument("--base-url", default="http://localhost:8000/v1", help="OpenAI互換API base URL")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct", help="モデル名")
    args = ap.parse_args()

    if args.live:
        asyncio.run(run_live_tests(args.base_url, args.model))
    else:
        asyncio.run(run_mock_tests())


if __name__ == "__main__":
    main()
