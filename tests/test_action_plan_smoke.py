# -*- coding: utf-8 -*-
"""
action_plan（dict固定）の guided decoding スモークテスト。
- デフォルト: モックで通す
- --live: vLLM 実機 (OpenAI互換) で本当に JSON だけ＆パターン合致するか確認
"""
from __future__ import annotations
import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from llm_plan.controller.async_gpt_controller import AsyncGPTController

# 仕様と同等の正規表現
_INT = r"-?\d+"
_WS = r"\s*"
MOVE_TO = re.compile(
    r"^move_to\("
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)" + r"," + _WS +
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)"
    r"\)\)$"
)
INTERACT = re.compile(
    r"^interact\("
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)"
    r"\)$"
)
WAIT = re.compile(
    r"^wait\("
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)"
    r"\)$"
)

def _ok(cmd: str) -> bool:
    return bool(MOVE_TO.match(cmd) or INTERACT.match(cmd) or WAIT.match(cmd))


# -------- 実機 vLLM --------
class LiveLLM:
    def __init__(self, base_url: str, model: str, api_key: str = "EMPTY"):
        from openai import OpenAI
        self.model = model
        self.client = OpenAI(base_url=base_url, api_key=api_key)
    async def __call__(self, *, messages, **kwargs):
        # Controller 側で kwargs に model が入っているので、重複させない
        kwargs.setdefault("model", self.model)
        return self.client.chat.completions.create(messages=messages, **kwargs)

async def run_live(base_url: str, model: str):
    print("=== LIVE ===")
    ctl = AsyncGPTController(LiveLLM(base_url, model), "live")
    prompt = (
        "出力は JSON オブジェクトのみ。"
        "キーは 'action_plan' のみ。値はコマンド文字列の配列。"
        "各要素は次のいずれかに厳密一致: "
        "move_to((x1, y1), (x2, y2)) / interact((x, y)) / wait((x, y))。"
        "x,y は整数。説明やコードフェンスは出力しない。"
    )
    raw = await ctl.run("You are an agent.", prompt, 0.0, force="dict")
    s = raw.strip()
    assert s.startswith("{") and s.endswith("}"), f"JSON以外が混入: {raw!r}"
    data = json.loads(s)
    assert list(data.keys()) == ["action_plan"], f"余計なキー: {data.keys()}"
    assert isinstance(data["action_plan"], list) and len(data["action_plan"]) >= 1
    assert all(_ok(cmd) for cmd in data["action_plan"]), f"パターン不一致: {data['action_plan']!r}"
    print("[PASS] live dict guided & JSON only & pattern OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    args = ap.parse_args()
    if args.live:
        asyncio.run(run_live(args.base_url, args.model))
    else:
        asyncio.run(run_mock())

if __name__ == "__main__":
    main()
