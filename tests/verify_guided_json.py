# -*- coding: utf-8 -*-
from __future__ import annotations
import json
from typing import Any, Dict
from jsonschema import validate, ValidationError
from openai import OpenAI

# vLLM OpenAI互換サーバへ接続
client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

# === スキーマ定義（dict / tuple） ==============================
DICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "args":   {"type": "object"},
    },
    "required": ["action", "args"],
    "additionalProperties": False,
}

TUPLE2_SCHEMA: Dict[str, Any] = {
    "type": "array",
    "prefixItems": [
        {"type": "string"},   # action
        {"type": "integer"},  # priority
    ],
    "minItems": 2,
    "maxItems": 2,
}

def call(prompt: str, extra_body: Dict[str, Any] | None = None) -> str:
    resp = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{"role":"user","content":prompt}],
        extra_body=extra_body,
        temperature=0.2,
    )
    return resp.choices[0].message.content

def try_parse_json(s: str) -> Any:
    return json.loads(s)

def show(title: str, content: str):
    print("="*80)
    print(title)
    print("-"*80)
    print(content)

def main():
    # A) guided_json なし（フリーテキストを意図）
    prompt_free = "次の一手を説明してください（日本語の文章で教えて。JSONは不要）"
    out_free = call(prompt_free)  # extra_body を渡さない
    show("A) guided_json なし（自由出力）", out_free)
    # JSON でない可能性が高いので、json.loads は失敗し得る（ここでは敢えてパースしない）

    # B) guided_json（dict）あり：必ず {"action":..., "args":{...}} になるはず
    prompt_dict = "次の一手を 'action'(str) と 'args'(object) で返してください。"
    out_dict = call(
        prompt_dict,
        extra_body={
            "guided_json": DICT_SCHEMA,
            "guided_decoding_backend": "lm-format-enforcer",
        },
    )
    show("B) guided_json=DICT_SCHEMA", out_dict)
    try:
        obj = try_parse_json(out_dict)
        validate(instance=obj, schema=DICT_SCHEMA)
        print("→ ✅ dict スキーマ適合を確認しました。")
    except (json.JSONDecodeError, ValidationError) as e:
        print("→ ❌ dict スキーマ検証に失敗:", e)

    # C) guided_json（tuple=固定長配列）あり：必ず ["<action>", <int>] になるはず
    prompt_tuple = "['action'(string), priority(int)] の配列で返してください。"
    out_tuple = call(
        prompt_tuple,
        extra_body={
            "guided_json": TUPLE2_SCHEMA,
            "guided_decoding_backend": "lm-format-enforcer",
        },
    )
    show("C) guided_json=TUPLE2_SCHEMA", out_tuple)
    try:
        arr = try_parse_json(out_tuple)
        validate(instance=arr, schema=TUPLE2_SCHEMA)
        print("→ ✅ tuple(固定長配列) スキーマ適合を確認しました。")
    except (json.JSONDecodeError, ValidationError) as e:
        print("→ ❌ tuple(固定長配列) スキーマ検証に失敗:", e)

if __name__ == "__main__":
    main()
