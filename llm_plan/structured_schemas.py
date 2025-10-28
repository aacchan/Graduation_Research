# -*- coding: utf-8 -*-
"""
Structured output schemas for vLLM guided decoding.

- "dict"     : 汎用の JSON object（Python の dict 相当）
- "tuple"    : 固定長2要素の配列（Python の tuple 相当として受け取る想定）
- "subgoal"  : HM の subgoal_module 用（action_plan: list[str] を必須）
- "none"     : スキーマ無し（出力制約しない）

vLLM の OpenAI互換エンドポイントに送る際は、extra_body に
    {"guided_json": <schema>, "guided_decoding_backend": "<backend>"}
のように渡します。backend は "lm-format-enforcer" 等。
"""
from __future__ import annotations
from typing import Literal, Optional, Dict, Any

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# === 1) 汎用 dict 固定: {"action": string, "args": object} を必須、余計なキー禁止 ===
STEP_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "action": {"type": "string", "description": "次に実行するアクション名"},
        "args":   {"type": "object", "description": "アクション引数（任意のJSONオブジェクト）"},
    },
    "required": ["action", "args"],
    "additionalProperties": False,
    "description": "エージェントの次の一手（dict固定）"
}

# === 2) tuple 固定（固定長2要素の配列）: ["action": string, "priority": integer] ===
TUPLE2_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "array",
    "prefixItems": [
        {"type": "string",  "title": "action",   "description": "アクション名"},
        {"type": "integer", "title": "priority", "description": "優先度（整数）"}
    ],
    "minItems": 2,
    "maxItems": 2,
    "description": "アクションと優先度の固定長タプル（配列）"
}

# === 3) HM subgoal 用: action_plan(list[str]) を必須 ===
SUBGOAL_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "action_plan": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": "次に実行するアクション列（文字列リスト）"
        }
    },
    "required": ["action_plan"],
    # subgoal では追加の補助キー（goal/notes 等）がつくこともあるため許可
    "additionalProperties": True,
    "description": "LLM subgoal planner output. Must include action_plan (list of action strings)."
}

# === 4) 入口 ===
def get_schema(mode: Literal["dict", "tuple", "none", "subgoal"] = "none") -> Optional[Dict[str, Any]]:
    if mode == "subgoal":
        return SUBGOAL_SCHEMA
    if mode == "dict":
        return STEP_JSON_SCHEMA
    if mode == "tuple":
        return TUPLE2_SCHEMA
    return None
