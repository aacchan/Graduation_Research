# -*- coding: utf-8 -*-
"""
Structured output schemas for vLLM guided decoding.

- "dict"  : JSON object（Python の dict 相当）
- "tuple" : 固定長の配列（Python の tuple 相当として受け取る想定）
- "none"  : スキーマ無し（= 従来どおり制約しない）

vLLM の OpenAI互換エンドポイントに送る際は、extra_body に
    {"guided_json": <schema>, "guided_decoding_backend": "<backend>"}
のように渡します。backend は "lm-format-enforcer" や "outlines" など。

参考:
- vLLM Structured Outputs（guided_json / guided_regex / guided_choice ...）
- JSON Schema の tuple 検証は prefixItems + minItems/maxItems を使用
"""
from __future__ import annotations
from typing import Literal, Optional, Dict, Any

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# === 1) dict 固定: {"action": string, "args": object} を必須、余計なキー禁止 ===
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

# === 3) “指定なし” 用のダミー（None を返す） ===
def get_schema(mode: Literal["dict", "tuple", "none"] = "none") -> Optional[Dict[str, Any]]:
    """
    使い方:
        schema = get_schema("dict")   # or "tuple" / "none"
        extra_body = {"guided_json": schema, "guided_decoding_backend": "lm-format-enforcer"}
        # ※ mode="none" の場合は schema=None を返すので、extra_body は付けない

    戻り値:
        dict | None
    """
    if mode == "dict":
        return STEP_JSON_SCHEMA
    if mode == "tuple":
        return TUPLE2_SCHEMA
    return None
