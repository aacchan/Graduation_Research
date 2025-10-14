# -*- coding: utf-8 -*-
"""
Structured output schemas for vLLM guided decoding.

- "dict"  : {"action_plan": [ "<command>", ... ]} 固定（本ファイルではこれを "dict" として採用）
- "tuple" : 固定長2要素の配列（["action", priority]）
- "none"  : スキーマ無し（制約しない）

使い方（OpenAI互換API; vLLM）:
  extra_body = {
      "guided_json": get_schema("dict"),  # または "tuple"/"none"
      "guided_decoding_backend": "lm-format-enforcer"
  }

本ファイルの "dict" は、以下のコマンド列を表す action_plan に最適化しています:

  そのまま使える最小テンプレ
    {
      "action_plan": [
        "move_to((x_start, y_start), (x_goal, y_goal))",
        "interact((x_resource, y_resource))",
        "wait((x_spot, y_spot))"
      ]
    }

例1：近くの Paper を確保 → 中央へ
    {
      "action_plan": [
        "move_to((3, 2), (5, 2))",
        "interact((6, 2))",
        "move_to((5, 2), (8, 3))"
      ]
    }

例2：相手が Rock 在庫多そう → Paper 連取り
    {
      "action_plan": [
        "move_to((4, 1), (6, 1))",
        "interact((7, 1))",
        "move_to((6, 1), (7, 4))",
        "interact((8, 4))"
      ]
    }

例3：不利マッチアップ回避 → Scissors へスイッチ
    {
      "action_plan": [
        "move_to((8, 5), (5, 5))",
        "move_to((5, 5), (2, 4))",
        "interact((1, 4))"
      ]
    }

例4：射線確保のために一時待機
    {
      "action_plan": [
        "move_to((7, 2), (9, 3))",
        "wait((9, 3))"
      ]
    }
"""
from __future__ import annotations
from typing import Literal, Optional, Dict, Any

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# ------------------------------------------------------------
# 1) dict 固定: {"action_plan": [ "<command>", ... ] }
#    <command> は以下の3種類のいずれか（厳密チェックは regex）
#      - move_to((x1, y1), (x2, y2))
#      - interact((x, y))
#      - wait((x, y))
#
#    ※ x,y は整数（正/負/0）を許容。空白は任意。
# ------------------------------------------------------------
# 正規表現（JSON Schema用; ^...$ で全体一致）
# - move_to: move_to((  int , int  ), (  int , int  ))
# - interact: interact(( int , int ))
# - wait:     wait(( int , int ))
# Pythonの生文字列で書き、JSON Schema の "pattern" に渡す
_INT = r"-?\d+"
_WS = r"\s*"
MOVE_TO_PATTERN = (
    r"^move_to\("
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)" + r"," + _WS +
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)"
    r"\)\)$"
)
INTERACT_PATTERN = (
    r"^interact\("
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)"
    r"\)$"
)
WAIT_PATTERN = (
    r"^wait\("
    r"\(" + _WS + _INT + _WS + "," + _WS + _INT + _WS + r"\)"
    r"\)$"
)

STEP_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "action_plan": {
            "type": "array",
            "description": "実行すべきコマンド列",
            "items": {
                # 3種類のコマンドのいずれか一つにマッチ
                "type": "string",
                "oneOf": [
                    {"type": "string", "pattern": MOVE_TO_PATTERN, "description": "move_to((x1,y1),(x2,y2))"},
                    {"type": "string", "pattern": INTERACT_PATTERN, "description": "interact((x,y))"},
                    {"type": "string", "pattern": WAIT_PATTERN, "description": "wait((x,y))"},
                ],
            },
            "minItems": 1,
            # 必要に応じて上限を付けたい場合は有効化:
            # "maxItems": 32
        }
    },
    "required": ["action_plan"],
    "additionalProperties": False,
    "description": "エージェントの行動計画（action_plan配列）"
}

# ------------------------------------------------------------
# 2) tuple 固定（固定長2要素の配列）: ["action"(str), priority(int)]
#    既存の "tuple" 用スキーマも維持します。
# ------------------------------------------------------------
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

# ------------------------------------------------------------
# 3) API
# ------------------------------------------------------------
def get_schema(mode: Literal["dict", "tuple", "none"] = "none") -> Optional[Dict[str, Any]]:
    """
    使い方:
        schema = get_schema("dict")    # {"action_plan": [...] } のスキーマ
        schema = get_schema("tuple")   # ["action", priority] のスキーマ
        schema = get_schema("none")    # 制約なし（None 返す）

    返り値:
        dict | None
    """
    if mode == "dict":
        return STEP_JSON_SCHEMA
    if mode == "tuple":
        return TUPLE2_SCHEMA
    return None

