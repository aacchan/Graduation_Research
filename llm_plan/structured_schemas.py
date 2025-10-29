# -*- coding: utf-8 -*-
"""
Structured output schemas for vLLM guided decoding.

mode:
- "subgoal"           : {"action_plan": [str, ...], "subgoal_num"?: int}
- "next_inventories"  : {"my_next_inventory":[int,int,int], "opponent_next_inventory"?:[int,int,int]}
- "dict" / "tuple"    : 既存の学習用（任意で利用）
- "none"              : 非強制
"""
from __future__ import annotations
from typing import Literal, Optional, Dict, Any

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

# ---- 既存の例（任意で残す） -------------------------------------------------
STEP_JSON_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "args":   {"type": "object"},
    },
    "required": ["action", "args"],
    "additionalProperties": False,
}

TUPLE2_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "array",
    "prefixItems": [
        {"type": "string",  "title": "action"},
        {"type": "integer", "title": "priority"},
    ],
    "minItems": 2,
    "maxItems": 2,
}

# ---- 本件で使うスキーマ -----------------------------------------------------
SUBGOAL_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "action_plan": {"type": "array", "items": {"type": "string"}},
        "subgoal_num": {"type": "integer"},
    },
    "required": ["action_plan"],
    "additionalProperties": False,
    "description": "Subgoal/action plan for the agent.",
}

NEXT_INV_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "my_next_inventory": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
            "description": "[Y, P, B] integers",
        },
        "opponent_next_inventory": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 3,
            "maxItems": 3,
        },
    },
    "required": ["my_next_inventory"],
    "additionalProperties": False,
    "description": "Next inventories prediction: JSON only.",
}

def get_schema(mode: Literal[
    "subgoal", "next_inventories", "dict", "tuple", "none"
] = "none") -> Optional[Dict[str, Any]]:
    if mode == "subgoal":
        return SUBGOAL_SCHEMA
    if mode == "next_inventories":
        return NEXT_INV_SCHEMA
    if mode == "dict":
        return STEP_JSON_SCHEMA
    if mode == "tuple":
        return TUPLE2_SCHEMA
    return None
