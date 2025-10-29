# -*- coding: utf-8 -*-
from __future__ import annotations
from typing import Literal, Optional, Dict, Any

JSON_SCHEMA_DRAFT = "https://json-schema.org/draft/2020-12/schema"

def _inventory_object_schema(title: str) -> Dict[str, Any]:
    # 在庫辞書: {'rock/yellow': int>=1, 'paper/purple': int>=1, 'scissors/blue': int>=1}
    return {
        "title": title,
        "type": "object",
        "properties": {
            "rock/yellow":   {"type": "integer", "minimum": 1},
            "paper/purple":  {"type": "integer", "minimum": 1},
            "scissors/blue": {"type": "integer", "minimum": 1},
        },
        "required": ["rock/yellow", "paper/purple", "scissors/blue"],
        "additionalProperties": False,
    }

# --- A) 直近インタラクションの推定在庫 ---
OPPONENT_INV_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "possible_opponent_inventory": _inventory_object_schema("Opponent inventory (last round)")
    },
    "required": ["possible_opponent_inventory"],
    "additionalProperties": False,
    "description": "Infer opponent's inventory from last interaction.",
}

# --- B) 次ラウンドの双方在庫 ---
NEXT_INVENTORIES_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "predicted_opponent_next_inventory": _inventory_object_schema("Predicted opponent next inventory"),
        "my_next_inventory":                 _inventory_object_schema("My next inventory"),
    },
    "required": ["predicted_opponent_next_inventory", "my_next_inventory"],
    "additionalProperties": False,
    "description": "Plan for next round (both sides' inventories).",
}

# --- C) サブゴール出力（行動列） ---
SUBGOAL_SCHEMA: Dict[str, Any] = {
    "$schema": JSON_SCHEMA_DRAFT,
    "type": "object",
    "properties": {
        "action_plan": {
            "type": "array",
            "minItems": 1,
            "maxItems": 6,
            "items": {
                "type": "string",
                # 2種類の関数のみ許容: move_to((x,y), (x,y)) または fire_at((x,y))
                "pattern": r"^(move_to\(\(\d+,\s*\d+\),\s*\(\d+,\s*\d+\)\)|fire_at\(\(\d+,\s*\d+\)\))$"
            }
        }
    },
    "required": ["action_plan"],
    "additionalProperties": False,
    "description": "Low-level action plan (only move_to / fire_at).",
}

def get_schema(
    mode: Literal[
        "none",
        "opponent_inventory",
        "next_inventories",
        "subgoal",
        "opponent_strategy",
    ] = "none"
) -> Optional[Dict[str, Any]]:
    if mode == "opponent_inventory":
        return OPPONENT_INV_SCHEMA
    if mode == "next_inventories":
        return NEXT_INVENTORIES_SCHEMA
    if mode == "subgoal":
        return SUBGOAL_SCHEMA
    if mode == "opponent_strategy":
        return OPP_STRATEGY_SCHEMA
    return None

