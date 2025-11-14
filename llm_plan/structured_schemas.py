# structured_schemas.py  —— 全置き換え

from typing import Any, Dict, Optional

def _inventory_schema() -> Dict[str, Any]:
    # 在庫は3キー固定のオブジェクトにする（整数・0以上。足切りが厳しすぎると壊れるので minimum=0）
    return {
        "type": "object",
        "properties": {
            "rock/yellow":   {"type": "integer", "minimum": 0},
            "paper/purple":  {"type": "integer", "minimum": 0},
            "scissors/blue": {"type": "integer", "minimum": 0},
        },
        "required": ["rock/yellow", "paper/purple", "scissors/blue"],
        "additionalProperties": False,
    }

SCHEMAS: Dict[str, Dict[str, Any]] = {
    # hls_user_msg1 の後：対戦相手の直近在庫
    "opponent_inventory": {
        "type": "object",
        "properties": {
            "possible_opponent_inventory": _inventory_schema()
        },
        "required": ["possible_opponent_inventory"],
        "additionalProperties": False,
    },

    # hls_user_msg2 の後：相手方の戦略要約（文字列でOK）
    "opponent_strategy": {
        "type": "object",
        "properties": {
            "Opponent_strategy": {"type": "string", "minLength": 1}
        },
        "required": ["Opponent_strategy"],
        "additionalProperties": False,
    },

    # hls_user_msg3 の後：次ラウンドの相手と自分の在庫推定
    "next_inventories": {
        "type": "object",
        "properties": {
            "predicted_opponent_next_inventory": _inventory_schema(),
            "my_next_inventory": _inventory_schema(),
        },
        "required": ["predicted_opponent_next_inventory", "my_next_inventory"],
        "additionalProperties": False,
    },

    # subgoal_module 用：アクション計画は「文字列の配列」を JSON で返す
    "subgoal": {
        "type": "object",
        "properties": {
            "action_plan": {
                "type": "array",
                "minItems": 1,
                "maxItems": 6,
                "items": {
                    "type": "string",
                    # move_to(...) / fire_at(...) などの関数呼び出しを素直に文字列で受ける
                    "minLength": 5
                }
            }
        },
        "required": ["action_plan"],
        "additionalProperties": False,
    },
}

class StructuredSchemaFactory:
    @staticmethod
    def get(tag: str) -> Optional[Dict[str, Any]]:
        return SCHEMAS.get(tag)


