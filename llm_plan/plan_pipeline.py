# llm_plan/plan_pipeline.py
import json, re
from typing import Dict
from llm_plan.schema import ActionPlan

JSON_RULES = """
FORMAT RULES (STRICT):
- Output ONLY ONE JSON object. No prose before/after.
- Must match:
  {"action_plan":[
    {"type":"move_to","src":[x,y],"dst":[x,y]},
    {"type":"fire_at","dst":[x,y]}
  ]}
- Use double quotes for all keys/strings. No comments. No trailing commas.
"""

def _extract_outer_json(text: str) -> str:
    # フェンスや余文が混ざっても最外 { ... } を抽出
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found")
    return text[start:end+1]

def generate_action_plan_json(controller, system_message: str, user_message: str,
                              temperature: float = 0.2, max_tries: int = 4) -> Dict:
    prompt = f"{user_message}\n\n{JSON_RULES}"
    last_txt = ""
    for _ in range(max_tries):
        # 1) 生成（低温度推奨：構造重視）
        txt = controller.batch_prompt(system_message, [prompt], temperature=temperature)[0]
        last_txt = txt
        try:
            raw = _extract_outer_json(txt)
            # 2) 厳密検証（ここが“文法で縛る”の要）
            plan = ActionPlan.model_validate_json(raw)  # ← 形・型・必須キーをチェック
            return plan.model_dump()
        except Exception as e:
            # 3) 失敗したら、エラー内容をヒントに再生成（プロンプト追記）
            prompt = (
                f"{user_message}\n\n{JSON_RULES}\n"
                f"Your previous output was invalid: {e}\n"
                f"Fix it and return ONLY one valid JSON object that matches the schema."
            )
            continue
    # 最後まで通らなかった場合は、最後の出力を例外付きで落とす（呼び出し側で拾う）
    raise ValueError(f"Could not produce valid JSON after retries. Last output:\n{last_txt[:400]}")
