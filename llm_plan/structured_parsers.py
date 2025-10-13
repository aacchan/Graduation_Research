# -*- coding: utf-8 -*-
from __future__ import annotations
import json, re
from typing import Any, Dict, List, Tuple

class ParseError(RuntimeError):
    pass

_JSON_FENCE_OBJ = re.compile(r"```json\s*(\{[\s\S]*?\})\s*```", re.I)
_JSON_FENCE_ARR = re.compile(r"```json\s*(\[[\s\S]*?\])\s*```", re.I)
_OBJ_FALLBACK   = re.compile(r"(\{[\s\S]*\})")

def parse_as_dict(text: str) -> Dict[str, Any]:
    """
    LLMの返答（文字列）から JSON object を取り出して dict にする。
    ```json ...``` を最優先、なければ最初の {} を抽出して json.loads。
    """
    m = _JSON_FENCE_OBJ.search(text)
    raw = m.group(1) if m else None
    if raw is None:
        m2 = _OBJ_FALLBACK.search(text)
        if not m2:
            raise ParseError("Valid JSON object not found in LLM output.")
        raw = m2.group(1)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON decode failed: {e}") from e
    if not isinstance(data, dict):
        raise ParseError(f"Expected JSON object, got: {type(data)}")
    return data

def parse_as_tuple2(text: str) -> Tuple[str, int]:
    """
    LLMの返答（文字列）から 固定長2要素の配列 を取り出して (str, int) にする。
    ```json [...]``` を優先、なければ全文を json.loads。
    """
    m = _JSON_FENCE_ARR.search(text)
    raw = m.group(1) if m else text.strip()
    try:
        arr = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ParseError(f"JSON decode failed: {e}") from e
    if not (isinstance(arr, list) and len(arr) == 2 and isinstance(arr[0], str) and isinstance(arr[1], int)):
        raise ParseError(f"Expected ['action'(str), priority(int)], got: {arr!r}")
    return (arr[0], arr[1])
