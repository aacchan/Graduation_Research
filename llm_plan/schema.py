# llm_plan/schema.py
from typing import List, Literal, Tuple, Union
from pydantic import BaseModel, conlist

Coord = conlist(int, min_items=2, max_items=2)

class MoveTo(BaseModel):
    type: Literal["move_to"]
    src: Tuple[int, int]
    dst: Tuple[int, int]

class FireAt(BaseModel):
    type: Literal["fire_at"]
    dst: Tuple[int, int]

Action = Union[MoveTo, FireAt]

class ActionPlan(BaseModel):
    action_plan: List[Action]
