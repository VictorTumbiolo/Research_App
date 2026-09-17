from pydantic import BaseModel
from typing import List, Literal

class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class UserQuery(BaseModel):
    question: str
    filename: str
    history: List[ChatTurn] = []