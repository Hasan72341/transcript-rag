"""Request and response types shared by the API and retrieval adapter."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

MessageText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=4000)]
HistoryText = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=8000)]


class HistoryMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    text: HistoryText


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: MessageText
    history: list[HistoryMessage] = Field(default_factory=list, max_length=12)


class Turn(BaseModel):
    turn_index: int = Field(ge=0)
    speaker: str
    text: str


class Evidence(BaseModel):
    transcript_id: str
    turn_numbers: list[int]
    relevant_turns: list[Turn]
    full_turns: list[Turn] = Field(default_factory=list)


class TextRecord(BaseModel):
    id: str
    text: str


class BotAnswer(BaseModel):
    answer: str = Field(min_length=1)
    time: float = Field(ge=0, allow_inf_nan=False)
    node_ids: list[str]
    unique_text: list[TextRecord]
    leave_text: list[TextRecord]


class AnalysisResponse(BaseModel):
    bot_answer: BotAnswer
    evidence: list[Evidence]
