from typing import TypeVar

from pydantic import BaseModel


class GSM8KData(BaseModel):
    question_id: int
    question: str
    answer: str
    answer_num: float


class ARCData(BaseModel):
    question_id: str
    question: str
    choices: dict[str, str]
    answer_key: str


class LogiQAData(BaseModel):
    question_id: int
    passage: str
    question: str
    choices: dict[str, str]
    answer_key: str


class GAOKAOData(BaseModel):
    question_id: int
    question_and_choices: str
    answer_keys: str


class TimeTablingData(BaseModel):
    question_id: int
    question: str
    answers: dict[str, str]
    answer_count: int


Data = TypeVar("Data", GSM8KData, ARCData, LogiQAData, GAOKAOData, TimeTablingData)
