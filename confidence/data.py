from typing import TypeVar

from pydantic import BaseModel


class GSM8KData(BaseModel):
    id: int
    question: str
    answer: str
    answer_num: float


class ARCData(BaseModel):
    id: str
    question: str
    choices: dict[str, str]
    answer_key: str


class LogiQAData(BaseModel):
    id: int
    passage: str
    question: str
    choices: dict[str, str]
    answer_key: str


class GAOKAOData(BaseModel):
    id: int
    question_and_choices: str
    answer_keys: str


Data = TypeVar("Data", GSM8KData, ARCData, LogiQAData, GAOKAOData)
