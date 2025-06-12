from typing import TypeVar

from pydantic import BaseModel


class TimeTablingData(BaseModel):
    question_id: int
    question: str
    answers: dict[str, str]
    answer_count: int


class SubsetSumData(BaseModel):
    question_id: int
    question: str
    answer_count: int


Data = TypeVar("Data", TimeTablingData, SubsetSumData)
