from abc import ABC, abstractmethod, ABCMeta
from enum import Enum, EnumMeta
from typing import TypeVar, assert_never, Type

from pydantic import BaseModel


class ABCEnumMeta(ABCMeta, EnumMeta): ...


class IData(BaseModel): ...


class GSM8KData(IData):
    id: int
    question: str
    answer: str
    answer_num: float


class ARCData(IData):
    id: str
    question: str
    choices: dict[str, str]
    answer_key: str


Data = TypeVar("Data", GSM8KData, ARCData)


class ITemplate(ABC, Enum, metaclass=ABCEnumMeta):
    def __str__(self):
        return self.value

    @abstractmethod
    def prompt(self, data_fields: Type[IData]) -> str: ...


class GSM8KTemplate(ITemplate):
    BigGSM = "biggsm"

    def prompt(self, data: "GSM8KData") -> str:
        if self == self.BigGSM:
            # https://github.com/LightChen233/reasoning-boundary/blob/908b7bf0348d6f3b0696c3cef58d7ba4aeb45314/request_marp.py#L21
            return (
                "You need to perform multi-step reasoning, with each step carrying out as many basic operations as possible.\n"
                "\n"
                "Remember, you can only complete tasks that contain up to 5 basic operations per step, and multiplication operations must be less than 1.5e5. The upper limit of the multiplication operations decreases as the number of operations per step increases.\n"
                "\n"
                "[EXAMPLE]\n"
                "Question: Leo's assignment was divided into three parts. He finished the first part of his assignment in 25 minutes. It took him twice as long to finish the second part. If he was able to finish his assignment in 2 hours, how many minutes did Leo finish the third part of the assignment?\n"
                "Answer: Leo finished the first and second parts of the assignment in 25 + 25*2 = <<25+25*2=75>>75 minutes.\n"
                "Therefore, it took Leo 60 x 2 - 75 = <<60*2-75=45>>45 minutes to finish the third part of the assignment.\n"
                "#### 45\n"
                "\n"
                "Question: Liza bought 10 kilograms of butter to make cookies. She used one-half of it for chocolate chip cookies, one-fifth of it for peanut butter cookies, and one-third of the remaining butter for sugar cookies. How many kilograms of butter are left after making those three kinds of cookies?\n"
                "Answer: Liza used 10 / 2 + 10 / 5 = <<10/2+10/5=7>>7 kilograms of butter for the chocolate and peanut butter cookies.\n"
                "Then, Liza used (10 - 7) / 3 = <<(10-7)/3=1>>1 kilograms of butter for the sugar cookies.\n"
                "Therefore, only 10-7-1 = <<10-7-1=2>>2 kilograms of butter were left.\n"
                "#### 2\n"
                "\n"
                "Question: Tina makes $18 an hour.  If she works more than 8 hours per shift, she is eligible for overtime, which is paid by your hourly wage + 1/2 your hourly wage.  If she works 10 hours every day for 5 days, how much money does she make?\n"
                "Answer: She works 5 days and makes 5 * 8 * $18 = $<<8*18*5=720>>720 regular pay.\n"
                "Her overtime pay is 18+18*0.5 = $<<18+18*0.5=27>>27.\n"
                "She works 2 hours of overtime for 5 days and makes 27*2*5 = $<<27*(10-8)*5=270>>270 in overtime pay.\n"
                "She makes $720 + $270 = $<<720+270=990>>990.\n"
                "#### 990\n"
                "\n"
                "[REQUEST]\n"
                f"Question: {data.question}"
            )
        else:
            assert_never(self)


class ARCTemplate(ITemplate):
    OpenCompass = "opencompass"

    def prompt(self, data: "ARCData") -> str:
        if self == self.OpenCompass:
            # https://github.com/open-compass/opencompass/blob/277d7946f5ac314138b8c30e985ebde87552e474/opencompass/configs/datasets/ARC_c/ARC_c_gen_1e0de5.py#L20
            return (
                f"Question: {data.question}\n" + "\n".join([f"{k}. {v}" for k, v in data.choices.items()]) + "\nAnswer:"
            )
        else:
            assert_never(self)


Template = TypeVar("Template", GSM8KTemplate, ARCTemplate)


def string_to_template(string: str) -> GSM8KTemplate | ARCTemplate:
    for t in GSM8KTemplate:
        if t.value == string:
            return t
    for t in ARCTemplate:
        if t.value == string:
            return t
    raise ValueError(f"Unknown template: {string}")


class IRecord(ABC, BaseModel):
    model_answer_response: str
    model_answer_extracted: str
    model_confidence_response: str
    model_confidence_extracted: float
    template: str
    method: str
    history: dict  # dict[str, str]
    model: str


class GSM8KRecord(IRecord, GSM8KData): ...


class ARCRecord(IRecord, ARCData): ...


Record = TypeVar("Record", GSM8KRecord, ARCRecord)
