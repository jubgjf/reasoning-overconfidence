from abc import ABC, ABCMeta, abstractmethod
from enum import Enum, EnumMeta
from typing import Type, assert_never

from .data import ARCData, Data, GAOKAOData, GSM8KData, LogiQAData, TimeTablingData


class ABCEnumMeta(ABCMeta, EnumMeta): ...


class ITemplate(ABC, Enum, metaclass=ABCEnumMeta):
    def __str__(self):
        return self.value

    @abstractmethod
    def prompt(self, data: Type[Data]) -> str: ...


class GSM8KTemplate(ITemplate):
    BigGSM = "biggsm"

    def prompt(self, data: GSM8KData) -> str:
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
    OpenCompassCoT = "opencompass-cot"

    def prompt(self, data: ARCData) -> str:
        if self == self.OpenCompass:
            # https://github.com/open-compass/opencompass/blob/277d7946f5ac314138b8c30e985ebde87552e474/opencompass/configs/datasets/ARC_c/ARC_c_gen_1e0de5.py#L20
            return (
                f"Question: {data.question}\n" + "\n".join([f"{k}. {v}" for k, v in data.choices.items()]) + "\nAnswer:"
            )
        elif self == self.OpenCompassCoT:
            # https://github.com/open-compass/opencompass/blob/277d7946f5ac314138b8c30e985ebde87552e474/opencompass/configs/datasets/ARC_c/ARC_c_cot_gen_926652.py#L8
            return (
                "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.\n"
                "\n"
                f"{data.question}\n"
                "\n"
            ) + "\n".join([f"{k}. {v}" for k, v in data.choices.items()])
        else:
            assert_never(self)


class LogiQATemplate(ITemplate):
    CoTEval = "coteval"
    CoTEvalCoT = "coteval-cot"

    def prompt(self, data: LogiQAData) -> str:
        if self == self.CoTEval:
            # https://github.com/logikon-ai/cot-eval/blob/5df942c22f4222fe449ac9e413ce5c318f3af08d/eleuther/tasks/logikon/utils_logikon.py#L2
            return (
                (
                    "Answer the following question about the given passage.\n\n"
                    f"Passage: {data.passage}\n\n"
                    f"Question: {data.question}\n"
                )
                + "\n".join([f"{k}. {v}" for k, v in data.choices.items()])
                + "\nAnswer:"
            )
        elif self == self.CoTEvalCoT:
            return (
                (
                    "Answer the following question about the given passage. Think step by step before answering.\n\n"
                    f"Passage: {data.passage}\n\n"
                    f"Question: {data.question}\n"
                )
                + "\n".join([f"{k}. {v}" for k, v in data.choices.items()])
                + "\nAnswer:"
            )


class GAOKAOTemplate(ITemplate):
    OpenCompass = "opencompass"
    OpenCompassCoT = "opencompass-cot"

    def prompt(self, data: GAOKAOData) -> str:
        if self == self.OpenCompass:
            return (
                "请你做一道物理选择题。\n"
                "你将从A，B，C，D中选出所有符合题意的答案，并写在【答案】和<eoa>之间。\n"
                "例如：【答案】 AB <eoa>\n"
                "完整的题目回答的格式如下：\n"
                "【答案】... <eoa>\n"
                "请你严格按照上述格式作答。\n"
                f"{data.question_and_choices}"
            )
        elif self == self.OpenCompassCoT:
            # https://github.com/open-compass/opencompass/blob/b9de8b0e2b47f9561395c6e5fd23bd4ca1e5e4f6/opencompass/configs/datasets/GaokaoBench/GaokaoBench_prompts.py#L33
            return (
                "请你做一道物理选择题。\n"
                "请你一步一步思考并将思考过程写在【解析】和<eoe>之间。你将从A，B，C，D中选出所有符合题意的答案，并写在【答案】和<eoa>之间。\n"
                "例如：【答案】 AB <eoa>\n"
                "完整的题目回答的格式如下：\n"
                "【解析】 ... <eoe>\n"
                "【答案】... <eoa>\n"
                "请你严格按照上述格式作答。\n"
                f"{data.question_and_choices}"
            )
        else:
            assert_never(self)


class TimeTablingTemplate(ITemplate):
    simple = "simple"
    cot = "cot"

    def prompt(self, data: TimeTablingData) -> str:
        if self == self.simple:
            return f"{data.question}\nPlease provide all feasible schedules that satisfies all constraints one by one."
        elif self == self.cot:
            return f"{data.question}\nPlease provide all feasible schedules that satisfies all constraints one by one. Think step by step before answering."
        else:
            assert_never(self)


Template = GSM8KTemplate | ARCTemplate | LogiQATemplate | GAOKAOTemplate | TimeTablingTemplate


def string_to_template(string: str) -> Template:
    template_cls = [GSM8KTemplate, ARCTemplate, LogiQATemplate, GAOKAOTemplate | TimeTablingTemplate]
    for cls in template_cls:
        for t in cls:
            if t.value == string:
                return t
    raise ValueError(f"Unknown template: {string}")
