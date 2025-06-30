from typing import Literal, TypeVar

from pydantic import BaseModel

Template = Literal["simple", "cot"]


class TimeTablingData(BaseModel):
    question_id: int
    question: str
    answers: dict[str, str]
    answer_count: int

    def ask_for_solve(self, template_name: Template) -> str:
        packed_question = (
            "You are asked to perform a timetabling task.\n"
            "Please find ALL FEASIBLE SCHEDULES that satisfies all constraints one by one and output the number of feasible schedules.\n"
            "Output format example:\n"
            "Solution 1:\n"
            "| Course  | Time  | Room  | Teacher  |\n"
            "|---------|-------|-------|----------|\n"
            "| Course0 | T2    | R0    | P0       |\n"
            "| Course1 | T3    | R2    | P2       |\n"
            "| Course2 | T0    | R2    | P1       |\n"
            "Solution 2:\n"
            "| Course  | Time  | Room  | Teacher  |\n"
            "|---------|-------|-------|----------|\n"
            "| Course0 | T2    | R0    | P0       |\n"
            "| Course1 | T3    | R2    | P2       |\n"
            "| Course2 | T1    | R2    | P1       |\n"
            "...\n"
            "\n"
            "Total xxx feasible solutions shown above.\n"
            "\n"
            "The question is\n"
            f"{self.question}\n"
            "You must output all feasible solutions without using ellipsis, etc.\n"
            "The most important thing is to FIND THE SPECIFIC CONTENT OF EACH SOLUTION, "
            "rather than just counting the number of solutions.\n"
            "Please note that the examples I gave you are just to show the format, "
            "the actual answer may be different from the examples shown."
        )
        maps = {"simple": packed_question, "cot": f"{packed_question}\nThink step by step before answering."}
        return maps[template_name]


class SubsetSumData(BaseModel):
    question_id: int
    question: str
    answer_count: int

    def ask_for_solve(self, template_name: Template) -> str:
        packed_question = (
            "You are asked to perform a subset-sum task.\n"
            "Please find ALL FEASIBLE SUBSETS that meet the requirements one by one and output the number of feasible subsets.\n"
            "Output format example:\n"
            "Solution 1: {1, 3, 5}\n"
            "Solution 2: {1, 4, 4}\n"
            "...\n"
            "\n"
            "Total xxx feasible solutions shown above.\n"
            "\n"
            "The question is\n"
            f"{self.question}\n"
            "You must output all feasible solutions without using ellipsis, etc.\n"
            "The most important thing is to FIND THE SPECIFIC CONTENT OF EACH SOLUTION, "
            "rather than just counting the number of solutions.\n"
            "Please note that the examples I gave you are just to show the format, "
            "the actual answer may be different from the examples shown."
        )
        maps = {"simple": packed_question, "cot": f"{packed_question}\nThink step by step before answering."}
        return maps[template_name]


Data = TypeVar("Data", TimeTablingData, SubsetSumData)
