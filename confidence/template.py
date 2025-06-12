from abc import ABC, ABCMeta, abstractmethod
from enum import Enum, EnumMeta
from typing import Type, assert_never

from .data import Data, TimeTablingData, SubsetSumData


class ABCEnumMeta(ABCMeta, EnumMeta): ...


class ITemplate(ABC, Enum, metaclass=ABCEnumMeta):
    def __str__(self):
        return self.value

    @abstractmethod
    def prompt(self, data: Type[Data]) -> str: ...


class TimeTablingTemplate(ITemplate):
    simple = "simple"
    cot = "cot"

    def prompt(self, data: TimeTablingData) -> str:
        if self == self.simple:
            return (
                "You are asked to perform a timetabling task.\n"
                "Please provide all feasible schedules that satisfies all constraints one by one and output the number of feasible schedules.\n"
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
                "The question is\n"
                f"{data.question}\n"
                "You must output all feasible solutions without using ellipsis, etc."
                "Please note that the examples I gave you are just to show the format, "
                "the actual number of answers may be more than the examples shown."
            )
        elif self == self.cot:
            return (
                "You are asked to perform a timetabling task.\n"
                "Please provide all feasible schedules that satisfies all constraints one by one and output the number of feasible schedules.\n"
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
                "Total xxx feasible solutions shown above.\n"
                "\n"
                "The question is\n"
                f"{data.question}\n"
                "You must output all feasible solutions without using ellipsis, etc."
                "Please note that the examples I gave you are just to show the format, "
                "the actual number of answers may be more than the examples shown."
                "Think step by step before answering."
            )
        else:
            assert_never(self)


class SubsetSumTemplate(ITemplate):
    simple = "simple-subsetsum"
    cot = "cot-subsetsum"

    def prompt(self, data: SubsetSumData) -> str:
        if self == self.simple:
            return (
                "You are asked to perform a subset-sum task.\n"
                "Please provide all feasible subsets that meet the requirements one by one and output the number of feasible subsets.\n"
                "Output format example:\n"
                "Solution 1: {1, 3, 5}\n"
                "Solution 2: {1, 4, 4}\n"
                "...\n"
                "Total xxx feasible solutions shown above.\n"
                "\n"
                "The question is\n"
                f"{data.question}\n"
                "You must output all feasible solutions without using ellipsis, etc."
                "Please note that the examples I gave you are just to show the format, "
                "the actual number of answers may be more than the examples shown."
            )
        elif self == self.cot:
            return (
                "You are asked to perform a subset-sum task.\n"
                "Please provide all feasible subsets that meet the requirements one by one and output the number of feasible subsets.\n"
                "Output format example:\n"
                "Solution 1: {1, 3, 5}\n"
                "Solution 2: {1, 4, 4}\n"
                "...\n"
                "Total xxx feasible solutions shown above.\n"
                "\n"
                "The question is\n"
                f"{data.question}\n"
                "You must output all feasible solutions without using ellipsis, etc."
                "Please note that the examples I gave you are just to show the format, "
                "the actual number of answers may be more than the examples shown."
                "Think step by step before answering."
            )
        else:
            assert_never(self)


Template = TimeTablingTemplate | SubsetSumTemplate


def string_to_template(string: str) -> Template:
    template_cls = [TimeTablingTemplate, SubsetSumTemplate]
    for cls in template_cls:
        for t in cls:
            if t.value == string:
                return t
    raise ValueError(f"Unknown template: {string}")
