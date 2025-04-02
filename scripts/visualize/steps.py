import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import LogiQATemplate


def count_steps(history_thinking_content: str) -> int:
    return count_steps_and_reflections_positions(history_thinking_content)[0]


def reflection_positions(history_thinking_content: str) -> list[int]:
    return count_steps_and_reflections_positions(history_thinking_content)[1]


def count_steps_and_reflections_positions(history_thinking_content: str) -> tuple[int, list[int]]:
    steps = history_thinking_content.split("\n\n")
    reflection_pos = []

    reflection_patterns = [
        r"Wait,",
        r"Let me double-check",
        r"Let me think again",
    ]
    combined_pattern = "|".join(reflection_patterns)

    for i, step in enumerate(steps):
        if re.match(combined_pattern, step):
            reflection_pos.append(i)

    return len(steps), reflection_pos


async def main():
    model = ModelName.DEEPSEEK_R1_DISTILL_QWEN2_5_MATH_7B
    dataset = DatasetName.LogiQA
    template = LogiQATemplate.CoTEval
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    records_list = []

    # ===== original =====
    record_cls = dataset.record_cls
    db_logger = Logger(
        db_name=dataset.value,
        table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()
    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    if method == MethodName.Verbal_0_100:
        df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
    df["reflection_positions"] = df["model_thinking_response"].apply(reflection_positions)
    df["steps"] = df["model_thinking_response"].apply(count_steps)
    records_list.append(df)

    # ===== no reflection =====
    record_cls = dataset.record_cls
    db_logger = Logger(
        db_name=dataset.value,
        table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}-no-reflection",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()
    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    if method == MethodName.Verbal_0_100:
        df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
    df["reflection_positions"] = df["model_thinking_response"].apply(reflection_positions)
    df["steps"] = df["model_thinking_response"].apply(count_steps)
    records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)
    df = df[df["steps"] <= 100]

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df, x="steps", y="model_confidence_extracted")
    plt.title("Relationship between Reflection Times and Model Confidence for Top 10 Questions")
    plt.xlabel("Steps")
    plt.ylabel("Model Confidence")
    plt.show()


if __name__ == "__main__":
    run_async(main())
