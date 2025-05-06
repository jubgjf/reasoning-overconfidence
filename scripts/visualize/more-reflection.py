import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import TimeTablingTemplate


def count_reflections(history_thinking_content: str) -> int:
    reflection_patterns = [
        r"^Wait,.*\n\n",
        r"^Let me double - check.*\n\n",
        r"^Let me think again.*\n\n",
    ]
    combined_pattern = "|".join(reflection_patterns)

    # thinking_steps_by_reflection =
    #     0: thinking...  1: reflection...
    #     2: thinking...  3: reflection...
    #     4: thinking ...                    # Last step must not be reflection
    last_step_start_index, thinking_steps_by_reflection = 0, []
    if history_thinking_content.startswith("<think>\n"):
        history_thinking_content = history_thinking_content.lstrip("<think>\n")
    for m in re.finditer(combined_pattern, history_thinking_content, re.M):
        thinking_steps_by_reflection.append(history_thinking_content[last_step_start_index : m.start()])
        thinking_steps_by_reflection.append(m.group())
        last_step_start_index = m.end()
    thinking_steps_by_reflection.append(history_thinking_content[last_step_start_index:])
    if len(history_thinking_content) == last_step_start_index:
        # Last step is reflection, although this might be impossible. Remove it.
        thinking_steps_by_reflection = thinking_steps_by_reflection[:-2]

    thinking_with_reduced_reflection = []
    for i in range(0, len(thinking_steps_by_reflection), 2):
        thinking_with_reduced_reflection.append(thinking_steps_by_reflection[: i + 1])

    return len(thinking_with_reduced_reflection)


async def main():
    model = ModelName.QWQ_32B
    judge_model = ModelName.QWQ_32B
    dataset = DatasetName.TimeTabling
    template = TimeTablingTemplate.simple
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    records_list = []

    # ===== original =====
    record_cls = dataset.record_cls
    db_logger = Logger(
        db_name=dataset.value,
        table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}--evaluate-by-{judge_model}",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()
    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    if method == MethodName.Verbal_0_100:
        df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    df["setting"] = "original"
    records_list.append(df)

    # ===== more reflection =====
    record_cls = dataset.record_cls
    db_logger = Logger(
        db_name=dataset.value,
        table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}--more-reflection--evaluate-by-{judge_model}",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()
    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    if method == MethodName.Verbal_0_100:
        df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    df["setting"] = "more"
    records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    df["reflection_times_bin"] = df["reflection_times"].apply(lambda x: int(x // 10) if int(x // 10) < 8 else 8)

    df["correct_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[0]))
    df["total_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[1]))

    df = df[df["correct_solution_count"] <= df["total_solution_count"]]
    df = df[df["correct_solution_count"] <= df["answer_count"]]
    df = df[df["total_solution_count"] <= df["answer_count"]]

    df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50) if int(x // 50) < 8 else 8)

    df["correct_accuracy"] = df["correct_solution_count"] / df["answer_count"]
    df["total_accuracy"] = df["total_solution_count"] / df["answer_count"]

    print("original", df[df["setting"] == "original"]["correct_accuracy"].mean())
    print("more    ", df[df["setting"] == "more"]["correct_accuracy"].mean())

    plt.figure()
    sns.lineplot(data=df, x="answer_count_bin", y="reflection_times", hue="setting")
    plt.xlabel("answer count bin")
    plt.ylabel("reflection times")
    plt.title("answer count vs reflection")
    plt.show()

    plt.figure()
    sns.lineplot(data=df, x="reflection_times", y="correct_accuracy", hue="setting")
    plt.xlabel("reflection times")
    plt.ylabel("correct_accuracy")
    plt.title("reflection vs accuracy")
    plt.show()


if __name__ == "__main__":
    run_async(main())
