import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ttest_ind
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import LogiQATemplate


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

    return len(thinking_with_reduced_reflection) - 1


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
    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
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
    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    confidence_with_reflection = df[df["reflection_times"] > 0]["model_confidence_extracted"]
    confidence_without_reflection = df[df["reflection_times"] == 0]["model_confidence_extracted"]
    t_statistic, p_value = ttest_ind(confidence_with_reflection, confidence_without_reflection)
    print(f"t - statistic: {t_statistic:.2f}")
    print(f"p - value: {p_value:.2f}")

    mean_with_reflection = confidence_with_reflection.mean()
    mean_without_reflection = confidence_without_reflection.mean()
    print(mean_with_reflection - mean_without_reflection)

    top_10_questions = df["question_id"].value_counts().nlargest(5).index
    df_top_10 = df[df["question_id"].isin(top_10_questions)]

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_top_10, x="reflection_times", y="model_confidence_extracted", hue="question_id")
    plt.xlabel("Reflection Times")
    plt.ylabel("Model Confidence")
    plt.legend(title="Question ID")
    plt.show()

    df_top_10 = df_top_10.sort_values(by=["question_id", "reflection_times"])
    df_top_10["confidence_difference"] = df_top_10.groupby("question_id")["model_confidence_extracted"].diff()

    plt.figure(figsize=(12, 8))
    sns.lineplot(data=df_top_10, x="reflection_times", y="confidence_difference", hue="question_id")
    plt.xlabel("Reflection Times")
    plt.ylabel("Model Confidence Difference")
    plt.legend(title="Question ID")
    plt.show()


if __name__ == "__main__":
    run_async(main())
