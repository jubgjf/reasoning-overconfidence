import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence import MethodName
from confidence.model import ModelName
from confidence.template import SubsetSumTemplate, TimeTablingTemplate
from scripts.visualize.metrics import prf


def count_reflections(history_thinking_content: str) -> int:
    reflection_patterns = [
        r"^Wait,.*\n\n",
        r"^But wait,.*\n\n",
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
    model = ModelName.QWEN3_8B_THINK
    # template = SubsetSumTemplate.simple
    template = TimeTablingTemplate.simple
    judge_model = ModelName.QWEN3_32B_NO_THINK
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False
    turn = 0
    temperature = 0.2

    record_cls = dataset.record_cls
    db_logger = Logger(
        db_name=dataset.value + f"--turn{turn}",
        table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}--{temperature}--less-reflection--evaluate-by-{judge_model}",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()

    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    df = prf(df, method, dataset)

    if dataset == DatasetName.TimeTabling:
        # df = df[df["answer_count_bin"] < 3]  # easy
        df = df[df["answer_count_bin"] > 7]  # hard
        pass  # total

    df["overconfidence_rate"] = df["model_confidence_extracted"] - df["recall"]

    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    df["reflection_times_bin"] = df["reflection_times"].apply(lambda x: int(x // 5) if int(x // 5) < 9 else 9)

    # 只保留数据量大于50的reflection_times_bin
    df = df[
        df["reflection_times_bin"].isin(
            df["reflection_times_bin"].value_counts()[df["reflection_times_bin"].value_counts() > 50].index
        )
    ]

    # 假设“做对”标准为recall==1（可根据实际情况调整）
    min_correct_reflection = (
        df[df["recall"] == 1].groupby("question_id")["reflection_times"].min().rename("min_correct_reflection_times")
    )
    df = df.merge(min_correct_reflection, on="question_id", how="left")
    df["reflection_times_norm"] = df["reflection_times"] - df["min_correct_reflection_times"]
    grouped = df.groupby("reflection_times_norm")[["precision", "recall", "overconfidence_rate"]].mean().reset_index()

    plt.figure(figsize=(15, 5))
    for i, metric in enumerate(["precision", "recall", "overconfidence_rate"]):
        valid_df = df[["reflection_times_norm", metric]].dropna()
        corr, p = spearmanr(valid_df["reflection_times_norm"], valid_df[metric])
        significant = False if p >= 0.05 else True
        plt.subplot(1, 3, i + 1)
        sns.lineplot(data=grouped, x="reflection_times_norm", y=metric, marker="o")
        plt.xlabel("Normalized Reflection Times")
        plt.ylabel(metric.capitalize())
        plt.title(
            f"{metric.capitalize()} vs Normalized Reflection Times\n"
            f"Spearmanr Corr: {corr:.2f}, p: {p:.2g} {'(Significant)' if significant else '(Not Significant)'}"
        )
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
