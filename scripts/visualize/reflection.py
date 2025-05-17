import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import SubsetSumTemplate, TimeTablingTemplate


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
    template = SubsetSumTemplate.simple
    # template = TimeTablingTemplate.simple
    judge_model = ModelName.QWEN3_32B_NO_THINK
    dataset = DatasetName.SubsetSum
    # dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False

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
    df["reflection_times_bin"] = df["reflection_times"].apply(lambda x: int(x // 5) if int(x // 5) < 9 else 9)

    df["correct_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[0]))
    df["total_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[1]))

    df = df[df["correct_solution_count"] <= df["total_solution_count"]]
    df = df[df["correct_solution_count"] <= df["answer_count"]]
    df = df[df["total_solution_count"] <= df["answer_count"]]

    if dataset == DatasetName.TimeTabling:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50))
        answer_count_bin_range = 10
    elif dataset == DatasetName.SubsetSum:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50) if int(x // 50) < 6 else 6)
        answer_count_bin_range = 7
    else:
        raise NotImplementedError

    df["recall"] = df["correct_solution_count"] / df["answer_count"]
    df["precision"] = df["correct_solution_count"] / df["total_solution_count"]

    # 只保留数据量大于50的reflection_times_bin
    df = df[
        df["reflection_times_bin"].isin(
            df["reflection_times_bin"].value_counts()[df["reflection_times_bin"].value_counts() > 50].index
        )
    ]

    if dataset == DatasetName.TimeTabling:
        df = df[~df["answer_count_bin"].isin([3, 4, 5, 6, 7])]
        df["difficulty"] = df["answer_count_bin"].apply(lambda x: "easy" if x < 3 else "hard")
    elif dataset == DatasetName.SubsetSum:
        df = df[~df["answer_count_bin"].isin([2, 3, 4, 5])]
        df["difficulty"] = df["answer_count_bin"].apply(lambda x: "easy" if x < 2 else "hard")

    plt.figure()
    sns.lineplot(data=df, x="reflection_times_bin", y="recall", hue="difficulty")
    plt.xlabel("Reflection Times Bin")
    plt.ylabel("Recall")
    # plt.title("TimeTabling")
    plt.title("SubsetSum")
    plt.show()

    plt.figure()
    sns.lineplot(data=df, x="reflection_times_bin", y="precision", hue="difficulty")
    plt.xlabel("Reflection Times Bin")
    plt.ylabel("Precision")
    # plt.title("TimeTabling")
    plt.title("SubsetSum")
    plt.show()


if __name__ == "__main__":
    run_async(main())
