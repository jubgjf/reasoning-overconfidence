import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
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

    return len(thinking_with_reduced_reflection) - 1


async def main():
    model = ModelName.QWQ_32B
    dataset = DatasetName.TimeTabling
    template = TimeTablingTemplate.simple
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    records_list = []

    # ===== original =====
    # record_cls = dataset.record_cls
    # db_logger = Logger(
    #     db_name=dataset.value,
    #     table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}",
    #     record_cls=record_cls,
    # )
    # async with db_logger:
    #     records = await db_logger.fetch()
    # method_records = [record.model_dump() for record in records]
    # df = pd.DataFrame(method_records)
    # if method == MethodName.Verbal_0_100:
    #     df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
    # df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    # df["setting"] = "original"
    # records_list.append(df)

    # ===== less reflection =====
    record_cls = dataset.record_cls
    db_logger = Logger(
        db_name=dataset.value,
        table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}--less-reflection",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()
    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    if method == MethodName.Verbal_0_100:
        df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    df["setting"] = "fake"
    records_list.append(df)

    # ===== more reflection =====
    # record_cls = dataset.record_cls
    # db_logger = Logger(
    #     db_name=dataset.value,
    #     table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}--more-reflection",
    #     record_cls=record_cls,
    # )
    # async with db_logger:
    #     records = await db_logger.fetch()
    # method_records = [record.model_dump() for record in records]
    # df = pd.DataFrame(method_records)
    # if method == MethodName.Verbal_0_100:
    #     df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
    # df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    # df["setting"] = "more"
    # records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    spearman_corr = df["reflection_times"].corr(df["model_confidence_extracted"], method="spearman")
    print(f"Spearman correlation coefficient: {spearman_corr:.4f}")

    top_5_questions = df["question_id"].value_counts().nlargest(5).index
    df_top_5 = df[df["question_id"].isin(top_5_questions)]

    plt.figure(figsize=(12, 8))
    sns.scatterplot(
        data=df_top_5, x="reflection_times", y="model_confidence_extracted", hue="question_id", palette="viridis"
    )
    plt.xlabel("Reflection Times")
    plt.ylabel("Model Confidence")
    plt.legend(title="Question ID")
    plt.title(f"Reflection Times vs. Model Confidence\nSpearman Correlation: {spearman_corr:.4f}")
    plt.show()

    df["has_reflection"] = df["reflection_times"] > 0
    plt.figure(figsize=(5, 8))
    # sns.boxplot(data=df, x="setting", y="model_confidence_extracted", hue="setting")
    sns.boxplot(
        data=df[df["answer_count"] > 80], x="has_reflection", y="model_confidence_extracted", hue="has_reflection"
    )
    # sns.histplot(data=df, x="model_confidence_extracted", hue="setting")
    plt.xlabel("Model Confidence")
    plt.ylabel("Density")
    plt.title("Model Confidence Distribution without Reflections")
    plt.show()

    confidence_with_reflection = df[df["has_reflection"]]["model_confidence_extracted"]
    confidence_without_reflection = df[~df["has_reflection"]]["model_confidence_extracted"]

    # 正态性检验
    _, p_with_reflection = stats.shapiro(confidence_with_reflection)
    _, p_without_reflection = stats.shapiro(confidence_without_reflection)

    # 方差齐性检验
    _, p_levene = stats.levene(confidence_with_reflection, confidence_without_reflection)

    # 根据检验结果选择合适的统计检验方法
    if p_with_reflection > 0.05 and p_without_reflection > 0.05 and p_levene > 0.05:
        # 数据满足正态性和方差齐性，使用独立样本 t 检验
        t_stat, p_t = stats.ttest_ind(confidence_with_reflection, confidence_without_reflection)
        print(f"独立样本 t 检验: t = {t_stat:.4f}, p = {p_t:.4f}")
    else:
        # 数据不满足条件，使用 Mann - Whitney U 检验
        u_stat, p_u = stats.mannwhitneyu(confidence_with_reflection, confidence_without_reflection)
        print(f"Mann - Whitney U 检验: U = {u_stat:.4f}, p = {p_u:.4f}")


if __name__ == "__main__":
    run_async(main())
