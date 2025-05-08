import re

import pandas as pd
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import Template, TimeTablingTemplate


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


# 设置模型和模板
class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.cot),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
        df["setting"] = f"{setting.model}--{setting.template}"

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)

    df["correct_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[0]))
    df["total_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[1]))

    df = df[df["correct_solution_count"] <= df["total_solution_count"]]
    df = df[df["correct_solution_count"] <= df["answer_count"]]
    df = df[df["total_solution_count"] <= df["answer_count"]]

    if dataset == DatasetName.TimeTabling:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50))
    elif dataset == DatasetName.SubsetSum:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50) if int(x // 50) < 6 else 6)
    else:
        raise NotImplementedError

    df["completeness"] = df["correct_solution_count"] / df["answer_count"]
    df["accuracy"] = df["correct_solution_count"] / df["total_solution_count"]

    df["confidence_bin"] = pd.cut(df["model_confidence_extracted"], bins=10, include_lowest=True, labels=False)
    grouped = (
        df.groupby(["setting", "confidence_bin"])
        .agg(
            mean_confidence=("model_confidence_extracted", "mean"),
            mean_accuracy=("accuracy", "mean"),
            count=("confidence_bin", "size"),
        )
        .reset_index()
    )

    # short cot not work but long cot works
    # short_cot_df = df[df["setting"] == "qwen3-8b-no_think--cot"]
    # overconfident_short_cot_ids = short_cot_df[
    #     (short_cot_df["completeness"] > 0)
    #     & (short_cot_df["completeness"] < 0.1)
    #     & (short_cot_df["model_confidence_extracted"] > 0.8)
    # ]["question_id"]
    # long_cot_df = df[df["setting"] == "qwen3-8b-think--simple"]
    # improved_long_cot_ids = long_cot_df[
    #     (long_cot_df["completeness"] > 0.8) & (long_cot_df["model_confidence_extracted"] > 0.7)
    # ]["question_id"]
    # inner = pd.merge(overconfident_short_cot_ids, improved_long_cot_ids, how="inner")
    # print(inner["question_id"])

    # more reflections, worse performance on easy tasks
    # simple_questions = df[df["answer_count_bin"] < 3]
    # decrease_in_completeness = simple_questions[simple_questions["completeness"] < 0.5]
    # many_reflections = decrease_in_completeness[decrease_in_completeness["reflection_times"] > 20]
    # print(many_reflections["question_id"])

    # ### 4. Scaled Long CoT相比 Long CoT Base有显著提升
    scaled_long_cot_df = df[df["setting"] == "qwen3-8b-think--cot-scaling"]
    long_cot_base_df = df[df["setting"] == "qwen3-8b-think--cot"]
    scaled_improvement = scaled_long_cot_df[scaled_long_cot_df["completeness"] > long_cot_base_df["completeness"].mean()]
    print("Scaled Long CoT Improved Instances:")
    print(scaled_improvement["question"])


if __name__ == "__main__":
    run_async(main())
