import pandas as pd
import random
import difflib
import numpy as np
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import Template, TimeTablingTemplate


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False
    turn = 0

    settings = [
        # Setting(model=ModelName.QWEN3_8B_THINK, template=SubsetSumTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value + f"--turn{turn}",
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
        df["scaling"] = "none"
        records_list.append(df)

        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value + f"--turn{turn}",
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--more-reflection--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()
        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
        df["setting"] = f"{setting.model}--{setting.template}"
        df["scaling"] = "vertical"
        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

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

    df["recall"] = df["correct_solution_count"] / df["answer_count"]
    df["precision"] = df["correct_solution_count"] / df["total_solution_count"]

    if dataset == DatasetName.TimeTabling:
        # df = df[df["answer_count_bin"] < 3]  # easy
        # df = df[df["answer_count_bin"] > 7]  # hard
        pass  # total
    elif dataset == DatasetName.SubsetSum:
        # df = df[df["answer_count_bin"] < 2]  # easy
        # df = df[df["answer_count_bin"] > 5]  # hard
        pass  # total
    else:
        raise NotImplementedError

    vertical_recall_mean = df[df["scaling"] == "vertical"]["recall"].mean()
    none_recall_mean = df[df["scaling"] == "none"]["recall"].mean()
    print(f"Vertical Recall: {vertical_recall_mean * 100:.2f}")
    print(f"None Recall: {none_recall_mean * 100:.2f}")

    vertical_precision_mean = df[df["scaling"] == "vertical"]["precision"].mean()
    none_precision_mean = df[df["scaling"] == "none"]["precision"].mean()
    print(f"Vertical Precision: {vertical_precision_mean * 100:.2f}")
    print(f"None Precision: {none_precision_mean * 100:.2f}")

    vertical_confidence_mean = df[df["scaling"] == "vertical"]["model_confidence_extracted"].mean()
    none_confidence_mean = df[df["scaling"] == "none"]["model_confidence_extracted"].mean()
    print(f"Vertical Confidence: {vertical_confidence_mean}")
    print(f"None Confidence: {none_confidence_mean}")

    ece_dict = {}
    for scaling, group in df.groupby("scaling"):
        bins = np.linspace(0, 1, 11)
        group["bin"] = pd.cut(group["model_confidence_extracted"], bins=bins, include_lowest=True, labels=False)
        ece = 0
        N = len(group)
        for b in range(10):
            bin_data = group[group["bin"] == b]
            if len(bin_data) == 0:
                continue
            acc = (bin_data["recall"] == 1).mean()  # 或用 precision
            conf = bin_data["model_confidence_extracted"].mean()
            ece += len(bin_data) / N * abs(acc - conf)
        ece_dict[scaling] = ece

    print(f"Vertical ECE: {ece_dict.get('vertical', float('nan')) * 100:.2f}")
    print(f"None ECE: {ece_dict.get('none', float('nan')) * 100:.2f}")

    # 1. 找到同时存在vertical和none的question_id
    scaling_counts = df.groupby(["question_id"])["scaling"].nunique()
    valid_qids = scaling_counts[scaling_counts == 2].index

    # 2. 只保留这些question_id的数据
    df_valid = df[df["question_id"].isin(valid_qids)]

    # 3. 对每个question_id，判断vertical和none的answer_count是否不同
    def has_diff_answer_count(group):
        counts = group.groupby("scaling")["total_solution_count"].unique()
        if "vertical" in counts and "none" in counts:
            return set(counts["vertical"]) != set(counts["none"])
        return False

    def has_diff_total_solution_count(group):
        counts = group.groupby("scaling")["total_solution_count"].unique()
        if "vertical" in counts and "none" in counts:
            return set(counts["vertical"]) != set(counts["none"])
        return False

    qids_with_diff = [qid for qid, group in df_valid.groupby("question_id") if has_diff_total_solution_count(group)]
    df_result = df_valid[df_valid["question_id"].isin(qids_with_diff)]

    # 从df_result中获取所有question_id
    qids = df_result["question_id"].unique()
    # 随机选择10个
    sample_qids = random.sample(list(qids), min(10, len(qids)))

    for qid in sample_qids:
        group = df_result[df_result["question_id"] == qid]
        vertical_row = group[group["scaling"] == "vertical"]
        none_row = group[group["scaling"] == "none"]
        if not vertical_row.empty and not none_row.empty:
            print(f"Question ID: {qid}")
            v = vertical_row["model_thinking_response"].values[0]
            n = none_row["model_thinking_response"].values[0]
            diff = difflib.unified_diff(v.splitlines(), n.splitlines())
            print("\n".join(diff))
            print("=" * 40)


if __name__ == "__main__":
    run_async(main())
