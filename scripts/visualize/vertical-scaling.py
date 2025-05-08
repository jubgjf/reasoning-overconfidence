import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    settings = [
        # Setting(model=ModelName.QWEN3_8B_THINK, template=SubsetSumTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
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
        df["scaling"] = "none"
        records_list.append(df)

        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
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

    df["completeness"] = df["correct_solution_count"] / df["answer_count"]
    df["accuracy"] = df["correct_solution_count"] / df["total_solution_count"]

    # df = df[df["answer_count_bin"] < 3]  # easy
    # df = df[df["answer_count_bin"] > 7]  # hard

    vertical_completeness_mean = df[df["scaling"] == "vertical"]["completeness"].mean()
    none_completeness_mean = df[df["scaling"] == "none"]["completeness"].mean()
    print(vertical_completeness_mean, none_completeness_mean)
    plt.figure()
    sns.lineplot(data=df, x="answer_count_bin", y="completeness", hue="scaling")
    plt.xlabel("Answer Count Bin")
    plt.ylabel("completeness")
    plt.title("completeness")
    plt.show()

    vertical_accuracy_mean = df[df["scaling"] == "vertical"]["accuracy"].mean()
    none_accuracy_mean = df[df["scaling"] == "none"]["accuracy"].mean()
    print(vertical_accuracy_mean, none_accuracy_mean)
    plt.figure()
    sns.lineplot(data=df, x="answer_count_bin", y="accuracy", hue="scaling")
    plt.xlabel("Answer Count Bin")
    plt.ylabel("accuracy")
    plt.title("accuracy")
    plt.show()

    vertical_confidence_mean = df[df["scaling"] == "vertical"]["model_confidence_extracted"].mean()
    none_confidence_mean = df[df["scaling"] == "none"]["model_confidence_extracted"].mean()
    print(vertical_confidence_mean, none_confidence_mean)
    plt.figure()
    sns.scatterplot(df, x="model_confidence_extracted", y="completeness", hue="scaling")
    plt.xlabel("confidence")
    plt.ylabel("completeness")
    plt.title("confidence vs completeness")
    plt.show()


if __name__ == "__main__":
    run_async(main())
