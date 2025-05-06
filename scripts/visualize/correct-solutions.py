import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import TimeTablingTemplate, Template


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWQ_32B
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    settings = [
        Setting(model=ModelName.QWQ_32B, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN2_5_7B, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN2_5_7B, template=TimeTablingTemplate.cot),
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

    df = pd.concat(records_list)

    df["correct_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[0]))
    df["total_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[1]))

    df = df[df["correct_solution_count"] <= df["total_solution_count"]]
    df = df[df["correct_solution_count"] <= df["answer_count"]]
    df = df[df["total_solution_count"] <= df["answer_count"]]

    df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50) if int(x // 50) < 8 else 8)

    df["correct_accuracy"] = df["correct_solution_count"] / df["answer_count"]
    df["total_accuracy"] = df["total_solution_count"] / df["answer_count"]

    plt.figure()
    # sns.scatterplot(data=df, x="answer_count_bin", y="correct_accuracy", hue="setting")
    sns.lineplot(data=df, x="answer_count_bin", y="correct_accuracy", hue="setting")
    plt.xlabel("Answer Count Bin")
    plt.ylabel("Correct Acc")
    plt.title("Answer Count Bin vs. Correct Acc")
    plt.show()

    plt.figure()
    sns.lineplot(data=df, x="answer_count_bin", y="total_solution_count", hue="setting")
    plt.xlabel("Answer Count Bin")
    plt.ylabel("Solution Count")
    plt.title("Answer Count Bin vs. Solution Count")
    plt.show()


if __name__ == "__main__":
    run_async(main())
