import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import Template, TimeTablingTemplate
from scripts.visualize.metrics import prf


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False
    model = ModelName.QWEN3_8B_NO_THINK
    template = TimeTablingTemplate.cot
    turn = 0

    records_list = []
    for temperature in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value + f"--turn{turn}",
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}--{temperature}--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()
        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        df["temperature"] = temperature
        records_list.append(df)
    df = pd.concat(records_list, ignore_index=True)

    df = prf(df, method, dataset)

    plt.figure()
    sns.scatterplot(df, x="temperature", y="precision")
    sns.lineplot(df, x="temperature", y="precision")
    plt.xlabel("temperature")
    plt.ylabel("precision")
    plt.show()

    plt.figure()
    sns.scatterplot(df, x="temperature", y="recall")
    sns.lineplot(df, x="temperature", y="recall")
    plt.xlabel("temperature")
    plt.ylabel("recall")
    plt.show()


if __name__ == "__main__":
    run_async(main())
