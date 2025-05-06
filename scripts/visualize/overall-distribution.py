import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.template import TimeTablingTemplate, Template
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName


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

    bins = [i / 10 for i in range(11)]

    plt.figure(figsize=(10, 6))
    sns.histplot(df, x="model_confidence_extracted", hue="setting", bins=bins, multiple="dodge", alpha=0.25)
    plt.xlim(0 - 0.05, 1 + 0.05)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title(f"Confidence Distribution Comparison\n{dataset}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
