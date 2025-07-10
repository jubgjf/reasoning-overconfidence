import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, prf
from confidence.logger import Logger
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template="simple"),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        title = f"{dataset}--{setting.template}--{setting.model}--{temperature}--{turn}"
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)

        # 计算 precision, recall 等指标
        df = prf(df, dataset)
        df = add_confidence_column(df)

        if setting.model == ModelName.QWEN3_8B_NO_THINK and setting.template == "cot":
            df["setting"] = "Short-CoT"
        elif setting.model == ModelName.QWEN3_8B_THINK and setting.template == "simple":
            df["setting"] = "Long-CoT"
        else:
            raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    plt.figure()
    sns.histplot(
        data=df, x="model_confidence_extracted", hue="setting", bins=10, alpha=0.7, kde=False, multiple="dodge"
    )
    plt.title("Model Confidence Distribution by Setting")
    plt.xlabel("Model Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
