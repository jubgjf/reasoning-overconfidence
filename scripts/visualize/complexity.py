import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import prf, add_confidence_column
from confidence.logger import Logger
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    dataset = DatasetName.SubsetSum
    # dataset = DatasetName.TimeTabling
    temperature = 0.2
    turn = 0

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template="simple"),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
        # Setting(model=ModelName.DEEPSEEK_R1, template="simple"),
        # Setting(model=ModelName.DEEPSEEK_V3, template="cot"),
        # Setting(model=ModelName.O4_MINI, template="simple"),
        # Setting(model=ModelName.GPT_4O_MINI, template="cot"),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        title = f"{dataset}--{setting.template}--{setting.model}--{temperature}--{turn}".replace("/", "_")
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)

        if setting.template == "cot":
            df["setting"] = "Short-CoT"
        elif setting.template == "simple":
            df["setting"] = "Long-CoT"
        else:
            raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)
    df = prf(df, dataset)
    df = add_confidence_column(df)

    plt.figure(figsize=(6, 3))
    sns.lineplot(data=df, x="answer_count_bin", y="recall", hue="setting")
    plt.xlabel("Complexity Bin")
    plt.ylabel("Recall")
    # plt.title("TimeTabling")
    plt.savefig(f"figures/complexity-qwen-{dataset}-recall.pdf")
    # plt.show()

    plt.figure(figsize=(6, 3))
    sns.lineplot(data=df, x="answer_count_bin", y="model_confidence_extracted", hue="setting")
    plt.xlabel("Complexity Bin")
    plt.ylabel("Confidence")
    # plt.title("TimeTabling")
    plt.savefig(f"figures/complexity-qwen-{dataset}-confidence.pdf")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
