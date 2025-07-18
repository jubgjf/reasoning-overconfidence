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
    dataset: DatasetName


async def main():
    turn = 0
    temperature = 0.2

    # 定义所有实验设置
    settings = [
        Setting(dataset=DatasetName.SubsetSum, model=ModelName.QWEN3_8B_THINK, template="simple"),
        Setting(dataset=DatasetName.SubsetSum, model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
        Setting(dataset=DatasetName.TimeTabling, model=ModelName.QWEN3_8B_THINK, template="simple"),
        Setting(dataset=DatasetName.TimeTabling, model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
    ]

    records_list = []
    for setting in settings:
        record_cls = setting.dataset.record_cls
        title = f"{setting.dataset}--{setting.template}--{setting.model}--{temperature}--{turn}".replace("/", "_")
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)

        # 计算 precision, recall 等指标
        df = prf(df, setting.dataset)
        df = add_confidence_column(df)

        # 确定数据集名称
        if setting.dataset == DatasetName.TimeTabling:
            dataset_name = "TimeTabling"
        else:
            dataset_name = "SubsetSum"

        # 根据模板设置标签，包含数据集信息
        if setting.template == "cot":
            df["setting"] = f"Short-CoT ({dataset_name})"
        elif setting.template == "simple":
            df["setting"] = f"Long-CoT ({dataset_name})"
        else:
            raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    # 自定义颜色调色板
    # Long-CoT 使用蓝色系，Short-CoT 使用橙色系
    unique_settings = df["setting"].unique()
    color_palette = {}

    # Long-CoT 使用蓝色系，Short-CoT 使用橙色系
    long_cot_colors = ["#1f77b4", "#aec7e8"]  # 深蓝、浅蓝
    short_cot_colors = ["#ff7f0e", "#ffbb78"]  # 深橙、浅橙

    long_cot_count = 0
    short_cot_count = 0

    for setting in unique_settings:
        if "Long-CoT" in setting:
            color_palette[setting] = long_cot_colors[long_cot_count % len(long_cot_colors)]
            long_cot_count += 1
        elif "Short-CoT" in setting:
            color_palette[setting] = short_cot_colors[short_cot_count % len(short_cot_colors)]
            short_cot_count += 1

    plt.figure(figsize=(8, 4))
    sns.histplot(
        data=df,
        x="model_confidence_extracted",
        hue="setting",
        bins=10,
        alpha=0.7,
        kde=False,
        multiple="dodge",
        palette=color_palette,
    )
    plt.xlabel("Model Confidence")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figures/confidence-distribution.pdf")
    plt.show()


if __name__ == "__main__":
    run_async(main())
