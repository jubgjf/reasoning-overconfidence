import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, ece, prf
from confidence.logger import Logger
from confidence.model import ModelName


async def main():
    model = ModelName.QWEN3_8B_NO_THINK
    template = "cot"
    turn = 0

    # 定义两个数据集的实验设置
    datasets = [DatasetName.SubsetSum, DatasetName.TimeTabling]

    records_list = []
    for dataset in datasets:
        for temperature in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            record_cls = dataset.record_cls
            title = f"{dataset}--{template}--{model}--{temperature}--{turn}"
            db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
            async with db_logger:
                records = await db_logger.fetch()
            method_records = [record.model_dump() for record in records]
            df = pd.DataFrame(method_records)
            df["temperature"] = temperature

            # 添加数据集标识
            if dataset == DatasetName.TimeTabling:
                df["dataset"] = "TimeTabling"
            else:
                df["dataset"] = "SubsetSum"

            # 计算指标
            df = prf(df, dataset)
            df = add_confidence_column(df)
            
            # 计算 ECE 值
            ece_value = ece(df, metric_column="recall")
            df["ece"] = ece_value

            records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    # 绘制 Precision vs Temperature
    plt.figure(figsize=(6, 3))
    sns.lineplot(data=df, x="temperature", y="precision", hue="dataset", marker="o")
    plt.xlabel("Temperature")
    plt.ylabel("Precision")
    plt.legend(title="Dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/temperature-{model.series_name.lower()}-precision.pdf")
    # plt.show()

    # 绘制 Recall vs Temperature
    plt.figure(figsize=(6, 3))
    sns.lineplot(data=df, x="temperature", y="recall", hue="dataset", marker="o")
    plt.xlabel("Temperature")
    plt.ylabel("Recall")
    plt.legend(title="Dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/temperature-{model.series_name.lower()}-recall.pdf")
    # plt.show()

    # 绘制 Model Confidence vs Temperature
    plt.figure(figsize=(6, 3))
    sns.lineplot(data=df, x="temperature", y="model_confidence_extracted", hue="dataset", marker="o")
    plt.xlabel("Temperature")
    plt.ylabel("Model Confidence")
    plt.legend(title="Dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/temperature-{model.series_name.lower()}-confidence.pdf")
    # plt.show()

    # 绘制 ECE vs Temperature
    plt.figure(figsize=(6, 3))
    sns.lineplot(data=df, x="temperature", y="ece", hue="dataset", marker="o")
    plt.xlabel("Temperature")
    plt.ylabel("ECE (r)")
    plt.legend(title="Dataset")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/temperature-{model.series_name.lower()}-ece.pdf")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
