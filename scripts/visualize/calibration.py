import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, ece_by_groups, prf
from confidence.logger import Logger
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

    settings = [
        # Setting(model=ModelName.QWEN3_8B_THINK, template="simple"),
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
        # Setting(model=ModelName.DEEPSEEK_R1, template="simple"),
        # Setting(model=ModelName.DEEPSEEK_V3, template="cot"),
        Setting(model=ModelName.O4_MINI, template="simple"),
        Setting(model=ModelName.GPT_4O_MINI, template="cot"),
    ]
    model_series_name = settings[0].model.series_name
    assert all(setting.model.series_name == model_series_name for setting in settings)

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        title = f"{dataset}--{setting.template}--{setting.model}--{temperature}--{turn}".replace("/", "_")
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)

        # 计算 precision, recall 等指标
        df = prf(df, dataset)
        df = add_confidence_column(df)

        if setting.template == "cot":
            df["setting"] = "Short-CoT"
        elif setting.template == "simple":
            df["setting"] = "Long-CoT"
        else:
            raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    # ============================ RECALL ============================

    ece_dict = ece_by_groups(df, "setting", "recall")

    # 同样需要归一化置信度用于可视化
    df["confidence_bin"] = pd.cut(df["model_confidence_extracted"], bins=10, include_lowest=True, labels=False)
    grouped = (
        df.groupby(["setting", "confidence_bin"])
        .agg(
            mean_confidence=("model_confidence_extracted", "mean"),
            mean_accuracy=("recall", "mean"),  # precision or recall
            count=("confidence_bin", "size"),
        )
        .reset_index()
    )

    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = 0.035

    color_map = {"Long-CoT": "tab:blue", "Short-CoT": "tab:orange"}
    for i, (setting, group) in enumerate(grouped.groupby("setting")):
        # 提取原始名称用于配色
        if "Long-CoT" in setting:
            color = color_map["Long-CoT"]
        else:
            color = color_map["Short-CoT"]

        mean_accuracys = np.full(10, np.nan)
        for j in range(10):
            bin_group = group[group["confidence_bin"] == j]
            if not bin_group.empty:
                acc = bin_group["mean_accuracy"].values[0]
                if acc == 0:
                    continue
                mean_accuracys[j] = acc

        valid = ~np.isnan(mean_accuracys)
        ece_value = ece_dict[str(setting)] * 100

        ax.bar(
            bin_centers,
            mean_accuracys,
            width=0.07,
            alpha=0.6,
            label=f"{setting}",
            align="center",
            edgecolor="black",
            color=color,
        )
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)

        print(f"{setting} (ECE={ece_value:.2f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Recall")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_title(f"{model_series_name} on {dataset.name}")
    ax.grid(True)
    ax.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"figures/calibration-{model_series_name.lower()}-{dataset}-recall.pdf")
    # plt.show()

    # # ============================ PRECISION ============================

    # ece_dict = ece_by_groups(df, "setting", "precision")

    # # 同样需要归一化置信度用于可视化
    # df["confidence_bin"] = pd.cut(df["model_confidence_extracted"], bins=10, include_lowest=True, labels=False)
    # grouped = (
    #     df.groupby(["setting", "confidence_bin"])
    #     .agg(
    #         mean_confidence=("model_confidence_extracted", "mean"),
    #         mean_accuracy=("precision", "mean"),  # precision or recall
    #         count=("confidence_bin", "size"),
    #     )
    #     .reset_index()
    # )

    # fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    # bin_edges = np.linspace(0, 1, 11)
    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # bar_width = 0.035

    # color_map = {"Long-CoT": "tab:blue", "Short-CoT": "tab:orange"}
    # for i, (setting, group) in enumerate(grouped.groupby("setting")):
    #     # 提取原始名称用于配色
    #     if "Long-CoT" in setting:
    #         color = color_map["Long-CoT"]
    #     else:
    #         color = color_map["Short-CoT"]
    #     mean_accuracys = np.full(10, np.nan)
    #     for j in range(10):
    #         bin_group = group[group["confidence_bin"] == j]
    #         if not bin_group.empty:
    #             acc = bin_group["mean_accuracy"].values[0]
    #             # 如果acc为0则跳过该点
    #             if acc == 0:
    #                 continue
    #             mean_accuracys[j] = acc
    #     valid = ~np.isnan(mean_accuracys)
    #     ece_value = ece_dict[str(setting)] * 100
    #     ax.bar(
    #         bin_centers + offset,
    #         mean_accuracys,
    #         width=bar_width,
    #         alpha=0.6,
    #         label=f"{setting} (ECE={ece_value:.2f})",
    #         align="center",
    #         edgecolor="black",
    #         color=color,
    #     )
    #     ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)

    # ax.set_xlabel("Confidence")
    # ax.set_ylabel("Precision")
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_title(f"{model_series_name} on {dataset.name}")
    # ax.grid(True)
    # ax.legend()

    # plt.tight_layout(rect=[0, 0, 1, 0.96])
    # plt.savefig(f"figures/calibration-deepseek-{dataset}-precision.pdf")
    # # plt.show()


if __name__ == "__main__":
    run_async(main())
if __name__ == "__main__":
    run_async(main())
