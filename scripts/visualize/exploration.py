import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, ece_by_groups, prf, show_metrics
from confidence.logger import Logger
from confidence.model import ModelName


async def main():
    dataset = DatasetName.SubsetSum
    turn = 0
    temperature = 0.2
    model = ModelName.QWEN3_8B_NO_THINK
    template = "cot"

    records_list = []
    record_cls = dataset.record_cls
    title = f"{dataset}--{template}--{model}--{temperature}--{turn}"
    db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
    async with db_logger:
        records = await db_logger.fetch()
    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    df["scaling"] = "short-cot"
    records_list.append(df)

    title = f"{dataset}--{template}--{model}--{temperature}--{turn}--more"
    db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
    async with db_logger:
        records = await db_logger.fetch()
    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    df["scaling"] = "exploration"
    records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    df = prf(df, dataset)
    df = add_confidence_column(df)

    short_cot_df = df[df["scaling"] == "short-cot"].copy()
    show_metrics(short_cot_df, "Short-CoT Baseline")

    exploration_df = df[df["scaling"] == "exploration"].copy()
    show_metrics(exploration_df, "Exploration")

    # 使用 evaluate.py 中的 ECE 计算函数
    ece_dict = ece_by_groups(df, "scaling", "recall")
    df["scaling_label"] = df["scaling"].apply(lambda s: f"{s.capitalize()} (ECE={ece_dict[s] * 100:.2f})")

    # 同样需要归一化置信度用于可视化
    df["confidence_bin"] = pd.cut(df["model_confidence_extracted"], bins=10, include_lowest=True, labels=False)
    grouped = (
        df.groupby(["scaling_label", "confidence_bin"])
        .agg(
            mean_confidence=("model_confidence_extracted", "mean"),
            mean_accuracy=("recall", "mean"),
            count=("confidence_bin", "size"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.5), sharey=True)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {"short-cot": "tab:blue", "exploration": "tab:orange"}
    for ax, (scaling_label, group) in zip(axes, grouped.groupby("scaling_label")):
        # 提取原始名称用于配色
        scaling_label_str = str(scaling_label)
        if "Short-cot" in scaling_label_str:
            color = color_map["short-cot"]
        elif "Exploration" in scaling_label_str:
            color = color_map["exploration"]
        else:
            color = "tab:gray"  # 默认颜色
        mean_accuracys = np.full(10, np.nan)
        for i in range(10):
            bin_group = group[group["confidence_bin"] == i]
            if not bin_group.empty:
                acc = bin_group["mean_accuracy"].values[0]
                # 如果acc为0则跳过该点
                if acc == 0:
                    continue
                mean_accuracys[i] = acc
        valid = ~np.isnan(mean_accuracys)
        ax.bar(
            bin_centers,
            mean_accuracys,
            width=0.07,
            alpha=0.6,
            label=scaling_label,
            align="center",
            edgecolor="black",
            color=color,
        )
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{scaling_label}")
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel("Recall")
    # plt.suptitle("Short-CoT vs Exploration Calibration")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(f"figures/exploration-qwen-{dataset}.pdf")
    plt.show()


if __name__ == "__main__":
    run_async(main())
