import pandas as pd
import matplotlib.pyplot as plt
from pydantic import BaseModel
from tortoise import run_async
import numpy as np

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import Template, TimeTablingTemplate
from scripts.visualize.metrics import prf, show_metrics


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False
    turn = 0
    temperature = 0.2

    settings = [
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template=SubsetSumTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.cot),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value + f"--turn{turn}",
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--{temperature}--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()
        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        df["setting"] = f"{setting.model}--{setting.template}"
        df["scaling"] = "short-cot"
        records_list.append(df)

        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value + f"--turn{turn}",
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--{temperature}--more-reflection--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()
        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        df["setting"] = f"{setting.model}--{setting.template}"
        df["scaling"] = "exploration"
        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)
    df = prf(df, method, dataset)

    short_cot_df = df[df["scaling"] == "short-cot"].copy()
    show_metrics(short_cot_df, "Short-CoT Baseline")

    exploration_df = df[df["scaling"] == "exploration"].copy()
    show_metrics(exploration_df, "Exploration")

    # 计算 ECE 并准备绘图
    ece_dict = {}
    for scaling, group in df.groupby("scaling"):
        bins = np.linspace(0, 1, 11)
        group["bin"] = pd.cut(group["model_confidence_extracted"], bins=bins, include_lowest=True, labels=False)
        ece = 0
        N = len(group)
        for b in range(10):
            bin_data = group[group["bin"] == b]
            if len(bin_data) == 0:
                continue
            acc = (bin_data["recall"] == 1).mean()
            conf = bin_data["model_confidence_extracted"].mean()
            ece += len(bin_data) / N * abs(acc - conf)
        ece_dict[scaling] = ece
    df["scaling_label"] = df["scaling"].apply(lambda s: f"{s.capitalize()} (ECE={ece_dict[s]:.3f})")

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

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {"Short-CoT": "tab:blue", "Exploration": "tab:orange"}
    for ax, (scaling_label, group) in zip(axes, grouped.groupby("scaling_label")):
        # 提取原始名称用于配色
        if "Short-CoT" in scaling_label:
            color = color_map["Short-CoT"]
        else:
            color = color_map["Exploration"]
        mean_accuracys = np.full(10, np.nan)
        for i in range(10):
            bin_group = group[group["confidence_bin"] == i]
            if not bin_group.empty:
                mean_accuracys[i] = bin_group["mean_accuracy"].values[0]
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
        valid = ~np.isnan(mean_accuracys)
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(scaling_label)
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel("Recall")
    plt.suptitle("Short-CoT vs Exploration Calibration")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    run_async(main())
