import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import SubsetSumTemplate, Template, TimeTablingTemplate
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
    temperature = 0.2

    settings = [
        # Setting(model=ModelName.QWEN3_8B_THINK, template=SubsetSumTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
    ]

    records_list = []
    turns = [0, 1, 2, 3, 4]
    for setting in settings:
        for turn in turns:
            record_cls = dataset.record_cls
            db_logger = Logger(
                db_name=dataset.value + "--turn" + str(turn),
                table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--{temperature}--evaluate-by-{judge_model}",
                record_cls=record_cls,
            )
            async with db_logger:
                records = await db_logger.fetch()

            method_records = [record.model_dump() for record in records]
            df = pd.DataFrame(method_records)
            df["setting"] = f"{setting.model}--{setting.template}"
            df["turn"] = turn
            df["consistency_choose"] = False

            records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)
    df = prf(df, method, dataset)

    long_cot_df = df[df["turn"] == 0].copy()
    show_metrics(long_cot_df, "Long-CoT Baseline")

    df["max_model_thinking_response"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_thinking_response"
    ].transform(lambda x: x.max())
    self_consistency_df = df[df["model_thinking_response"] == df["max_model_thinking_response"]].copy()
    show_metrics(self_consistency_df, "Self-Consistency (max model thinking length)")

    df["max_model_confidence"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_confidence_extracted"
    ].transform(lambda x: x.max())
    self_consistency_df = df[df["model_confidence_extracted"] == df["max_model_confidence"]].copy()
    show_metrics(self_consistency_df, "Self-Consistency (max model confidence)")

    df["min_model_confidence"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_confidence_extracted"
    ].transform(lambda x: x.min())
    self_consistency_df = df[df["model_confidence_extracted"] == df["min_model_confidence"]].copy()
    show_metrics(self_consistency_df, "Self-Consistency (min model confidence)")

    df["median_model_confidence"] = df.groupby(["question_id", "answer_count_bin"])[
        "model_confidence_extracted"
    ].transform("median")
    df["confidence_diff"] = (df["model_confidence_extracted"] - df["median_model_confidence"]).abs()
    df["min_confidence_diff"] = df.groupby(["question_id", "answer_count_bin"])["confidence_diff"].transform("min")
    self_consistency_df = df[df["confidence_diff"] == df["min_confidence_diff"]].copy()
    show_metrics(self_consistency_df, "Self-Consistency (median model confidence)")

    # 标记不同方法
    df["method"] = "Long-CoT"
    df.loc[df["model_thinking_response"] == df["max_model_thinking_response"], "method"] = (
        "Self-Consistency (max thinking)"
    )
    df.loc[df["model_confidence_extracted"] == df["max_model_confidence"], "method"] = (
        "Self-Consistency (max confidence)"
    )
    df.loc[df["model_confidence_extracted"] == df["min_model_confidence"], "method"] = (
        "Self-Consistency (min confidence)"
    )
    df.loc[df["confidence_diff"] == df["min_confidence_diff"], "method"] = "Self-Consistency (median confidence)"

    # 只保留每种方法的唯一样本
    plot_df = df.drop_duplicates(subset=["question_id", "answer_count_bin", "method"])

    # 计算ECE
    ece_dict = {}
    for method, group in plot_df.groupby("method"):
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
        ece_dict[method] = ece
    plot_df["method_label"] = plot_df["method"].apply(lambda s: f"{s}\n(ECE={ece_dict[s]:.3f})")

    plot_df["confidence_bin"] = pd.cut(
        plot_df["model_confidence_extracted"], bins=10, include_lowest=True, labels=False
    )
    grouped = (
        plot_df.groupby(["method_label", "confidence_bin"])
        .agg(
            mean_confidence=("model_confidence_extracted", "mean"),
            mean_accuracy=("recall", "mean"),
            count=("confidence_bin", "size"),
        )
        .reset_index()
    )

    method_order = [
        "Long-CoT",
        "Self-Consistency (max thinking)",
        "Self-Consistency (max confidence)",
        "Self-Consistency (min confidence)",
        "Self-Consistency (median confidence)",
    ]
    # 保证顺序
    method_labels = [f"{m}\n(ECE={ece_dict[m]:.3f})" for m in method_order]

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {
        "Long-CoT": "tab:blue",
        "Self-Consistency (max thinking)": "tab:orange",
        "Self-Consistency (max confidence)": "tab:green",
        "Self-Consistency (min confidence)": "tab:red",
        "Self-Consistency (median confidence)": "tab:purple",
    }

    for ax, method_label, method in zip(axes, method_labels, method_order):
        group = grouped[grouped["method_label"] == method_label]
        color = color_map[method]
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
            label=method_label,
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
        ax.set_title(method_label, fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Recall")
    plt.suptitle("Long-CoT vs Self-Consistency Calibration")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# ...existing code...

if __name__ == "__main__":
    run_async(main())
