import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

    # ============================ RECALL ============================

    ece_dict = {}
    for setting, group in df.groupby("setting"):
        confidence_normalized = group["model_confidence_extracted"]
        bins = [i / 10.0 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
        group_copy = group.copy()
        group_copy["bin"] = pd.cut(confidence_normalized, bins=bins, include_lowest=True, labels=False)
        ece = 0
        N = len(group_copy)
        for b in range(10):
            bin_data = group_copy[group_copy["bin"] == b]
            if len(bin_data) == 0:
                continue
            acc = (bin_data["recall"] == 1).mean()  # 或用 precision
            conf = confidence_normalized[bin_data.index].mean()
            ece += len(bin_data) / N * abs(acc - conf)
        ece_dict[setting] = ece
    print("ECE结果:", ece_dict)
    df["setting"] = df["setting"].apply(lambda s: f"{s} (ECE={ece_dict[s]:.3f})")

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

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {"Long-CoT": "tab:blue", "Short-CoT": "tab:orange"}
    for ax, (setting, group) in zip(axes, grouped.groupby("setting")):
        # 提取原始名称用于配色
        if "Long-CoT" in setting:
            color = color_map["Long-CoT"]
        else:
            color = color_map["Short-CoT"]
        mean_accuracys = np.full(10, np.nan)
        for i in range(10):
            bin_group = group[group["confidence_bin"] == i]
            if not bin_group.empty:
                acc = bin_group["mean_accuracy"].values[0]
                # 如果acc为0则跳过该点
                if acc == 0:
                    continue
                mean_accuracys[i] = acc
        # 拟合直线
        valid = ~np.isnan(mean_accuracys)
        x = bin_centers[valid]
        y = mean_accuracys[valid]
        if len(x) > 1:
            k, b = np.polyfit(x, y, 1)
            angle_rad = np.arctan(k) - np.arctan(1)
            angle_deg = np.degrees(angle_rad)
            # 绘制拟合直线
            fit_y = k * x + b
            ax.plot(
                x, fit_y, color=color, linestyle=":", linewidth=2, label=f"Fitted line (angle diff={angle_deg:.2f}°)"
            )
        else:
            angle_deg = float("nan")
        ax.bar(
            bin_centers,
            mean_accuracys,
            width=0.07,
            alpha=0.6,
            label=setting,
            align="center",
            edgecolor="black",
            color=color,
        )
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{setting}\nAngle diff: {angle_deg:.2f}°")
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel("Recall")
    plt.suptitle("TimeTabling Calibration")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # ============================ PRECISION ============================

    ece_dict = {}
    for setting, group in df.groupby("setting"):
        confidence_normalized = group["model_confidence_extracted"]
        bins = [i / 10.0 for i in range(11)]  # [0.0, 0.1, 0.2, ..., 1.0]
        group_copy = group.copy()
        group_copy["bin"] = pd.cut(confidence_normalized, bins=bins, include_lowest=True, labels=False)
        ece = 0
        N = len(group_copy)
        for b in range(10):
            bin_data = group_copy[group_copy["bin"] == b]
            if len(bin_data) == 0:
                continue
            acc = (bin_data["precision"] == 1).mean()  # 或用 precision
            conf = confidence_normalized[bin_data.index].mean()
            ece += len(bin_data) / N * abs(acc - conf)
        ece_dict[setting] = ece
    print("ECE结果:", ece_dict)
    df["setting"] = df["setting"].apply(lambda s: f"{s} (ECE={ece_dict[s]:.3f})")

    # 同样需要归一化置信度用于可视化
    df["confidence_bin"] = pd.cut(df["model_confidence_extracted"], bins=10, include_lowest=True, labels=False)
    grouped = (
        df.groupby(["setting", "confidence_bin"])
        .agg(
            mean_confidence=("model_confidence_extracted", "mean"),
            mean_accuracy=("precision", "mean"),  # precision or recall
            count=("confidence_bin", "size"),
        )
        .reset_index()
    )

    fig, axes = plt.subplots(1, 2, figsize=(8, 5), sharey=True)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {"Long-CoT": "tab:blue", "Short-CoT": "tab:orange"}
    for ax, (setting, group) in zip(axes, grouped.groupby("setting")):
        # 提取原始名称用于配色
        if "Long-CoT" in setting:
            color = color_map["Long-CoT"]
        else:
            color = color_map["Short-CoT"]
        mean_accuracys = np.full(10, np.nan)
        for i in range(10):
            bin_group = group[group["confidence_bin"] == i]
            if not bin_group.empty:
                acc = bin_group["mean_accuracy"].values[0]
                # 如果acc为0则跳过该点
                if acc == 0:
                    continue
                mean_accuracys[i] = acc
        # 拟合直线
        valid = ~np.isnan(mean_accuracys)
        x = bin_centers[valid]
        y = mean_accuracys[valid]
        if len(x) > 1:
            k, b = np.polyfit(x, y, 1)
            angle_rad = np.arctan(k) - np.arctan(1)
            angle_deg = np.degrees(angle_rad)
            # 绘制拟合直线
            fit_y = k * x + b
            ax.plot(
                x, fit_y, color=color, linestyle=":", linewidth=2, label=f"Fitted line (angle diff={angle_deg:.2f}°)"
            )
        else:
            angle_deg = float("nan")
        ax.bar(
            bin_centers,
            mean_accuracys,
            width=0.07,
            alpha=0.6,
            label=setting,
            align="center",
            edgecolor="black",
            color=color,
        )
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{setting}\nAngle diff: {angle_deg:.2f}°")
        ax.grid(True)
        ax.legend()
    axes[0].set_ylabel("Precision")
    plt.suptitle("TimeTabling Calibration")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


if __name__ == "__main__":
    run_async(main())
