import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from pydantic import BaseModel
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, ece_by_groups, prf
from confidence.logger import Logger
from confidence.model import ModelName

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

    settings_group = [
        [
            Setting(model=ModelName.QWEN3_8B_THINK, template="simple"),
            Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
        ],
        [
            Setting(model=ModelName.DEEPSEEK_R1, template="simple"),
            Setting(model=ModelName.DEEPSEEK_V3, template="cot"),
        ],
        [
            Setting(model=ModelName.O4_MINI, template="simple"),
            Setting(model=ModelName.GPT_4O_MINI, template="cot"),
        ],
    ]

    color_map = {"Long-CoT": "tab:blue", "Short-CoT": "tab:orange"}  # 在函数开始定义color_map

    # 遍历每个设置组
    for settings in settings_group:
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

        plt.figure(figsize=(3, 3))
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        ece_legend_items = []  # 用于存储ECE图例项
        
        for i, (setting_name, group) in enumerate(grouped.groupby("setting")):
            # 提取原始名称用于配色
            if "Long-CoT" in str(setting_name):
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
            ece_value = ece_dict[str(setting_name)] * 100

            plt.bar(
                bin_centers,
                mean_accuracys,
                width=0.07,
                alpha=0.6,
                label=f"{setting_name}",
                align="center",
                edgecolor="black",
                color=color,
            )
            plt.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)

            # 收集ECE值用于图例
            ece_legend_items.append((setting_name, ece_value, color))
            print(f"{setting_name} (ECE={ece_value:.2f})")

        plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        plt.xlabel("Confidence")
        plt.ylabel("Recall")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(f"{model_series_name}", fontsize=15, pad=10)
        plt.grid(True)
        
        # 添加ECE值图例到右上角
        ece_legend_text = []
        for setting_name, ece_value, color in ece_legend_items:
            ece_legend_text.append(f"{setting_name}: {ece_value:.2f}%")
        
        # 在左上角添加文本框显示ECE值
        legend_text = '\n'.join(ece_legend_text)
        plt.text(0.05, 0.95, legend_text, transform=plt.gca().transAxes, 
                verticalalignment='top', horizontalalignment='left',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                fontsize=13)

        plt.tight_layout()

        # 保存无图例的主图
        plt.savefig(f"figures/calibration-{model_series_name.lower()}-{dataset}-recall-main.pdf", bbox_inches="tight")
        # plt.show()

    # 创建并保存图例（在循环外只运行一次）
    _create_and_save_legend(color_map, dataset)


def _create_and_save_legend(color_map, dataset):
    """创建并保存单独的图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(5, 0.2))
    ax_legend.axis("off")

    # 重新创建图例项
    handles = []
    labels = []

    # 为每种CoT类型创建图例项
    for setting_name in ["Long-CoT", "Short-CoT"]:
        color = color_map[setting_name]
        handle = Line2D([0], [0], marker="o", color=color, linewidth=2, markersize=6, label=setting_name)
        handles.append(handle)
        labels.append(setting_name)

    # 添加完美校准线
    handles.append(Line2D([0], [0], linestyle="--", color="gray", label="Perfectly Calibrated"))
    labels.append("Perfectly Calibrated")

    ax_legend.legend(handles, labels, loc="center", frameon=True, ncol=3)

    # 保存单独的图例（移除模型名）
    plt.savefig(f"figures/calibration-{dataset}-recall-legend.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
