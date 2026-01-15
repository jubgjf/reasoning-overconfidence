import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, ece_by_groups, prf, show_metrics
from confidence.logger import Logger
from confidence.model import ModelName

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


async def main():
    dataset = DatasetName.TimeTabling
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
    show_metrics(exploration_df, "w/ Exploration")

    # 使用 evaluate.py 中的 ECE 计算函数
    ece_dict = ece_by_groups(df, "scaling", "recall")
    
    # 创建正确的标签映射
    def get_display_label(scaling_type):
        if scaling_type == "short-cot":
            return f"Short-CoT (ECE={ece_dict[scaling_type] * 100:.2f})"
        elif scaling_type == "exploration":
            return f"w/ Exploration (ECE={ece_dict[scaling_type] * 100:.2f})"
        else:
            return f"{scaling_type.capitalize()} (ECE={ece_dict[scaling_type] * 100:.2f})"
    
    df["scaling_label"] = df["scaling"].apply(get_display_label)

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

    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {"short-cot": "tab:orange", "exploration": "tab:green"}
    
    # 分别绘制每个子图
    for scaling_label, group in grouped.groupby("scaling_label"):
        plt.figure(figsize=(3, 3))
        
        # 提取原始名称用于配色
        scaling_label_str = str(scaling_label)
        if "Short-CoT" in scaling_label_str:
            color = color_map["short-cot"]
        elif "w/ Exploration" in scaling_label_str:
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
        
        plt.bar(
            bin_centers,
            mean_accuracys,
            width=0.07,
            alpha=0.6,
            align="center",
            edgecolor="black",
            color=color,
        )
        plt.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Confidence")
        plt.ylabel("Recall")
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(f"{scaling_label}", fontsize=15, pad=10)
        plt.grid(True)
        plt.tight_layout()
        
        # 根据scaling类型保存不同的文件名
        if "Short-CoT" in scaling_label_str:
            plt.savefig(f"figures/exploration-{model.series_name.lower()}-{dataset}-short-cot.pdf", bbox_inches="tight")
        elif "w/ Exploration" in scaling_label_str:
            plt.savefig(f"figures/exploration-{model.series_name.lower()}-{dataset}-exploration.pdf", bbox_inches="tight")
        # plt.show()

    # 创建并保存单独的图例
    _create_and_save_legend(color_map, model, dataset)


def _create_and_save_legend(color_map, model, dataset):
    """创建并保存单独的图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(5, 0.2))
    ax_legend.axis("off")

    # 重新创建图例项
    handles = []
    labels = []

    # 为每种scaling类型创建图例项
    for scaling_name in ["Short-CoT", "w/ Exploration"]:
        if scaling_name == "Short-CoT":
            color = color_map["short-cot"]
        else:
            color = color_map["exploration"]
        handle = Line2D([0], [0], marker="o", color=color, linewidth=2, markersize=6, label=scaling_name)
        handles.append(handle)
        labels.append(scaling_name)

    # 添加完美校准线
    handles.append(Line2D([0], [0], linestyle="--", color="gray", label="Perfectly Calibrated"))
    labels.append("Perfectly Calibrated")

    ax_legend.legend(handles, labels, loc="center", frameon=True, ncol=3)

    # 保存单独的图例
    plt.savefig(f"figures/exploration-{model.series_name.lower()}-{dataset}-legend.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
