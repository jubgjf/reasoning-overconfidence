import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from confidence.dataset import DatasetName
from confidence.model import ModelName

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


def load_self_consistency_data(data_path: str) -> pd.DataFrame:
    """
    加载self-consistency数据

    Args:
        data_path: 数据文件路径

    Returns:
        处理后的DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_pickle(data_path)
    print(f"Loaded data from: {data_path}")
    print(f"Total samples: {len(df)}")
    print(f"Methods: {df['method'].unique()}")

    return df


def plot_calibration_charts(
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    output_dir: str = "figures",
    save_fig: bool = True,
    show_fig: bool = False,
):
    """
    绘制校准图表

    Args:
        df: 包含self-consistency结果的DataFrame
        dataset_name: 数据集名称
        model_name: 模型名称
        output_dir: 输出目录
        save_fig: 是否保存图片
        show_fig: 是否显示图片
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 归一化置信度用于可视化
    df["confidence_bin"] = pd.cut(df["model_confidence_extracted"], bins=10, include_lowest=True, labels=False)
    grouped = (
        df.groupby(["method", "confidence_bin"])
        .agg(
            mean_confidence=("model_confidence_extracted", "mean"),
            mean_accuracy=("recall", "mean"),
            count=("confidence_bin", "size"),
        )
        .reset_index()
    )

    method_order = [
        "Long-CoT",
        # "Self-Consistency (max thinking)",
        # "Self-Consistency (max confidence)",
        # "Self-Consistency (min confidence)",
        "Self-Consistency (median confidence)",
        "Self-Consistency (majority voting)",
    ]

    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {
        "Long-CoT": "tab:blue",
        # "Self-Consistency (max thinking)": "tab:orange",
        # "Self-Consistency (max confidence)": "tab:green",
        # "Self-Consistency (min confidence)": "tab:red",
        "Self-Consistency (median confidence)": "tab:purple",
        "Self-Consistency (majority voting)": "tab:brown",
    }

    # 只处理未注释的方法
    active_methods = [method for method in method_order if not method.startswith("#")]

    for method in active_methods:
        if method not in df["method"].values:
            continue

        # 为每个方法创建单独的图
        fig, ax = plt.subplots(1, 1, figsize=(3, 3))

        group = grouped[grouped["method"] == method]
        color = color_map[method]
        mean_accuracys = np.full(10, np.nan)

        for i in range(10):
            bin_group = group[group["confidence_bin"] == i]
            if not bin_group.empty:
                acc = bin_group["mean_accuracy"].values[0]
                if acc == 0:
                    continue
                mean_accuracys[i] = acc

        # 只对有实际数据的点进行绘制
        valid = ~np.isnan(mean_accuracys)

        if method == "Long-CoT":
            method_short_name = "Long-CoT"
        # elif method == "Self-Consistency (max thinking)":
        #     method_short_name = "w/ longest"
        # elif method == "Self-Consistency (max confidence)":
        #     method_short_name = "w/ max confidence"
        # elif method == "Self-Consistency (min confidence)":
        #     method_short_name = "w/ min confidence"
        elif method == "Self-Consistency (median confidence)":
            method_short_name = "w/ Median Conf"
        elif method == "Self-Consistency (majority voting)":
            method_short_name = "w/ Voting"
        else:
            raise ValueError(f"Unknown method: {method}")

        # 绘制柱状图
        ax.bar(
            bin_centers,
            mean_accuracys,
            width=0.07,
            alpha=0.6,
            label=f"{method_short_name}",
            align="center",
            edgecolor="black",
            color=color,
        )
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.set_title(f"{method_short_name}", fontsize=15, pad=10)
        ax.grid(True)
        # 不添加图例到主图

        plt.tight_layout()

        if save_fig:
            # 为每个方法生成单独的文件名
            method_suffix = method_short_name.replace(" ", "-").replace("/", "-").lower()
            output_filename = (
                f"self-consistency-{model_name.lower()}-{dataset_name.lower()}-{method_suffix}-recall-main.pdf"
            )
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path, bbox_inches="tight")
            print(f"Figure saved to: {output_path}")

        if show_fig:
            plt.show()
        else:
            plt.close()

    # 在循环结束后创建并保存共用的图例
    if save_fig:
        _create_and_save_legend(color_map, dataset_name, output_dir)


def _create_and_save_legend(color_map: dict, dataset_name: str, output_dir: str):
    """创建并保存单独的图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(4, 0.6))
    ax_legend.axis("off")

    # 重新创建图例项
    handles = []
    labels = []

    # 为每种方法创建图例项
    method_names = [
        ("Long-CoT", "Long-CoT"),
        ("Self-Consistency (median confidence)", "w/ Median Conf"),
        ("Self-Consistency (majority voting)", "w/ Voting"),
    ]

    for method_full_name, method_short_name in method_names:
        if method_full_name in color_map:
            color = color_map[method_full_name]
            handle = Line2D([0], [0], marker="o", color=color, linewidth=2, markersize=6, label=method_short_name)
            handles.append(handle)
            labels.append(method_short_name)

    # 添加完美校准线
    handles.append(Line2D([0], [0], linestyle="--", color="gray", label="Perfectly Calibrated"))
    labels.append("Perfectly Calibrated")

    ax_legend.legend(handles, labels, loc="center", frameon=True, ncol=2)

    # 保存单独的图例
    legend_filename = f"self-consistency-{dataset_name.lower()}-recall-legend.pdf"
    legend_path = os.path.join(output_dir, legend_filename)
    plt.savefig(legend_path, bbox_inches="tight")
    print(f"Legend saved to: {legend_path}")
    plt.close()


def main():
    """主函数：读取数据并生成所有图表"""
    # 配置参数
    dataset = DatasetName.TimeTabling
    model = ModelName.QWEN3_8B_THINK
    template = "simple"
    temperature = 0.2

    # 构建数据文件路径
    data_filename = f"{dataset}_{model.series_name.lower()}_{template}_temp{temperature}.pkl"
    data_path = os.path.join("tmp/self_consistency", data_filename)

    # 加载数据
    try:
        df = load_self_consistency_data(data_path)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please run generate_self_consistency_data.py first to generate the data.")
        return

    # 生成图表
    dataset_name = str(dataset)
    model_name = model.series_name

    plot_calibration_charts(df, dataset_name, model_name, save_fig=True, show_fig=False)


if __name__ == "__main__":
    main()
