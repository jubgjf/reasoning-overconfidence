import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence.dataset import DatasetName
from confidence.model import ModelName


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
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))

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
            method_short_name = "w/ median confidence"
        elif method == "Self-Consistency (majority voting)":
            method_short_name = "w/ majority voting"
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
        ax.set_title(f"Self-Consistency ({method_short_name})")
        ax.grid(True)
        ax.legend()

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        if save_fig:
            # 为每个方法生成单独的文件名
            method_suffix = method_short_name.replace(" ", "-").replace("/", "-").lower()
            output_filename = f"self-consistency-{model_name.lower()}-{dataset_name.lower()}-{method_suffix}-recall.pdf"
            output_path = os.path.join(output_dir, output_filename)
            plt.savefig(output_path)
            print(f"Figure saved to: {output_path}")

        if show_fig:
            plt.show()
        else:
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
