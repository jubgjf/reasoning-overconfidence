import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from confidence.dataset import DatasetName
from confidence.evaluate import ece_by_groups
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

    # 使用 evaluate.py 中的 ECE 计算函数
    ece_dict = ece_by_groups(df, "method", "recall")

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
        "Self-Consistency (max thinking)",
        "Self-Consistency (max confidence)",
        "Self-Consistency (min confidence)",
        "Self-Consistency (median confidence)",
    ]

    fig, axes = plt.subplots(1, 5, figsize=(18, 4), sharey=True)
    bin_edges = np.linspace(0, 1, 11)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    color_map = {
        "Long-CoT": "tab:blue",
        "Self-Consistency (max thinking)": "tab:orange",
        "Self-Consistency (max confidence)": "tab:green",
        "Self-Consistency (min confidence)": "tab:red",
        "Self-Consistency (median confidence)": "tab:purple",
    }

    for ax, method in zip(axes, method_order):
        if method not in df["method"].values:
            ax.set_visible(False)
            continue

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
        ece_value = ece_dict[method] * 100  # 乘以100转换为百分比

        if method == "Long-CoT":
            method_short_name = "Long-CoT"
        elif method == "Self-Consistency (max thinking)":
            method_short_name = "w/ longest"
        elif method == "Self-Consistency (max confidence)":
            method_short_name = "w/ max confidence"
        elif method == "Self-Consistency (min confidence)":
            method_short_name = "w/ min confidence"
        elif method == "Self-Consistency (median confidence)":
            method_short_name = "w/ mid confidence"
        else:
            raise ValueError(f"Unknown method: {method}")

        # 绘制柱状图
        ax.bar(
            bin_centers,
            mean_accuracys,
            width=0.07,
            alpha=0.6,
            label=f"{method_short_name} (ECE={ece_value:.2f})",
            align="center",
            edgecolor="black",
            color=color,
        )
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"Self-Consistency ({method_short_name})")
        ax.grid(True)
        ax.legend()

    axes[0].set_ylabel("Recall")
    plt.tight_layout(rect=(0, 0, 1, 0.96))

    if save_fig:
        output_filename = f"self-consistency-{model_name.lower()}-{dataset_name.lower()}-recall.pdf"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        print(f"Figure saved to: {output_path}")

    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_3d_density_scatter(
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    output_dir: str = "figures",
    save_fig: bool = True,
    show_fig: bool = False,
):
    """
    绘制三维密度散点图

    Args:
        df: 包含self-consistency结果的DataFrame
        dataset_name: 数据集名称
        model_name: 模型名称
        output_dir: 输出目录
        save_fig: 是否保存图片
        show_fig: 是否显示图片
    """
    # 计算每个方法的密度
    for method in df["method"].unique():
        mask = df["method"] == method
        sub_df = df[mask]
        x = np.array(sub_df["model_confidence_extracted"], dtype=float)
        y = np.array(sub_df["recall"], dtype=float)
        if len(x) > 1:
            kde = gaussian_kde(np.vstack([x, y]))
            density = kde(np.vstack([x, y]))
        else:
            density = np.ones_like(x)
        df.loc[mask, "density"] = density

    # 绘制三维散点图
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 颜色映射
    color_map = {
        "Long-CoT": "tab:blue",
        "Self-Consistency (max thinking)": "tab:orange",
        "Self-Consistency (max confidence)": "tab:green",
        "Self-Consistency (min confidence)": "tab:red",
        "Self-Consistency (median confidence)": "tab:purple",
    }

    for method, color in color_map.items():
        if method not in df["method"].values:
            continue
        sub = df[df["method"] == method]
        if len(sub) > 0:
            ax.scatter(
                sub["model_confidence_extracted"],
                sub["recall"],
                sub["density"],
                c=color,
                label=method,
                alpha=0.6,
                edgecolor="k",
                linewidth=0.5,
            )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Recall")
    ax.set_zlabel("Density")  # type: ignore
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.set_title("3D Scatter: Confidence vs Recall vs Density")
    plt.tight_layout()

    if save_fig:
        output_filename = f"self-consistency-3d-{model_name.lower()}-{dataset_name.lower()}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        print(f"3D figure saved to: {output_path}")

    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_individual_3d_charts(
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    output_dir: str = "figures",
    save_fig: bool = True,
    show_fig: bool = False,
):
    """
    绘制每种方法的单独三维图

    Args:
        df: 包含self-consistency结果的DataFrame
        dataset_name: 数据集名称
        model_name: 模型名称
        output_dir: 输出目录
        save_fig: 是否保存图片
        show_fig: 是否显示图片
    """
    method_order = [
        "Long-CoT",
        "Self-Consistency (max thinking)",
        "Self-Consistency (max confidence)",
        "Self-Consistency (min confidence)",
        "Self-Consistency (median confidence)",
    ]

    color_map = {
        "Long-CoT": "tab:blue",
        "Self-Consistency (max thinking)": "tab:orange",
        "Self-Consistency (max confidence)": "tab:green",
        "Self-Consistency (min confidence)": "tab:red",
        "Self-Consistency (median confidence)": "tab:purple",
    }

    # 绘制单独的三维图：每种方法一个图
    fig = plt.figure(figsize=(20, 12))

    for i, method in enumerate(method_order):
        if method not in df["method"].values:
            continue

        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        sub = df[df["method"] == method]

        if len(sub) > 0:
            color = color_map[method]
            ax.scatter(
                sub["model_confidence_extracted"],
                sub["recall"],
                sub["density"],
                c=color,
                alpha=0.7,
                edgecolor="k",
                linewidth=0.5,
            )

            ax.set_xlabel("Confidence")
            ax.set_ylabel("Recall")
            ax.set_zlabel("Density")  # type: ignore
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{method}\n3D Scatter: Confidence vs Recall vs Density")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_fig:
        output_filename = f"self-consistency-individual-3d-{model_name.lower()}-{dataset_name.lower()}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        print(f"Individual 3D figure saved to: {output_path}")

    if show_fig:
        plt.show()
    else:
        plt.close()


def plot_2d_density_charts(
    df: pd.DataFrame,
    dataset_name: str,
    model_name: str,
    output_dir: str = "figures",
    save_fig: bool = True,
    show_fig: bool = False,
):
    """
    绘制二维密度图

    Args:
        df: 包含self-consistency结果的DataFrame
        dataset_name: 数据集名称
        model_name: 模型名称
        output_dir: 输出目录
        save_fig: 是否保存图片
        show_fig: 是否显示图片
    """
    method_order = [
        "Long-CoT",
        "Self-Consistency (max thinking)",
        "Self-Consistency (max confidence)",
        "Self-Consistency (min confidence)",
        "Self-Consistency (median confidence)",
    ]

    color_map = {
        "Long-CoT": "tab:blue",
        "Self-Consistency (max thinking)": "tab:orange",
        "Self-Consistency (max confidence)": "tab:green",
        "Self-Consistency (min confidence)": "tab:red",
        "Self-Consistency (median confidence)": "tab:purple",
    }

    # 创建多个子图显示不同方法的二维密度图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, method in enumerate(method_order):
        if i >= len(axes) or method not in df["method"].values:
            break

        ax = axes[i]
        sub = df[df["method"] == method]

        if len(sub) > 0:
            # 创建JointGrid风格的密度图
            x = np.array(sub["model_confidence_extracted"], dtype=float)
            y = np.array(sub["recall"], dtype=float)
            density = np.array(sub["density"], dtype=float)

            # 散点图，用密度作为点的大小
            ax.scatter(
                x,
                y,
                s=50 + 200 * (density - density.min()) / (density.max() - density.min() + 1e-6),
                alpha=0.6,
                color=color_map[method],
                edgecolor="k",
                linewidth=0.5,
            )

            # 绘制等高线
            if len(x) > 5:  # 需要足够的点来绘制等高线
                try:
                    kde = gaussian_kde(np.vstack([x, y]))
                    x_grid = np.linspace(0, 1, 50)
                    y_grid = np.linspace(0, 1, 50)
                    xx, yy = np.meshgrid(x_grid, y_grid)
                    z = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
                    ax.contour(xx, yy, z, levels=5, alpha=0.3, colors=color_map[method])
                except Exception:
                    pass  # 如果KDE失败，跳过等高线绘制

        ax.set_xlabel("Confidence")
        ax.set_ylabel("Recall")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{method}")
        ax.grid(True, alpha=0.3)

    # 隐藏多余的子图
    for i in range(len(method_order), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()

    if save_fig:
        output_filename = f"self-consistency-2d-density-{model_name.lower()}-{dataset_name.lower()}.pdf"
        output_path = os.path.join(output_dir, output_filename)
        plt.savefig(output_path)
        print(f"2D density figure saved to: {output_path}")

    if show_fig:
        plt.show()
    else:
        plt.close()


def main():
    """主函数：读取数据并生成所有图表"""
    # 配置参数
    dataset = DatasetName.SubsetSum
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

    print("Generating calibration charts...")
    plot_calibration_charts(df, dataset_name, model_name, save_fig=True, show_fig=False)

    # print("Generating 3D density scatter plot...")
    # plot_3d_density_scatter(df, dataset_name, model_name, save_fig=True, show_fig=False)

    # print("Generating individual 3D charts...")
    # plot_individual_3d_charts(df, dataset_name, model_name, save_fig=True, show_fig=False)

    # print("Generating 2D density charts...")
    # plot_2d_density_charts(df, dataset_name, model_name, save_fig=True, show_fig=False)

    print("All visualizations completed!")


if __name__ == "__main__":
    main()
