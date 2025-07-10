import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.evaluate import prf, show_metrics, add_confidence_column
from confidence.logger import Logger
from confidence.model import ModelName


async def main():
    dataset = DatasetName.TimeTabling
    temperature = 0.2
    model = ModelName.QWEN3_8B_THINK
    template = "simple"

    records_list = []
    turns = range(0, 32)
    for turn in turns:
        record_cls = dataset.record_cls
        title = f"{dataset}--{template}--{model}--{temperature}--{turn}"
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()
        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        df["turn"] = turn
        df["consistency_choose"] = False

        records_list.append(df)
    df = pd.concat(records_list, ignore_index=True)

    df = prf(df, dataset)
    df = add_confidence_column(df)

    if dataset == DatasetName.TimeTabling:
        # df = df[df["answer_count_bin"] == 3]
        pass  # total

    df["model_thinking_response"] = df["thinking_history"].apply(lambda x: x[1])

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
            acc = (bin_data["recall"] == 1).mean()
            conf = confidence_normalized[bin_data.index].mean()
            ece += len(bin_data) / N * abs(acc - conf)
        ece_dict[method] = ece
    plot_df["method_label"] = plot_df["method"].apply(lambda s: f"{s}\n(ECE={ece_dict[s]:.3f})")

    # 同样需要归一化置信度用于可视化
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
        mean_accuracys = np.full(10, 0.0)  # 用0填充
        has_data = np.full(10, False)  # 标记哪些bin有实际数据

        for i in range(10):
            bin_group = group[group["confidence_bin"] == i]
            if not bin_group.empty:
                acc = bin_group["mean_accuracy"].values[0]
                mean_accuracys[i] = acc
                has_data[i] = True
            # 如果bin_group为空，mean_accuracys[i]保持为0，has_data[i]保持为False

        # 拟合直线 - 只对有实际数据的点进行拟合
        valid = has_data
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

        # 绘制所有点，包括0值
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
        ax.plot(bin_centers[valid], mean_accuracys[valid], marker="o", linestyle="-", linewidth=2, color=color)
        ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly Calibrated")
        ax.set_xlabel("Confidence")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{method_label}\nAngle diff: {angle_deg:.2f}°", fontsize=10)
        ax.grid(True)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Recall")
    plt.suptitle("Long-CoT vs Self-Consistency Calibration")
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()

    # 绘制三维图：Confidence vs Recall vs Density
    # 计算每个方法的密度
    for method in plot_df["method"].unique():
        mask = plot_df["method"] == method
        sub_df = plot_df[mask]
        x = np.array(sub_df["model_confidence_extracted"], dtype=float)
        y = np.array(sub_df["recall"], dtype=float)
        if len(x) > 1:
            kde = gaussian_kde(np.vstack([x, y]))
            density = kde(np.vstack([x, y]))
        else:
            density = np.ones_like(x)
        plot_df.loc[mask, "density"] = density

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
        sub = plot_df[plot_df["method"] == method]
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
    plt.show()

    # 绘制单独的三维图：每种方法一个图
    fig = plt.figure(figsize=(20, 12))

    for i, method in enumerate(method_order):
        ax = fig.add_subplot(2, 3, i + 1, projection="3d")
        sub = plot_df[plot_df["method"] == method]

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
    plt.show()

    # 创建多个子图显示不同方法的二维密度图
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, method in enumerate(method_order):
        if i >= len(axes):
            break

        ax = axes[i]
        sub = plot_df[plot_df["method"] == method]

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
    plt.show()


if __name__ == "__main__":
    run_async(main())
