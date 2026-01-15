import matplotlib.pyplot as plt
import numpy as np
from matplotlib import hatch
from matplotlib.patches import Rectangle

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


def _create_and_save_legend():
    """创建并保存单独的图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(4, 0.9))
    ax_legend.axis("off")

    # 第一行：颜色对应的指标
    metrics = ["Recall (↑)", "ECE (↓)", "ESC (↑)", "NSD (↑)"]
    metric_colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]

    handles_metrics = []
    labels_metrics = []

    # 添加指标颜色
    for metric, color in zip(metrics, metric_colors):
        handle = Rectangle(
            (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.8, alpha=0.7
        )
        handles_metrics.append(handle)
        labels_metrics.append(metric)

    # 方法对应的花纹
    handles_methods = []
    labels_methods = []

    # Short-CoT (实心柱子)
    handle1 = Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", linewidth=0.8, alpha=0.7)
    handles_methods.append(handle1)
    labels_methods.append("Short-CoT")

    # w/ Exploration (带花纹的柱子)
    handle2 = Rectangle((0, 0), 1, 1, facecolor="gray", edgecolor="black", linewidth=0.8, hatch="ooooo", alpha=0.5)
    handles_methods.append(handle2)
    labels_methods.append("w/ Exploration")

    # 合并所有 handles 和 labels
    # matplotlib legend 按列填充，要实现：
    # 第一行：Recall, ECE
    # 第二行：ESC, NSD
    # 第三行：Short-CoT, w/ Exploration
    # 需要按列顺序排列：[Recall, ESC, Short-CoT, ECE, NSD, w/ Exploration]
    all_handles = [handles_metrics[0], handles_metrics[2], handles_methods[0], 
                   handles_metrics[1], handles_metrics[3], handles_methods[1]]
    all_labels = [labels_metrics[0], labels_metrics[2], labels_methods[0],
                  labels_metrics[1], labels_metrics[3], labels_methods[1]]

    # 使用统一的 legend，ncol=2 实现三行显示（6个元素，每行2个）
    ax_legend.legend(all_handles, all_labels, loc="center", frameon=True, ncol=2)

    # 保存单独的图例
    plt.savefig("figures/exploration-all-qwen-timetabling-legend.pdf", bbox_inches="tight")
    # plt.show()


def _plot_dataset(dataset_name, short_cot_values, exploration_values, filename_prefix):
    """绘制单个数据集的柱状图"""
    # 准备数据进行绘图
    metrics = ["Recall", "ECE", "ESC", "NSD"]
    
    # 设置图形
    x = np.arange(len(metrics))
    width = 0.30  # 更细的柱子

    # 为每个指标设置相同的颜色
    metric_colors = ["tab:orange", "tab:green", "tab:purple", "tab:brown"]

    # 在创建图形之前设置花纹属性
    plt.rcParams["hatch.color"] = "white"
    plt.rcParams["hatch.linewidth"] = 0.8  # 调整花纹线条粗细

    # 创建窄图
    plt.figure(figsize=(3, 3))
    bars1 = plt.bar(
        x - width / 2,
        short_cot_values,
        width,
        label="Short-CoT",
        color=metric_colors,
        edgecolor="black",
        linewidth=0.8,
        alpha=0.7,
    )
    bars2 = plt.bar(
        x + width / 2,
        exploration_values,
        width,
        label="w/ Exploration",
        color=metric_colors,
        edgecolor="black",
        linewidth=0.8,
        hatch="ooooo",  # 使用斜线花纹，可能更明显
        alpha=0.5,  # 稍微降低透明度以突出花纹
    )

    plt.xlabel("Metrics")
    plt.ylabel("Values")
    plt.title("w/ Exploration", fontsize=15, pad=10)
    plt.xticks(x, metrics, fontsize=13)
    plt.yticks(fontsize=13)
    # 不添加图例到主图
    plt.grid(True, alpha=0.5, axis="y")

    # 固定 y 轴范围为 0-1
    plt.ylim(0, 1.0)

    plt.tight_layout()

    # 保存无图例的主图
    plt.savefig(f"figures/{filename_prefix}-main.pdf", bbox_inches="tight")
    plt.close()  # 关闭当前图形以释放内存


# 柱状图比较 Short-CoT 和 Exploration 的各种指标
if __name__ == "__main__":
    # \toprule
    # Dataset                      & Method         & Precision (\%) $\uparrow$ & Recall (\%) $\uparrow$ & ECE (r) ($\downarrow$) & CSR (\%) $\uparrow$ & ESC (\%) $\uparrow$ & NSD (\%) $\uparrow$ \\
    # \midrule
    # \multirow{2}{*}{TimeTabling} & Short-CoT      & 37.25                     & 3.68                   & 95.51                  & 43.15               & \textbf{65.22}      & 0.19                \\
    #                              & w/ Exploration & \textbf{61.31}            & \textbf{6.76}          & \textbf{93.59}         & \textbf{76.93}      & 39.47               & \textbf{0.25}       \\
    # \midrule
    # \multirow{2}{*}{SubsetSum}   & Short-CoT      & 4.51                      & 2.19                   & \textbf{56.44}         & 14.89               & \textbf{15.63}      & \textbf{0.20}       \\
    #                              & w/ Exploration & \textbf{4.64}             & \textbf{2.21}          & 80.36                  & \textbf{16.35}      & 12.10               & 0.00                \\
    # \bottomrule

    # TimeTabling数据集数据
    timetabling_data = {
        "Method": ["Short-CoT", "w/ Exploration"],
        "Precision (%)": [37.25, 61.31],
        "Recall (%)": [3.68, 6.76],
        "ECE (r)": [95.51, 93.59],
        "CSR (%)": [43.15, 76.93],
        "ESC (%)": [65.22, 39.47],
        "NSD (%)": [0.19, 0.25],
    }

    # SubsetSum数据集数据
    subsetsum_data = {
        "Method": ["Short-CoT", "w/ Exploration"],
        "Precision (%)": [4.51, 4.64],
        "Recall (%)": [2.19, 2.21],
        "ECE (r)": [56.44, 80.36],
        "CSR (%)": [14.89, 16.35],
        "ESC (%)": [15.63, 12.10],
        "NSD (%)": [0.20, 0.00],
    }

    # 绘制TimeTabling数据集 (转换为0-1范围)
    timetabling_short_cot = [3.68 / 100, 95.51 / 100, 65.22 / 100, 1.00 / 100]
    timetabling_exploration = [6.76 / 100, 93.59 / 100, 39.47 / 100, 1.50 / 100]
    _plot_dataset("TimeTabling", timetabling_short_cot, timetabling_exploration, "exploration-all-qwen-timetabling")

    # 绘制SubsetSum数据集 (转换为0-1范围)
    subsetsum_short_cot = [2.19 / 100, 56.44 / 100, 15.63 / 100, 1.00 / 100]
    subsetsum_exploration = [2.21 / 100, 80.36 / 100, 12.10 / 100, 0.50 / 100]
    _plot_dataset("SubsetSum", subsetsum_short_cot, subsetsum_exploration, "exploration-all-qwen-subsetsum")

    # 创建并保存单独的图例
    _create_and_save_legend()
