import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


def _create_and_save_legend(color_map):
    """创建并保存单独的图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(3, 0.2))
    ax_legend.axis("off")

    # 重新创建图例项
    handles = []
    labels = []

    # 为每种CoT类型创建图例项
    for setting_name in ["Long-CoT", "Short-CoT"]:
        color = color_map[setting_name]
        # 创建矩形图例项（用于柱状图）
        handle = patches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", alpha=0.6)
        handles.append(handle)
        labels.append(setting_name)

    ax_legend.legend(handles, labels, loc="center", frameon=True, ncol=2)

    # 保存单独的图例
    plt.savefig("figures/short-long-cot-timetabling-legend.pdf", bbox_inches="tight")
    # plt.show()


# 柱状图比 Short-CoT 和 Long-CoT 的 和 Recall
if __name__ == "__main__":
    # TimeTabling
    # Recall (result from performance.py):
    #     Qwen
    #         Long-CoT (Recall=9.99)
    #         Short-CoT (Recall=3.68)
    #     DeepSeek
    #         Long-CoT (Recall=11.96)
    #         Short-CoT (Recall=3.10)
    #     GPT
    #         Long-CoT (Recall=9.60)
    #         Short-CoT (Recall=1.88)
    # SubsetSum
    # Recall (result from performance.py):
    #     Qwen
    #         Long-CoT (Recall=21.24)
    #         Short-CoT (Recall=2.19)
    #     DeepSeek
    #         Long-CoT (Recall=22.43)
    #         Short-CoT (Recall=19.40)
    #     GPT
    #         Long-CoT (Recall=23.15)
    #         Short-CoT (Recall=2.34)

    # 组织数据
    models = ["Qwen", "DeepSeek", "GPT"]

    # Recall 数据
    recall_long_cot = [9.99, 11.96, 9.60]
    recall_short_cot = [3.68, 3.10, 1.88]

    # 设置图形
    x = np.arange(len(models))
    width = 0.35

    # 颜色映射
    color_map = {"Long-CoT": "tab:blue", "Short-CoT": "tab:orange"}
    edge_color_map = {"Long-CoT": "black", "Short-CoT": "black"}

    plt.figure(figsize=(3, 3))
    bars3 = plt.bar(
        x - width / 2,
        recall_long_cot,
        width,
        label="Long-CoT",
        alpha=0.6,
        color=color_map["Long-CoT"],
        edgecolor=edge_color_map["Long-CoT"],
        linewidth=1,
    )
    bars4 = plt.bar(
        x + width / 2,
        recall_short_cot,
        width,
        label="Short-CoT",
        alpha=0.6,
        color=color_map["Short-CoT"],
        edgecolor=edge_color_map["Short-CoT"],
        linewidth=1,
    )

    plt.xlabel("Model")
    plt.ylabel("Recall")
    plt.title("TimeTabling", fontsize=15, pad=10)
    plt.xticks(x, models, fontsize=13)
    plt.yticks(fontsize=13)
    # 不添加图例到主图
    plt.grid(True, alpha=0.3, axis="y")

    # 调整y轴上限
    plt.ylim(0, max(max(recall_long_cot), max(recall_short_cot)) * 1.1)

    plt.tight_layout()
    plt.savefig("figures/short-long-cot-timetabling-recall-main.pdf", bbox_inches="tight")
    # plt.show()

    # 创建并保存图例
    _create_and_save_legend(color_map)

    # SubsetSum 图形
    # Recall 数据 (SubsetSum)
    subsetsum_recall_long_cot = [21.24, 22.43, 23.15]
    subsetsum_recall_short_cot = [2.19, 19.40, 2.34]

    plt.figure(figsize=(3, 3))
    bars5 = plt.bar(
        x - width / 2,
        subsetsum_recall_long_cot,
        width,
        label="Long-CoT",
        alpha=0.6,
        color=color_map["Long-CoT"],
        edgecolor=edge_color_map["Long-CoT"],
        linewidth=1,
    )
    bars6 = plt.bar(
        x + width / 2,
        subsetsum_recall_short_cot,
        width,
        label="Short-CoT",
        alpha=0.6,
        color=color_map["Short-CoT"],
        edgecolor=edge_color_map["Short-CoT"],
        linewidth=1,
    )

    plt.xlabel("Model")
    plt.ylabel("Recall")
    plt.title("SubsetSum", fontsize=15, pad=10)
    plt.xticks(x, models, fontsize=13)
    plt.yticks(fontsize=13)
    # 不添加图例到主图
    plt.grid(True, alpha=0.3, axis="y")

    # 调整y轴上限
    plt.ylim(0, max(max(subsetsum_recall_long_cot), max(subsetsum_recall_short_cot)) * 1.1)

    plt.tight_layout()
    plt.savefig("figures/short-long-cot-subsetsum-recall-main.pdf", bbox_inches="tight")
    # plt.show()

    # SubsetSum 图例 (可以复用相同的图例)
    plt.figure(figsize=(3, 0.2))
    plt.axis("off")

    # 重新创建图例项
    handles = []
    labels = []

    # 为每种CoT类型创建图例项
    for setting_name in ["Long-CoT", "Short-CoT"]:
        color = color_map[setting_name]
        # 创建矩形图例项（用于柱状图）
        handle = patches.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="black", alpha=0.6)
        handles.append(handle)
        labels.append(setting_name)

    plt.legend(handles, labels, loc="center", frameon=True, ncol=2)

    # 保存SubsetSum的图例
    plt.savefig("figures/short-long-cot-subsetsum-legend.pdf", bbox_inches="tight")
    # plt.show()
