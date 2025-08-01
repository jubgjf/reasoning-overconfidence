import matplotlib.pyplot as plt
import numpy as np

# 柱状图比 Short-CoT 和 Long-CoT 的 ECE 和 Recall
if __name__ == "__main__":
    # TimeTabling
    # ECE (result from calibration.py):
    #     Qwen
    #         Long-CoT (ECE=77.23)
    #         Short-CoT (ECE=95.51)
    #     DeepSeek
    #         Long-CoT (ECE=56.46)
    #         Short-CoT (ECE=82.37)
    #     GPT
    #         Long-CoT (ECE=4.98)
    #         Short-CoT (ECE=78.22)
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

    # 组织数据
    models = ["Qwen", "DeepSeek", "GPT"]

    # ECE 数据
    ece_long_cot = [77.23, 56.46, 4.98]
    ece_short_cot = [95.51, 82.37, 78.22]

    # Recall 数据
    recall_long_cot = [9.99, 11.96, 9.60]
    recall_short_cot = [3.68, 3.10, 1.88]

    # 设置图形
    x = np.arange(len(models))
    width = 0.35

    # 颜色映射
    color_map = {"Long-CoT": "tab:blue", "Short-CoT": "tab:orange"}
    edge_color_map = {"Long-CoT": "black", "Short-CoT": "black"}

    # ECE 独立图
    plt.figure(figsize=(3, 4))
    bars1 = plt.bar(
        x - width / 2,
        ece_long_cot,
        width,
        label="Long-CoT",
        alpha=0.6,
        color=color_map["Long-CoT"],
        edgecolor=edge_color_map["Long-CoT"],
        linewidth=1,
    )
    bars2 = plt.bar(
        x + width / 2,
        ece_short_cot,
        width,
        label="Short-CoT",
        alpha=0.6,
        color=color_map["Short-CoT"],
        edgecolor=edge_color_map["Short-CoT"],
        linewidth=1,
    )

    plt.xlabel("Model")
    plt.ylabel("ECE (%)")
    plt.title("Models on TimeTabling")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    # 在柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, height + 1, f"{height:.1f}", ha="center", va="bottom", fontsize=9)

    # 调整y轴上限以确保标签不被截断
    plt.ylim(0, max(max(ece_long_cot) * 1.2, max(ece_short_cot)) * 1.2)

    plt.tight_layout()
    plt.savefig("figures/short-long-cot-timetabling-ece.pdf")
    # plt.show()

    # Recall 独立图
    plt.figure(figsize=(3, 4))
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
    plt.ylabel("Recall (%)")
    plt.title("Models on TimeTabling")
    plt.xticks(x, models)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    # 在柱子上添加数值标签
    for bar in bars3:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.2, f"{height:.1f}", ha="center", va="bottom", fontsize=9
        )
    for bar in bars4:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0, height + 0.2, f"{height:.1f}", ha="center", va="bottom", fontsize=9
        )

    # 调整y轴上限以确保标签不被截断
    plt.ylim(0, max(max(recall_long_cot) * 1.2, max(recall_short_cot)) * 1.2)

    plt.tight_layout()
    plt.savefig("figures/short-long-cot-timetabling-recall.pdf")
    # plt.show()
