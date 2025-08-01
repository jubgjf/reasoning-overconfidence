import matplotlib.pyplot as plt
import numpy as np

# 柱状图比 Short-CoT 和 Exploration 的各种指标
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

    # 准备数据进行绘图
    metrics = ["Precision (↑)", "Recall (↑)", "ECE (↓)", "CSR (↑)", "ESC (↑)", "NSD (↑)"]
    short_cot_values = [37.25, 3.68, 95.51, 43.15, 65.22, 0.19]
    exploration_values = [61.31, 6.76, 93.59, 76.93, 39.47, 0.25]

    # 设置图形
    x = np.arange(len(metrics))
    width = 0.35

    # 颜色映射
    color_map = {"Short-CoT": "tab:orange", "w/ Exploration": "tab:green"}
    edge_color_map = {"Short-CoT": "black", "w/ Exploration": "black"}

    # 创建柱状图
    plt.figure(figsize=(6, 3))
    bars1 = plt.bar(
        x - width / 2,
        short_cot_values,
        width,
        label="Short-CoT",
        alpha=0.6,
        color=color_map["Short-CoT"],
        edgecolor=edge_color_map["Short-CoT"],
        linewidth=1,
    )
    bars2 = plt.bar(
        x + width / 2,
        exploration_values,
        width,
        label="w/ Exploration",
        alpha=0.6,
        color=color_map["w/ Exploration"],
        edgecolor=edge_color_map["w/ Exploration"],
        linewidth=1,
    )

    plt.xlabel("Metrics")
    plt.ylabel("Values (%)")
    plt.title("Short-CoT vs w/ Exploration on TimeTabling")
    plt.xticks(x, metrics, fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3, axis="y")

    # 在柱子上添加数值标签
    for bar in bars1:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(short_cot_values + exploration_values) * 0.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(short_cot_values + exploration_values) * 0.01,
            f"{height:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # 调整y轴上限以确保标签不被截断
    plt.ylim(0, max(short_cot_values + exploration_values) * 1.15)

    plt.tight_layout()
    plt.savefig("figures/exploration-all-qwen-timetabling.pdf")
    # plt.show()
