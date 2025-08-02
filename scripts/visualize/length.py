import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from tortoise import run_async
from transformers import AutoTokenizer

from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, prf
from confidence.logger import Logger
from confidence.model import ModelName

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


async def main():
    model = ModelName.QWEN3_8B_THINK
    template = "simple"
    dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

    record_cls = dataset.record_cls
    title = f"{dataset}--{template}--{model}--{temperature}--{turn}--less"
    db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
    async with db_logger:
        records = await db_logger.fetch()

    method_records = [record.model_dump() for record in records]
    df = pd.DataFrame(method_records)
    if model == ModelName.QWEN3_8B_NO_THINK and template == "cot":
        df["setting"] = "Short-CoT"
    elif model == ModelName.QWEN3_8B_THINK and template == "simple":
        df["setting"] = "Long-CoT"
    else:
        raise ValueError(f"Unknown setting: {model}--{template}")
    df = pd.DataFrame(method_records)

    df = prf(df, dataset)
    df = add_confidence_column(df)

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")

    df["model_thinking_response"] = df["thinking_history"].apply(lambda x: x[1])
    df["model_thinking_length"] = df["model_thinking_response"].apply(lambda x: len(tokenizer.encode(x)))

    plt.figure(figsize=(3, 3))
    df["length_bin"] = pd.cut(df["model_thinking_length"], bins=10, include_lowest=True)
    length_grouped = df.groupby("length_bin", observed=False)["model_confidence_extracted"].mean().reset_index()
    length_grouped["length_center"] = length_grouped["length_bin"].apply(lambda x: x.mid)

    # 绘制原始数据点
    sns.lineplot(
        x="length_center",
        y="model_confidence_extracted",
        data=length_grouped,
        marker="o",
        linewidth=2,
        markersize=6,
        label="Binned data",
    )

    # 添加直线拟合 (在log空间中进行线性拟合)
    # 转换为数值类型以避免类型错误
    log_x = np.log(length_grouped["length_center"].astype(float))
    y = length_grouped["model_confidence_extracted"]

    # 线性拟合
    coeffs = np.polyfit(log_x, y, 1)
    poly_func = np.poly1d(coeffs)

    # 计算相关系数和显著性
    corr_coefficient, p_value = pearsonr(log_x, y)
    r_squared = corr_coefficient**2
    significant = "Significant" if p_value < 0.05 else "Not Significant"

    # 生成拟合线
    log_x_range = np.linspace(log_x.min(), log_x.max(), 100)
    x_range = np.exp(log_x_range)
    y_fit = poly_func(log_x_range)

    plt.plot(
        x_range,
        y_fit,
        "--",
        color="red",
        linewidth=2,
        label="Log-linear fit",
    )

    plt.xscale("log")

    # 设置更规律的x轴刻度 - 使用科学计数法格式
    x_min, x_max = length_grouped["length_center"].min(), length_grouped["length_center"].max()
    # 计算合适的数量级范围
    log_min, log_max = np.log10(x_min), np.log10(x_max)

    # 生成规律的刻度位置：1, 2, 5, 10, 20, 50, 100, ...
    tick_positions = []
    for exp in range(int(log_min), int(log_max) + 2):
        for base in [2, 5]:
            # for base in [1, 2, 5]:
            tick_val = base * (10**exp)
            if x_min <= tick_val <= x_max * 1.5:  # 稍微扩展范围以包含边界值
                tick_positions.append(tick_val)

    tick_positions = sorted(list(set(tick_positions)))  # 去重并排序

    plt.xticks(tick_positions, [f"{x / 1000:.1f}×10³" if x >= 1000 else f"{int(x)}" for x in tick_positions])

    plt.xlabel("Model Thinking Length (Tokens)")
    plt.ylabel("Model Confidence")
    plt.title(
        f"Corr: {corr_coefficient:.2f}, p: {p_value:.2g}\n({'Significant' if significant else 'Not Significant'})"
    )
    plt.legend(loc="best")

    # 打印统计结果
    print(f"\nLinear correlation analysis:")
    print(f"  Pearson correlation coefficient: r = {corr_coefficient:.3f}")
    print(f"  R-squared: R² = {r_squared:.3f}")
    print(f"  P-value: p = {p_value:.3f}")
    print(f"  Significance: {significant}")
    print(f"  Slope in log space: {coeffs[0]:.3f}")

    plt.tight_layout()
    plt.savefig(f"figures/length-{model.series_name.lower()}-{dataset}.pdf")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
