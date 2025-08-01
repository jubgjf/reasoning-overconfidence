import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, ece, prf
from confidence.logger import Logger
from confidence.model import ModelName


def count_reflections(history_thinking_content: str) -> int:
    reflection_patterns = [
        r"^Wait,.*\n\n",
        r"^But wait,.*\n\n",
        r"^Let me double - check.*\n\n",
        r"^Let me think again.*\n\n",
    ]
    combined_pattern = "|".join(reflection_patterns)

    # thinking_steps_by_reflection =
    #     0: thinking...  1: reflection...
    #     2: thinking...  3: reflection...
    #     4: thinking ...                    # Last step must not be reflection
    last_step_start_index, thinking_steps_by_reflection = 0, []
    if history_thinking_content.startswith("<think>\n"):
        history_thinking_content = history_thinking_content.lstrip("<think>\n")
    for m in re.finditer(combined_pattern, history_thinking_content, re.M):
        thinking_steps_by_reflection.append(history_thinking_content[last_step_start_index : m.start()])
        thinking_steps_by_reflection.append(m.group())
        last_step_start_index = m.end()
    thinking_steps_by_reflection.append(history_thinking_content[last_step_start_index:])
    if len(history_thinking_content) == last_step_start_index:
        # Last step is reflection, although this might be impossible. Remove it.
        thinking_steps_by_reflection = thinking_steps_by_reflection[:-2]

    thinking_with_reduced_reflection = []
    for i in range(0, len(thinking_steps_by_reflection), 2):
        thinking_with_reduced_reflection.append(thinking_steps_by_reflection[: i + 1])

    return len(thinking_with_reduced_reflection)


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

    df["model_thinking_response"] = df["thinking_history"].apply(lambda x: x[1])
    df["reflection_times"] = df["model_thinking_response"].apply(count_reflections)
    df["reflection_times_bin"] = df["reflection_times"].apply(lambda x: int(x // 5) if int(x // 5) < 9 else 9)

    # 只保留数据量大于100的reflection_times_bin
    df = df[
        df["reflection_times_bin"].isin(
            df["reflection_times_bin"].value_counts()[df["reflection_times_bin"].value_counts() > 100].index
        )
    ]

    # 假设“做对”标准为recall==1（可根据实际情况调整）
    min_correct_reflection = (
        df[df["recall"] == 1].groupby("question_id")["reflection_times"].min().rename("min_correct_reflection_times")
    )
    df = df.merge(min_correct_reflection, on="question_id", how="left")
    df["reflection_times_norm"] = df["reflection_times"] - df["min_correct_reflection_times"]

    # 计算每个 reflection_times_norm 组的统计量
    grouped_basic = (
        df.groupby("reflection_times_norm")[["precision", "recall", "model_confidence_extracted"]].mean().reset_index()
    )

    # 添加样本量统计，用于过滤噪声
    sample_counts = df.groupby("reflection_times_norm").size().reset_index(name="sample_count")
    grouped_basic = grouped_basic.merge(sample_counts, on="reflection_times_norm")

    # 添加标准误差计算
    grouped_std = (
        df.groupby("reflection_times_norm")[["precision", "recall", "model_confidence_extracted"]].std().reset_index()
    )
    grouped_std.columns = ["reflection_times_norm", "precision_std", "recall_std", "confidence_std"]
    grouped_basic = grouped_basic.merge(grouped_std, on="reflection_times_norm")

    # 为每个 reflection_times_norm 组计算 ECE
    ece_values = []
    for norm_time in df["reflection_times_norm"].unique():
        if pd.isna(norm_time):
            ece_values.append((norm_time, float("nan"), 0))
        else:
            group_data = df[df["reflection_times_norm"] == norm_time]
            ece_value = ece(group_data, metric_column="recall")
            ece_values.append((norm_time, ece_value, len(group_data)))

    ece_df = pd.DataFrame(ece_values, columns=["reflection_times_norm", "ece", "ece_sample_count"])
    grouped = grouped_basic.merge(ece_df, on="reflection_times_norm")

    # 过滤样本量过少的数据点（保留样本量>=5的数据点）
    min_sample_threshold = 5
    grouped_filtered = grouped[grouped["sample_count"] >= min_sample_threshold].copy()

    # 为ECE单独过滤
    grouped_ece_filtered = grouped[grouped["ece_sample_count"] >= min_sample_threshold].copy()

    plt.figure(figsize=(12, 3.2))
    for i, metric in enumerate(["precision", "recall", "model_confidence_extracted", "ece"]):
        if metric == "precision":
            ylabel = "Precision"
        elif metric == "recall":
            ylabel = "Recall"
        elif metric == "model_confidence_extracted":
            ylabel = "Confidence"
        elif metric == "ece":
            ylabel = "ECE"
        else:
            raise ValueError(f"Unknown metric: {metric}")

        # 选择过滤后的数据
        if metric == "ece":
            plot_data = grouped_ece_filtered
            valid_df = grouped_ece_filtered[["reflection_times_norm", metric]].dropna()
        else:
            plot_data = grouped_filtered
            valid_df = grouped_filtered[["reflection_times_norm", metric]].dropna()

        # 计算相关性
        try:
            result = spearmanr(valid_df["reflection_times_norm"].values, valid_df[metric].values)
            corr = result[0]
            p_value = result[1]
            significant = str(p_value) != "nan" and float(str(p_value)) < 0.05
        except Exception:
            corr = 0.0
            p_value = 1.0
            significant = False

        plt.subplot(1, 4, i + 1)

        # 绘制数据点和连线（统一风格）
        sns.lineplot(
            data=plot_data,
            x="reflection_times_norm",
            y=metric,
            marker="o",
            markersize=6,
            linewidth=2,
            label="Data",
        )

        # 线性拟合
        if len(valid_df) > 1:
            X = valid_df["reflection_times_norm"].values.reshape(-1, 1)
            y = valid_df[metric].values
            reg = LinearRegression().fit(X, y)

            # 生成拟合直线的x值
            x_range = np.linspace(valid_df["reflection_times_norm"].min(), valid_df["reflection_times_norm"].max(), 100)
            y_pred = reg.predict(x_range.reshape(-1, 1))

            # 绘制拟合直线
            plt.plot(
                x_range,
                y_pred,
                "--",
                color="red",
                alpha=0.7,
                linewidth=2,
                label=f"Fit: y={reg.coef_[0]:.3f}x+{reg.intercept_:.3f}",
            )

        plt.xlabel("Normalized Reflection Times")
        plt.ylabel(ylabel)
        plt.title(f"Corr: {corr:.2f}, p: {p_value:.2g}\n({'Significant' if significant else 'Not Significant'})")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"figures/reflection-{model.series_name.lower()}-{dataset}.pdf")
    # plt.show()

    # 打印过滤信息
    print(f"原始数据点数: {len(grouped)}")
    print(f"过滤后数据点数: {len(grouped_filtered)}")
    print(f"过滤掉的数据点占比: {(len(grouped) - len(grouped_filtered)) / len(grouped) * 100:.1f}%")


if __name__ == "__main__":
    run_async(main())
