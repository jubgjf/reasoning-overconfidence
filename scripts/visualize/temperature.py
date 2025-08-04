import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, ece, prf
from confidence.logger import Logger
from confidence.model import ModelName

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


async def main():
    model = ModelName.QWEN3_8B_NO_THINK
    template = "cot"
    turn = 0
    dataset = DatasetName.TimeTabling

    records_list = []
    for temperature in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        record_cls = dataset.record_cls
        title = f"{dataset}--{template}--{model}--{temperature}--{turn}"
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()
        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        df["temperature"] = temperature

        # 添加数据集标识
        if dataset == DatasetName.TimeTabling:
            df["dataset"] = "TimeTabling"
        else:
            df["dataset"] = "SubsetSum"

        # 计算指标
        df = prf(df, dataset)
        df = add_confidence_column(df)

        # 计算 ECE 值
        ece_value = ece(df, metric_column="recall")
        df["ece"] = ece_value

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    # 绘制 Recall vs Temperature 和 ECE vs Temperature 在同一个图中
    fig, (ax1_top, ax1_bottom) = plt.subplots(
        2, 1, figsize=(3, 3), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0.15}
    )

    # 上半部分 - 显示 0.8-1.0 范围
    color1 = "tab:blue"
    color2 = "tab:red"

    # 上半部分的左y轴 - Recall
    sns.lineplot(data=df, x="temperature", y="recall", color=color1, marker="o", label="Recall", ax=ax1_top, ci=None)
    ax1_top.tick_params(axis="y", labelcolor=color1)
    ax1_top.set_ylim(0.8, 1.0)
    ax1_top.grid(True, alpha=0.3)
    ax1_top.set_ylabel("")  # 移除默认的y轴标签

    # 上半部分的右y轴 - ECE
    ax2_top = ax1_top.twinx()
    sns.lineplot(data=df, x="temperature", y="ece", color=color2, marker="s", label="ECE", ax=ax2_top, ci=None)
    ax2_top.tick_params(axis="y", labelcolor=color2)
    if dataset == DatasetName.TimeTabling:
        ax2_top.set_ylim(0.8, 1.0)
    elif dataset == DatasetName.SubsetSum:
        ax2_top.set_ylim(0.5, 1.0)
    ax2_top.set_ylabel("")  # 移除默认的y轴标签

    # 下半部分 - 显示 0-0.2 范围
    # 下半部分的左y轴 - Recall
    ax1_bottom.set_xlabel("Temperature")
    sns.lineplot(data=df, x="temperature", y="recall", color=color1, marker="o", ax=ax1_bottom, ci=None)
    ax1_bottom.tick_params(axis="y", labelcolor=color1)
    ax1_bottom.set_ylim(0, 0.2)
    ax1_bottom.grid(True, alpha=0.3)
    ax1_bottom.set_ylabel("")  # 移除默认的y轴标签

    # 下半部分的右y轴 - ECE
    ax2_bottom = ax1_bottom.twinx()
    sns.lineplot(data=df, x="temperature", y="ece", color=color2, marker="s", ax=ax2_bottom, ci=None)
    ax2_bottom.tick_params(axis="y", labelcolor=color2)
    if dataset == DatasetName.TimeTabling:
        ax2_bottom.set_ylim(0, 0.2)
    elif dataset == DatasetName.SubsetSum:
        ax2_bottom.set_ylim(0, 0.5)
    ax2_bottom.set_ylabel("")  # 移除默认的y轴标签

    # 添加断轴标记
    d = 0.015  # 断轴标记的大小
    kwargs = dict(transform=ax1_top.transAxes, color="k", clip_on=False)
    ax1_top.plot((-d, +d), (-d, +d), **kwargs)
    ax1_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax1_bottom.transAxes)
    ax1_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax1_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

    # 添加整体的y轴标签
    fig.text(0.02, 0.52, "Recall", va="center", rotation="vertical", color=color1, fontsize=10)
    fig.text(0.94, 0.52, "ECE (r)", va="center", rotation="vertical", color=color2, fontsize=10)

    # 添加图例到上半部分
    lines1, labels1 = ax1_top.get_legend_handles_labels()
    lines2, labels2 = ax2_top.get_legend_handles_labels()
    ax1_top.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    # 移除自动生成的图例
    for ax in [ax1_bottom, ax2_top, ax2_bottom]:
        legend = ax.get_legend()
        if legend:
            legend.remove()

    plt.tight_layout()
    plt.subplots_adjust(left=0.17, right=0.83, bottom=0.18, top=0.85)
    plt.savefig(f"figures/temperature-{model.series_name.lower()}-{dataset.name.lower()}-recall-ece.pdf")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
