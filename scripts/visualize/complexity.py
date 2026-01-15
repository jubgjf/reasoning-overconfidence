import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, prf
from confidence.logger import Logger
from confidence.model import ModelName

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 15


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    dataset = DatasetName.TimeTabling
    temperature = 0.2
    turn = 0

    settings_group = [
        [
            Setting(model=ModelName.QWEN3_8B_THINK, template="simple"),
            Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
        ],
        [
            Setting(model=ModelName.DEEPSEEK_R1, template="simple"),
            Setting(model=ModelName.DEEPSEEK_V3, template="cot"),
        ],
        [
            Setting(model=ModelName.O4_MINI, template="simple"),
            Setting(model=ModelName.GPT_4O_MINI, template="cot"),
        ],
    ]

    # 遍历每个设置组
    for settings in settings_group:
        model_series_name = settings[0].model.series_name
        assert all(setting.model.series_name == model_series_name for setting in settings)

        records_list = []
        for setting in settings:
            record_cls = dataset.record_cls
            title = f"{dataset}--{setting.template}--{setting.model}--{temperature}--{turn}".replace("/", "_")
            db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
            async with db_logger:
                records = await db_logger.fetch()

            method_records = [record.model_dump() for record in records]
            df = pd.DataFrame(method_records)

            if setting.template == "cot":
                df["setting"] = "Short-CoT"
            elif setting.template == "simple":
                df["setting"] = "Long-CoT"
            else:
                raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

            records_list.append(df)

        df = pd.concat(records_list, ignore_index=True)
        df = prf(df, dataset)
        df = add_confidence_column(df)

        # 绘制recall图（主图，不包含图例）
        plt.figure(figsize=(3, 3))
        sns.lineplot(data=df, x="answer_count_bin", y="recall", hue="setting")
        plt.xlabel("Complexity Level")
        plt.ylabel("Recall")
        plt.xticks(ticks=range(0, 10, 2), fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(f"{model_series_name}", fontsize=15, pad=10)
        plt.legend().remove()  # 移除图例
        plt.tight_layout()
        plt.savefig(f"figures/complexity-{model_series_name.lower()}-{dataset}-recall-main.pdf")
        # plt.show()

        # 绘制confidence图（主图，不包含图例）
        plt.figure(figsize=(3, 3))
        sns.lineplot(data=df, x="answer_count_bin", y="model_confidence_extracted", hue="setting")
        plt.xlabel("Complexity Level")
        plt.ylabel("Confidence")
        plt.xticks(ticks=range(0, 10, 2), fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(f"{model_series_name}", fontsize=15, pad=10)
        plt.legend().remove()  # 移除图例
        plt.tight_layout()
        plt.savefig(f"figures/complexity-{model_series_name.lower()}-{dataset}-confidence-main.pdf")
        # plt.show()

    # 创建并保存共用的图例（在循环外只运行一次）
    _create_and_save_legend(dataset)


def _create_and_save_legend(dataset):
    """创建并保存单独的图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(5, 0.2))
    ax_legend.axis("off")

    # 创建图例项（对应Short-CoT和Long-CoT）
    handles = []
    labels = []

    # 为每种CoT类型创建图例项
    for setting_name in ["Long-CoT", "Short-CoT"]:
        # 使用seaborn的默认颜色
        color = "tab:blue" if setting_name == "Long-CoT" else "tab:orange"
        handle = mlines.Line2D([0], [0], marker="o", color=color, linewidth=2, markersize=6, label=setting_name)
        handles.append(handle)
        labels.append(setting_name)

    ax_legend.legend(handles, labels, loc="center", frameon=True, ncol=2)

    # 保存单独的图例
    plt.savefig(f"figures/complexity-{dataset}-legend.pdf", bbox_inches="tight")
    # plt.show()


if __name__ == "__main__":
    run_async(main())
