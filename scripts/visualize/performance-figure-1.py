import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pydantic import BaseModel
from scipy.stats import gaussian_kde
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
    turn = 0
    temperature = 0.2

    settings = [
        Setting(model=ModelName.O4_MINI, template="simple"),
        Setting(model=ModelName.GPT_4O_MINI, template="cot"),
    ]
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

        # 计算 precision, recall 等指标
        df = prf(df, dataset)
        df = add_confidence_column(df)

        if setting.template == "cot":
            df["setting"] = "Short-CoT"
        elif setting.template == "simple":
            df["setting"] = "Long-CoT"
        else:
            raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    # ============================ RECALL ============================

    palette = {"Short-CoT": "tab:orange", "Long-CoT": "tab:blue"}
    # 计算每个 setting 的密度
    for setting in ["Short-CoT", "Long-CoT"]:
        mask = df["setting"] == setting
        x = df.loc[mask, "model_confidence_extracted"].values.astype(float)
        y = df.loc[mask, "recall"].values.astype(float)
        if len(x) > 1:
            kde = gaussian_kde(np.vstack([x, y]))
            density = kde(np.vstack([x, y]))
        else:
            density = np.ones_like(x)
        df.loc[mask, "density"] = density

    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111, projection="3d")

    point_size = max(40, int(plt.rcParams["font.size"] * 3))
    # 定义深色边框颜色
    edge_palette = {"Short-CoT": "#8B4500", "Long-CoT": "#00008B"}  # 更深的颜色
    for setting, color in palette.items():
        sub = df[df["setting"] == setting]
        ax.scatter(
            sub["model_confidence_extracted"],
            sub["recall"],
            sub["density"],
            c=color,
            label=setting,
            alpha=0.6,
            edgecolor=edge_palette[setting],
            linewidth=1.0,
            s=point_size,
        )

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    # 隐藏坐标轴刻度值
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    # 调整轴标签位置
    # 增大轴标签与刻度/数值之间的间距以适应较大的字体
    ax.xaxis.labelpad = 15
    ax.yaxis.labelpad = 15
    # z 轴的 labelpad 在 3D 中通常需要正值来向外移动标签
    ax.zaxis.labelpad = 15

    # 使用subplots_adjust而不是tight_layout
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.show()


if __name__ == "__main__":
    run_async(main())
