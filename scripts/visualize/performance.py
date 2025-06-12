import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import Template, SubsetSumTemplate, TimeTablingTemplate
from scripts.visualize.metrics import prf


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False
    turn = 0

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.cot),
        # Setting(model=ModelName.QWEN3_8B_THINK, template=SubsetSumTemplate.simple),
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template=SubsetSumTemplate.simple),
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template=SubsetSumTemplate.cot),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value + f"--turn{turn}",
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        if isinstance(setting.template, SubsetSumTemplate):
            df["setting"] = f"{setting.model}--{setting.template.value.replace('-subsetsum', '')}"
        else:
            df["setting"] = f"{setting.model}--{setting.template}"

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)
    df = prf(df, method, dataset)

    df.loc[df["setting"] == "qwen3-8b-no_think--cot", "setting"] = "Short-CoT"
    df.loc[df["setting"] == "qwen3-8b-think--simple", "setting"] = "Long-CoT"

    # 只保留 Short-CoT 和 Long-CoT
    plot_df = df[df["setting"].isin(["Short-CoT", "Long-CoT"])].copy()
    palette = {"Short-CoT": "tab:orange", "Long-CoT": "tab:blue"}

    # 计算每个 setting 的密度
    for setting in ["Short-CoT", "Long-CoT"]:
        mask = plot_df["setting"] == setting
        x = plot_df.loc[mask, "model_confidence_extracted"].values
        y = plot_df.loc[mask, "recall"].values
        if len(x) > 1:
            kde = gaussian_kde(np.vstack([x, y]))
            density = kde(np.vstack([x, y]))
        else:
            density = np.ones_like(x)
        plot_df.loc[mask, "density"] = density

    # 创建 JointGrid
    g = sns.JointGrid(
        data=plot_df,
        x="model_confidence_extracted",
        y="recall",
        hue="setting",
        height=6,
        space=0,
    )

    # 主图：密度为点大小的散点图
    for setting in ["Long-CoT", "Short-CoT"]:
        color = palette[setting]
        sub = plot_df[plot_df["setting"] == setting]
        g.ax_joint.scatter(
            sub["model_confidence_extracted"],
            sub["recall"],
            s=50 + 200 * (sub["density"] - sub["density"].min()) / (sub["density"].max() - sub["density"].min() + 1e-6),
            alpha=0.5,
            color=color,
            label=setting,
            edgecolor="k",
            linewidth=0.5,
        )

    # x 轴密度
    for setting, color in palette.items():
        sns.kdeplot(
            plot_df[plot_df["setting"] == setting]["model_confidence_extracted"],
            ax=g.ax_marg_x,
            color=color,
            fill=True,
            alpha=0.3,
            label=setting,
        )

    # y 轴密度
    for setting, color in palette.items():
        sns.kdeplot(
            plot_df[plot_df["setting"] == setting]["recall"],
            ax=g.ax_marg_y,
            color=color,
            fill=True,
            alpha=0.3,
            label=setting,
            vertical=True,
        )

    g.ax_joint.set_xlabel("Confidence")
    g.ax_joint.set_ylabel("Recall")
    g.ax_joint.set_xlim(-0.02, 1.02)
    g.ax_joint.set_ylim(-0.02, 1.02)
    g.ax_joint.legend(title="Setting")
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for setting, color in palette.items():
        sub = plot_df[plot_df["setting"] == setting]
        ax.scatter(
            sub["model_confidence_extracted"],
            sub["recall"],
            sub["density"],
            c=color,
            label=setting,
            alpha=0.6,
            edgecolor="k",
            s=40,
            linewidth=0.5,
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Recall")
    ax.set_zlabel("Density")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Setting")
    plt.title("3D Scatter: Confidence vs Recall vs Density")
    plt.tight_layout()
    plt.show()

    # 1. 构建网格
    x_min, x_max = plot_df["model_confidence_extracted"].min(), plot_df["model_confidence_extracted"].max()
    y_min, y_max = plot_df["recall"].min(), plot_df["recall"].max()
    n_grid = 20  # 网格数
    grid_x = np.linspace(x_min, x_max, n_grid)
    grid_y = np.linspace(y_min, y_max, n_grid)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_width = (x_max - x_min) / (n_grid - 1)

    # 2. 取交集
    short_df = plot_df[plot_df["setting"] == "Short-CoT"].set_index("question_id")
    long_df = plot_df[plot_df["setting"] == "Long-CoT"].set_index("question_id")
    common_ids = np.intersect1d(short_df.index, long_df.index)
    short_common = short_df.loc[common_ids]
    long_common = long_df.loc[common_ids]

    # 3. 计算每个点的移动向量
    short_points = short_common[["model_confidence_extracted", "recall"]].values
    long_points = long_common[["model_confidence_extracted", "recall"]].values
    move_vecs = long_points - short_points

    # 4. 对每个网格点，统计附近的点，计算平均向量
    radius = grid_width * 1.2  # 邻域半径
    U = np.zeros_like(xx)
    V = np.zeros_like(yy)
    for i, (gx, gy) in enumerate(grid_points):
        dists = np.sqrt((short_points[:, 0] - gx) ** 2 + (short_points[:, 1] - gy) ** 2)
        mask = dists < radius
        if np.any(mask):
            mean_vec = move_vecs[mask].mean(axis=0)
            # 限制箭头长度
            norm = np.linalg.norm(mean_vec)
            if norm > grid_width:
                mean_vec = mean_vec / norm * grid_width
            U.ravel()[i] = mean_vec[0]
            V.ravel()[i] = mean_vec[1]

    # 5. 绘制
    plt.figure(figsize=(8, 7))
    plt.scatter(short_points[:, 0], short_points[:, 1], c="tab:orange", alpha=0.1, label="Short-CoT")
    plt.scatter(long_points[:, 0], long_points[:, 1], c="tab:blue", alpha=0.1, label="Long-CoT")
    plt.quiver(xx, yy, U, V, angles="xy", scale_units="xy", scale=1.5, color="gray", width=0.005, alpha=1.0)
    plt.xlabel("Confidence")
    plt.ylabel("Recall")
    plt.xlim(x_min - 0.02, x_max + 0.02)
    plt.ylim(y_min - 0.02, y_max + 0.02)
    plt.legend()
    plt.title("2D Short-CoT to Long-CoT Movement Field")
    plt.tight_layout()
    plt.show()

    # plt.figure(figsize=(8, 7))
    # plt.scatter(short_points[:, 0], short_points[:, 1], c="tab:orange", alpha=0.1, label="Short-CoT")
    # plt.scatter(long_points[:, 0], long_points[:, 1], c="tab:blue", alpha=0.1, label="Long-CoT")
    # plt.streamplot(xx, yy, U, V, color="gray", density=1.2, linewidth=1.2, arrowsize=1.5)
    # plt.xlabel("Confidence")
    # plt.ylabel("Recall")
    # plt.xlim(x_min - 0.02, x_max + 0.02)
    # plt.ylim(y_min - 0.02, y_max + 0.02)
    # plt.legend()
    # plt.title("2D Short-CoT to Long-CoT Movement Field (Streamplot)")
    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    run_async(main())
