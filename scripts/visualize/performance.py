import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from scipy.stats import gaussian_kde
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import prf, add_confidence_column
from confidence.logger import Logger
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template="simple"),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        title = f"{dataset}--{setting.template}--{setting.model}--{temperature}--{turn}"
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)

        # 计算 precision, recall 等指标
        df = prf(df, dataset)
        df = add_confidence_column(df)

        if setting.model == ModelName.QWEN3_8B_NO_THINK and setting.template == "cot":
            df["setting"] = "Short-CoT"
        elif setting.model == ModelName.QWEN3_8B_THINK and setting.template == "simple":
            df["setting"] = "Long-CoT"
        else:
            raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    # 计算两种setting下的平均precision和recall
    print("=== 平均指标统计 ===")
    for setting in ["Short-CoT", "Long-CoT"]:
        setting_data = df[df["setting"] == setting]
        avg_precision = setting_data["precision"].mean()
        avg_recall = setting_data["recall"].mean()
        print(f"{setting}:")
        print(f"  平均 Precision: {avg_precision:.4f}")
        print(f"  平均 Recall: {avg_recall:.4f}")
        print(f"  样本数量: {len(setting_data)}")
        print()

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

    # 创建 JointGrid
    g = sns.JointGrid(
        data=df,
        x="model_confidence_extracted",
        y="recall",
        hue="setting",
        height=6,
        space=0,
    )

    # 主图：密度为点大小的散点图
    for setting in ["Long-CoT", "Short-CoT"]:
        color = palette[setting]
        sub = df[df["setting"] == setting]
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
        data = df[df["setting"] == setting]["model_confidence_extracted"].values
        sns.kdeplot(
            x=data,
            ax=g.ax_marg_x,
            color=color,
            fill=True,
            alpha=0.3,
            label=setting,
        )

    # y 轴密度
    for setting, color in palette.items():
        data = df[df["setting"] == setting]["recall"].values
        sns.kdeplot(
            y=data,
            ax=g.ax_marg_y,
            color=color,
            fill=True,
            alpha=0.3,
            label=setting,
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
        sub = df[df["setting"] == setting]
        ax.scatter(  # 改为 ax.scatter 而不是 plt.scatter
            sub["model_confidence_extracted"],
            sub["recall"],
            sub["density"],
            c=color,
            label=setting,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Recall")
    ax.set_zlabel("Density")  # type: ignore
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Setting")
    ax.set_title("3D Scatter: Confidence vs Recall vs Density")  # 改为 ax.set_title
    plt.tight_layout()
    plt.show()

    # 1. 构建网格
    x_min, x_max = df["model_confidence_extracted"].min(), df["model_confidence_extracted"].max()
    y_min, y_max = df["recall"].min(), df["recall"].max()
    n_grid = 20  # 网格数
    grid_x = np.linspace(x_min, x_max, n_grid)
    grid_y = np.linspace(y_min, y_max, n_grid)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_width = (x_max - x_min) / (n_grid - 1)

    # 2. 取交集
    short_df = df[df["setting"] == "Short-CoT"].set_index("question_id")
    long_df = df[df["setting"] == "Long-CoT"].set_index("question_id")
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

    # ============================ PRECISION ============================

    palette = {"Short-CoT": "tab:orange", "Long-CoT": "tab:blue"}
    # 计算每个 setting 的密度
    for setting in ["Short-CoT", "Long-CoT"]:
        mask = df["setting"] == setting
        x = df.loc[mask, "model_confidence_extracted"].values.astype(float)
        y = df.loc[mask, "precision"].values.astype(float)
        if len(x) > 1:
            kde = gaussian_kde(np.vstack([x, y]))
            density = kde(np.vstack([x, y]))
        else:
            density = np.ones_like(x)
        df.loc[mask, "density"] = density

    # 创建 JointGrid
    g = sns.JointGrid(
        data=df,
        x="model_confidence_extracted",
        y="precision",
        hue="setting",
        height=6,
        space=0,
    )

    # 主图：密度为点大小的散点图
    for setting in ["Long-CoT", "Short-CoT"]:
        color = palette[setting]
        sub = df[df["setting"] == setting]
        g.ax_joint.scatter(
            sub["model_confidence_extracted"],
            sub["precision"],
            s=50 + 200 * (sub["density"] - sub["density"].min()) / (sub["density"].max() - sub["density"].min() + 1e-6),
            alpha=0.5,
            color=color,
            label=setting,
            edgecolor="k",
            linewidth=0.5,
        )

    # x 轴密度
    for setting, color in palette.items():
        data = df[df["setting"] == setting]["model_confidence_extracted"].values
        sns.kdeplot(
            x=data,
            ax=g.ax_marg_x,
            color=color,
            fill=True,
            alpha=0.3,
            label=setting,
        )

    # y 轴密度
    for setting, color in palette.items():
        data = df[df["setting"] == setting]["precision"].values
        sns.kdeplot(
            y=data,
            ax=g.ax_marg_y,
            color=color,
            fill=True,
            alpha=0.3,
            label=setting,
        )

    g.ax_joint.set_xlabel("Confidence")
    g.ax_joint.set_ylabel("Precision")
    g.ax_joint.set_xlim(-0.02, 1.02)
    g.ax_joint.set_ylim(-0.02, 1.02)
    g.ax_joint.legend(title="Setting")
    plt.show()

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    for setting, color in palette.items():
        sub = df[df["setting"] == setting]
        ax.scatter(  # 改为 ax.scatter 而不是 plt.scatter
            sub["model_confidence_extracted"],
            sub["precision"],
            sub["density"],
            c=color,
            label=setting,
            alpha=0.6,
            edgecolor="k",
            linewidth=0.5,
        )

    ax.set_xlabel("Confidence")
    ax.set_ylabel("Precision")
    ax.set_zlabel("Density")  # type: ignore
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.legend(title="Setting")
    ax.set_title("3D Scatter: Confidence vs Precision vs Density")  # 改为 ax.set_title
    plt.tight_layout()
    plt.show()

    # 1. 构建网格
    x_min, x_max = df["model_confidence_extracted"].min(), df["model_confidence_extracted"].max()
    y_min, y_max = df["precision"].min(), df["precision"].max()
    n_grid = 20  # 网格数
    grid_x = np.linspace(x_min, x_max, n_grid)
    grid_y = np.linspace(y_min, y_max, n_grid)
    xx, yy = np.meshgrid(grid_x, grid_y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_width = (x_max - x_min) / (n_grid - 1)

    # 2. 取交集
    short_df = df[df["setting"] == "Short-CoT"].set_index("question_id")
    long_df = df[df["setting"] == "Long-CoT"].set_index("question_id")
    common_ids = np.intersect1d(short_df.index, long_df.index)
    short_common = short_df.loc[common_ids]
    long_common = long_df.loc[common_ids]

    # 3. 计算每个点的移动向量
    short_points = short_common[["model_confidence_extracted", "precision"]].values
    long_points = long_common[["model_confidence_extracted", "precision"]].values
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
    plt.ylabel("Precision")
    plt.xlim(x_min - 0.02, x_max + 0.02)
    plt.ylim(y_min - 0.02, y_max + 0.02)
    plt.legend()
    plt.title("2D Short-CoT to Long-CoT Movement Field")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
