import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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


def _create_and_save_3d_legend(palette, region_color, dataset):
    """创建并保存3D图的单独图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(4, 0.3))
    ax_legend.axis("off")

    # 重新创建图例项
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["Short-CoT"], markersize=8, label="Short-CoT"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=palette["Long-CoT"], markersize=8, label="Long-CoT"),
        Patch(facecolor=region_color, alpha=0.2, label="Overconfidence Region"),
    ]

    ax_legend.legend(handles=legend_elements, loc="center", frameon=True, ncol=3, fontsize=15)

    # 保存单独的图例（移除模型名）
    plt.savefig(f"figures/performance-3d-{dataset}-recall-legend.pdf", bbox_inches="tight")
    # plt.show()


def _create_and_save_movement_legend(dataset):
    """创建并保存movement图的单独图例"""
    fig_legend, ax_legend = plt.subplots(figsize=(5, 0.3))
    ax_legend.axis("off")

    # 重新创建图例项
    legend_elements = [
        Line2D(
            [0], [0], marker="o", color="w", markerfacecolor="tab:orange", markersize=8, alpha=1.0, label="Short-CoT"
        ),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="tab:blue", markersize=8, alpha=1.0, label="Long-CoT"),
        Line2D([0], [0], color="k", linestyle="--", alpha=1.0, label="Perfectly Calibrated"),
        Line2D([0], [0], color="red", marker=">", markersize=8, alpha=1.0, label="Toward Calibration"),
    ]

    ax_legend.legend(handles=legend_elements, loc="center", frameon=True, ncol=4, fontsize=15)

    # 保存单独的图例（移除模型名）
    plt.savefig(f"figures/performance-movement-{dataset}-recall-legend.pdf", bbox_inches="tight")
    # plt.show()


async def main():
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

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

        # 计算两种setting下的平均precision和recall
        print(f"=== {model_series_name} 平均指标统计 ===")
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

        fig = plt.figure(figsize=(4, 4))
        ax = fig.add_subplot(111, projection="3d")

        for setting, color in palette.items():
            sub = df[df["setting"] == setting]
            ax.scatter(
                sub["model_confidence_extracted"],
                sub["recall"],
                sub["density"],
                c=color,
                label=setting,
                alpha=0.6,
                edgecolor="k",
                linewidth=0.0,
            )

        # 自动计算高密度区域的边界
        short_cot_data = df[df["setting"] == "Short-CoT"]

        # 找出Short-CoT中density最高的点
        density_threshold = short_cot_data["density"].quantile(0.3)  # 密度前70%的点
        high_density_points = short_cot_data[short_cot_data["density"] >= density_threshold]

        if len(high_density_points) > 0:
            # 计算高密度点的边界
            conf_values = high_density_points["model_confidence_extracted"]
            recall_values = high_density_points["recall"]

            # 增加边界扩展幅度，使方框更宽
            conf_margin = max(0.1, (conf_values.max() - conf_values.min()) * 0.5)  # 最小0.1，或50%扩展
            recall_margin = max(0.1, (recall_values.max() - recall_values.min()) * 0.5)  # 最小0.1，或50%扩展

            x_range = [max(0, conf_values.min() - conf_margin), min(1.0, conf_values.max() + conf_margin)]
            y_range = [max(0, recall_values.min() - recall_margin), min(1.0, recall_values.max() + recall_margin)]

            print("自动检测到高密度区域:")
            print(f"  Confidence范围: [{x_range[0]:.3f}, {x_range[1]:.3f}]")
            print(f"  Recall范围: [{y_range[0]:.3f}, {y_range[1]:.3f}]")
            print(f"  包含 {len(high_density_points)} 个高密度点")
        else:
            # 备用方案：使用更宽的固定范围
            x_range = [0.6, 1.0]  # 从0.7调整到0.6
            y_range = [0.0, 0.4]  # 从0.3调整到0.4
            print("未检测到明显的高密度区域，使用默认范围")

        z_max = df["density"].max()

        vertices = [
            # 底面
            [x_range[0], y_range[0], 0],
            [x_range[1], y_range[0], 0],
            [x_range[1], y_range[1], 0],
            [x_range[0], y_range[1], 0],
            # 顶面
            [x_range[0], y_range[0], z_max],
            [x_range[1], y_range[0], z_max],
            [x_range[1], y_range[1], z_max],
            [x_range[0], y_range[1], z_max],
        ]

        # 定义方框的六个面
        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
            [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
            [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
            [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
            [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
            [vertices[0], vertices[3], vertices[7], vertices[4]],  # 左面
        ]

        # 添加半透明方框
        # 柔和颜色选择：可以选择以下任一颜色
        # 'lightcoral' - 浅珊瑚色，比红色柔和
        # 'lightpink' - 浅粉色，温和
        # 'lightsalmon' - 浅鲑鱼色，暖色调
        # 'peachpuff' - 桃色，非常柔和
        # 'mistyrose' - 雾玫瑰色，淡雅
        region_color = "lightcoral"  # 可以替换为上述任意颜色

        poly = Poly3DCollection(faces, alpha=0.2, facecolor=region_color, edgecolor=region_color, linewidth=1)
        ax.add_collection3d(poly)

        ax.set_xlabel("Confidence", fontsize=15)
        ax.set_ylabel("Recall", fontsize=15)
        ax.set_zlabel("Density", fontsize=15)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)

        # 设置刻度标签字号
        ax.tick_params(axis='x', labelsize=13)
        ax.tick_params(axis='y', labelsize=13)
        ax.tick_params(axis='z', labelsize=13)

        # 移除重复的导入（已在文件开头导入）
        # 不添加图例到3D主图

        ax.set_title(f"{model_series_name}", fontsize=15, y=1.02)

        # 调整轴标签位置
        ax.xaxis.labelpad = 5
        ax.yaxis.labelpad = 5
        ax.zaxis.labelpad = -147

        # 使用subplots_adjust而不是tight_layout
        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
        plt.savefig(
            f"figures/performance-3d-{model_series_name.lower()}-{dataset}-recall-main.pdf", bbox_inches="tight"
        )
        # plt.show()

        # 1. 构建网格
        x_min, x_max = df["model_confidence_extracted"].min(), df["model_confidence_extracted"].max()
        y_min, y_max = df["recall"].min(), df["recall"].max()
        n_grid = 10  # 网格数
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
        arrow_colors = np.full(xx.shape, "gray", dtype=object)  # 默认颜色为灰色

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

                # 判断箭头是否指向对角线方向
                if norm > 0:
                    arrow_dir = mean_vec / norm

                    # 判断当前点相对于对角线的位置
                    # 对角线方程: y = x，即 x - y = 0
                    distance_to_diagonal = gx - gy

                    if distance_to_diagonal > 0:
                        # 点在对角线右下方（overconfident区域，confidence > recall）
                        # 指向校准的方向应该是向左上（减少confidence或增加recall）
                        # 目标方向向量 (-1, 1)，归一化后为 (-1/√2, 1/√2)
                        target_dir = np.array([-1, 1]) / np.sqrt(2)
                    else:
                        # 点在对角线左上方（underconfident区域，confidence < recall）
                        # 指向校准的方向应该是向右下（增加confidence或减少recall）
                        # 目标方向向量 (1, -1)，归一化后为 (1/√2, -1/√2)
                        target_dir = np.array([1, -1]) / np.sqrt(2)

                    # 计算箭头方向与目标方向的夹角余弦值
                    cos_angle = np.dot(arrow_dir, target_dir)
                    # 如果夹角小于90度（余弦值大于0），则认为指向校准方向
                    if cos_angle > 0:
                        row, col = np.unravel_index(i, xx.shape)
                        arrow_colors[row, col] = "red"

        # 5. 绘制
        plt.figure(figsize=(4, 4))
        plt.scatter(short_points[:, 0], short_points[:, 1], c="tab:orange", alpha=0.1, label="Short-CoT")
        plt.scatter(long_points[:, 0], long_points[:, 1], c="tab:blue", alpha=0.1, label="Long-CoT")

        # 分别绘制不同颜色的箭头
        for color in ["gray", "red"]:
            mask = arrow_colors == color
            if np.any(mask):
                plt.quiver(
                    xx[mask],
                    yy[mask],
                    U[mask],
                    V[mask],
                    angles="xy",
                    scale_units="xy",
                    scale=1.5,
                    color=color,
                    width=0.01,
                    alpha=1.0,
                )

        # 添加对角线
        plt.plot([x_min, x_max], [y_min, y_max], "k--", alpha=0.5, linewidth=1, label="Perfectly Calibrated")

        # 创建自定义图例，使用不透明的颜色
        # 移除重复的导入（已在文件开头导入）
        # 不添加图例到movement主图

        plt.xlabel("Confidence", fontsize=15)
        plt.ylabel("Recall", fontsize=15)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.xlim(x_min - 0.02, x_max + 0.02)
        plt.ylim(y_min - 0.02, y_max + 0.02)
        # 不添加图例到主图
        plt.title(f"{model_series_name}", fontsize=15, pad=10)
        plt.tight_layout()
        plt.savefig(
            f"figures/performance-movement-{model_series_name.lower()}-{dataset}-recall-main.pdf", bbox_inches="tight"
        )
        # plt.show()

    # 创建并保存分离的图例（在循环外只运行一次）
    palette = {"Short-CoT": "tab:orange", "Long-CoT": "tab:blue"}
    region_color = "lightcoral"
    _create_and_save_3d_legend(palette, region_color, dataset)
    _create_and_save_movement_legend(dataset)


if __name__ == "__main__":
    run_async(main())
