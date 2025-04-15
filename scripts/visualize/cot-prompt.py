import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from scipy.stats import ttest_rel, wilcoxon
from tortoise import run_async

from confidence.template import LogiQATemplate, Template
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    dataset = DatasetName.LogiQA
    method = MethodName.Verbal_0_100
    settings = [
        Setting(model=ModelName.QWEN2_5_7B, template=LogiQATemplate.CoTEval),
        Setting(model=ModelName.QWEN2_5_7B, template=LogiQATemplate.CoTEvalCoT),
    ]
    # dataset = DatasetName.GSM8K
    # method = MethodName.LogProb
    # settings = [
    #     Setting(model=ModelName.QWEN2_5_7B, template=GSM8KTemplate.BigGSM),
    #     Setting(model=ModelName.QWQ_32B, template=GSM8KTemplate.BigGSM),
    # ]
    # dataset = DatasetName.ARC
    # method = MethodName.LogProb
    # settings = [
    #     Setting(model=ModelName.QWEN2_5_7B, template=ARCTemplate.OpenCompass),
    #     Setting(model=ModelName.QWEN2_5_7B, template=ARCTemplate.OpenCompassCoT),
    # ]

    records_list = []
    for setting in settings:
        model = setting.model
        template = setting.template

        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--{template}--{model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)

        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)

        df = df[["id", "model_confidence_extracted"]].copy()
        df.rename(columns={"model_confidence_extracted": f"conf_{template.value}"}, inplace=True)
        records_list.append(df)

    merged_df = pd.merge(records_list[0], records_list[1], on="id", how="inner")
    template_cols = [col for col in merged_df if col.startswith("conf_")]

    avg_0, std_1 = merged_df[template_cols[0]].mean(), merged_df[template_cols[0]].std()
    avg_1, std_1 = merged_df[template_cols[1]].mean(), merged_df[template_cols[1]].std()
    print(f"Avg confidence for {settings[0].template.value} = {avg_0:.3f}, Std = {std_1:.3f}")
    print(f"Avg confidence for {settings[1].template.value} = {avg_1:.3f}, Std = {std_1:.3f}")

    t_stat, p_t = ttest_rel(merged_df[template_cols[0]], merged_df[template_cols[1]])
    w_stat, p_w = wilcoxon(merged_df[template_cols[0]], merged_df[template_cols[1]])
    print(f"Paired t-test: t = {t_stat:.3f}, p = {p_t:.4f}")
    print(f"Wilcoxon test: W = {w_stat}, p = {p_w:.4f}")

    def cohens_d(x, y):
        nx = len(x)
        ny = len(y)
        dof = nx + ny - 2
        return (x.mean() - y.mean()) / np.sqrt(((nx - 1) * x.std() ** 2 + (ny - 1) * y.std() ** 2) / dof)

    d = cohens_d(merged_df[template_cols[0]], merged_df[template_cols[1]])
    print(f"Cohen's d: {d:.3f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.boxplot(data=merged_df[template_cols])
    plt.title("Confidence Distribution")
    plt.ylabel("Confidence Score")
    plt.xticks(ticks=[0, 1], labels=[t.template.name for t in settings])

    plt.subplot(1, 2, 2)
    differences = merged_df[template_cols[0]] - merged_df[template_cols[1]]
    sns.histplot(differences, kde=True, bins=15)
    plt.title("Paired Differences")
    plt.xlabel(f"Difference ({settings[0].template.name} - {settings[1].template.name})")
    plt.axvline(differences.mean(), color="r", linestyle="--", label=f"Mean: {differences.mean():.2f}")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
