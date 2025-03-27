from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tortoise import run_async
from confidence.template import ARCTemplate, GSM8KTemplate, LogiQATemplate
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from scipy.stats import ttest_ind


class Setting(BaseModel):
    method: MethodName
    no_cot_memory: bool


async def main():
    model = ModelName.QWQ_32B
    # dataset = DatasetName.GSM8K
    # template = GSM8KTemplate.BigGSM
    dataset = DatasetName.ARC
    template = ARCTemplate.OpenCompass

    settings = [
        Setting(method=MethodName.Verbal_0_100, no_cot_memory=False),
        Setting(method=MethodName.Verbal_0_100, no_cot_memory=True),
        # Setting(method=MethodName.LogProb, no_cot_memory=False),
        # Setting(method=MethodName.LogProb, no_cot_memory=True),
        Setting(method=MethodName.P_True, no_cot_memory=False),
        Setting(method=MethodName.P_True, no_cot_memory=True),
    ]

    records_list = []

    for setting in settings:
        method = setting.method
        no_cot_memory = setting.no_cot_memory

        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        records = [record.model_dump() for record in records]
        df = pd.DataFrame(records)

        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)

        df["no-cot-memory"] = str(no_cot_memory)

        records_list.append(df)

    df = pd.concat(records_list)

    true_data = df[df["no-cot-memory"] == 'True']["model_confidence_extracted"]
    false_data = df[df["no-cot-memory"] == 'False']["model_confidence_extracted"]
    t_stat, p_value = ttest_ind(true_data, false_data)

    plt.figure(figsize=(8, 6))
    sns.boxplot(data=df, x="method", y="model_confidence_extracted", hue="no-cot-memory")

    plt.xlabel("Method")
    plt.ylabel("Model Confidence Extracted")
    plt.title(f"Model Confidence Distribution by Method and No-Cot-Memory\n{model}--{dataset}--{template}\nt = {t_stat:.2f}, p = {p_value:.2f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())

