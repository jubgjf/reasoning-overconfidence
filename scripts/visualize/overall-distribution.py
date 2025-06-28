import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence import MethodName
from confidence.model import ModelName
from confidence.template import Template, TimeTablingTemplate
from scripts.visualize.metrics import prf


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False
    temperature = 0.2

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.cot),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--{temperature}--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        df["setting"] = f"{setting.model}--{setting.template}"

        records_list.append(df)

    df = pd.concat(records_list)
    df = prf(df, method, dataset)

    df.loc[df["setting"] == "qwen3-8b-no_think--cot", "setting"] = "Short-CoT"
    df.loc[df["setting"] == "qwen3-8b-think--simple", "setting"] = "Long-CoT"

    bins = np.linspace(0, 1, 11)
    df["confidence_bin"] = pd.cut(df["model_confidence_extracted"], bins=bins, include_lowest=True, right=True)
    df["bin_right"] = df["confidence_bin"].apply(lambda x: x.right)
    grouped = df.groupby(["bin_right", "setting"]).size().reset_index(name="count")
    plt.figure(figsize=(8, 4))
    sns.barplot(data=grouped, x="bin_right", y="count", hue="setting", palette="muted")
    plt.xlabel("Confidence (bin right edge)")
    plt.ylabel("Frequency")
    # plt.title(f"Overall Confidence Distribution\n{dataset}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
