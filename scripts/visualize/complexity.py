import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence import MethodName
from confidence.model import ModelName
from confidence.template import SubsetSumTemplate, Template, TimeTablingTemplate
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
    temperature = 0.2

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.cot),
        # Setting(model=ModelName.QWEN3_8B_THINK, template=SubsetSumTemplate.simple),
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template=SubsetSumTemplate.cot),
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
        if isinstance(setting.template, SubsetSumTemplate):
            df["setting"] = f"{setting.model}--{setting.template.value.replace('-subsetsum', '')}"
        else:
            df["setting"] = f"{setting.model}--{setting.template}"

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)
    df = prf(df, method, dataset)

    df.loc[df["setting"] == "qwen3-8b-no_think--cot", "setting"] = "Short-CoT"
    df.loc[df["setting"] == "qwen3-8b-think--simple", "setting"] = "Long-CoT"

    plt.figure()
    sns.lineplot(data=df, x="answer_count_bin", y="recall", hue="setting")
    plt.xlabel("Complexity Bin")
    plt.ylabel("Recall")
    # plt.title("SubsetSum")
    plt.title("TimeTabling")
    plt.show()

    plt.figure()
    sns.lineplot(data=df, x="answer_count_bin", y="model_confidence_extracted", hue="setting")
    plt.xlabel("Complexity Bin")
    plt.ylabel("Confidence")
    # plt.title("SubsetSum")
    plt.title("TimeTabling")
    plt.show()


if __name__ == "__main__":
    run_async(main())
