import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from tortoise import run_async
from scipy.stats import gaussian_kde

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName
from confidence.template import Template, SubsetSumTemplate, TimeTablingTemplate


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    judge_model = ModelName.QWEN3_32B_NO_THINK
    dataset = DatasetName.SubsetSum
    # dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False

    settings = [
        # Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.simple),
        # Setting(model=ModelName.QWEN3_8B_NO_THINK, template=TimeTablingTemplate.cot),
        Setting(model=ModelName.QWEN3_8B_THINK, template=SubsetSumTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=SubsetSumTemplate.simple),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template=SubsetSumTemplate.cot),
    ]

    records_list = []
    for setting in settings:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{setting.template}--{setting.model}--evaluate-by-{judge_model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)
        if isinstance(setting.template, SubsetSumTemplate):
            df["setting"] = f"{setting.model}--{setting.template.value.replace('-subsetsum', '')}"
        else:
            df["setting"] = f"{setting.model}--{setting.template}"

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    df["correct_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[0]))
    df["total_solution_count"] = df["eval_result"].apply(lambda x: int(x.split("/")[1]))

    df = df[df["correct_solution_count"] <= df["total_solution_count"]]
    df = df[df["correct_solution_count"] <= df["answer_count"]]
    df = df[df["total_solution_count"] <= df["answer_count"]]

    if dataset == DatasetName.TimeTabling:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50))
    elif dataset == DatasetName.SubsetSum:
        df["answer_count_bin"] = df["answer_count"].apply(lambda x: int(x // 50) if int(x // 50) < 6 else 6)
    else:
        raise NotImplementedError

    df["recall"] = df["correct_solution_count"] / df["answer_count"]
    df["precision"] = df["correct_solution_count"] / df["total_solution_count"]

    df.loc[df["setting"] == "qwen3-8b-no_think--cot", "setting"] = "Short-CoT"
    df.loc[df["setting"] == "qwen3-8b-no_think--simple", "setting"] = "no-CoT"
    df.loc[df["setting"] == "qwen3-8b-think--simple", "setting"] = "Long-CoT"

    x = df[df["setting"] == "no-CoT"]["recall"].values
    y = df[df["setting"] == "no-CoT"]["model_confidence_extracted"].values
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=x, y=y, hue=density, size=density, sizes=(10, 200), palette="viridis", legend=False)
    plt.xlim(0 - 0.05, 1 + 0.05)
    plt.ylim(0 - 0.05, 1 + 0.05)
    plt.xlabel("Recall")
    plt.ylabel("Confidence")
    plt.title("no-CoT SubsetSum")
    # plt.title("no-CoT TimeTabling")
    plt.show()

    x = df[df["setting"] == "Short-CoT"]["recall"].values
    y = df[df["setting"] == "Short-CoT"]["model_confidence_extracted"].values
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=x, y=y, hue=density, size=density, sizes=(10, 200), palette="viridis", legend=False)
    plt.xlim(0 - 0.05, 1 + 0.05)
    plt.ylim(0 - 0.05, 1 + 0.05)
    plt.xlabel("Recall")
    plt.ylabel("Confidence")
    plt.title("no-CoT SubsetSum")
    # plt.title("no-CoT TimeTabling")
    plt.show()

    x = df[df["setting"] == "Long-CoT"]["recall"].values
    y = df[df["setting"] == "Long-CoT"]["model_confidence_extracted"].values
    xy = np.vstack([x, y])
    density = gaussian_kde(xy)(xy)
    plt.figure(figsize=(5, 4))
    sns.scatterplot(x=x, y=y, hue=density, size=density, sizes=(10, 200), palette="viridis", legend=False)
    plt.xlim(0 - 0.05, 1 + 0.05)
    plt.ylim(0 - 0.05, 1 + 0.05)
    plt.xlabel("Recall")
    plt.ylabel("Confidence")
    plt.title("no-CoT SubsetSum")
    # plt.title("no-CoT TimeTabling")
    plt.show()


if __name__ == "__main__":
    run_async(main())
