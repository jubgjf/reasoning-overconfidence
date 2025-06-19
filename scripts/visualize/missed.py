import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import numpy as np
import pandas as pd
import seaborn as sns
from pydantic import BaseModel
from scipy.stats import gaussian_kde
from tortoise import run_async

from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
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
    turn = 0
    temperature = 0.2

    settings = [
        # Setting(model=ModelName.QWEN3_8B_THINK, template=TimeTablingTemplate.simple),
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

    df["missed_solutions"] = (df["answer_count"] - df["total_solution_count"]) / df["answer_count"]
    valid = df[["model_confidence_extracted", "missed_solutions"]].dropna()
    corr, p = spearmanr(valid["model_confidence_extracted"], valid["missed_solutions"])
    print(f"模型置信度与遗落解数量的Spearman相关系数: {corr:.4f}，p值: {p:.4f}")


if __name__ == "__main__":
    run_async(main())
