import pandas as pd
from pydantic import BaseModel
from scipy.stats import spearmanr
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import add_confidence_column, prf
from confidence.logger import Logger
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    dataset = DatasetName.SubsetSum
    # dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template="simple"),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot"),
        # Setting(model=ModelName.DEEPSEEK_R1, template="simple"),
        # Setting(model=ModelName.DEEPSEEK_V3, template="cot"),
        # Setting(model=ModelName.O4_MINI, template="simple"),
        # Setting(model=ModelName.GPT_4O_MINI, template="cot"),
    ]

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

        print(df["setting"])
        df["missed_solutions"] = (df["answer_count"] - df["total_solution_count"]) / df["answer_count"]
        valid = df[["model_confidence_extracted", "missed_solutions"]].dropna()
        corr, p = spearmanr(valid["model_confidence_extracted"], valid["missed_solutions"])
        print(f"{corr:.4f} ({p:.4f})")
        # print(f"模型置信度与遗落解数量的Spearman相关系数: {corr:.4f}，p值: {p:.4f}")


if __name__ == "__main__":
    run_async(main())
