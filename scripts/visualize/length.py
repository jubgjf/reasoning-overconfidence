import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tortoise import run_async
from confidence.template import ARCTemplate, Template
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    # dataset = DatasetName.LogiQA
    # method = MethodName.LogProb
    # settings = [
    #     Setting(model=ModelName.QWEN2_5_7B, template=LogiQATemplate.CoTEval),
    #     Setting(model=ModelName.QWEN2_5_7B, template=LogiQATemplate.CoTEvalCoT),
    #     Setting(model=ModelName.QWQ_32B, template=LogiQATemplate.CoTEval),
    # ]
    # dataset = DatasetName.GSM8K
    # method = MethodName.LogProb
    # settings = [
    #     Setting(model=ModelName.QWEN2_5_7B, template=GSM8KTemplate.BigGSM),
    #     Setting(model=ModelName.QWQ_32B, template=GSM8KTemplate.BigGSM),
    # ]
    dataset = DatasetName.ARC
    method = MethodName.LogProb
    settings = [
        Setting(model=ModelName.QWEN2_5_7B, template=ARCTemplate.OpenCompass),
        Setting(model=ModelName.QWEN2_5_7B, template=ARCTemplate.OpenCompassCoT),
        Setting(model=ModelName.QWQ_32B, template=ARCTemplate.OpenCompass),
    ]

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

        tokenizer = AutoTokenizer.from_pretrained(model.hf_name)
        df["model_answer_response_len"] = df["model_answer_response"].apply(lambda x: len(tokenizer.encode(x)))

        spearman_corr = df["model_answer_response_len"].corr(df["model_confidence_extracted"], method="spearman")
        print(f"Spearman correlation coefficient: {spearman_corr:.4f}")

        df["setting"] = f"(spearman={spearman_corr:.2f}) {template}--{model}"

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        x=df["model_answer_response_len"], y=df["model_confidence_extracted"], hue=df["setting"], alpha=0.25
    )
    plt.xscale("log")
    plt.ylim(0 - 0.05, 1 + 0.05)
    plt.xlabel("log(Response Length)")
    plt.ylabel("Confidence")
    plt.title(f"Response Length vs. Confidence\n{dataset}--{method}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
