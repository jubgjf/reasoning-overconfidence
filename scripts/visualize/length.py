import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from pydantic import BaseModel
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from tortoise import run_async
from confidence.template import Template, TimeTablingTemplate
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template


async def main():
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    no_cot_memory = False
    settings = [
        Setting(model=ModelName.QWQ_32B, template=TimeTablingTemplate.simple),
    ]

    records_list = []

    for setting in settings:
        model = setting.model
        template = setting.template

        method_records = []

        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()
        method_records.extend([record.model_dump() for record in records])
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}--fake-reflection",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()
        method_records.extend([record.model_dump() for record in records])

        df = pd.DataFrame(method_records)

        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)

        tokenizer = AutoTokenizer.from_pretrained(model.hf_name)
        df["model_output"] = df["model_thinking_response"] + df["model_answer_response"]
        df["model_output_len"] = df["model_output"].apply(lambda x: len(tokenizer.encode(x)))

        spearman_corr = df["model_output_len"].corr(df["model_confidence_extracted"], method="spearman")
        print(f"Spearman correlation coefficient: {spearman_corr:.4f}")

        df["setting"] = f"(spearman={spearman_corr:.2f}) {template}--{model}"

        records_list.append(df)

    df = pd.concat(records_list, ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=df["model_output_len"], y=df["model_confidence_extracted"], hue=df["setting"], alpha=0.25)
    plt.xscale("log")
    plt.ylim(0 - 0.05, 1 + 0.05)
    plt.xlabel("log(Response Length)")
    plt.ylabel("Confidence")
    plt.title(f"Response Length vs. Confidence\n{dataset}--{method}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
