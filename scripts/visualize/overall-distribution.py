import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tortoise import run_async

from confidence.template import TimeTablingTemplate
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName


async def main():
    model = ModelName.QWQ_32B
    dataset = DatasetName.TimeTabling
    template = TimeTablingTemplate.simple
    no_cot_memory = False

    methods = [MethodName.Verbal_0_100]
    # methods = [MethodName.Verbal_0_100, MethodName.LogProb, MethodName.P_True]

    records_list = []
    for method in methods:
        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name=dataset.value,
            table_name=f"{dataset}--{method}--no-cot-memory-{no_cot_memory}--{template}--{model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)

        records_list.append(df)

    df = pd.concat(records_list)

    bins = [i / 10 for i in range(11)]

    plt.figure(figsize=(10, 6))
    sns.histplot(df, x="model_confidence_extracted", hue="method", bins=bins, multiple="dodge", alpha=0.25)
    plt.xlim(0 - 0.05, 1 + 0.05)
    plt.xlabel("Confidence")
    plt.ylabel("Frequency")
    plt.title(f"Confidence Distribution Comparison\n{dataset}--{template}--{model}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
