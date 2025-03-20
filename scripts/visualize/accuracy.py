import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tortoise import run_async

from confidence.data import GSM8KTemplate
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName


async def main():
    model = ModelName.QWQ_32B
    dataset = DatasetName.GSM8K
    template = GSM8KTemplate.BigGSM
    method = MethodName.Verbal_0_100

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
    if dataset == DatasetName.GSM8K:
        df["model_answer_extracted"] = df["model_answer_extracted"].apply(float)

    if dataset == DatasetName.GSM8K:
        df["is_correct"] = df["model_answer_extracted"] == df["answer_num"]
    elif dataset in [DatasetName.ARC, DatasetName.LogiQA]:
        df["is_correct"] = df["model_answer_extracted"] == df["answer_key"]
    acc = df["is_correct"].sum() / len(df["is_correct"])
    df["confidence_bin"] = pd.cut(
        df["model_confidence_extracted"],
        bins=np.linspace(0, 1, 11),
        labels=np.arange(10),
        include_lowest=True,
        right=True,
    )
    cm = confusion_matrix(df["is_correct"], df["confidence_bin"], labels=np.arange(10))
    cm = cm[:2]  # only keep the first two rows: correct/incorrect

    n_rows, n_cols = cm.shape
    fig, ax = plt.subplots(figsize=(n_cols, n_rows * 1.6))

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
    plt.xlabel("Confidence Bin")
    plt.ylabel("Correctness")
    plt.title(f"Confidence vs. Correctness\n{dataset}--{method}--{template}--{model}\nAcc = {acc:.2f}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run_async(main())
