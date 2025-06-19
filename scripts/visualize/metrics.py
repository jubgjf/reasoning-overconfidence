import numpy as np
import pandas as pd

from confidence.dataset import DatasetName
from confidence.method import MethodName


def prf(df: pd.DataFrame, method: MethodName, dataset: DatasetName) -> pd.DataFrame:
    if method == MethodName.Verbal_0_100:
        df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)

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

    # 准确率：完成多解题任务的正确性，计算precision
    df["precision"] = df["correct_solution_count"] / df["total_solution_count"]
    # 答案多样性：正确答案数/总生成数量，衡量输出多样化答案的能力
    df["recall"] = df["correct_solution_count"] / df["answer_count"]

    return df


def ece(df: pd.DataFrame, only_over_confidence: bool = False) -> float:
    bins = np.linspace(0, 1, 11)
    df["bin"] = pd.cut(df["model_confidence_extracted"], bins=bins, include_lowest=True, labels=False)
    ece = 0
    N = len(df)
    for b in range(10):
        bin_data = df[df["bin"] == b]
        if len(bin_data) == 0:
            continue
        acc = (bin_data["recall"] == 1).mean()  # 或用 precision
        conf = bin_data["model_confidence_extracted"].mean()
        if only_over_confidence:
            ece += len(bin_data) / N * max(conf - acc, 0)
        else:
            ece += len(bin_data) / N * abs(acc - conf)
    return ece


def route_dependence(df: pd.DataFrame):
    # 路径依赖性：同一问题多次回答，计算推理步骤的相似度，衡量模型推理过程的固执程度
    ...


def stability(df: pd.DataFrame):
    # 稳定性：同一问题多次回答，模型在多论反思重新考虑时，是否经常改变答案，如果从不改变并且给出的答案是错的，说明过度自信
    ...


def show_metrics(df: pd.DataFrame, setting_name: str):
    print(setting_name)
    metrics = {
        "Precision": df["precision"].mean(),
        "Recall": df["recall"].mean(),
        "ECE(r)": ece(df),
        "EOE(r)": ece(df, only_over_confidence=True),
    }
    print(",".join(metrics.keys()))
    print(",".join(f"{value * 100:.2f}" for value in metrics.values()))
    print("==========================================")
