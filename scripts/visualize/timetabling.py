import json
import pandas as pd
import matplotlib.pyplot as plt
from confidence.data import TimeTablingData

if __name__ == "__main__":
    with open("./dataset/timetabling.jsonl") as f:
        dataset = [TimeTablingData.model_validate(json.loads(line)) for line in f]

    # 将数据集转换为 Pandas DataFrame
    df = pd.DataFrame([data.model_dump() for data in dataset])

    # 定义新的分桶边界和标签
    bins = [1, 51, 101, 151, 201, 251, 301, 351, 401, 451, 501]
    labels = ["1-50", "51-100", "101-150", "151-200", "201-250", "251-300", "301-350", "351-400", "401-450", "451-500"]

    # 使用 pandas 的 cut 函数将 answer_count 分到不同的桶中
    df["difficulty"] = pd.cut(df["answer_count"], bins=bins, labels=labels, right=False)

    # 统计每个难度桶中的样本数量
    difficulty_counts = df["difficulty"].value_counts().sort_index()

    # 绘制分布图
    plt.figure(figsize=(12, 6))  # 稍微调整图形大小以适应更多标签
    difficulty_counts.plot(kind="bar")
    plt.title("Dataset Distribution by Difficulty (Answer Count)")
    plt.xlabel("Difficulty (Number of Answers)")
    plt.ylabel("Number of Questions")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

    print(dataset[0].question)
    print(dataset[0].answers["0"])
