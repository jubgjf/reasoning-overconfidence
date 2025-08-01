import os

import pandas as pd

from confidence.dataset import DatasetName
from confidence.evaluate import extract_all_solutions, is_solution_changed
from confidence.model import ModelName


def load_self_consistency_data(data_path: str) -> pd.DataFrame:
    """
    加载self-consistency数据

    Args:
        data_path: 数据文件路径

    Returns:
        处理后的DataFrame
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_pickle(data_path)
    print(f"Loaded data from: {data_path}")
    print(f"Total samples: {len(df)}")
    print(f"Methods: {df['method'].unique()}")

    return df


def main():
    # 配置参数
    dataset = DatasetName.TimeTabling
    model = ModelName.QWEN3_8B_THINK
    template = "simple"
    temperature = 0.2

    # 构建数据文件路径
    data_filename = f"{dataset}_{model.series_name.lower()}_{template}_temp{temperature}.pkl"
    data_path = os.path.join("tmp/self_consistency", data_filename)

    # 加载数据
    try:
        df = load_self_consistency_data(data_path)
    except FileNotFoundError:
        print(f"Data file not found: {data_path}")
        print("Please run generate_self_consistency_data.py first to generate the data.")
        return

    # 获取所有方法
    methods = df["method"].unique()
    print(f"Processing methods: {methods}")

    # 为每个方法计算指标
    method_results = {}

    for method in methods:
        print(f"\nProcessing method: {method}")
        method_df = df[df["method"] == method]

        # 跳过majority voting方法，因为它没有真实的多轮对话历史
        if method == "Self-Consistency (majority voting)":
            print(f"Skipping {method} - no multi-round chat history available")
            continue

        chat_history = method_df["chat_history"].values.tolist()
        answer_counts = method_df["answer_count"].values.tolist()

        # 初始化统计变量
        correct_preservation_rates = []
        error_correction_rates = []
        new_solution_discovery_rates = []

        # 中间指标统计
        first_round_correct_counts = []
        second_round_correct_counts = []
        intersection_correct_counts = []
        first_round_error_counts = []
        second_round_error_counts = []
        intersection_error_counts = []

        for i, (history, answer_count) in enumerate(zip(chat_history, answer_counts)):
            # 提取第一轮和第二轮的模型输出
            first_round_output = history[1]["content"]  # 第一轮回答
            second_round_output = history[5]["content"]  # 第二轮回答

            # 获取真实答案（如果有的话）
            if "answers" in method_df.columns and i < len(method_df):
                ground_truth_answers = str(method_df.iloc[i]["answers"]["0"])
                ground_truth_solutions = extract_all_solutions(ground_truth_answers)
            else:
                # 如果没有真实答案，假设所有的解都是正确的（这种情况下无法真正评估）
                ground_truth_solutions = set()

            # 提取第一轮和第二轮的解
            first_round_solutions = extract_all_solutions(first_round_output)
            second_round_change_result = is_solution_changed(second_round_output)
            if second_round_change_result.is_ok():
                if second_round_change_result.ok_value:
                    second_round_solutions = extract_all_solutions(second_round_output)
                else:
                    second_round_solutions = first_round_solutions
            else:
                # 抽取失败，假设第二轮解与第一轮相同
                second_round_solutions = first_round_solutions

            if ground_truth_solutions:
                # 计算正确解和错误解
                first_round_correct = first_round_solutions.intersection(ground_truth_solutions)
                first_round_error = first_round_solutions - ground_truth_solutions

                second_round_correct = second_round_solutions.intersection(ground_truth_solutions)
                second_round_error = second_round_solutions - ground_truth_solutions

                # 两轮正确解的交集
                intersection_correct = first_round_correct.intersection(second_round_correct)
                # 两轮错误解的交集
                intersection_error = first_round_error.intersection(second_round_error)

                # 记录中间指标
                first_round_correct_counts.append(len(first_round_correct))
                second_round_correct_counts.append(len(second_round_correct))
                intersection_correct_counts.append(len(intersection_correct))
                first_round_error_counts.append(len(first_round_error))
                second_round_error_counts.append(len(second_round_error))
                intersection_error_counts.append(len(intersection_error))

                # 计算三个指标
                # 1. 正解保留率 = 两轮正确解交集的大小 / 第一轮正确解的数量（若第一轮全错则直接为0）
                if len(first_round_correct) > 0:
                    correct_preservation_rate = len(intersection_correct) / len(first_round_correct)
                else:
                    correct_preservation_rate = 0.0
                correct_preservation_rates.append(correct_preservation_rate)

                # 2. 错误修正率 = 1 - 两轮错误解交集大小 / 第一轮错误解的数量（若第一轮全对则为1）
                if len(first_round_error) > 0:
                    error_correction_rate = 1 - (len(intersection_error) / len(first_round_error))
                else:
                    error_correction_rate = 1.0
                error_correction_rates.append(error_correction_rate)

                # 3. 新解发现率 = （第二轮正确解的数量 - 两轮正确解交集的数量） / 解空间总大小
                new_correct_solutions = len(second_round_correct) - len(intersection_correct)
                new_solution_discovery_rate = (
                    new_correct_solutions / int(answer_count) if int(answer_count) > 0 else 0.0
                )
                new_solution_discovery_rates.append(new_solution_discovery_rate)
            else:
                # 如果没有真实答案，设置为默认值
                first_round_correct_counts.append(0)
                second_round_correct_counts.append(0)
                intersection_correct_counts.append(0)
                first_round_error_counts.append(0)
                second_round_error_counts.append(0)
                intersection_error_counts.append(0)
                correct_preservation_rates.append(0.0)
                error_correction_rates.append(0.0)
                new_solution_discovery_rates.append(0.0)

        # 计算该方法的平均指标
        avg_correct_preservation = (
            sum(correct_preservation_rates) / len(correct_preservation_rates) if correct_preservation_rates else 0.0
        )
        avg_error_correction = (
            sum(error_correction_rates) / len(error_correction_rates) if error_correction_rates else 0.0
        )
        avg_new_solution_discovery = (
            sum(new_solution_discovery_rates) / len(new_solution_discovery_rates)
            if new_solution_discovery_rates
            else 0.0
        )

        # 保存该方法的结果
        method_results[method] = {
            "correct_preservation": avg_correct_preservation,
            "error_correction": avg_error_correction,
            "new_solution_discovery": avg_new_solution_discovery,
        }

        print(
            f"{method}: {avg_correct_preservation * 100:.2f} {avg_error_correction * 100:.2f} {avg_new_solution_discovery * 100:.2f}"
        )

    # 输出所有方法的结果汇总
    print("\n=== Results Summary ===")
    print("Method\t\t\t\t\tCorrect Preservation\tError Correction\tNew Solution Discovery")
    print("-" * 100)
    for method, results in method_results.items():
        print(
            f"{method:<40}\t{results['correct_preservation'] * 100:.2f}\t\t{results['error_correction'] * 100:.2f}\t\t{results['new_solution_discovery'] * 100:.2f}"
        )


if __name__ == "__main__":
    main()
