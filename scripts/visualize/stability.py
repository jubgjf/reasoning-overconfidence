from typing import Literal

import pandas as pd
from pydantic import BaseModel
from tortoise import run_async

from confidence.data import Template
from confidence.dataset import DatasetName
from confidence.evaluate import extract_all_solutions, is_solution_changed
from confidence.logger import Logger
from confidence.model import ModelName


class Setting(BaseModel):
    model: ModelName
    template: Template
    mode: Literal["--more", ""]


async def main():
    # dataset = DatasetName.SubsetSum
    dataset = DatasetName.TimeTabling
    turn = 0
    temperature = 0.2

    settings = [
        Setting(model=ModelName.QWEN3_8B_THINK, template="simple", mode=""),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot", mode=""),
        Setting(model=ModelName.QWEN3_8B_NO_THINK, template="cot", mode="--more"),
        # Setting(model=ModelName.DEEPSEEK_R1, template="simple", mode=""),
        # Setting(model=ModelName.DEEPSEEK_V3, template="cot", mode=""),
        # Setting(model=ModelName.O4_MINI, template="simple",mode=""),
        # Setting(model=ModelName.GPT_4O_MINI, template="cot",mode=""),
    ]

    for setting in settings:
        record_cls = dataset.record_cls
        title = f"{dataset}--{setting.template}--{setting.model}--{temperature}--{turn}{setting.mode}".replace("/", "_")
        db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls)
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)
        if setting.template == "cot":
            if setting.mode == "":
                df["setting"] = "Short-CoT"
            else:
                df["setting"] = "Short-CoT-Explore"
        elif setting.template == "simple":
            df["setting"] = "Long-CoT"
        else:
            raise ValueError(f"Unknown setting: {setting.model}--{setting.template}")

        chat_history = df["chat_history"].values.tolist()
        answer_counts = df["answer_count"].values.tolist()

        # 初始化统计变量
        total_samples = len(chat_history)
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
            if "answers" in df.columns and i < len(df):
                ground_truth_answers = str(df.iloc[i]["answers"]["0"])
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

        # 计算平均指标
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

        # 计算中间指标的平均值
        avg_first_round_correct = (
            sum(first_round_correct_counts) / len(first_round_correct_counts) if first_round_correct_counts else 0.0
        )
        avg_second_round_correct = (
            sum(second_round_correct_counts) / len(second_round_correct_counts) if second_round_correct_counts else 0.0
        )
        avg_intersection_correct = (
            sum(intersection_correct_counts) / len(intersection_correct_counts) if intersection_correct_counts else 0.0
        )
        avg_first_round_error = (
            sum(first_round_error_counts) / len(first_round_error_counts) if first_round_error_counts else 0.0
        )
        avg_second_round_error = (
            sum(second_round_error_counts) / len(second_round_error_counts) if second_round_error_counts else 0.0
        )
        avg_intersection_error = (
            sum(intersection_error_counts) / len(intersection_error_counts) if intersection_error_counts else 0.0
        )

        # 将answer_count转换为int类型求平均值
        avg_answer_count = sum(int(ac) for ac in answer_counts) / len(answer_counts) if answer_counts else 0.0

        # 打印结果
        # print(f"\n=== {df['setting'].iloc[0]} ===")
        # print(f"样本数量: {total_samples}")
        # print(f"解空间总大小（平均）: {avg_answer_count:.2f}")
        # print("\n--- 中间指标 ---")
        # print(f"第一轮正确解数量（平均）: {avg_first_round_correct:.2f}")
        # print(f"第二轮正确解数量（平均）: {avg_second_round_correct:.2f}")
        # print(f"两轮正确解交集大小（平均）: {avg_intersection_correct:.2f}")
        # print(f"第一轮错误解数量（平均）: {avg_first_round_error:.2f}")
        # print(f"第二轮错误解数量（平均）: {avg_second_round_error:.2f}")
        # print(f"两轮错误解交集大小（平均）: {avg_intersection_error:.2f}")
        # print("\n--- 主要指标 ---")
        # print(f"正解保留率: {avg_correct_preservation:.4f}")
        # print(f"错误修正率: {avg_error_correction:.4f}")
        # print(f"新解发现率: {avg_new_solution_discovery:.4f}")
        print(
            f"{avg_correct_preservation * 100:.2f} {avg_error_correction * 100:.2f} {avg_new_solution_discovery * 100:.2f}"
        )


if __name__ == "__main__":
    run_async(main())
