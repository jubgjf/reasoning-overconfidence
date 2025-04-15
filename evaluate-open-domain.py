import asyncio
import seaborn as sns
import matplotlib.pyplot as plt
import re
from result import Ok, Err, Result
from tortoise import run_async
from tqdm.auto import tqdm

from confidence.utils import split_thinking_answer, limit_concurrency
from pydantic import BaseModel
import pandas as pd
from confidence.template import GSM8KTemplate, ARCTemplate, LogiQATemplate, TimeTablingTemplate, GAOKAOTemplate
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName, Model


async def evaluate(judge_model: Model, df: pd.DataFrame) -> list[tuple[int, int, float]]:
    async def evaluate_once(question: str, answer: str, confidence: float) -> Result[tuple[int, int, float], str]:
        judge_prompt = (
            "You are now a large language model referee. "
            "I will provide you with a question and the answer from another model, "
            "and you need to determine whether the model's answer is correct.\n\n"
            "Question:\n"
            f"{question}\n"
            "Answer:\n"
            f"{answer}\n\n"
            "Please tell me how many solutions the model obtained for this question. "
            "According to the constraints in the question, is each solution of the model correct? "
            "Please verify each solution using the constraints one by one, and output [[x/y]], "
            "where x is the number of correct solutions output by the model "
            "and y is the number of solutions output by the model. "
            "Besides, x or y can be 0 if no solution is output by the model."
        )
        judge_response_result = await judge_model.request(messages=[{"role": "user", "content": judge_prompt}])
        if judge_response_result.is_err():
            return judge_response_result

        _, judge_response = split_thinking_answer(judge_response_result.ok_value.message_content)
        matches = [(int(x), int(y)) for x, y in re.findall(r"\[\[(\d+)/(\d+)]]", judge_response)]
        if len(matches) < 1:
            return Err(f"No solution count found in judge model response: {judge_response}")
        correct_count, total_count = matches[0][0], matches[0][1]
        if correct_count > total_count:
            return Err(
                f"Correct count {correct_count} cannot be greater than total count {total_count}\n{judge_response}"
            )

        return Ok((correct_count, total_count, confidence))

    tasks = [
        evaluate_once(question=q, answer=a, confidence=c)
        for q, a, c in zip(df["question"], df["model_answer_response"], df["model_confidence_extracted"])
    ]
    tasks = limit_concurrency(tasks, 100)
    results = []
    async for result in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        result: Result[tuple[int, int, float], str] = await result
        if result.is_err():
            print(result.err_value)
            continue
        results.append(result.ok_value)
    return results


class Setting(BaseModel):
    model: ModelName
    template: GSM8KTemplate | ARCTemplate | LogiQATemplate | GAOKAOTemplate | TimeTablingTemplate


async def main():
    dataset = DatasetName.TimeTabling
    method = MethodName.Verbal_0_100
    settings = [
        Setting(model=ModelName.QWEN2_5_7B, template=TimeTablingTemplate.simple),
        Setting(model=ModelName.QWEN2_5_7B, template=TimeTablingTemplate.cot),
        Setting(model=ModelName.QWQ_32B, template=TimeTablingTemplate.simple),
    ]

    all_results = []

    for setting in settings:
        model = setting.model
        template = setting.template

        record_cls = dataset.record_cls
        db_logger = Logger(
            db_name="debug",
            table_name=f"{dataset}--{method}--no-cot-memory-False--{template}--{model}",
            record_cls=record_cls,
        )
        async with db_logger:
            records = await db_logger.fetch()

        method_records = [record.model_dump() for record in records]
        df = pd.DataFrame(method_records)

        if method == MethodName.Verbal_0_100:
            df["model_confidence_extracted"] = df["model_confidence_extracted"].apply(lambda x: x / 100)

        solution_counts = await evaluate(judge_model=Model(ModelName.QWEN2_5_7B), df=df)
        for c in solution_counts:
            print(c)  # (correct_count, total_count, confidence)
            correct_count, total_count, confidence = c
            all_results.append(
                {
                    "model_name": model,
                    "template": template,
                    "model_name--template": f"{model}--{template}",
                    "correct_count": correct_count,
                    "total_count": total_count,
                    "confidence": confidence,
                }
            )

    df = pd.DataFrame(all_results)

    sns.lineplot(data=df, x="confidence", y="total_count", hue="model_name--template", estimator="mean")
    plt.title("Relationship between Model Confidence and Number of Solutions")
    plt.xlabel("Model Confidence")
    plt.ylabel("Number of Solutions")
    plt.show()
    sns.lineplot(data=df, x="confidence", y="correct_count", hue="model_name--template", estimator="mean")
    plt.title("Relationship between Model Confidence and Number of Correct Solutions")
    plt.xlabel("Model Confidence")
    plt.ylabel("Number of Correct Solutions")
    plt.show()


if __name__ == "__main__":
    run_async(main())
