import asyncio

from loguru import logger

import re
from result import Ok, Err, Result
from tap import Tap
from tortoise import run_async
from tqdm.auto import tqdm

from confidence.utils import split_thinking_answer, limit_concurrency
from confidence.template import TimeTablingTemplate, Template, string_to_template
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.method import MethodName
from confidence.model import ModelName, Model


class Argument(Tap):
    model: ModelName = ModelName.QWQ_32B
    judge_model: ModelName = ModelName.QWQ_32B
    dataset: DatasetName = DatasetName.TimeTabling
    template: Template = TimeTablingTemplate.simple
    method: MethodName = MethodName.Verbal_0_100
    no_cot_memory: bool = False
    concurrency: int = 200
    debug: bool = False

    def configure(self) -> None:
        self.add_argument("--template", type=string_to_template)


async def evaluate(judge_model: Model, record: dict) -> Result[tuple[int, int, dict], str]:
    judge_prompt = (
        "You are now a large language model referee. "
        "I will provide you with a question and the answer from another model, "
        "and you need to determine whether the model's answer is correct.\n\n"
        "Question:\n"
        f"{record['question']}\n"
        "Answer:\n"
        f"{record['model_answer_extracted']}\n\n"
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
    matches = [(int(x), int(y)) for x, y in re.findall(r"[{\[]\[(\d+)/(\d+)][]}]", judge_response)]
    if len(matches) < 1:
        return Err(f"No solution count found in judge model response: {judge_response}")
    correct_count, total_count = matches[0][0], matches[0][1]
    if correct_count > total_count:
        return Err(f"Correct count {correct_count} cannot be greater than total count {total_count}\n{judge_response}")

    return Ok((correct_count, total_count, record))


async def main(args: Argument):
    judge_model = Model(args.judge_model)
    record_cls = args.dataset.record_cls
    db_logger = Logger(
        db_name="debug",
        table_name=f"{args.dataset}--{args.method}--no-cot-memory-{args.no_cot_memory}--{args.template}--{args.model}",
        record_cls=record_cls,
    )
    async with db_logger:
        records = await db_logger.fetch()
        records = [record.model_dump() for record in records]

    tasks = [evaluate(judge_model=judge_model, record=record) for record in records]
    tasks = limit_concurrency(tasks, args.concurrency)

    db_logger = Logger(
        db_name="debug",
        table_name=f"{args.dataset}--{args.method}--no-cot-memory-{args.no_cot_memory}--{args.template}--{args.model}--evaluate-by-{args.judge_model}",
        record_cls=record_cls,
    )
    async with db_logger:
        async for zip_response_result in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Evaluating"):
            zip_response_result: Result[tuple[int, int, dict], str] = await zip_response_result
            if zip_response_result.is_err():
                logger.error(zip_response_result.err_value)
                continue

            correct_count: int
            total_count: int
            record: dict
            correct_count, total_count, record = zip_response_result.ok_value
            await db_logger.insert(record_cls(**{**record, "eval_result": f"{correct_count}/{total_count}"}))


if __name__ == "__main__":
    args = Argument().parse_args()

    run_async(main(args))
