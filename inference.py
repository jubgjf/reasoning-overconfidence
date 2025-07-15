import asyncio
import os

from loguru import logger
from sglang.utils import launch_server_cmd, terminate_process, wait_for_server
from tap import Tap
from tortoise import run_async
from tqdm.auto import tqdm

import confidence
from confidence.data import Data, Template
from confidence.dataset import DatasetName
from confidence.logger import Logger
from confidence.model import ChatResponse, Model, ModelName
from confidence.result import Result
from confidence.utils import last_git_hash, limit_concurrency


class Argument(Tap):
    model: ModelName = ModelName.QWEN3_8B_THINK
    model_name_or_path: str = "Qwen/Qwen3-8B"
    dataset: DatasetName = DatasetName.TimeTabling
    template: Template = "simple"
    temperature: float = 0.2
    max_completion_tokens: int = 20480
    force_update: bool = False
    concurrency: int = 100
    turn: int = 0


async def request(
    model: Model,
    template: Template,
    data: Data,
    temperature: float,
    max_completion_tokens: int = 20480,
) -> tuple[Data, Result[ChatResponse, str]]:
    response_result = await confidence.request(
        model, data, template, temperature=temperature, max_completion_tokens=max_completion_tokens
    )
    return data, response_result


async def main(args: Argument):
    record_cls = args.dataset.record_cls
    title = f"{args.dataset}--{args.template}--{args.model}--{args.temperature}--{args.turn}".replace("/", "_")
    db_logger = Logger(db_name=title, table_name=title, record_cls=record_cls, force_update=args.force_update)
    async with db_logger:
        # ===== model =====
        model = Model(args.model, args.model_name_or_path)

        # ===== dataset =====
        dataset_cls = args.dataset.dataset_cls
        dataset = dataset_cls().load_resume_dataset(
            await db_logger.already_processed_question_ids(),
            force_restart=args.force_update,
        )
        if len(dataset) > 0:
            logger.info(f"Example: {dataset[0]}")

        # ===== task =====
        tasks = [
            request(
                model=model,
                template=args.template,
                data=data,
                temperature=args.temperature,
                max_completion_tokens=args.max_completion_tokens,
            )
            for data in dataset
        ]
        tasks = limit_concurrency(tasks, args.concurrency)

        async for zip_response in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=title):
            data: Data
            response_result: Result[ChatResponse, str]
            data, response_result = await zip_response
            if response_result.is_err():
                logger.error(response_result.err_value)
                continue

            thinking = response_result.ok_value.thinking
            record = record_cls(
                **data.model_dump(),
                chat_history=response_result.ok_value.messages,
                thinking_history=thinking if thinking is not None else [],
                model=args.model.value,
                dataset=args.dataset.value,
                template=args.template,
                temperature=args.temperature,
                git_hash=last_git_hash(),
            )
            await db_logger.insert(record)


if __name__ == "__main__":
    args = Argument().parse_args()

    os.environ["SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN"] = "1"
    server_process, port = launch_server_cmd(
        (
            "python3 -m sglang.launch_server "
            "--tp 8 "
            "--dp 1 "
            f"--model-path {args.model_name_or_path} "
            f"--served-model-name {args.model} "
            "--reasoning-parser qwen3 "
            "--context-length 131072 "
            """--json-model-override-args {"rope_scaling":{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}} """
            "--host 0.0.0.0 "
            "--port 33333 "
            "--log-level warning "
        )
    )
    wait_for_server(f"http://localhost:{port}")

    os.environ["BASE_URL"] = f"http://localhost:{port}/v1"
    os.environ["API_KEY"] = "sglang"

    run_async(main(args))

    terminate_process(server_process)
