import asyncio

from loguru import logger
from result import Result
from tap import Tap
from tortoise import run_async
from tqdm.auto import tqdm

from confidence.data import Data
from confidence.dataset import DatasetName
from confidence.extractor import extract_answer_and_confidence
from confidence.logger import Logger, list_history_to_dict
from confidence.method import Method, MethodName, Response
from confidence.model import Model, ModelName
from confidence.template import ARCTemplate, GAOKAOTemplate, GSM8KTemplate, LogiQATemplate, Template, string_to_template
from confidence.utils import limit_concurrency


class Argument(Tap):
    model: ModelName = ModelName.QWQ_32B
    dataset: DatasetName = DatasetName.LogiQA
    template: GSM8KTemplate | ARCTemplate | LogiQATemplate | GAOKAOTemplate = LogiQATemplate.CoTEval
    method: MethodName = MethodName.Verbal_0_100
    max_samples: int | None = None
    force_update: bool = False
    concurrency: int = 100
    debug: bool = False

    def configure(self) -> None:
        self.add_argument("--template", type=string_to_template)


async def request(method: Method, model: Model, template: Template, data: Data) -> tuple[str, Result[Response, str]]:
    response_result = await method.request(model, data, template, temperature=0.2, max_tokens=16384)
    return data, response_result


async def main(args: Argument):
    record_cls = args.dataset.record_cls
    db_logger = Logger(
        db_name=args.dataset.value if not args.debug else "debug",
        table_name=f"{args.dataset}--{args.method}--{args.template}--{args.model}",
        record_cls=record_cls,
        force_update=args.force_update,
    )
    async with db_logger:
        # ===== model =====
        model = Model(args.model)

        # ===== method =====
        method = Method(name=args.method)

        # ===== dataset =====
        dataset_cls = args.dataset.dataset_cls
        dataset = dataset_cls().load_resume_dataset(
            await db_logger.already_processed_ids(),
            force_restart=args.force_update,
        )

        if args.max_samples is not None:
            dataset = dataset[: args.max_samples]
            if len(dataset) < args.max_samples:
                logger.warning(f"Dataset has only {len(dataset)} samples")

        if len(dataset) > 0:
            logger.info(f"Example: {dataset[0]}")

        # ===== task =====
        tasks = [request(method=method, model=model, template=args.template, data=data) for data in dataset]
        tasks = limit_concurrency(tasks, args.concurrency)

        async for zip_response in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc=f"{args.dataset}--{args.method}--{args.template}--{args.model}",
        ):
            data: Data
            response_result: Result[Response, str]
            data, response_result = await zip_response
            if response_result.is_err():
                logger.error(response_result.err_value)
                continue

            history = response_result.ok_value.messages
            assert history[-1]["role"] == "assistant"
            answer_and_confidence_result = extract_answer_and_confidence(
                method_name=args.method,
                dataset_name=args.dataset,
                answer_response=history[1]["content"],
                confidence_response=history[-1]["content"] if args.method.need_another_turn else None,
                logprobs=response_result.ok_value.logprobs,
            )
            if answer_and_confidence_result.is_err():
                logger.error(answer_and_confidence_result.err_value)
                continue

            model_answer_extracted, confidence_extracted = answer_and_confidence_result.ok_value
            record = record_cls(
                **data.model_dump(),
                model_answer_response=history[1]["content"],
                model_answer_extracted=model_answer_extracted,
                model_confidence_response=history[-1]["content"] if args.method.need_another_turn else "NONE",
                model_confidence_extracted=confidence_extracted,
                template=args.template.value,
                method=args.method.value,
                history=list_history_to_dict(history),
                model=args.model.value,
            )
            await db_logger.insert(record)


if __name__ == "__main__":
    args = Argument().parse_args()

    run_async(main(args))
