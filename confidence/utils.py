import asyncio
from typing import assert_never

from openai.types.completion_choice import Logprobs
import git
from collections.abc import Coroutine, Sequence

from openai.types.chat import ChatCompletionTokenLogprob
from loguru import logger


def limit_concurrency(coroutines: Sequence[Coroutine], concurrency: int) -> list[Coroutine]:
    semaphore = asyncio.Semaphore(concurrency)

    async def with_concurrency_limit(coroutine: Coroutine) -> Coroutine:
        async with semaphore:
            return await coroutine

    return [with_concurrency_limit(coroutine) for coroutine in coroutines]


def last_git_hash() -> str:
    repo = git.Repo(".")
    commit = repo.head.commit
    return commit.hexsha[:7]


def split_thinking_answer(text: str) -> tuple[str, str]:
    if text.count("</think>") > 1:
        logger.warning(f'text.count("</think>") = {text.count("</think>")}, text = {text}')

    if "</think>" in text:
        splits = text.split("</think>")
        thinking_content = "</think>".join(splits[:-1]) + "</think>"
        answer_content = splits[-1]
    else:
        thinking_content, answer_content = "", text

    return thinking_content, answer_content


def split_thinking_answer_logprobs(
    logprobs: list[ChatCompletionTokenLogprob] | Logprobs | None,
) -> tuple[list[ChatCompletionTokenLogprob] | Logprobs | None, list[ChatCompletionTokenLogprob] | Logprobs | None]:
    if logprobs is None:
        return None, None

    im_end_tokens = ["<|im_end|>", "<｜end▁of▁sentence｜>"]
    if isinstance(logprobs, list):
        logprobs = [t for t in logprobs if t.token not in im_end_tokens]

        if not any([t.token == "</think>" for t in logprobs]):
            return None, logprobs

        found_think_end = False
        thinking_logprobs, answer_logprobs = [], []
        for t in logprobs:
            if not found_think_end:
                thinking_logprobs.append(t)
            else:
                answer_logprobs.append(t)

            if t.token == "</think>":
                found_think_end = True

        return thinking_logprobs, answer_logprobs
    elif isinstance(logprobs, Logprobs):
        drop_index = []
        for t in im_end_tokens:
            try:
                drop_index.append(logprobs.tokens.index(t))
            except ValueError:
                pass
        logprobs = Logprobs(
            text_offset=[elem for i, elem in enumerate(logprobs.text_offset) if i not in drop_index],
            token_logprobs=[elem for i, elem in enumerate(logprobs.token_logprobs) if i not in drop_index],
            tokens=[elem for i, elem in enumerate(logprobs.tokens) if i not in drop_index],
            top_logprobs=[elem for i, elem in enumerate(logprobs.top_logprobs) if i not in drop_index],
        )

        if not any([t == "</think>" for t in logprobs.tokens]):
            return None, logprobs

        think_end_index = None
        for i, t in enumerate(logprobs.tokens):
            if t == "</think>":
                think_end_index = i
        assert think_end_index is not None

        thinking_logprobs = Logprobs(
            text_offset=[elem for i, elem in enumerate(logprobs.text_offset) if i <= think_end_index],
            token_logprobs=[elem for i, elem in enumerate(logprobs.token_logprobs) if i <= think_end_index],
            tokens=[elem for i, elem in enumerate(logprobs.tokens) if i <= think_end_index],
            top_logprobs=[elem for i, elem in enumerate(logprobs.top_logprobs) if i <= think_end_index],
        )
        answer_logprobs = Logprobs(
            text_offset=[elem for i, elem in enumerate(logprobs.text_offset) if i > think_end_index],
            token_logprobs=[elem for i, elem in enumerate(logprobs.token_logprobs) if i > think_end_index],
            tokens=[elem for i, elem in enumerate(logprobs.tokens) if i > think_end_index],
            top_logprobs=[elem for i, elem in enumerate(logprobs.top_logprobs) if i > think_end_index],
        )

        return thinking_logprobs, answer_logprobs
    else:
        assert_never(logprobs)


def flatten[T](ll: list[list[T]]) -> list[T]:
    flattened = []
    for l in ll:
        flattened.extend(l)
    return flattened
