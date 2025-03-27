import asyncio
import git
import re
from collections.abc import Coroutine, Sequence

from openai.types.chat import ChatCompletionTokenLogprob
from result import Err, Ok, Result


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


def gsm8k_postprocess(text: str) -> Result[tuple[str, int, int], str]:
    if "</think>" in text:
        thinking_content, answer_content = text.split("</think>")
        thinking_content += "</think>"
    else:
        thinking_content, answer_content = "", text

    # pattern is from https://github.com/open-compass/opencompass/blob/854c6bf025ed53e332ae58a7ee66807eae48618d/opencompass/datasets/gsm8k.py#L44
    pattern = r"-?\d+\.\d+|-?\d+"
    numbers = re.finditer(pattern, answer_content)
    numbers = [n for n in numbers]
    if len(numbers) == 0:
        return Err(f"No number found in response: {answer_content}")
    extracted_answer = numbers[-1].group()
    answer_num_index = numbers[-1].start() + answer_content[numbers[-1].start() : numbers[-1].end()].index(
        extracted_answer
    )
    answer_num_len = len(extracted_answer)
    assert (
        text[len(thinking_content) + answer_num_index : len(thinking_content) + answer_num_index + answer_num_len]
        == extracted_answer
    )
    return Ok((extracted_answer, len(thinking_content) + answer_num_index, answer_num_len))


def first_option_postprocess(
    text: str, options: str = "ABCD", cushion: bool = True
) -> Result[tuple[str, int, int], str]:
    thinking_content, answer_content = split_thinking_answer(text)

    # https://github.com/open-compass/opencompass/blob/854c6bf025ed53e332ae58a7ee66807eae48618d/opencompass/utils/text_postprocessors.py#L73
    patterns = [
        rf"答案是?\s*([{options}])",
        rf"答案是?\s*：\s*([{options}])",
        rf"答案是?\s*:\s*([{options}])",
        rf"答案选项应?该?是\s*([{options}])",
        rf"答案选项应?该?为\s*([{options}])",
        rf"答案应该?是\s*([{options}])",
        rf"答案应该?选\s*([{options}])",
        rf"答案选项为?\s*：\s*([{options}])",
        rf"答案选项为?\s+\(?\*?\*?([{options}])\*?\*?\)?",
        rf"答案选项是?\s*:\s*([{options}])",
        rf"答案为\s*([{options}])",
        rf"答案选\s*([{options}])",
        rf"选择?\s*([{options}])",
        rf"故选?\s*([{options}])只有选?项?\s?([{options}])\s?是?对",
        rf"只有选?项?\s?([{options}])\s?是?错",
        rf"只有选?项?\s?([{options}])\s?不?正确",
        rf"只有选?项?\s?([{options}])\s?错误",
        rf"说法不?对选?项?的?是\s?([{options}])",
        rf"说法不?正确选?项?的?是\s?([{options}])",
        rf"说法错误选?项?的?是\s?([{options}])",
        rf"([{options}])\s?是正确的",
        rf"([{options}])\s?是正确答案",
        rf"选项\s?([{options}])\s?正确",
        rf"所以答\s?([{options}])",
        rf"所以\s?([{options}][.。$]?$)",
        rf"所有\s?([{options}][.。$]?$)",
        rf"[\s，：:,]([{options}])[。，,\.]?$",
        rf"[\s，,：:][故即]([{options}])[。\.]?$",
        rf"[\s，,：:]因此([{options}])[。\.]?$",
        rf"[是为。]\s?([{options}])[。\.]?$",
        rf"因此\s?([{options}])[。\.]?$",
        rf"显然\s?([{options}])[。\.]?$",
        r"答案是\s?(\S+)(?:。|$)",
        r"答案应该是\s?(\S+)(?:。|$)",
        r"答案为\s?(\S+)(?:。|$)",
        rf"(?i)ANSWER\s*:\s*([{options}])",
        rf"(?i)ANSWER\*?\*?:?\s+\*?\*?([{options}])",
        rf"[Tt]he answer is:?\s+\(?([{options}])\)?",
        rf"[Tt]he answer is:?\s+\(?\*?\*?([{options}])\*?\*?\)?",
        rf"[Tt]he answer is option:?\s+\(?([{options}])\)?",
        rf"[Tt]he correct answer is:?\s+\(?([{options}])\)?",
        rf"[Tt]he correct answer is:?\*?\*?([{options}])",
        rf"[Tt]he correct answer is option:?\s+\(?([{options}])\)?",
        rf"[Tt]he correct answer is:?.*?boxed{{([{options}])}}",
        rf"[Tt]he correct option is:?.*?boxed{{([{options}])}}",
        rf"[Tt]he correct answer option is:?.*?boxed{{([{options}])}}",
        rf"[Tt]he answer to the question is:?\s+\(?([{options}])\)?",
        rf"^选项\s?([{options}])",
        rf"^([{options}])\s?选?项",
        rf"(\s|^)[{options}][\s。，,：:\.$]",
        r"1.\s?(.*?)$",
        rf"1.\s?([{options}])[.。$]?$",
    ]
    cushion_patterns = [
        rf"\*\*([{options}])[\s\n\.:]",
        rf"([{options}])[\s\n\.:]",
        rf"([{options}]):",
        rf"([{options}])",
    ]

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        match = re.search(pattern, answer_content, re.DOTALL)
        if match:
            if match.group(1) is not None and match.group(1) != "":
                outputs = match.group(1)
            else:
                outputs = match.group(0)
            for i in options:
                if i in outputs:
                    answer_char_index = match.start() + answer_content[match.start() : match.end()].index(i)
                    assert text[len(thinking_content) + answer_char_index] == i
                    return Ok((i, len(thinking_content) + answer_char_index, 1))
    return Err(f"No option found in response: {answer_content}")


def gaokao_postprocess(text: str, options: str = "ABCD") -> Result[tuple[str, int, int], str]:
    thinking_content, answer_content = split_thinking_answer(text)

    match = re.search(r"【答案】\s?(.*)\s?<eoa>", answer_content, re.DOTALL)
    if match:
        if match.group(1) is not None and match.group(1) != "":
            outputs = match.group(1)
        else:
            outputs = match.group(0)
        choices, first_answer_char_index = [], 99999
        for i in options:
            if i in outputs:
                answer_char_index = match.start() + answer_content[match.start() : match.end()].index(i)
                assert text[len(thinking_content) + answer_char_index] == i
                choices.append(i)
                first_answer_char_index = (
                    answer_char_index if first_answer_char_index > answer_char_index else first_answer_char_index
                )
        return Ok((" ".join(choices), len(thinking_content) + first_answer_char_index, len(choices)))
    return Err(f"No option found in response: {answer_content}")


def split_thinking_answer(text: str) -> tuple[str, str]:
    if "</think>" in text:
        thinking_content, answer_content = text.split("</think>")
        thinking_content += "</think>"
    else:
        thinking_content, answer_content = "", text

    return thinking_content, answer_content


def split_thinking_answer_logprobs(
    logprobs: list[ChatCompletionTokenLogprob] | None,
) -> tuple[list[ChatCompletionTokenLogprob] | None, list[ChatCompletionTokenLogprob] | None]:
    if logprobs is None:
        return None, None

    logprobs = [t for t in logprobs if t.token != "<|im_end|>"]

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
