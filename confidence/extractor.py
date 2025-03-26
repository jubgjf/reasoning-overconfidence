import itertools
import math
import re
from functools import partial
from typing import Callable

from loguru import logger
from openai.types.chat import ChatCompletionTokenLogprob
from result import Err, Ok, Result

from .dataset import DatasetName
from .method import MethodName
from .utils import first_option_postprocess, gaokao_postprocess, gsm8k_postprocess


def extract_answer_and_verbal_confidence(
    answer_response: str,
    confidence_response: str | None,
    postprocessor: Callable[[str], Result[tuple[str, int, int], str]],
) -> Result[tuple[str, float], str]:
    extracted_result = postprocessor(answer_response)
    if extracted_result.is_err():
        return extracted_result
    extracted_answer, _, _ = extracted_result.ok_value

    if "</think>" in confidence_response:
        _, confidence_response_no_thinking = confidence_response.split("</think>")
    else:
        confidence_response_no_thinking = confidence_response
    confidence_score_matches = re.findall(r"\[\[(100|[1-9]?[0-9])]]", confidence_response_no_thinking)
    if len(confidence_score_matches) < 1:
        return Err(f"No confidence score found in confidence response: {confidence_response_no_thinking}")
    return Ok((extracted_answer, float(confidence_score_matches[0])))


def extract_answer_and_logprob_confidence(
    answer_response: str,
    postprocessor: Callable[[str], Result[tuple[str, int, int], str]],
    logprobs: list[ChatCompletionTokenLogprob] | None,
) -> Result[tuple[str, float], str]:
    answer_response_from_tokens = "".join([t.token for t in logprobs]).rstrip("<|im_end|>")
    if answer_response.strip() != answer_response_from_tokens.strip():
        # Sometimes Unicode chars are divided into multiple tokens, so len(tokens) may not match len(answer)
        logger.warning("Answer token mismatch:")
        logger.warning(f"answer = {answer_response.replace('\n', '\\n')}")
        logger.warning(f"tokens = {answer_response_from_tokens.replace('\n', '\\n')}")
        if "�" in answer_response_from_tokens:
            logger.warning("Broken Unicode character (�) found, might be normal")
        else:
            logger.warning("Broken Unicode character (�) not found, might be abnormal")
    extracted_result = postprocessor(answer_response_from_tokens)
    if extracted_result.is_err():
        return extracted_result
    extracted_answer, answer_char_index, answer_char_len = extracted_result.ok_value
    token_index, char_index = 0, 0
    for tok in logprobs:
        token_index += 1
        char_index += len(tok.token)
        if char_index >= answer_char_index:
            break
    else:
        return Err(f"Answer token not found in {answer_response_from_tokens}")
    if char_index == answer_char_index:
        # Token is "A" or "B" etc.
        answer_tokens = logprobs[token_index : token_index + answer_char_len]
    else:
        # Token is " A" or " B" etc.
        answer_tokens = logprobs[token_index - 1 : token_index - 1 + answer_char_len]
    if "".join([t.token.strip() for t in answer_tokens]) != extracted_answer:
        return Err(f"Answer token mismatch: {''.join([t.token.strip() for t in answer_tokens])} != {extracted_answer}")
    top_logprobs = [t.top_logprobs for t in answer_tokens]
    all_number_pairs = list(itertools.product(*top_logprobs))
    logprobs_unnormalized_unfiltered = {
        "".join([p.token.strip() for p in pair]): sum([p.logprob for p in pair]) for pair in all_number_pairs
    }
    logprobs_unnormalized = dict(
        filter(
            lambda item: item[0].lstrip("-").replace(".", "", 1).isdigit(),
            logprobs_unnormalized_unfiltered.items(),
        )
    )
    logprobs_unnormalized = {k: math.exp(v) for k, v in logprobs_unnormalized.items()}
    logprobs_normalized = {k: v / sum(logprobs_unnormalized.values()) for k, v in logprobs_unnormalized.items()}
    return Ok((extracted_answer, logprobs_normalized[extracted_answer]))


def extract_answer_and_p_true_confidence(
    answer_response: str,
    confidence_response: str | None,
    postprocessor: Callable[[str], Result[tuple[str, int, int], str]],
    logprobs: list[ChatCompletionTokenLogprob] | None,
) -> Result[tuple[str, float], str]:
    extracted_result = postprocessor(answer_response)
    if extracted_result.is_err():
        return extracted_result
    extracted_answer, _, _ = extracted_result.ok_value

    if "</think>" in confidence_response:
        _, confidence_response_no_thinking = confidence_response.split("</think>")
    else:
        confidence_response_no_thinking = confidence_response
    confidence_score_matches = re.findall(r"\[\[([01])]]", confidence_response_no_thinking)
    if len(confidence_score_matches) < 1:
        return Err(f"No confidence score found in confidence response: {confidence_response_no_thinking}")
    found_left_quote, found_number, found_right_quote, number_token_index = False, False, False, None
    for i, tok in enumerate(logprobs):
        if tok.token.strip() == "[[":
            found_left_quote = True
        elif tok.token.strip() in ["0", "1"]:
            found_number = True
            number_token_index = i
        elif tok.token.strip() == "]]":
            found_right_quote = True
        if found_left_quote and found_number and found_right_quote and number_token_index is not None:
            break
    else:
        return Err(f"No confidence score found in confidence response: {confidence_response}")
    confidence_token = logprobs[number_token_index]
    logprobs_unnormalized = {"0": float("-inf"), "1": float("-inf")}
    for top_token in confidence_token.top_logprobs:
        if top_token.token.strip() in logprobs_unnormalized.keys():
            logprobs_unnormalized[top_token.token.strip()] = top_token.logprob
    logprobs_unnormalized = {k: math.exp(v) for k, v in logprobs_unnormalized.items()}
    logprobs_normalized = {k: v / sum(logprobs_unnormalized.values()) for k, v in logprobs_unnormalized.items()}
    return Ok((extracted_answer, logprobs_normalized["1"]))


def extract_answer_and_confidence(
    method_name: MethodName,
    dataset_name: DatasetName,
    answer_response: str,
    confidence_response: str | None,
    logprobs: list[ChatCompletionTokenLogprob] | None,
) -> Result[tuple[str, float], str]:
    assert method_name.need_logprobs == (logprobs is not None)

    postprocess_map = {
        DatasetName.GSM8K: gsm8k_postprocess,
        DatasetName.ARC: first_option_postprocess,
        DatasetName.LogiQA: first_option_postprocess,
        DatasetName.GAOKAO_Physics: gaokao_postprocess,
    }
    postprocessor = postprocess_map[dataset_name]

    extractor_map = {
        MethodName.Verbal_0_100: partial(extract_answer_and_verbal_confidence, confidence_response=confidence_response),
        MethodName.LogProb: partial(extract_answer_and_logprob_confidence, logprobs=logprobs),
        MethodName.P_True: partial(
            extract_answer_and_p_true_confidence, confidence_response=confidence_response, logprobs=logprobs
        ),
    }
    extractor = extractor_map[method_name]
    return extractor(answer_response=answer_response, postprocessor=postprocessor)
