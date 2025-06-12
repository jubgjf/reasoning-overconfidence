import math
import re
from functools import partial
from typing import Callable

from result import Err, Ok, Result

from .dataset import DatasetName
from .method import ChatResponsePerTurn, CompleteResponsePerTurn, MethodName
from .utils import split_thinking_answer


def extract_answer_and_verbal_confidence(
    question_turn: ChatResponsePerTurn | CompleteResponsePerTurn,
    confidence_turn: ChatResponsePerTurn,
    postprocessor: Callable[[str], Result[tuple[str, int, int], str]],
) -> Result[tuple[str, float], str]:
    extracted_result = postprocessor(question_turn.answer_content)
    if extracted_result.is_err():
        return extracted_result
    extracted_answer, _, _ = extracted_result.ok_value

    confidence_score_matches = re.findall(r"\[\[(100|[1-9]?[0-9])]]", confidence_turn.answer_content)
    if len(confidence_score_matches) < 1:
        return Err(f"No confidence score found in confidence response: {confidence_turn.answer_content}")
    return Ok((extracted_answer, float(confidence_score_matches[0])))


def extract_answer_and_p_true_confidence(
    question_turn: ChatResponsePerTurn | CompleteResponsePerTurn,
    confidence_turn: ChatResponsePerTurn,
    postprocessor: Callable[[str], Result[tuple[str, int, int], str]],
) -> Result[tuple[str, float], str]:
    extracted_result = postprocessor(question_turn.answer_content)
    if extracted_result.is_err():
        return extracted_result
    extracted_answer, _, _ = extracted_result.ok_value

    confidence_score_matches = re.findall(r"\[\[([01])]]", confidence_turn.answer_content)
    if len(confidence_score_matches) < 1:
        return Err(f"No confidence score found in confidence response: {confidence_turn.answer_content}")
    found_left_quote, found_number, found_right_quote, number_token_index = False, False, False, None
    for i, tok in enumerate(confidence_turn.answer_logprobs):
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
        return Err(f"No confidence score found in confidence response: {confidence_turn.answer_content}")
    confidence_token = confidence_turn.answer_logprobs[number_token_index]
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
    question_turn: ChatResponsePerTurn | CompleteResponsePerTurn,
    confidence_turn: ChatResponsePerTurn | None,
) -> Result[tuple[str, float], str]:
    if dataset_name in [DatasetName.TimeTabling, DatasetName.SubsetSum]:
        assert method_name == MethodName.Verbal_0_100

    postprocess_map = {
        DatasetName.TimeTabling: lambda x: Ok((split_thinking_answer(x)[-1], -1, -1)),
        DatasetName.SubsetSum: lambda x: Ok((split_thinking_answer(x)[-1], -1, -1)),
    }
    postprocessor = postprocess_map[dataset_name]

    extractor_map = {
        MethodName.Verbal_0_100: partial(extract_answer_and_verbal_confidence, confidence_turn=confidence_turn),
        MethodName.P_True: partial(extract_answer_and_p_true_confidence, confidence_turn=confidence_turn),
    }
    extractor = extractor_map[method_name]
    return extractor(question_turn=question_turn, postprocessor=postprocessor)
