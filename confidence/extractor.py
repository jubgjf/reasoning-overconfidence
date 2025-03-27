import itertools
import math
import re
from functools import partial
from typing import Callable, assert_never

from result import Err, Ok, Result

from .dataset import DatasetName
from .method import ChatResponsePerTurn, CompleteResponsePerTurn, MethodName
from .utils import first_option_postprocess, gaokao_postprocess, gsm8k_postprocess


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


def extract_answer_and_logprob_confidence(
    dataset_name: DatasetName,
    question_turn: ChatResponsePerTurn | CompleteResponsePerTurn,
    postprocessor: Callable[[str], Result[tuple[str, int, int], str]],
) -> Result[tuple[str, float], str]:
    if isinstance(question_turn, ChatResponsePerTurn):
        extracted_result = postprocessor(question_turn.answer_content)
        if extracted_result.is_err():
            return extracted_result
        extracted_answer, answer_char_index, answer_char_len = extracted_result.ok_value
        token_index, char_index = 0, 0
        for tok in question_turn.answer_logprobs:
            token_index += 1
            char_index += len(tok.token)
            if char_index >= answer_char_index:
                break
        else:
            return Err(f"Answer token not found in {question_turn.answer_content}")
        if char_index == answer_char_index:
            # Token is "A" or "B" etc.
            answer_tokens = question_turn.answer_logprobs[token_index : token_index + answer_char_len]
            candidate_answer_tokens = question_turn.answer_logprobs[token_index + 1 : token_index + answer_char_len + 1]
        else:
            # Token is " A" or " B" etc.
            answer_tokens = question_turn.answer_logprobs[token_index - 1 : token_index - 1 + answer_char_len]
            candidate_answer_tokens = question_turn.answer_logprobs[token_index : token_index + answer_char_len]
        if "".join([t.token.strip() for t in answer_tokens]) != extracted_answer:
            if "".join([t.token.strip() for t in candidate_answer_tokens]) != extracted_answer:
                # Sometimes "�" in "".join(tokens), so char_index may be wrong, and tokens may not match answer exactly
                return Err(
                    f"Answer token mismatch: {''.join([t.token.strip() for t in answer_tokens])} != {extracted_answer}"
                )
            answer_tokens = candidate_answer_tokens
        top_logprobs = [t.top_logprobs for t in answer_tokens]
        all_number_pairs = list(itertools.product(*top_logprobs))
        logprobs_unnormalized_unfiltered = {
            "".join([p.token.strip() for p in pair]): sum([p.logprob for p in pair]) for pair in all_number_pairs
        }
        if dataset_name in [DatasetName.GSM8K]:
            logprobs_unnormalized = dict(
                filter(
                    lambda item: item[0].lstrip("-").replace(".", "", 1).isdigit(),
                    logprobs_unnormalized_unfiltered.items(),
                )
            )
        elif dataset_name in [DatasetName.ARC, DatasetName.LogiQA]:
            logprobs_unnormalized = dict(
                filter(lambda item: item[0] in "ABCD", logprobs_unnormalized_unfiltered.items())
            )
        else:
            raise NotImplementedError
        logprobs_unnormalized = {k: math.exp(v) for k, v in logprobs_unnormalized.items()}
        logprobs_normalized = {k: v / sum(logprobs_unnormalized.values()) for k, v in logprobs_unnormalized.items()}
        return Ok((extracted_answer, logprobs_normalized[extracted_answer]))
    elif isinstance(question_turn, CompleteResponsePerTurn):
        raise NotImplementedError
    else:
        assert_never(question_turn)


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
    postprocess_map = {
        DatasetName.GSM8K: gsm8k_postprocess,
        DatasetName.ARC: first_option_postprocess,
        DatasetName.LogiQA: first_option_postprocess,
        DatasetName.GAOKAO_Physics: gaokao_postprocess,
    }
    postprocessor = postprocess_map[dataset_name]

    extractor_map = {
        MethodName.Verbal_0_100: partial(extract_answer_and_verbal_confidence, confidence_turn=confidence_turn),
        MethodName.LogProb: partial(extract_answer_and_logprob_confidence, dataset_name=dataset_name),
        MethodName.P_True: partial(extract_answer_and_p_true_confidence, confidence_turn=confidence_turn),
    }
    extractor = extractor_map[method_name]
    return extractor(question_turn=question_turn, postprocessor=postprocessor)
