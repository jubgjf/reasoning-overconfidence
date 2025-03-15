import itertools
import math
import re
from enum import Enum
from typing import assert_never

from loguru import logger
from openai.types.chat import ChatCompletionTokenLogprob
from pydantic import BaseModel
from result import Result, Ok, Err

from .data import Template, Data
from .dataset import DatasetName
from .model import Model
from .utils import first_option_postprocess, gsm8k_postprocess


class MethodName(Enum):
    Verbal_0_100 = "verbal_0_100"
    LogProb = "logprob"
    P_True = "p_true"

    def __str__(self) -> str:
        return self.value

    @property
    def need_another_turn(self) -> bool:
        return self in [self.Verbal_0_100, self.P_True]

    @property
    def need_logprobs(self) -> bool:
        return self in [self.LogProb, self.P_True]

    @property
    def prompt(self) -> str | None:
        if self == self.Verbal_0_100:
            return "Please rate your confidence in the proposed answer on a scale of 0-100. Give your confidence in format: [[xx]]"
        elif self == self.P_True:
            return "Please rate your confidence in the proposed answer as either 0 or 1. Give your confidence in format: [[xx]]"
        elif self == self.LogProb:
            return None
        else:
            assert_never(self)


class Response(BaseModel):
    messages: list[dict[str, str]]
    logprobs: list[ChatCompletionTokenLogprob] | None


class Method:
    def __init__(self, name: MethodName):
        self._name = name

    async def request(
        self,
        model: Model,
        data: Data,
        template: Template,
        temperature: float = 0,
        max_tokens: int = 16384,
    ) -> Result[Response, str]:
        # ===== First turn =====
        messages = [{"role": "user", "content": template.prompt(data)}]
        response_result = await model.request(
            messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return response_result
        messages.append({"role": "assistant", "content": response_result.ok_value.message_content})
        if not self._name.need_another_turn:
            assert len(messages) == 2
            return Ok(Response(messages=messages, logprobs=response_result.ok_value.logprobs_content))

        # ===== Second turn (Optional) =====
        messages.append({"role": "user", "content": self._name.prompt})
        response_result = await model.request(
            messages=messages, temperature=temperature, max_tokens=max_tokens, logprobs=self._name.need_logprobs
        )
        if response_result.is_err():
            return response_result
        messages.append({"role": "assistant", "content": response_result.ok_value.message_content})
        assert len(messages) == 4
        return Ok(Response(messages=messages, logprobs=response_result.ok_value.logprobs_content))

    def extract_answer_and_confidence(
        self,
        dataset_name: DatasetName,
        answer_response: str,
        confidence_response: str | None,
        logprobs: list[ChatCompletionTokenLogprob] | None,
    ) -> Result[tuple[str, float], str]:
        # TODO refactor

        assert self._name.need_logprobs == (logprobs is not None)

        if self._name == self._name.Verbal_0_100:
            if dataset_name == DatasetName.GSM8K:
                extracted_result = gsm8k_postprocess(answer_response)
                if extracted_result.is_err():
                    return extracted_result
                extracted_answer, _, _ = extracted_result.ok_value
            elif dataset_name == DatasetName.ARC:
                extracted_result = first_option_postprocess(answer_response)
                if extracted_result.is_err():
                    return extracted_result
                extracted_answer, _ = extracted_result.ok_value
            else:
                assert_never(self)
            confidence_score_matches = re.findall(r"\[\[(100|[1-9]?[0-9])]]", confidence_response)
            if len(confidence_score_matches) < 1:
                return Err(f"No confidence score found in confidence response: {confidence_response}")
            return Ok((extracted_answer, float(confidence_score_matches[0])))
        elif self._name == self._name.LogProb:
            if dataset_name == DatasetName.GSM8K:
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
                extracted_result = gsm8k_postprocess(answer_response_from_tokens)
                if extracted_result.is_err():
                    return extracted_result
                extracted_answer, answer_num_index, answer_num_len = extracted_result.ok_value
                token_index, num_index = 0, 0
                for tok in logprobs:
                    token_index += 1
                    num_index += len(tok.token)
                    if num_index >= answer_num_index:
                        break
                else:
                    return Err(f"Answer token not found in {answer_response_from_tokens}")
                if num_index == answer_num_index:
                    # Token is "A" or "B" etc.
                    answer_tokens = logprobs[token_index : token_index + answer_num_len]
                else:
                    # Token is " A" or " B" etc.
                    answer_tokens = logprobs[token_index - 1 : token_index - 1 + answer_num_len]
                if "".join([t.token.strip() for t in answer_tokens]) != extracted_answer:
                    return Err(
                        f"Answer token mismatch: {''.join([t.token.strip() for t in answer_tokens])} != {extracted_answer}"
                    )
                top_logprobs = [t.top_logprobs for t in answer_tokens]
                all_number_pairs = list(itertools.product(*top_logprobs))
                logprobs_unnormalized_unfiltered = {
                    "".join([p.token.strip() for p in pair]): sum([p.logprob for p in pair])
                    for pair in all_number_pairs
                }
                logprobs_unnormalized = dict(
                    filter(
                        lambda item: item[0].lstrip("-").replace(".", "", 1).isdigit(),
                        logprobs_unnormalized_unfiltered.items(),
                    )
                )
                logprobs_unnormalized = {k: math.exp(v) for k, v in logprobs_unnormalized.items()}
                logprobs_normalized = {
                    k: v / sum(logprobs_unnormalized.values()) for k, v in logprobs_unnormalized.items()
                }
                return Ok((extracted_answer, logprobs_normalized[extracted_answer]))
            elif dataset_name == DatasetName.ARC:
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
                extracted_result = first_option_postprocess(answer_response_from_tokens)
                if extracted_result.is_err():
                    return extracted_result
                extracted_answer, answer_char_index = extracted_result.ok_value
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
                    answer_token = logprobs[token_index]
                else:
                    # Token is " A" or " B" etc.
                    answer_token = logprobs[token_index - 1]
                if answer_token.token.strip() != extracted_answer:
                    return Err(f"Answer token mismatch: {answer_token.token.strip()} != {extracted_answer}")
                logprobs_unnormalized = {"A": float("-inf"), "B": float("-inf"), "C": float("-inf"), "D": float("-inf")}
                for top_token in answer_token.top_logprobs:
                    if top_token.token.strip() in logprobs_unnormalized.keys():
                        logprobs_unnormalized[top_token.token.strip()] = top_token.logprob
                logprobs_unnormalized = {k: math.exp(v) for k, v in logprobs_unnormalized.items()}
                logprobs_normalized = {
                    k: v / sum(logprobs_unnormalized.values()) for k, v in logprobs_unnormalized.items()
                }
                return Ok((extracted_answer, logprobs_normalized[extracted_answer]))
            else:
                assert_never(self)
        elif self._name == self._name.P_True:
            if dataset_name == DatasetName.GSM8K:
                extracted_result = gsm8k_postprocess(answer_response)
                if extracted_result.is_err():
                    return extracted_result
                extracted_answer, _, _ = extracted_result.ok_value
            elif dataset_name == DatasetName.ARC:
                extracted_result = first_option_postprocess(answer_response)
                if extracted_result.is_err():
                    return extracted_result
                extracted_answer, _ = extracted_result.ok_value
            else:
                assert_never(self)

            confidence_score_matches = re.findall(r"\[\[([01])]]", confidence_response)
            if len(confidence_score_matches) < 1:
                return Err(f"No confidence score found in confidence response: {confidence_response}")
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
        else:
            assert_never(self._name)
