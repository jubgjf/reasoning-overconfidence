import asyncio
import io
import os
from enum import Enum

from dotenv import load_dotenv
from loguru import logger
from openai import AsyncOpenAI
from pydantic import BaseModel
from transformers import AutoTokenizer

from confidence.result import Result
from confidence.utils import split_thinking_answer


class ModelName(Enum):
    QWEN3_8B_THINK = "qwen3-8b-think"
    QWEN3_8B_NO_THINK = "qwen3-8b-no_think"
    DEEPSEEK_R1 = "DeepSeekR1INT8"
    DEEPSEEK_V3 = "DeepSeekV3"
    GPT_4O_MINI = "gpt-4o-mini-2024-07-18"
    O4_MINI = "o4-mini-2025-04-16"

    def __str__(self) -> str:
        return self.value

    @property
    def model_id(self) -> str:
        return self.value

    @property
    def series_name(self) -> str:
        if self in [ModelName.QWEN3_8B_THINK, ModelName.QWEN3_8B_NO_THINK]:
            return "Qwen"
        elif self in [ModelName.DEEPSEEK_R1, ModelName.DEEPSEEK_V3]:
            return "DeepSeek"
        elif self in [ModelName.O4_MINI, ModelName.GPT_4O_MINI]:
            return "GPT"
        else:
            raise ValueError(f"Unknown model name: {self}")


class ChatResponse(BaseModel):
    messages: list[dict[str, str]]
    thinking: list[str] | None


class Model:
    def __init__(self, model_name: ModelName, model_name_or_path: str):
        self.model_name = model_name
        self._client = self._get_client()
        self._tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    @staticmethod
    def _get_client() -> AsyncOpenAI:
        def get_url_and_key() -> tuple[str | None, str | None]:
            return os.getenv("BASE_URL"), os.getenv("API_KEY")

        base_url, api_key = get_url_and_key()
        if base_url is None or api_key is None:
            load_dotenv()
            base_url, api_key = get_url_and_key()

        if base_url is None or api_key is None:
            raise ValueError("BASE_URL and API_KEY must be set in environment variables")

        return AsyncOpenAI(base_url=base_url, api_key=api_key, timeout=1800)

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0,
        max_completion_tokens: int = 32768,
    ) -> Result[ChatResponse, str]:
        """
        "[[ASSISTANT]]" is a placeholder for the assistant's response.

        Example of incoming `messages`:
        ```python
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "[[ASSISTANT]]"},
            {"role": "user", "content": "What is your confidence?"},
            {"role": "assistant", "content": "[[ASSISTANT]]"},
        ]
        ```

        And return is like:
        ```python
        (
            [
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"},                  # No thinking content
                {"role": "user", "content": "What is your confidence?"},
                {"role": "assistant", "content": "90%"},                    # No thinking content
            ],
            [
                "Okay, the capital of France is ...</think>",        # Thinking content is here
                "Okay, my confidence is ...</think>",                # Thinking content is here
            ]
        )
        ```

        len(thinking_content) is equal to "[[ASSISTANT]]" count in `messages`.
        """

        async def request_once(messages_no_placeholder: list[dict[str, str]]) -> Result[str, str]:
            try:
                response = await self._client.chat.completions.create(
                    model=self.model_name.model_id,
                    messages=messages_no_placeholder,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    stream=self.model_name in [ModelName.DEEPSEEK_R1],
                )
                if self.model_name in [ModelName.QWEN3_8B_THINK]:
                    assert response.choices[0].message.model_extra is not None
                    assert response.choices[0].message.model_extra["reasoning_content"] is not None
                    if response.choices[0].message.content is None:
                        return Result(
                            err="response.choices[0].message.content is None but reasoning_content is not None, consider increase max_new_tokens"
                        )
                    message_content = (
                        response.choices[0].message.model_extra["reasoning_content"]
                        + "</think>"
                        + response.choices[0].message.content
                    )
                elif self.model_name in [ModelName.DEEPSEEK_R1]:
                    # Wired, but `timeout` in AsyncOpenAI is not working
                    # DeepSeek-R1 API is slow, so we use streaming to keep the connection alive
                    buffer = io.StringIO()
                    async for chunk in response:
                        if chunk.choices[0].delta.content:
                            chunk_content = chunk.choices[0].delta.content
                            # print(chunk_content, end="", flush=True)
                            buffer.write(chunk_content)
                    full_response = buffer.getvalue()
                    buffer.close()
                    if "</think>" not in full_response:
                        return Result(err="</think> not found in response")
                    message_content = full_response
                elif self.model_name in [ModelName.O4_MINI]:
                    assert response.choices[0].message.content is not None
                    message_content = "REASONING CONTENT NOT RETURN" + "</think>" + response.choices[0].message.content
                else:
                    assert response.choices[0].message.content is not None
                    message_content = response.choices[0].message.content
                return Result(ok=message_content)
            except Exception as e:
                return Result(err=f"Error: {e}")

        def _extract_prompt(messages: list[dict[str, str]]) -> tuple[list[dict[str, str]] | None, int | None]:
            assert len(messages) % 2 == 0

            for i in range(0, len(messages), 2):
                user_turn, assistant_turn = messages[i], messages[i + 1]
                assert user_turn["role"] == "user"
                assert assistant_turn["role"] == "assistant"

                if "[[ASSISTANT]]" in assistant_turn["content"]:
                    return messages[: i + 1], i + 1

            return None, None

        placeholder_count = sum([1 if "[[ASSISTANT]]" in turn["content"] else 0 for turn in messages])

        thinking_list = []
        while True:
            extracted_messages, placeholder_index = _extract_prompt(messages)
            if extracted_messages is None and placeholder_index is None:
                break
            assert extracted_messages is not None and placeholder_index is not None

            retry, tolerance = 0, 3
            response_result = Result(err="Failed to request")
            while retry < tolerance:
                response_result = await request_once(extracted_messages)
                if response_result.is_ok():
                    break
                retry += 1
                logger.error(f"Retry: {retry}/{tolerance}. Failure: {response_result.err_value} ")
                await asyncio.sleep(0.1)
            else:
                return Result(err=response_result.err_value)
            thinking_content, answer_content = split_thinking_answer(response_result.ok_value)
            messages[placeholder_index]["content"] = answer_content
            thinking_list.append(thinking_content)

        assert messages[-1]["role"] == "assistant"
        assert len(thinking_list) == placeholder_count

        thinking = None if all(t == "" for t in thinking_list) else thinking_list
        return Result(ok=ChatResponse(messages=messages, thinking=thinking))

    async def complete(
        self,
        prompt: str,
        temperature: float = 0,
        max_tokens: int = 20480,
    ) -> Result[str, str]:
        async def request_once() -> Result[str, str]:
            try:
                response = await self._client.completions.create(
                    model=self.model_name.model_id,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return Result(ok=response.choices[0].text)
            except Exception as e:
                err_msg = f"Error: {e}"
                return Result(err=err_msg)

        retry, tolerance = 0, 3
        response_result = Result(err="Failed to request")
        while retry < tolerance:
            response_result = await request_once()
            if response_result.is_ok():
                break
            retry += 1
            logger.error(f"Retry: {retry}/{tolerance}. Failure: {response_result.err} ")
            await asyncio.sleep(0.1)

        return response_result

    def string_to_token_id(self, text: str) -> int:
        token_ids = self._tokenizer.encode(text)
        assert len(token_ids) == 1
        return token_ids[0]

    def token_ids_to_string(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)

    def apply_chat_template(self, messages: list[dict[str, str]]) -> str:
        return self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
