from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from openai import OpenAI


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


class OpenAICompatibleLLM:
    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str = "dummy",
        extra_body: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.extra_body = extra_body
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )

    def generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        top_k: int | None = None,
        min_p: float | None = None,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {}

        if temperature is not None:
            kwargs["temperature"] = temperature

        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        extra_body = dict(self.extra_body or {})

        if enable_thinking is not None:
            chat_template_kwargs = dict(
                extra_body.get(
                    "chat_template_kwargs",
                    {}
                )
            )
            chat_template_kwargs[
                "enable_thinking"
            ] = enable_thinking
            extra_body[
                "chat_template_kwargs"
            ] = chat_template_kwargs

        if reasoning_effort is not None:
            extra_body[
                "reasoning_effort"
            ] = reasoning_effort

        if top_k is not None:
            extra_body["top_k"] = top_k

        if min_p is not None:
            extra_body["min_p"] = min_p

        if extra_body:
            kwargs["extra_body"] = extra_body

        result = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            **kwargs,
        )

        usage = result.usage

        return LLMResponse(
            text=(
                result.choices[0].message.content
                or ""
            ),
            prompt_tokens=(
                usage.prompt_tokens
                if usage is not None
                else None
            ),
            completion_tokens=(
                usage.completion_tokens
                if usage is not None
                else None
            ),
        )
