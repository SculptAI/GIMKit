# Adapted from https://github.com/dottxt-ai/outlines/blob/main/outlines/models/openai.py

from typing import Any, Literal, overload

from openai import AsyncAzureOpenAI as AsyncAzureOpenAIClient
from openai import AsyncOpenAI as AsyncOpenAIClient
from openai import AzureOpenAI as AzureOpenAIClient
from openai import OpenAI as OpenAIClient
from outlines.models.openai import AsyncOpenAI as OutlinesAsyncOpenAI
from outlines.models.openai import OpenAI as OutlinesOpenAI

from gimkit.contexts import Query, Result
from gimkit.models.base import _acall, _call
from gimkit.models.types import ErrorMode, GenerationResult
from gimkit.schemas import ContextInput, TagField


class OpenAI(OutlinesOpenAI):
    @overload
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["raise"] = "raise",
        **inference_kwargs: Any,
    ) -> Result | list[Result]: ...

    @overload
    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["collect"],
        **inference_kwargs: Any,
    ) -> GenerationResult | list[GenerationResult]: ...

    def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: ErrorMode = "raise",
        **inference_kwargs: Any,
    ) -> Result | list[Result] | GenerationResult | list[GenerationResult]:
        return _call(
            self,
            model_input,
            output_type,
            backend,
            use_gim_prompt,
            visible_tag_fields,
            error_mode=error_mode,
            **inference_kwargs,
        )


class AsyncOpenAI(OutlinesAsyncOpenAI):
    @overload
    async def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["raise"] = "raise",
        **inference_kwargs: Any,
    ) -> Result | list[Result]: ...

    @overload
    async def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: Literal["collect"],
        **inference_kwargs: Any,
    ) -> GenerationResult | list[GenerationResult]: ...

    async def __call__(
        self,
        model_input: ContextInput | Query,
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        use_gim_prompt: bool = False,
        visible_tag_fields: list[TagField] | None = None,
        *,
        error_mode: ErrorMode = "raise",
        **inference_kwargs: Any,
    ) -> Result | list[Result] | GenerationResult | list[GenerationResult]:
        return await _acall(
            self,
            model_input,
            output_type,
            backend,
            use_gim_prompt,
            visible_tag_fields,
            **inference_kwargs,
            error_mode=error_mode,
        )


@overload
def from_openai(
    client: OpenAIClient | AzureOpenAIClient, model_name: str | None = None
) -> OpenAI: ...


@overload
def from_openai(
    client: AsyncOpenAIClient | AsyncAzureOpenAIClient, model_name: str | None = None
) -> AsyncOpenAI: ...


def from_openai(
    client: OpenAIClient | AsyncOpenAIClient | AzureOpenAIClient | AsyncAzureOpenAIClient,
    model_name: str | None = None,
) -> OpenAI | AsyncOpenAI:
    import openai

    if isinstance(client, openai.OpenAI):
        return OpenAI(client, model_name)
    elif isinstance(client, openai.AsyncOpenAI):
        return AsyncOpenAI(client, model_name)
    else:
        raise ValueError(
            "Invalid client type. The client must be an instance of "
            "`openai.OpenAI` or `openai.AsyncOpenAI`."
        )
