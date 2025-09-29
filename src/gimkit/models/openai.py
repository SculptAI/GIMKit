from typing import TYPE_CHECKING, Any, Literal, Union

from outlines.models.openai import AsyncOpenAI as OutlinesAsyncOpenAI
from outlines.models.openai import OpenAI as OutlinesOpenAI

from gimkit.contexts import Response
from gimkit.models.utils import _acall, _call
from gimkit.schemas import MaskedTag


if TYPE_CHECKING:
    from openai import (
        AsyncAzureOpenAI as AsyncAzureOpenAIClient,
    )
    from openai import (
        AsyncOpenAI as AsyncOpenAIClient,
    )
    from openai import (
        AzureOpenAI as AzureOpenAIClient,
    )
    from openai import (
        OpenAI as OpenAIClient,
    )


class OpenAI(OutlinesOpenAI):
    def __call__(
        self,
        model_input: str | MaskedTag | list[str | MaskedTag],
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        **inference_kwargs: Any,
    ) -> Response:
        return _call(self, model_input, output_type, backend, **inference_kwargs)


class AsyncOpenAI(OutlinesAsyncOpenAI):
    async def __call__(
        self,
        model_input: str | MaskedTag | list[str | MaskedTag],
        output_type: Literal["json"] | None = None,
        backend: str | None = None,
        **inference_kwargs: Any,
    ) -> Response:
        return await _acall(self, model_input, output_type, backend, **inference_kwargs)


def from_openai(
    client: Union[
        "OpenAIClient",
        "AsyncOpenAIClient",
        "AzureOpenAIClient",
        "AsyncAzureOpenAIClient",
    ],
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
            "+ `openai.OpenAI` or `openai.AsyncOpenAI`."
        )
