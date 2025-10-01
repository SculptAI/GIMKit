from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai import AsyncOpenAI, OpenAI
from outlines import AsyncOpenAI as OutlinesAsyncOpenAI
from outlines import OpenAI as OutlinesOpenAI

from gimkit.contexts import Response
from gimkit.guides import guide
from gimkit.models.openai import AsyncOpenAI as GIMAsyncOpenAI
from gimkit.models.openai import OpenAI as GIMOpenAI
from gimkit.models.openai import from_openai
from gimkit.schemas import MaskedTag


def test_from_openai():
    client = OpenAI(api_key="test")
    model = from_openai(client)
    assert type(model) is GIMOpenAI
    assert type(model) is not OutlinesOpenAI
    assert type(model) is not OpenAI
    assert model.client is client

    model2 = from_openai(client, model_name="gpt-4o")
    assert type(model2) is GIMOpenAI
    assert model2.client is client
    assert model2.model_name == "gpt-4o"

    async_client = AsyncOpenAI(api_key="test")
    async_model = from_openai(async_client)
    assert type(async_model) is GIMAsyncOpenAI
    assert type(async_model) is not OutlinesAsyncOpenAI
    assert type(async_model) is not AsyncOpenAI
    assert async_model.client is async_client

    with pytest.raises(
        ValueError,
        match=r"Invalid client type. The client must be an instance of `openai.OpenAI` or `openai.AsyncOpenAI`.",
    ):
        from_openai("not a client")


def test_sync_call():
    client = OpenAI(api_key="test", timeout=0, max_retries=0)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = '<|GIM_RESPONSE|><|MASKED id="m_0"|>world<|/MASKED|><|/GIM_RESPONSE|>'
    mock_response.choices[0].message.refusal = None

    with patch.object(client.chat.completions, "create", return_value=mock_response) as mock_create:
        model = from_openai(client, model_name="gpt-4o")
        response = model("Hello, " + guide(), output_type=None)
        assert isinstance(response, Response)
        assert response.tags[0] == MaskedTag(id=0, content="world")
        mock_create.assert_called_once()


@pytest.mark.asyncio
async def test_async_call():
    client = AsyncOpenAI(api_key="test", timeout=0, max_retries=0)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = '<|GIM_RESPONSE|><|MASKED id="m_0"|>world<|/MASKED|><|/GIM_RESPONSE|>'
    mock_response.choices[0].message.refusal = None

    with patch.object(
        client.chat.completions, "create", new_callable=AsyncMock, return_value=mock_response
    ) as mock_create:
        model = from_openai(client, model_name="gpt-4o")
        response = await model("Hello, " + guide(), output_type=None)
        assert isinstance(response, Response)
        assert response.tags[0] == MaskedTag(id=0, content="world")
        mock_create.assert_awaited_once()
