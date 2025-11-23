from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openai import AsyncOpenAI, OpenAI
from outlines import VLLM as OutlinesVLLM
from outlines import AsyncVLLM as OutlinesAsyncVLLM

from gimkit.contexts import Query, Result
from gimkit.guides import guide
from gimkit.models.vllm import VLLM as GIMVLLM
from gimkit.models.vllm import AsyncVLLM as GIMAsyncVLLM
from gimkit.models.vllm import from_vllm
from gimkit.schemas import MaskedTag


def test_from_vllm():
    client = OpenAI(api_key="test")
    model = from_vllm(client)
    assert type(model) is GIMVLLM
    assert type(model) is not OutlinesVLLM
    assert type(model) is not OpenAI
    assert model.client is client

    async_client = AsyncOpenAI(api_key="test")
    async_model = from_vllm(async_client)
    assert type(async_model) is GIMAsyncVLLM
    assert type(async_model) is not OutlinesAsyncVLLM
    assert type(async_model) is not AsyncOpenAI
    assert async_model.client is async_client

    with pytest.raises(
        ValueError,
        match=r"Unsupported client type: <class 'str'>.\s*Please provide an OpenAI or AsyncOpenAI instance.",
    ):
        from_vllm("not a client")


def test_sync_call():
    client = OpenAI(api_key="test", timeout=0, max_retries=0)

    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[
        0
    ].message.content = '<|GIM_RESPONSE|><|MASKED id="m_0"|>world<|/MASKED|><|/GIM_RESPONSE|>'
    mock_response.choices[0].message.refusal = None

    with patch.object(client.chat.completions, "create", return_value=mock_response) as mock_create:
        model = from_vllm(client, model_name="gpt-4o")

        result = model("Hello, " + guide())
        assert isinstance(result, Result)
        assert result.tags[0] == MaskedTag(id=0, content="world")
        mock_create.assert_called_once()
        # Verify that stop parameter is NOT passed
        assert "stop" not in mock_create.call_args[1]

        # Model can accept different input types
        model(Query("Hello, ", guide()))
        model(["Hello, " + guide()])

        # Model raises error on invalid output type
        with pytest.raises(ValueError, match="Invalid output type: xxx"):
            model(Query("hi"), output_type="xxx")


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
        model = from_vllm(client, model_name="gpt-4o")
        result = await model("Hello, " + guide())
        assert isinstance(result, Result)
        assert result.tags[0] == MaskedTag(id=0, content="world")
        mock_create.assert_awaited_once()
        # Verify that stop parameter is NOT passed
        assert "stop" not in mock_create.call_args[1]
