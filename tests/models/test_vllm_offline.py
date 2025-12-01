import sys

from unittest.mock import MagicMock, patch

import pytest

from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline

from gimkit.contexts import Result
from gimkit.models.vllm_offline import VLLMOffline as GIMVLLMOffline
from gimkit.models.vllm_offline import from_vllm_offline
from gimkit.schemas import MaskedTag


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="vLLM offline tests only run on Linux"
)


def test_from_vllm_offline():
    from vllm import LLM

    model = from_vllm_offline(MagicMock(spec=LLM))
    assert type(model) is GIMVLLMOffline
    assert type(model) is not OutlinesVLLMOffline
    assert type(model) is not LLM


def test_vllm_offline_call():
    from vllm import LLM

    mock_client = MagicMock(spec=LLM)
    model = from_vllm_offline(mock_client)

    with patch("gimkit.models.vllm_offline.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = '<|MASKED id="m_0"|>hi<|/MASKED|>'
        mock_generator.return_value = generator_instance

        returned = model(MaskedTag())
        assert isinstance(returned, Result)
        assert returned.tags[0].content == "hi"


def test_vllm_offline_call_invalid_response():
    from vllm import LLM, SamplingParams

    model = from_vllm_offline(MagicMock(spec=LLM))

    with patch("gimkit.models.vllm_offline.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = set()
        mock_generator.return_value = generator_instance
        with pytest.raises(TypeError, match="Expected responses to be str or list of str, got"):
            model(MaskedTag())

    with patch("gimkit.models.vllm_offline.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = [object, "response2"]
        mock_generator.return_value = generator_instance
        with pytest.raises(TypeError, match="All items in the response list must be strings, got"):
            model(MaskedTag(), sampling_params=SamplingParams(n=2))

    with patch("gimkit.models.vllm_offline.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = []
        mock_generator.return_value = generator_instance
        with pytest.raises(ValueError, match="Response list is empty"):
            model(MaskedTag())
