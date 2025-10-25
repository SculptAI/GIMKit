from unittest.mock import MagicMock, patch

import pytest

from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline
from vllm import LLM, SamplingParams

from gimkit.contexts import Result
from gimkit.models.vllm_offline import VLLMOffline as GIMVLLMOffline
from gimkit.models.vllm_offline import from_vllm_offline
from gimkit.schemas import MaskedTag


def test_from_vllm_offline():
    model = from_vllm_offline(MagicMock(spec=LLM))
    assert type(model) is GIMVLLMOffline
    assert type(model) is not OutlinesVLLMOffline
    assert type(model) is not LLM


def test_vllm_offline_call():
    mock_client = MagicMock(spec=LLM)
    model = from_vllm_offline(mock_client)
    result = Result(MaskedTag(id=0, content="hi"))

    with patch("gimkit.models.vllm_offline._call", return_value=result) as mock_call:
        returned = model(MaskedTag())
        assert isinstance(returned, Result)
        assert returned.tags[0].content == "hi"
        mock_call.assert_called_once_with(
            model,
            MaskedTag(),
            "cfg",
            None,
            False,
            sampling_params=SamplingParams(stop="<|/GIM_RESPONSE|>"),
        )

    with patch("gimkit.models.vllm_offline._call", return_value=result) as mock_call:
        sample_params = SamplingParams(temperature=0.2, stop=["another_stop"])
        returned = model(MaskedTag(), sampling_params=sample_params)
        assert isinstance(returned, Result)
        assert returned.tags[0].content == "hi"

        sample_params.stop.append("<|/GIM_RESPONSE|>")
        mock_call.assert_called_once_with(
            model, MaskedTag(), "cfg", None, False, sampling_params=sample_params
        )

    with patch("gimkit.models.utils.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = ["response1", "response2"]
        mock_generator.return_value = generator_instance

        results = model("", sampling_params=SamplingParams(n=2))

        assert isinstance(results, list)
        assert len(results) == 2
        mock_generator.assert_called_once()
        generator_instance.assert_called_once()


def test_vllm_offline_call_invalid_response():
    model = from_vllm_offline(MagicMock(spec=LLM))

    with patch("gimkit.models.utils.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = set()
        mock_generator.return_value = generator_instance
        with pytest.raises(TypeError, match="Expected responses to be str or list of str, got"):
            model(MaskedTag())

    with patch("gimkit.models.utils.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = [object, "response2"]
        mock_generator.return_value = generator_instance
        with pytest.raises(TypeError, match="All items in the response list must be strings, got"):
            model(MaskedTag(), sampling_params=SamplingParams(n=2))

    with patch("gimkit.models.utils.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = []
        mock_generator.return_value = generator_instance
        with pytest.raises(ValueError, match="Response list is empty"):
            model(MaskedTag())
