import sys

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline

from gimkit.contexts import Result
from gimkit.models.vllm_offline import VLLMOffline as GIMVLLMOffline
from gimkit.models.vllm_offline import from_vllm_offline
from gimkit.schemas import RESPONSE_SUFFIX, MaskedTag


pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="vLLM offline tests only run on Linux"
)


def _mock_vllm_client():
    from vllm import LLM

    mock_client = MagicMock(spec=LLM)
    mock_client.get_tokenizer.return_value = object()
    return mock_client


def _request_output(*texts: str):
    request_output = MagicMock()
    request_output.outputs = [MagicMock(text=text) for text in texts]
    return request_output


def test_from_vllm_offline():
    from vllm import LLM

    model = from_vllm_offline(_mock_vllm_client())
    assert type(model) is GIMVLLMOffline
    assert type(model) is not OutlinesVLLMOffline
    assert type(model) is not LLM


def test_vllm_offline_call():
    mock_client = _mock_vllm_client()
    model = from_vllm_offline(mock_client)

    with patch("gimkit.models.vllm_offline.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = '<|MASKED id="m_0"|>hi<|/MASKED|>'
        mock_generator.return_value = generator_instance

        returned = model(MaskedTag())
        assert isinstance(returned, Result)
        assert returned.tags[0].content == "hi"

        model(MaskedTag(), visible_tag_fields=["id", "desc", "content", "regex"])


def test_vllm_offline_batch():
    mock_client = _mock_vllm_client()
    mock_client.generate.return_value = [
        _request_output('<|MASKED id="m_0"|>world<|/MASKED|>'),
        _request_output('<|MASKED id="m_0"|>dear<|/MASKED|><|MASKED id="m_1"|>friend<|/MASKED|>'),
    ]
    model = from_vllm_offline(mock_client)

    returned = model.batch(
        [
            ["Hello, ", MaskedTag()],
            ["Goodbye, ", MaskedTag(), " ", MaskedTag()],
        ]
    )

    assert len(returned) == 2
    assert isinstance(returned[0], list)
    assert str(returned[0][0]) == "Hello, world"
    assert str(returned[1][0]) == "Goodbye, dear friend"

    mock_client.generate.assert_called_once()
    sampling_params = mock_client.generate.call_args.kwargs["sampling_params"]
    assert len(sampling_params) == 2
    assert (
        sampling_params[0].structured_outputs.grammar
        != sampling_params[1].structured_outputs.grammar
    )
    assert RESPONSE_SUFFIX in sampling_params[0].stop
    assert RESPONSE_SUFFIX in sampling_params[1].stop


def test_vllm_offline_batch_sampling_params_list():
    from vllm import SamplingParams

    mock_client = _mock_vllm_client()
    mock_client.generate.return_value = [
        _request_output('<|MASKED id="m_0"|>world<|/MASKED|>'),
        _request_output('<|MASKED id="m_0"|>friend<|/MASKED|>'),
    ]
    model = from_vllm_offline(mock_client)
    sampling_params = [SamplingParams(stop=["<END>"]), SamplingParams()]

    returned = model.batch(
        [
            ["Hello, ", MaskedTag()],
            ["Goodbye, ", MaskedTag()],
        ],
        sampling_params=sampling_params,
    )

    assert len(returned) == 2
    assert str(returned[0][0]) == "Hello, world"
    assert str(returned[1][0]) == "Goodbye, friend"
    assert RESPONSE_SUFFIX in sampling_params[0].stop
    assert RESPONSE_SUFFIX in sampling_params[1].stop


def test_vllm_offline_ensure_sampling_params_response_suffix():
    model = from_vllm_offline(_mock_vllm_client())

    sampling_params = SimpleNamespace(stop=None)
    model._ensure_sampling_params_response_suffix(sampling_params)
    assert sampling_params.stop == [RESPONSE_SUFFIX]

    sampling_params = SimpleNamespace(stop="<END>")
    model._ensure_sampling_params_response_suffix(sampling_params)
    assert sampling_params.stop == ["<END>", RESPONSE_SUFFIX]

    sampling_params = SimpleNamespace(stop=RESPONSE_SUFFIX)
    model._ensure_sampling_params_response_suffix(sampling_params)
    assert sampling_params.stop == RESPONSE_SUFFIX


def test_vllm_offline_batch_invalid_sampling_params_list_length():
    from vllm import SamplingParams

    model = from_vllm_offline(_mock_vllm_client())

    with pytest.raises(ValueError, match="sampling_params list must have the same length"):
        model.batch(
            [
                ["Hello, ", MaskedTag()],
                ["Goodbye, ", MaskedTag()],
            ],
            sampling_params=[SamplingParams()],
        )


def test_vllm_offline_batch_invalid_response():
    mock_client = _mock_vllm_client()
    mock_client.generate.return_value = [_request_output()]
    model = from_vllm_offline(mock_client)

    with pytest.raises(ValueError, match="Response list is empty"):
        model.batch([["Hello, ", MaskedTag()]])

    mock_client.generate.return_value = [
        MagicMock(outputs=[MagicMock(text=object())]),
    ]
    with pytest.raises(TypeError, match="All items in the response list must be strings"):
        model.batch([["Hello, ", MaskedTag()]])


def test_vllm_offline_batch_chat():
    mock_client = _mock_vllm_client()
    mock_client.chat.return_value = [
        _request_output('<|MASKED id="m_0"|>world<|/MASKED|>'),
        _request_output('<|MASKED id="m_0"|>friend<|/MASKED|>'),
    ]
    model = from_vllm_offline(mock_client)

    returned = model.batch(
        [
            ["Hello, ", MaskedTag()],
            ["Goodbye, ", MaskedTag()],
        ],
        use_gim_prompt=True,
    )

    assert len(returned) == 2
    assert str(returned[0][0]) == "Hello, world"
    assert str(returned[1][0]) == "Goodbye, friend"
    mock_client.chat.assert_called_once()


def test_vllm_offline_batch_flat_responses():
    model = from_vllm_offline(_mock_vllm_client())

    with patch.object(
        model,
        "_generate_batch_with_output_types",
        return_value=[
            '<|MASKED id="m_0"|>world<|/MASKED|>',
            '<|MASKED id="m_0"|>friend<|/MASKED|>',
        ],
    ):
        returned = model.batch(
            [
                ["Hello, ", MaskedTag()],
                ["Goodbye, ", MaskedTag()],
            ]
        )

        assert len(returned) == 2
        assert isinstance(returned[0], Result)
        assert str(returned[0]) == "Hello, world"
        assert str(returned[1]) == "Goodbye, friend"


def test_vllm_offline_call_invalid_response():
    from vllm import SamplingParams

    model = from_vllm_offline(_mock_vllm_client())

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
