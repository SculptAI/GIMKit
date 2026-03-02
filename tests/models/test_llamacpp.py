from unittest.mock import MagicMock, patch

import pytest

from outlines.models.llamacpp import LlamaCpp as OutlinesLlamaCpp

from gimkit.contexts import Result
from gimkit.models.llamacpp import LlamaCpp as GIMLlamaCpp
from gimkit.models.llamacpp import from_llamacpp
from gimkit.schemas import RESPONSE_SUFFIX, MaskedTag


@pytest.fixture(autouse=True)
def patch_tokenizer():
    """Patch LlamaCppTokenizer so tests run without llama-cpp-python installed."""
    with patch("outlines.models.llamacpp.LlamaCppTokenizer"):
        yield


def test_from_llamacpp():
    mock_llama = MagicMock()
    model = from_llamacpp(mock_llama)
    assert type(model) is GIMLlamaCpp
    assert type(model) is not OutlinesLlamaCpp
    assert model.model is mock_llama


def test_llamacpp_call():
    mock_llama = MagicMock()
    model = from_llamacpp(mock_llama)

    with patch("gimkit.models.llamacpp.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = '<|MASKED id="m_0"|>hi<|/MASKED|>'
        mock_generator.return_value = generator_instance

        returned = model(MaskedTag())
        assert isinstance(returned, Result)
        assert returned.tags[0].content == "hi"

        # Verify RESPONSE_SUFFIX is added to stop
        call_kwargs = generator_instance.call_args[1]
        assert RESPONSE_SUFFIX in call_kwargs["stop"]


def test_llamacpp_call_invalid_response():
    mock_llama = MagicMock()
    model = from_llamacpp(mock_llama)

    with patch("gimkit.models.llamacpp.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = set()
        mock_generator.return_value = generator_instance
        with pytest.raises(TypeError, match="Expected responses to be str or list of str, got"):
            model(MaskedTag())

    with patch("gimkit.models.llamacpp.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = []
        mock_generator.return_value = generator_instance
        with pytest.raises(ValueError, match="Response list is empty"):
            model(MaskedTag())


def test_ensure_response_suffix():
    mock_llama = MagicMock()
    model = from_llamacpp(mock_llama)

    # No stop provided — should add RESPONSE_SUFFIX
    kwargs = model._ensure_response_suffix({})
    assert kwargs["stop"] == [RESPONSE_SUFFIX]

    # stop is a list without RESPONSE_SUFFIX — should append it
    kwargs = model._ensure_response_suffix({"stop": ["other"]})
    assert RESPONSE_SUFFIX in kwargs["stop"]
    assert "other" in kwargs["stop"]

    # stop is a list already containing RESPONSE_SUFFIX — unchanged
    kwargs = model._ensure_response_suffix({"stop": [RESPONSE_SUFFIX]})
    assert kwargs["stop"] == [RESPONSE_SUFFIX]

    # stop is a string different from RESPONSE_SUFFIX — should wrap both
    kwargs = model._ensure_response_suffix({"stop": "other"})
    assert RESPONSE_SUFFIX in kwargs["stop"]
    assert "other" in kwargs["stop"]

    # stop is already RESPONSE_SUFFIX string — unchanged
    kwargs = model._ensure_response_suffix({"stop": RESPONSE_SUFFIX})
    assert kwargs["stop"] == RESPONSE_SUFFIX
