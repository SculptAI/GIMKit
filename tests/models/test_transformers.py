from unittest.mock import MagicMock, patch

import pytest

from outlines.models.transformers import Transformers as OutlinesTransformers

from gimkit.contexts import Result
from gimkit.models.transformers import Transformers as GIMTransformers
from gimkit.models.transformers import from_transformers
from gimkit.schemas import MaskedTag


def _make_model_and_tokenizer():
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    mock_model = MagicMock(spec=PreTrainedModel)
    mock_model.device = "cpu"
    mock_model.config = MagicMock()
    mock_model.config.is_encoder_decoder = False

    mock_tokenizer = MagicMock(spec=PreTrainedTokenizerBase)
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.eos_token = "</s>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.pad_token = "</s>"
    mock_tokenizer.all_special_tokens = []
    mock_tokenizer.get_vocab.return_value = {}
    mock_tokenizer.chat_template = None

    return mock_model, mock_tokenizer


def test_from_transformers():
    mock_model, mock_tokenizer = _make_model_and_tokenizer()
    model = from_transformers(mock_model, mock_tokenizer)
    assert type(model) is GIMTransformers
    assert type(model) is not OutlinesTransformers
    assert model.model is mock_model


def test_transformers_call():
    mock_model, mock_tokenizer = _make_model_and_tokenizer()
    model = from_transformers(mock_model, mock_tokenizer)

    with patch("gimkit.models.transformers.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = '<|MASKED id="m_0"|>hi<|/MASKED|>'
        mock_generator.return_value = generator_instance

        returned = model(MaskedTag())
        assert isinstance(returned, Result)
        assert returned.tags[0].content == "hi"


def test_transformers_call_invalid_response():
    mock_model, mock_tokenizer = _make_model_and_tokenizer()
    model = from_transformers(mock_model, mock_tokenizer)

    with patch("gimkit.models.transformers.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = set()
        mock_generator.return_value = generator_instance
        with pytest.raises(TypeError, match="Expected responses to be str or list of str, got"):
            model(MaskedTag())

    with patch("gimkit.models.transformers.Generator") as mock_generator:
        generator_instance = MagicMock()
        generator_instance.return_value = []
        mock_generator.return_value = generator_instance
        with pytest.raises(ValueError, match="Response list is empty"):
            model(MaskedTag())
