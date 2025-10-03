from unittest.mock import MagicMock, patch

from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline
from vllm import LLM

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
        result = model(MaskedTag())
        assert result.tags[0].content == "hi"
        mock_call.assert_called_once_with(model, MaskedTag(), "cfg", None)
