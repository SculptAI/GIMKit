from unittest.mock import MagicMock

from outlines.models.vllm_offline import VLLMOffline as OutlinesVLLMOffline
from vllm import LLM

from gimkit.models.vllm_offline import VLLMOffline as GIMVLLMOffline
from gimkit.models.vllm_offline import from_vllm_offline


def test_from_vllm_offline():
    model = from_vllm_offline(MagicMock(spec=LLM))
    assert type(model) is GIMVLLMOffline
    assert type(model) is not OutlinesVLLMOffline
    assert type(model) is not LLM


def test_vllm_offline_call():
    mock_model = MagicMock(spec=GIMVLLMOffline)

    mock_model("Hi")
