import pytest

from outlines.types.dsl import CFG

from gimkit.contexts import Query
from gimkit.models.utils import build_cfg, ensure_str


def test_build_cfg():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    grm = (
        'start: "<|GIM_RESPONSE|>" tag0 "<|/GIM_RESPONSE|>"\n'
        'tag0: "<|MASKED id=\\"m_0\\"|>" /(?s:.)*?/ "<|/MASKED|>"'
    )
    cfg = build_cfg(query)
    assert isinstance(cfg, CFG)
    assert cfg.definition == grm


def test_ensure_str():
    assert ensure_str("hello") == "hello"

    with pytest.raises(TypeError, match="Response is not a string"):
        ensure_str(123)
