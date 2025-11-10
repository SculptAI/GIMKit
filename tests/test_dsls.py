import pytest

from gimkit.contexts import Query
from gimkit.dsls import (
    build_cfg,
    build_json_schema,
)
from gimkit.schemas import MaskedTag


def test_build_cfg():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    grm = (
        'start: "<|GIM_RESPONSE|>" MASKED_TAG_0 "<|/GIM_RESPONSE|>"\n'
        'MASKED_TAG_0: "<|MASKED id=\\"m_0\\"|>" /(?s:.)*?/ "<|/MASKED|>"'
    )
    assert build_cfg(query) == grm

    with (
        pytest.warns(FutureWarning, match="Possible nested set at position 1"),
        pytest.raises(ValueError, match="Invalid CFG grammar constructed from the query object"),
    ):
        build_cfg(Query(MaskedTag(regex="[[]]")))


def test_build_json_schema():
    query = Query(
        "Name: ",
        MaskedTag(id=0, desc="user name", regex="[a-zA-Z]+"),
        ", Age: ",
        MaskedTag(id=1, desc="user age"),
    )
    schema = build_json_schema(query)
    expected_schema = {
        "type": "object",
        "properties": {
            "m_0": {
                "type": "string",
                "pattern": "[a-zA-Z]+",
                "description": "user name",
            },
            "m_1": {"type": "string", "description": "user age"},
        },
        "required": ["m_0", "m_1"],
        "additionalProperties": False,
    }
    assert schema == expected_schema
