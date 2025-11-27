import pytest

from gimkit.contexts import Query
from gimkit.dsls import (
    build_cfg,
    build_json_schema,
)
from gimkit.schemas import MaskedTag


def test_build_cfg():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    assert build_cfg(query) == (
        'start: "<|GIM_RESPONSE|>" masked_tag_0 "<|/GIM_RESPONSE|>"\n'
        'masked_tag_0: "<|MASKED id=\\"m_0\\"|>" /(?s:.)*?/ "<|/MASKED|>"'
    )

    # Test with regex
    query_with_regex = Query("Hello, ", MaskedTag(id=0, regex="[A-Za-z]{5}"), "!")
    assert build_cfg(query_with_regex) == (
        'start: "<|GIM_RESPONSE|>" masked_tag_0 "<|/GIM_RESPONSE|>"\n'
        'masked_tag_0: "<|MASKED id=\\"m_0\\"|>" /[A-Za-z]{5}/ "<|/MASKED|>"'
    )

    # Test with invalid regex
    with (
        pytest.warns(FutureWarning, match="Possible nested set at position 1"),
        pytest.raises(ValueError, match="Invalid CFG constructed from the query object"),
    ):
        build_cfg(Query(MaskedTag(regex="[[]]")))

    # Test with cfg
    cfg = 'start: obj1 ", " obj2\nobj1: "Hello" | "Hi"\nobj2: "World" | "Everyone"\n'
    query_with_cfg = Query(MaskedTag(id=0, cfg=cfg), "!")
    assert build_cfg(query_with_cfg) == (
        'start: "<|GIM_RESPONSE|>" masked_tag_0 "<|/GIM_RESPONSE|>"\n'
        'masked_tag_0: "<|MASKED id=\\"m_0\\"|>" masked_tag_0_start "<|/MASKED|>"\n'
        'masked_tag_0_start: obj1 ", " obj2\n'
        'obj1: "Hello" | "Hi"\n'
        'obj2: "World" | "Everyone"'
    )


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
                "pattern": "^([a-zA-Z]+)$",
                "description": "user name",
            },
            "m_1": {"type": "string", "description": "user age"},
        },
        "required": ["m_0", "m_1"],
        "additionalProperties": False,
    }
    assert schema == expected_schema
