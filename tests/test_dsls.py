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
        "%llguidance {}\n"
        'start: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\\"m_0\\"|>" m_0 REGEX "<|/GIM_RESPONSE|>"\n'
        "REGEX: /\\s*/\n"
        'm_0[capture, suffix="<|/MASKED|>"]: M_0\n'
        "M_0: /(?s:.*)/\n"
    )
    assert build_cfg(query) == grm

    # Test with regex
    query_with_regex = Query("Hello, ", MaskedTag(id=0, regex=r"\w+\.com"), "!")
    whole_grammar_regex = (
        "%llguidance {}\n"
        'start: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\\"m_0\\"|>" m_0 REGEX "<|/GIM_RESPONSE|>"\n'
        "REGEX: /\\s*/\n"
        'm_0[capture, suffix="<|/MASKED|>"]: M_0\n'
        "M_0: /\\w+\\.com/\n"
    )
    assert build_cfg(query_with_regex) == whole_grammar_regex

    # Test with invalid regex
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
                "pattern": "^([a-zA-Z]+)$",
                "description": "user name",
            },
            "m_1": {"type": "string", "description": "user age"},
        },
        "required": ["m_0", "m_1"],
        "additionalProperties": False,
    }
    assert schema == expected_schema
