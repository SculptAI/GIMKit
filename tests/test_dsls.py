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
        'm_0[capture, suffix="<|/MASKED|>"]: T_0\n'
        "T_0: /(?s:.*)/\n"
    )
    assert build_cfg(query) == grm

    # Test with regex
    query_with_regex = Query("Hello, ", MaskedTag(id=0, regex=r"\w+\.com"), "!")
    whole_grammar_regex = (
        "%llguidance {}\n"
        'start: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\\"m_0\\"|>" m_0 REGEX "<|/GIM_RESPONSE|>"\n'
        "REGEX: /\\s*/\n"
        'm_0[capture, suffix="<|/MASKED|>"]: T_0\n'
        "T_0: /\\w+\\.com/\n"
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


def test_build_cfg_terminal_reuse():
    """Test that tags with the same regex pattern reuse the same terminal."""
    # Test multiple tags with default pattern (no regex)
    query = Query(
        "Name: ",
        MaskedTag(id=0),
        ", Age: ",
        MaskedTag(id=1),
        ", City: ",
        MaskedTag(id=2),
    )
    grammar = build_cfg(query)

    # All three tags should reference the same terminal T_0
    assert 'm_0[capture, suffix="<|/MASKED|>"]: T_0' in grammar
    assert 'm_1[capture, suffix="<|/MASKED|>"]: T_0' in grammar
    assert 'm_2[capture, suffix="<|/MASKED|>"]: T_0' in grammar

    # Should only have one terminal definition for the default pattern
    assert grammar.count("T_0: /(?s:.*)/") == 1

    # Should not have M_0, M_1, M_2 (old naming scheme)
    assert "M_0:" not in grammar
    assert "M_1:" not in grammar
    assert "M_2:" not in grammar


def test_build_cfg_mixed_terminals():
    """Test that different regex patterns create different terminals but still reuse when possible."""
    query = Query(
        "Email: ",
        MaskedTag(id=0, regex=r"\w+@\w+\.com"),
        ", Name: ",
        MaskedTag(id=1),  # default pattern
        ", Age: ",
        MaskedTag(id=2),  # default pattern
        ", Website: ",
        MaskedTag(id=3, regex=r"\w+\.com"),
        ", Backup Email: ",
        MaskedTag(id=4, regex=r"\w+@\w+\.com"),  # same as id=0
    )
    grammar = build_cfg(query)

    # Email pattern should be T_0
    assert 'm_0[capture, suffix="<|/MASKED|>"]: T_0' in grammar
    assert "T_0: /\\w+@\\w+\\.com/" in grammar

    # Default pattern should be T_1
    assert 'm_1[capture, suffix="<|/MASKED|>"]: T_1' in grammar
    assert 'm_2[capture, suffix="<|/MASKED|>"]: T_1' in grammar
    assert "T_1: /(?s:.*)/" in grammar

    # Website pattern should be T_2
    assert 'm_3[capture, suffix="<|/MASKED|>"]: T_2' in grammar
    assert "T_2: /\\w+\\.com/" in grammar

    # Backup email should reuse T_0 (same pattern as first email)
    assert 'm_4[capture, suffix="<|/MASKED|>"]: T_0' in grammar

    # Verify we have exactly 3 unique terminals
    assert grammar.count("T_0:") == 1
    assert grammar.count("T_1:") == 1
    assert grammar.count("T_2:") == 1
    assert "T_3:" not in grammar  # Should not exist since we reused T_0
