from typing import ClassVar

import pytest

from gimkit.contexts import Query
from gimkit.dsls import (
    build_cfg,
    build_json_schema,
    get_grammar_spec,
    to_llguidance_regex,
)
from gimkit.schemas import MaskedTag


NO_MAGIC_LINE = (
    r"NO_MAGIC: ~/(?s:.*)(?:<\|GIM_QUERY\|>|<\|\/GIM_QUERY\|>|<\|GIM_RESPONSE\|>"
    r"|<\|\/GIM_RESPONSE\|>|<\|MASKED|<\|\/MASKED\|>)(?s:.*)/"
    "\n"
)


def test_build_cfg():
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    grm = (
        "%llguidance {}\n"
        'start: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\\"m_0\\"|>" m_0 REGEX "<|/GIM_RESPONSE|>"\n'
        "REGEX: /\\s*/\n"
        'm_0[capture, suffix="<|/MASKED|>"]: T_0\n'
        "T_0: /(?s:.*)/ & NO_MAGIC\n" + NO_MAGIC_LINE
    )
    assert build_cfg(query) == grm

    # Test with regex
    query_with_regex = Query("Hello, ", MaskedTag(id=0, regex=r"\w+\.com"), "!")
    whole_grammar_regex = (
        "%llguidance {}\n"
        'start: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\\"m_0\\"|>" m_0 REGEX "<|/GIM_RESPONSE|>"\n'
        "REGEX: /\\s*/\n"
        'm_0[capture, suffix="<|/MASKED|>"]: T_0\n'
        "T_0: /\\w+\\.com/ & NO_MAGIC\n" + NO_MAGIC_LINE
    )
    assert build_cfg(query_with_regex) == whole_grammar_regex

    # Test with invalid regex
    with (
        pytest.warns(FutureWarning, match="Possible nested set at position 1"),
        pytest.raises(ValueError, match="Invalid CFG grammar constructed from the query object"),
    ):
        build_cfg(Query(MaskedTag(regex="[[]]")))

    # Test with various complex patterns including repeated regexes
    query = Query(
        "Date: ",
        MaskedTag(id=0, regex=r"\d{4}-\d{2}-\d{2}"),
        ", AnotherDate: ",
        MaskedTag(id=1, regex=r"\d{4}-\d{2}-\d{2}"),  # same as id=0
        ", Time: ",
        MaskedTag(id=2, regex=r"\d{2}:\d{2}:\d{2}"),
        ", AnotherTime: ",
        MaskedTag(id=3, regex=r"\d{2}:\d{2}:\d{2}"),  # same as id=2
    )
    assert build_cfg(query) == (
        "%llguidance {}\n"
        'start: "<|GIM_RESPONSE|>" REGEX "<|MASKED id=\\"m_0\\"|>" m_0 REGEX "<|MASKED id=\\"m_1\\"|>" m_1 REGEX "<|MASKED id=\\"m_2\\"|>" m_2 REGEX "<|MASKED id=\\"m_3\\"|>" m_3 REGEX "<|/GIM_RESPONSE|>"\n'
        "REGEX: /\\s*/\n"
        'm_0[capture, suffix="<|/MASKED|>"]: T_0\n'
        'm_1[capture, suffix="<|/MASKED|>"]: T_0\n'
        'm_2[capture, suffix="<|/MASKED|>"]: T_1\n'
        'm_3[capture, suffix="<|/MASKED|>"]: T_1\n'
        "T_0: /\\d{4}-\\d{2}-\\d{2}/ & NO_MAGIC\n"
        "T_1: /\\d{2}:\\d{2}:\\d{2}/ & NO_MAGIC\n" + NO_MAGIC_LINE
    )


class _ByteTokenizer:
    """One token per byte, so grammar checks do not depend on a real model's vocabulary."""

    eos_token_id = 256
    bos_token_id = None
    tokens: ClassVar[list[bytes]] = [bytes([i]) for i in range(256)] + [b"<eos>"]
    special_token_ids: ClassVar[list[int]] = [256]

    def __call__(self, s: bytes) -> list[int]:
        return list(s)


def _grammar_accepts(grammar: str, text: str) -> bool:
    from llguidance import LLMatcher, LLTokenizer, TokenizerWrapper

    matcher = LLMatcher(LLTokenizer(TokenizerWrapper(_ByteTokenizer())), get_grammar_spec(grammar))
    tokens = list(text.encode("utf-8"))
    if matcher.validate_tokens(tokens) != len(tokens):
        return False
    matcher.consume_tokens(tokens)
    return matcher.is_accepting()


def test_to_llguidance_regex():
    assert to_llguidance_regex(r"\w+\.com") == r"\w+\.com"
    assert to_llguidance_regex(r"[一-鿿]+\d") == r"[一-鿿]+\d"
    # Python-only escapes become bare literals
    assert to_llguidance_regex(r"[ァ-ヶー]+\（[A-Z]+）") == r"[ァ-ヶー]+（[A-Z]+）"
    assert to_llguidance_regex(r"\<b\>") == r"<b>"
    # `/` would close the lark regex literal
    assert to_llguidance_regex(r"results/[a-z_/]+") == r"results\/[a-z_\/]+"
    assert to_llguidance_regex(r"a\/b") == r"a\/b"
    assert to_llguidance_regex(r"a\\/b") == r"a\\\/b"


def test_build_cfg_slot_cannot_run_past_its_end_tag():
    query = Query(
        "Topic: ",
        MaskedTag(id=0, regex="Robotic .* Leadership"),
        ", keyword: ",
        MaskedTag(id=1, regex="toxic .*"),
    )
    grammar = build_cfg(query)
    valid = (
        '<|GIM_RESPONSE|><|MASKED id="m_0"|>Robotic Servant Leadership<|/MASKED|>\n'
        '<|MASKED id="m_1"|>toxic waste<|/MASKED|><|/GIM_RESPONSE|>'
    )
    assert _grammar_accepts(grammar, valid)
    # m_0 does not satisfy its regex at its end tag. Before the fix, `.*` swallowed
    # "<|/MASKED|>" and every following tag, so m_1 was never constrained.
    escaped = (
        '<|GIM_RESPONSE|><|MASKED id="m_0"|>Robotic Prosthetics<|/MASKED|>'
        '<|MASKED id="m_1"|>prosthetics<|/MASKED|>'
        '<|MASKED id="m_0"|>x Leadership<|/MASKED|><|MASKED id="m_1"|>toxic x<|/MASKED|>'
        "<|/GIM_RESPONSE|>"
    )
    assert not _grammar_accepts(grammar, escaped)
    # Unconstrained slots cannot contain magic strings either.
    free = build_cfg(Query(MaskedTag(id=0)))
    assert _grammar_accepts(
        free, '<|GIM_RESPONSE|><|MASKED id="m_0"|>a |> b<|/MASKED|><|/GIM_RESPONSE|>'
    )
    assert not _grammar_accepts(
        free, '<|GIM_RESPONSE|><|MASKED id="m_0"|>a<|/GIM_RESPONSE|><|/MASKED|><|/GIM_RESPONSE|>'
    )


def test_build_cfg_python_regex_dialect():
    query = Query(
        MaskedTag(id=0, regex=r"[ァ-ヶー]+\（[A-Z]+）"),
        MaskedTag(id=1, regex=r"results/[a-z_/]+"),
    )
    grammar = build_cfg(query)
    assert _grammar_accepts(
        grammar,
        '<|GIM_RESPONSE|><|MASKED id="m_0"|>コンピュータ（CPU）<|/MASKED|>'
        '<|MASKED id="m_1"|>results/run_a/b<|/MASKED|><|/GIM_RESPONSE|>',
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
