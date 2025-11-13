import re

from dataclasses import fields

import pytest

from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import (
    ALL_ATTRS,
    ALL_FIELDS,
    COMMON_ATTRS,
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_END_PATTERN,
    TAG_FULL_PATTERN,
    TAG_OPEN_PATTERN,
    MaskedTag,
    TagField,
    parse_tags,
    validate,
)


def test_global_variables():
    assert COMMON_ATTRS == ("name", "desc", "regex")
    assert ALL_ATTRS == ("id", "name", "desc", "regex")
    assert ALL_FIELDS == ("id", "name", "desc", "regex", "content")
    assert tuple(f.name for f in fields(MaskedTag)) == ALL_FIELDS
    assert len(set(ALL_FIELDS)) == len(ALL_FIELDS)
    assert TagField.__args__ == ("id", "name", "desc", "regex", "content")


def test_regex_patterns():
    # Test TAG_OPEN_PATTERN
    match = re.fullmatch(TAG_OPEN_PATTERN, '<|MASKED id="m_0" name="test" desc="example"|>')
    assert match is not None
    assert match.group("id") == "0"
    assert match.group("name") == "test"
    assert match.group("desc") == "example"

    # Test TAG_END_PATTERN
    match = re.fullmatch(TAG_END_PATTERN, "<|/MASKED|>")
    assert match is not None

    # Test TAG_FULL_PATTERN
    test_string = '<|MASKED id="m_1" desc="sample" regex="[a-zA-Z]+"|>content here<|/MASKED|>'
    match = re.fullmatch(TAG_FULL_PATTERN, test_string)
    assert match is not None
    assert match.group("id") == "1"
    assert match.group("desc") == "sample"
    assert match.group("regex") == "[a-zA-Z]+"
    assert match.group("content") == "content here"


def test_masked_tag_init_invalid():
    with pytest.raises(ValueError, match="should be int, str of digits, or None"):
        MaskedTag(id="0=")
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(name=123)
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(desc=123)
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(content=object)
    with pytest.raises(
        ValueError,
        match=re.escape(
            "content should not contain special marks like `<|GIM_QUERY|>` or "
            "`<|/GIM_QUERY|>` or `<|GIM_RESPONSE|>` or `<|/GIM_RESPONSE|>` or "
            "`<|MASKED` or `<|/MASKED|>`"
        ),
    ):
        MaskedTag(content="<|MASKED|>")


def test_masked_tag_init_with_regex():
    with pytest.raises(ValueError, match="regex should not be an empty string"):
        MaskedTag(regex="")
    with pytest.raises(ValueError, match="regex should not start or end with /"):
        MaskedTag(regex="/abc/")
    with pytest.raises(ValueError, match="Invalid regex pattern"):
        MaskedTag(regex="[")


def test_masked_tag_attr_escape():
    original = "& < > \" ' \t \n \r"
    escaped = MaskedTag.attr_escape(original)
    unescaped = MaskedTag.attr_unescape(escaped)
    assert escaped == "&amp; &lt; &gt; &quot; &#x27; &#x09; &#x0a; &#x0d;"
    assert original == unescaped

    assert MaskedTag.attr_escape("&&") == "&amp;&amp;"
    assert MaskedTag.attr_unescape("&amp;#x09;") == "&#x09;"


def test_masked_tag_str():
    assert str(MaskedTag(id=0)) == '<|MASKED id="m_0"|><|/MASKED|>'
    assert (
        str(MaskedTag(id=0, desc="description"))
        == '<|MASKED id="m_0" desc="description"|><|/MASKED|>'
    )
    assert (
        str(MaskedTag(id=0, desc='desc with "quotes"'))
        == '<|MASKED id="m_0" desc="desc with &quot;quotes&quot;"|><|/MASKED|>'
    )
    assert str(MaskedTag(id=0, content="content")) == '<|MASKED id="m_0"|>content<|/MASKED|>'
    assert str(MaskedTag()) == "<|MASKED|><|/MASKED|>"
    assert str(MaskedTag(name="content")) == '<|MASKED name="content"|><|/MASKED|>'


def test_masked_tag_repr():
    assert repr(MaskedTag(id=0)) == '<|MASKED id="m_0"|><|/MASKED|>'
    assert repr(MaskedTag(id=0, content="content")) == '<|MASKED id="m_0"|>content<|/MASKED|>'


def test_parse_tags_valid():
    # Masked tag has desc
    tags = parse_tags(
        QUERY_PREFIX + """<|MASKED id="m_0" desc="xx"|><|/MASKED|>""" + QUERY_SUFFIX,
        QUERY_PREFIX,
        QUERY_SUFFIX,
    )
    assert tags[0].id == 0
    assert tags[0].desc == "xx"
    assert tags[0].name is None
    assert tags[0].content == ""

    # Without prefix. The id can start from any non-negative integer
    tags = parse_tags("""<|MASKED|><|/MASKED|><|MASKED id="m_3"|><|/MASKED|>""", None, None)
    assert tags[0].id is None
    assert tags[1].id == 3

    # Some masked tags have id, some don't
    tags = parse_tags(
        """<|MASKED id="m_0"|><|/MASKED|><|MASKED|><|/MASKED|><|MASKED id="m_2"|><|/MASKED|>""",
        None,
        None,
    )
    assert len(tags) == 3
    assert tags[1].id is None

    # With special mark
    tags = parse_tags("<|MASKED|>|><|/MASKED|>")
    assert tags[0].content == "|>"

    # A tricky example: the value in desc cannot contain a double quote,
    # so it will non-greedily match the first quote.
    tags = parse_tags('<|MASKED desc="xxx"|>"|><|/MASKED|>')
    assert tags[0].desc == "xxx"
    assert tags[0].content == '"|>'


def test_parse_tags_invalid():
    with pytest.raises(InvalidFormatError, match=f"String must start with the {QUERY_PREFIX} tag"):
        parse_tags("no prefix", QUERY_PREFIX, QUERY_SUFFIX)
    with pytest.raises(InvalidFormatError, match=r"Nested or duplicate .+ tag are not allowed."):
        parse_tags(f"{QUERY_PREFIX}example{QUERY_PREFIX}", QUERY_PREFIX, QUERY_SUFFIX)

    with pytest.raises(InvalidFormatError, match=f"String must end with the {QUERY_SUFFIX} tag"):
        parse_tags("no suffix", None, QUERY_SUFFIX)
    with pytest.raises(InvalidFormatError, match=r"Nested or duplicate .+ tag are not allowed."):
        parse_tags(f"{QUERY_SUFFIX}{QUERY_SUFFIX}", None, QUERY_SUFFIX)

    with pytest.raises(
        InvalidFormatError, match=r"Tag ids should be in order, got \d+ at position \d+\."
    ):
        parse_tags(
            f'{QUERY_PREFIX}<|MASKED id="m_0"|><|/MASKED|><|MASKED id="m_2"|><|/MASKED|>{QUERY_SUFFIX}',
            QUERY_PREFIX,
            QUERY_SUFFIX,
        )


def test_validate_wrapped_masked_io_yes():
    # Valid: simple case
    query = '<|GIM_QUERY|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/GIM_QUERY|>'
    response = '<|GIM_RESPONSE|><|MASKED id="m_0"|>example<|/MASKED|><|/GIM_RESPONSE|>'
    validate(query, response)

    # Valid: no id in query, id in response
    query = "<|GIM_QUERY|>This is an <|MASKED|><|/MASKED|> text.<|/GIM_QUERY|>"
    response = '<|GIM_RESPONSE|><|MASKED id="m_0"|>example<|/MASKED|><|/GIM_RESPONSE|>'
    validate(query, response)

    # Valid: empty query and response
    query = "<|GIM_QUERY|><|/GIM_QUERY|>"
    response = "<|GIM_RESPONSE|><|/GIM_RESPONSE|>"
    validate(query, response)

    # Valid: with whitespaces around
    query = '\n<|GIM_QUERY|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/GIM_QUERY|>\n\n \t'
    response = ' \n<|GIM_RESPONSE|>\n<|MASKED id="m_0"|>example<|/MASKED|><|/GIM_RESPONSE|>\n'
    validate(query, response)

    # Valid: only query or response
    validate(query, None)
    validate(None, response)

    # Valid: first tag has no id, second has id
    query = '\n<|GIM_QUERY|>This is an <|MASKED|><|/MASKED|><|MASKED id="m_1"|><|/MASKED|> text.<|/GIM_QUERY|>\n\n \t'
    response = " \n<|GIM_RESPONSE|>\n<|MASKED|>example<|/MASKED|><|MASKED|>example<|/MASKED|><|/GIM_RESPONSE|>\n"
    validate(query, response)


def test_validate_wrapped_masked_io_no():
    # Invalid: both None
    with pytest.raises(ValueError):
        validate(None, None)

    # Invalid: non-sequential ids
    query = '<|GIM_QUERY|>This is an <|MASKED id="m_1"|><|/MASKED|> text.<|/GIM_QUERY|>'
    response = '<|GIM_RESPONSE|><|MASKED id="m_1"|>example<|/MASKED|><|/GIM_RESPONSE|>'
    with pytest.raises(InvalidFormatError, match=r"Tag ids should be in order 0, 1, 2,"):
        validate(query, response)

    # Invalid: no prefix and suffix
    query = 'This is an <|MASKED id="m_0"|><|/MASKED|> text.'
    response = '<|MASKED id="m_0"|>example<|/MASKED|>'
    with pytest.raises(InvalidFormatError, match=r"String must start with the .+ tag\."):
        validate(query, response)

    # Invalid: nested tags
    query = '<|GIM_QUERY|>This is an <|MASKED id="m_0"|>nested <|MASKED id="m_1"|><|/MASKED|><|/MASKED|> text.<|/GIM_QUERY|>'
    response = '<|GIM_RESPONSE|><|MASKED id="m_0"|>example<|/MASKED|><|/GIM_RESPONSE|>'
    with pytest.raises(InvalidFormatError, match="Mismatched or nested masked tags"):
        validate(query, response)

    # Invalid: mismatched tags
    query = '<|GIM_QUERY|>This is an <|MASKED id="m_0"|><|/MASKED|><|/MASKED|> text.<|/GIM_QUERY|>'
    response = '<|GIM_RESPONSE|><|MASKED id="m_0"|>example<|/MASKED|><|/GIM_RESPONSE|>'
    with pytest.raises(InvalidFormatError, match="Mismatched or nested masked tags"):
        validate(query, response)

    query = f'{QUERY_PREFIX}<|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'
    response = f'{RESPONSE_PREFIX}<|MASKED id="m_0"|><|/MASKED|><|MASKED id="m_1"|><|/MASKED|>{RESPONSE_SUFFIX}'
    with pytest.raises(
        InvalidFormatError, match=r"Mismatched number of masked tags between query and response"
    ):
        validate(query, response)
