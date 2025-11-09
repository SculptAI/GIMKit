import pytest

from gimkit.exceptions import InvalidFormatError
from gimkit.parsing import parse_parts, parse_tags, validate
from gimkit.schemas import (
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    MaskedTag,
)


def test_parse_parts_simple():
    """Test parsing a simple string with a masked tag."""
    result = parse_parts('<|MASKED id="m_0"|>content<|/MASKED|>')
    assert len(result) == 1
    assert isinstance(result[0], MaskedTag)
    assert result[0].id == 0
    assert result[0].content == "content"


def test_parse_parts_multiple_tags():
    """Test parsing a string with multiple masked tags."""
    result = parse_parts(
        'Hello <|MASKED id="m_0"|>world<|/MASKED|> and <|MASKED id="m_1"|>universe<|/MASKED|>!'
    )
    assert len(result) == 5
    assert result[0] == "Hello "
    assert isinstance(result[1], MaskedTag)
    assert result[1].id == 0
    assert result[1].content == "world"
    assert result[2] == " and "
    assert isinstance(result[3], MaskedTag)
    assert result[3].id == 1
    assert result[3].content == "universe"
    assert result[4] == "!"


def test_parse_parts_with_attributes():
    """Test parsing tags with various attributes."""
    result = parse_parts(
        '<|MASKED id="m_0" name="test" desc="a test" regex="\\d+"|>123<|/MASKED|>'
    )
    assert len(result) == 1
    assert isinstance(result[0], MaskedTag)
    assert result[0].id == 0
    assert result[0].name == "test"
    assert result[0].desc == "a test"
    assert result[0].regex == r"\d+"
    assert result[0].content == "123"


def test_parse_parts_mismatched_tags():
    """Test that mismatched tags raise an error."""
    with pytest.raises(InvalidFormatError, match="Mismatched or nested masked tags"):
        parse_parts('<|MASKED id="m_0"|>content')


def test_parse_parts_invalid_tag_order():
    """Test that non-sequential tag IDs raise an error."""
    with pytest.raises(InvalidFormatError, match="Tag ids should be in order"):
        parse_parts('<|MASKED id="m_0"|>a<|/MASKED|><|MASKED id="m_2"|>b<|/MASKED|>')


def test_parse_tags_with_prefix_suffix():
    """Test parsing tags with prefix and suffix."""
    tags = parse_tags(
        f'{QUERY_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{QUERY_SUFFIX}',
        QUERY_PREFIX,
        QUERY_SUFFIX,
    )
    assert len(tags) == 1
    assert tags[0].id == 0
    assert tags[0].content == "content"


def test_parse_tags_missing_prefix():
    """Test that missing prefix raises an error."""
    with pytest.raises(InvalidFormatError, match="String must start with"):
        parse_tags('<|MASKED id="m_0"|>content<|/MASKED|>', QUERY_PREFIX, None)


def test_parse_tags_missing_suffix():
    """Test that missing suffix raises an error."""
    with pytest.raises(InvalidFormatError, match="String must end with"):
        parse_tags(
            f'{QUERY_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>', QUERY_PREFIX, QUERY_SUFFIX
        )


def test_parse_tags_duplicate_prefix():
    """Test that duplicate prefix raises an error."""
    with pytest.raises(InvalidFormatError, match=r"Nested or duplicate.*tag are not allowed"):
        parse_tags(
            f'{QUERY_PREFIX}{QUERY_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{QUERY_SUFFIX}',
            QUERY_PREFIX,
            QUERY_SUFFIX,
        )


def test_parse_tags_validates_sequential_ids():
    """Test that tag IDs must be sequential when prefix is provided."""
    with pytest.raises(InvalidFormatError, match="Tag ids should be in order 0, 1, 2"):
        parse_tags(
            f'{QUERY_PREFIX}<|MASKED id="m_1"|>content<|/MASKED|>{QUERY_SUFFIX}',
            QUERY_PREFIX,
            QUERY_SUFFIX,
        )


def test_validate_both_none():
    """Test that validate raises error when both query and response are None."""
    with pytest.raises(ValueError, match="At least one of query or response must be provided"):
        validate(None, None)


def test_validate_query_only():
    """Test validation with only query provided."""
    query = f'{QUERY_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{QUERY_SUFFIX}'
    validate(query, None)  # Should not raise


def test_validate_response_only():
    """Test validation with only response provided."""
    response = f'{RESPONSE_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{RESPONSE_SUFFIX}'
    validate(None, response)  # Should not raise


def test_validate_matching_tags():
    """Test validation with matching query and response tags."""
    query = f'{QUERY_PREFIX}<|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'
    response = f'{RESPONSE_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{RESPONSE_SUFFIX}'
    validate(query, response)  # Should not raise


def test_validate_mismatched_tag_count():
    """Test that mismatched tag counts raise an error."""
    query = f'{QUERY_PREFIX}<|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'
    response = (
        f'{RESPONSE_PREFIX}<|MASKED id="m_0"|>a<|/MASKED|>'
        f'<|MASKED id="m_1"|>b<|/MASKED|>{RESPONSE_SUFFIX}'
    )
    with pytest.raises(InvalidFormatError, match="Mismatched number of masked tags"):
        validate(query, response)
