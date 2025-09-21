import pytest

from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import (
    INPUT_PREFIX,
    INPUT_SUFFIX,
    OUTPUT_PREFIX,
    OUTPUT_SUFFIX,
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    MaskedTag,
    parse_tags,
    validate_wrapped_masked_io,
    validate_wrapped_masked_qr,
)


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


def test_masked_tag_init_invalid():
    with pytest.raises(ValueError, match="should be int or None"):
        MaskedTag(id="0")
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(name=123)
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(desc=123)
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(content=object)
    with pytest.raises(ValueError, match="content should not contain special marks like"):
        MaskedTag(content="<|MASKED|>")


def test_validate_wrapped_masked_io_yes():
    # Valid: simple case
    m_input = '<|M_INPUT|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/M_INPUT|>'
    m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
    validate_wrapped_masked_io(m_input, m_output)

    # Valid: no id in input, id in output
    m_input = "<|M_INPUT|>This is an <|MASKED|><|/MASKED|> text.<|/M_INPUT|>"
    m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
    validate_wrapped_masked_io(m_input, m_output)

    # Valid: empty input and output
    m_input = "<|M_INPUT|><|/M_INPUT|>"
    m_output = "<|M_OUTPUT|><|/M_OUTPUT|>"
    validate_wrapped_masked_io(m_input, m_output)

    # Valid: with whitespaces around
    m_input = '\n<|M_INPUT|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/M_INPUT|>\n\n \t'
    m_output = ' \n<|M_OUTPUT|>\n<|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>\n'
    validate_wrapped_masked_io(m_input, m_output)


def test_validate_wrapped_masked_io_no():
    # Invalid: non-sequential ids
    m_input = '<|M_INPUT|>This is an <|MASKED id="m_1"|><|/MASKED|> text.<|/M_INPUT|>'
    m_output = '<|M_OUTPUT|><|MASKED id="m_1"|>example<|/MASKED|><|/M_OUTPUT|>'
    with pytest.raises(InvalidFormatError, match=r"Tag ids should be in order 0, 1, 2,"):
        validate_wrapped_masked_io(m_input, m_output)

    # Invalid: no prefix and suffix
    m_input = 'This is an <|MASKED id="m_0"|><|/MASKED|> text.'
    m_output = '<|MASKED id="m_0"|>example<|/MASKED|>'
    with pytest.raises(InvalidFormatError, match=r"String must start with the .+ tag\."):
        validate_wrapped_masked_io(m_input, m_output)

    # Invalid: nested tags
    m_input = '<|M_INPUT|>This is an <|MASKED id="m_0"|>nested <|MASKED id="m_1"|><|/MASKED|><|/MASKED|> text.<|/M_INPUT|>'
    m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
    with pytest.raises(InvalidFormatError, match="Mismatched or nested masked tags"):
        validate_wrapped_masked_io(m_input, m_output)

    # Invalid: mismatched tags
    m_input = '<|M_INPUT|>This is an <|MASKED id="m_0"|><|/MASKED|><|/MASKED|> text.<|/M_INPUT|>'
    m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
    with pytest.raises(InvalidFormatError, match="Mismatched or nested masked tags"):
        validate_wrapped_masked_io(m_input, m_output)


def test_parse_tags():
    # Masked tag has desc
    parse_tags(
        INPUT_PREFIX + """<|MASKED id="m_0" desc="xx"|><|/MASKED|>""" + INPUT_SUFFIX,
        INPUT_PREFIX,
        INPUT_SUFFIX,
    )

    with pytest.raises(InvalidFormatError, match=f"String must start with the {INPUT_PREFIX} tag"):
        parse_tags("no prefix", INPUT_PREFIX, INPUT_SUFFIX)
    with pytest.raises(InvalidFormatError, match=r"Nested or duplicate .+ tag are not allowed."):
        parse_tags(f"{INPUT_PREFIX}example{INPUT_PREFIX}", INPUT_PREFIX, INPUT_SUFFIX)

    with pytest.raises(InvalidFormatError, match=f"String must end with the {INPUT_SUFFIX} tag"):
        parse_tags("no suffix", None, INPUT_SUFFIX)
    with pytest.raises(InvalidFormatError, match=r"Nested or duplicate .+ tag are not allowed."):
        parse_tags(f"{INPUT_SUFFIX}{INPUT_SUFFIX}", None, INPUT_SUFFIX)

    with pytest.raises(
        InvalidFormatError, match=r"Tag ids should be in order, got \d+ at position \d+\."
    ):
        parse_tags(
            f'{INPUT_PREFIX}<|MASKED id="m_0"|><|/MASKED|><|MASKED id="m_2"|><|/MASKED|>{INPUT_SUFFIX}',
            INPUT_PREFIX,
            INPUT_SUFFIX,
        )


def test_validate_wrapped_masked_io():
    with pytest.raises(ValueError):
        validate_wrapped_masked_io(None, None)

    m_input = f'{INPUT_PREFIX}<|MASKED id="m_0"|><|/MASKED|>{INPUT_SUFFIX}'
    m_output_mismatch = f'{OUTPUT_PREFIX}<|MASKED id="m_0"|><|/MASKED|><|MASKED id="m_1"|><|/MASKED|>{OUTPUT_SUFFIX}'
    with pytest.raises(
        InvalidFormatError, match=r"Mismatched number of masked tags between input and output."
    ):
        validate_wrapped_masked_io(m_input, m_output_mismatch)


def test_validate_wrapped_masked_qr_yes():
    # Valid: simple case with new query/response format
    m_query = '<|M_QUERY|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/M_QUERY|>'
    m_response = '<|M_RESPONSE|><|MASKED id="m_0"|>example<|/MASKED|><|/M_RESPONSE|>'
    validate_wrapped_masked_qr(m_query, m_response)

    # Valid: no id in query, id in response
    m_query = "<|M_QUERY|>This is an <|MASKED|><|/MASKED|> text.<|/M_QUERY|>"
    m_response = '<|M_RESPONSE|><|MASKED id="m_0"|>example<|/MASKED|><|/M_RESPONSE|>'
    validate_wrapped_masked_qr(m_query, m_response)

    # Valid: empty query and response
    m_query = "<|M_QUERY|><|/M_QUERY|>"
    m_response = "<|M_RESPONSE|><|/M_RESPONSE|>"
    validate_wrapped_masked_qr(m_query, m_response)


def test_validate_wrapped_masked_qr_no():
    # Invalid: mismatched number of tags between query and response
    m_query = f'{QUERY_PREFIX}<|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'
    m_response_mismatch = f'{RESPONSE_PREFIX}<|MASKED id="m_0"|><|/MASKED|><|MASKED id="m_1"|><|/MASKED|>{RESPONSE_SUFFIX}'
    with pytest.raises(
        InvalidFormatError, match=r"Mismatched number of masked tags between query and response."
    ):
        validate_wrapped_masked_qr(m_query, m_response_mismatch)

    with pytest.raises(ValueError):
        validate_wrapped_masked_qr(None, None)
