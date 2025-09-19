import pytest

from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import (
    INPUT_PREFIX,
    INPUT_SUFFIX,
    OUTPUT_PREFIX,
    OUTPUT_SUFFIX,
    MaskedTag,
    ParsedResult,
    parse_inp_or_outp,
    validate_wrapped_masked_io,
)


def test_some_input_output():
    # Valid: simple case
    m_input = '<|M_INPUT|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/M_INPUT|>'
    m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
    validate_wrapped_masked_io(m_input, m_output)

    # Invalid: non-sequential ids
    m_input = '<|M_INPUT|>This is an <|MASKED id="m_1"|><|/MASKED|> text.<|/M_INPUT|>'
    m_output = '<|M_OUTPUT|><|MASKED id="m_1"|>example<|/MASKED|><|/M_OUTPUT|>'
    with pytest.raises(InvalidFormatError, match=r"Tag ids should be in order 0, 1, 2, ..."):
        validate_wrapped_masked_io(m_input, m_output)

    # Valid: no id in input, id in output
    m_input = "<|M_INPUT|>This is an <|MASKED|><|/MASKED|> text.<|/M_INPUT|>"
    m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
    validate_wrapped_masked_io(m_input, m_output)

    # Valid: empty input and output
    m_input = "<|M_INPUT|><|/M_INPUT|>"
    m_output = "<|M_OUTPUT|><|/M_OUTPUT|>"
    validate_wrapped_masked_io(m_input, m_output)

    # Invalid: no prefix and suffix
    m_input = 'This is an <|MASKED id="m_0"|><|/MASKED|> text.'
    m_output = '<|MASKED id="m_0"|>example<|/MASKED|>'
    with pytest.raises(InvalidFormatError, match=r"Missing.+tags"):
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


def test_infill():
    query = f"{INPUT_PREFIX}Hello {MaskedTag(id=0, name='obj')}{INPUT_SUFFIX}"
    tags = [MaskedTag(id=0, name="obj", content="World")]
    result = ParsedResult(query, tags)
    assert result.infill() == "Hello World"

    tags_no_content = [MaskedTag(id=0, name="obj")]
    result_no_content = ParsedResult(query, tags_no_content)
    with pytest.raises(ValueError, match="no content to infill"):
        result_no_content.infill()


def test_parse_inp_or_outp():
    # Masked tag has desc
    parse_inp_or_outp(
        INPUT_PREFIX + """<|MASKED id="m_0" desc="xx\\"x"|><|/MASKED|>""" + INPUT_SUFFIX,
        INPUT_PREFIX,
        INPUT_SUFFIX,
    )

    with pytest.raises(InvalidFormatError, match=r"Missing.+tags"):
        parse_inp_or_outp("no prefix", INPUT_PREFIX, INPUT_SUFFIX)
    with pytest.raises(InvalidFormatError, match=r"Missing.+tags"):
        parse_inp_or_outp(f"{INPUT_PREFIX}example{INPUT_PREFIX}", INPUT_PREFIX, INPUT_SUFFIX)
    with pytest.raises(InvalidFormatError, match=r"Tag ids should be in order 0, 1, 2, ..."):
        parse_inp_or_outp(
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
