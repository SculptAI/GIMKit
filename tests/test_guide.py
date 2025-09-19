import pytest

from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import (
    INPUT_PREFIX,
    INPUT_SUFFIX,
    OUTPUT_PREFIX,
    OUTPUT_SUFFIX,
    guide,
)


def test_guide_call():
    g = guide()
    tag1 = g(name="first")
    assert tag1.id == 0
    assert tag1.name == "first"

    with pytest.raises(ValueError, match=r"Tag name .+ already exists."):
        g(name="first")  # duplicate name

    raw_query = "query"
    std_query = g.standardize(raw_query)
    assert std_query == f"{INPUT_PREFIX}query{INPUT_SUFFIX}"


def test_guide_parse():
    g = guide()

    # An invalid case
    query = g.standardize("invalid_str")
    response = f'{OUTPUT_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{OUTPUT_SUFFIX}'
    with pytest.raises(
        InvalidFormatError, match=r"Mismatched number of masked tags between input and output."
    ):
        g.parse(query, response)

    # Another invalid case
    query = INPUT_PREFIX + g.standardize("invalid_str")
    response = f'{OUTPUT_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{OUTPUT_SUFFIX}'
    with pytest.raises(InvalidFormatError, match=r"Nested .+ tags are not allowed."):
        g.parse(query, response)

    # A valid case
    query = f"{INPUT_PREFIX}Hello {g(name='obj')}{INPUT_SUFFIX}"
    result = g.parse(query, response)
    another_g = guide()
    assert result.tags._tags == [another_g(name="obj", content="content")]


def test_guide_prompt():
    # Test format string
    g = guide()
    prompt = f"""I'm {g(name="sub")}!"""
    assert prompt == """I'm <|MASKED id="m_0"|><|/MASKED|>!"""

    # Test __add__ and __radd__
    g = guide()
    prompt = g(desc="number") + " + " + g(desc="number") + " = 2"
    assert (
        prompt
        == """<|MASKED id="m_0" desc="number"|><|/MASKED|> + <|MASKED id="m_1" desc="number"|><|/MASKED|> = 2"""
    )

    # Test other types. No error should be raised.
    g = guide()
    g() + object
    object + g()
