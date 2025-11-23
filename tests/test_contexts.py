import pytest

from gimkit.contexts import Context, Query, Response, Result, infill
from gimkit.exceptions import InvalidFormatError
from gimkit.guides import guide as g
from gimkit.schemas import QUERY_PREFIX, QUERY_SUFFIX, RESPONSE_PREFIX, RESPONSE_SUFFIX, MaskedTag


def test_context_to_str_valid():
    query = Query(f"Hello, {g(name='obj')}")
    assert str(query) == f'{QUERY_PREFIX}Hello, <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'

    response = Response(f"{MaskedTag(id=0, content='world')}")
    assert str(response) == f'{RESPONSE_PREFIX}<|MASKED id="m_0"|>world<|/MASKED|>{RESPONSE_SUFFIX}'

    result = Result(f"Hello, {MaskedTag(id=0, name='obj', content='world')}")
    assert str(result) == "Hello, world"

    text = Context("prefix", "suffix", "Hello", g(name="xx", content=", world")).to_string(
        infill_mode=True
    )
    assert text == "Hello, world"

    text = Context("prefix", "suffix", "Hello", g(name="xx", content=", world")).to_string(
        fields="all"
    )
    assert text == 'prefixHello<|MASKED name="xx"|>, world<|/MASKED|>suffix'

    text = Context("", "", "Hello", g(content=", world")).to_string(infill_mode=True)
    assert text == "Hello, world"

    text = Context("p", "s", "Hello", g(name="obj", content=", world"))
    assert repr(text) == 'pHello<|MASKED name="obj"|>, world<|/MASKED|>s'


def test_context_to_str_invalid():
    with pytest.raises(ValueError, match="Exactly one of fields or infill_mode must be specified"):
        Context("prefix", "suffix", "Hello").to_string()

    with pytest.raises(ValueError, match="Exactly one of fields or infill_mode must be specified"):
        Context("prefix", "suffix", "Hello").to_string(fields="all", infill_mode=True)


def test_query_init():
    raw_query1 = f"Hello, {g(desc='world', name='obj')}"
    raw_query2 = 'Hello, <|MASKED id="m_0" name="obj" desc="world"|><|/MASKED|>'
    query1 = Query(raw_query1)
    query2 = Query(raw_query2)
    assert query1.parts == query2.parts
    assert query1.tags[:] == query2.tags[:]
    assert query1.tags[0] == MaskedTag(id=0, desc="world", name="obj")
    assert query1.parts == [
        QUERY_PREFIX,
        "Hello, ",
        MaskedTag(id=0, name="obj", desc="world"),
        QUERY_SUFFIX,
    ]
    assert str(query1) == str(query2)


def test_query_init_invalid():
    with pytest.raises(InvalidFormatError, match=r"Tag name '.+' already exists."):
        Query(f"Hello, {g(name='dup_obj')}{g(name='dup_obj')}")

    with pytest.raises(InvalidFormatError, match=r"Tag ids must be sequential starting from 0."):
        Query(MaskedTag(id=2))

    with pytest.raises(TypeError, match=r"List items must be str or MaskedTag"):
        Query([MaskedTag(), 123])

    with pytest.raises(
        TypeError, match=r"Arguments must be str, MaskedTag, or list of str/MaskedTag"
    ):
        Query(123)

    with pytest.raises(
        InvalidFormatError, match=r"Nested or duplicate <\|GIM_QUERY\|> tags are not allowed"
    ):
        Query("<|GIM_QUERY|><|GIM_QUERY|>Hello<|/GIM_QUERY|>")

    with pytest.raises(
        InvalidFormatError, match=r"Nested or duplicate <\|/GIM_QUERY\|> tags are not allowed"
    ):
        Query("<|GIM_QUERY|>Hello<|/GIM_QUERY|><|/GIM_QUERY|>")


def test_query_infill_different_types():
    query = Query(f"Hello, {g(name='obj')}")
    assert str(query.infill(Response(g(name="obj", content="world")))) == "Hello, world"
    assert str(query.infill([MaskedTag(id=0, name="obj", content="world")])) == "Hello, world"
    assert str(Result(f"Hello, {g(content='world')}")) == str(
        query.infill("Hello, " + g(name="obj", content="world"))
    )


def test_query_infill_invalid():
    with pytest.warns(UserWarning, match=r"Mismatch in number of tags between query and response"):
        Query(g(name="obj")).infill("")

    with pytest.warns(UserWarning, match=r"Mismatch in number of tags between query and response"):
        Query("string").infill(
            f'{RESPONSE_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{RESPONSE_SUFFIX}'
        )

    with pytest.raises(InvalidFormatError, match=r"Mismatched or nested masked tags in .+"):
        Query("<|MASKED|>string").infill(f"{RESPONSE_PREFIX}{RESPONSE_SUFFIX}")

    # With the repair functionality, malformed tag IDs are now repaired with a warning
    # instead of raising an error (when strict=False, which is the default)
    with pytest.warns(
        UserWarning, match=r"Response has malformed tag IDs\. Attempting to repair\."
    ):
        result = Query(g(), g()).infill(
            '<|MASKED id="m_2"|><|/MASKED|><|MASKED id="m_4"|><|/MASKED|>'
        )
        # Verify the result is valid after repair
        assert isinstance(result, Result)

    with pytest.raises(
        TypeError, match=r"Arguments must be str, MaskedTag, or list of str/MaskedTag\. Got .+"
    ):
        Query(g()).infill(123)


def test_response_init():
    r = Response(f"{RESPONSE_PREFIX}{g(name='obj', content='world')}{RESPONSE_SUFFIX}")
    assert str(r) == "<|GIM_RESPONSE|><|MASKED|>world<|/MASKED|><|/GIM_RESPONSE|>"


def test_response_infill():
    response = Response(g(content="world"))
    assert str(response.infill(["Hello, ", g()])) == "Hello, world"


def test_result_tags():
    tag1 = MaskedTag(id=0, name="obj1", content="world")
    tag2 = MaskedTag(id=1, name="obj2", content="universe")
    result = Result(f"Hello, {tag1}", "and ", tag2)
    tags = result.tags

    assert len(tags) == 2
    assert tags[0] == tag1
    assert tags["obj2"] == tag2
    assert tags[0:1] == [tag1]
    assert list(tags) == [tag1, tag2]

    with pytest.raises(TypeError, match="Key must be int, slice, or str"):
        tags[None]

    with pytest.raises(KeyError, match=r"Tag name '.+' does not exist."):
        tags["non_exist"]


def test_result_tags_modify():
    tag1 = MaskedTag(id=0, name="obj1", content="world")
    tag2 = MaskedTag(id=1, name="obj2", content="universe")
    result = Result(f"Hello, {tag1}", "and ", tag2)
    tags = result.tags

    tags[0] = "mars "
    assert len(tags) == 1
    assert str(result) == f"Hello, mars and {tag2.content}"

    tags["obj2"].content = "milky way"
    assert str(result) == "Hello, mars and milky way"

    tags["obj2"] = MaskedTag(name="obj3", content="galaxy")
    assert len(tags) == 1
    assert str(result) == "Hello, mars and galaxy"

    with pytest.raises(TypeError, match=r"New value must be a ContextPart \(str or MaskedTag\)"):
        tags[0] = 123

    with pytest.raises(KeyError, match=r"Tag name '.+' does not exist."):
        tags["non_exist"] = "new value"

    with pytest.raises(IndexError, match="list index out of range"):
        tags[10] = "new value"

    with pytest.raises(TypeError, match="Key must be int or str"):
        tags[None] = "new value"


def test_infill_strict():
    query = Query(f"Hello, {g(name='obj1')}{g(name='obj2')}")
    response = Response(g(name="obj1", content="world"))

    with pytest.warns(UserWarning, match=r"Mismatch in number of tags between query and response"):
        result = infill(query, response, strict=False)
        assert str(result) == 'Hello, world<|MASKED id="m_1" name="obj2"|><|/MASKED|>'

    with pytest.raises(
        InvalidFormatError, match=r"Mismatch in number of tags between query and response"
    ):
        infill(query, response, strict=True)


def test_infill_repair_malformed_ids():
    """Test that infill can repair responses with malformed tag IDs when strict=False."""
    query = Query(f"Hello, {g(name='name')} {g(name='greeting')}")
    response_str = (
        f'{RESPONSE_PREFIX}<|MASKED id="m_5"|>John<|/MASKED|>'
        f'<|MASKED id="m_10"|>world<|/MASKED|>{RESPONSE_SUFFIX}'
    )

    # With strict=False, malformed IDs should be repaired
    with pytest.warns(
        UserWarning, match=r"Response has malformed tag IDs\. Attempting to repair\."
    ):
        result = infill(query, response_str, strict=False)
        assert str(result) == "Hello, John world"

    # With strict=True, should raise an error
    with pytest.raises(InvalidFormatError, match=r"Tag ids should be in order"):
        infill(query, response_str, strict=True)


def test_infill_repair_skipped_ids():
    """Test repairing responses with non-sequential IDs."""
    query = Query(f"A {g()} B {g()} C {g()}")
    response_str = (
        f'{RESPONSE_PREFIX}<|MASKED id="m_0"|>one<|/MASKED|>'
        f'<|MASKED id="m_2"|>two<|/MASKED|>'
        f'<|MASKED id="m_5"|>three<|/MASKED|>{RESPONSE_SUFFIX}'
    )

    with pytest.warns(UserWarning, match=r"Response has malformed tag IDs"):
        result = infill(query, response_str, strict=False)
        assert str(result) == "A one B two C three"


def test_infill_repair_with_attributes():
    """Test that repair preserves other tag attributes."""
    query = Query(f"Hello, {g(name='obj1')} {g(name='obj2')}")
    response_str = (
        f'{RESPONSE_PREFIX}<|MASKED id="m_10" name="x"|>John<|/MASKED|>'
        f'<|MASKED id="m_20" name="y"|>world<|/MASKED|>{RESPONSE_SUFFIX}'
    )

    with pytest.warns(UserWarning, match=r"Response has malformed tag IDs"):
        result = infill(query, response_str, strict=False)
        assert str(result) == "Hello, John world"
        # Check that the result tags have the correct attributes from the query
        assert result.tags[0].name == "obj1"
        assert result.tags[1].name == "obj2"


def test_infill_repair_response_object():
    """Test that repair is not triggered when passing a Response object."""
    query = Query(f"Hello, {g()} {g()}")
    # Create a valid Response object
    response = Response(f"{g(content='John')}{g(content='world')}")

    # Should work without warnings since Response is already parsed
    result = infill(query, response, strict=False)
    assert str(result) == "Hello, John world"


def test_infill_repair_no_tags():
    """Test repair with empty response (no tags)."""
    query = Query(f"Hello, {g()}")
    # Response with no tags - should not trigger repair
    response_str = f"{RESPONSE_PREFIX}{RESPONSE_SUFFIX}"

    with pytest.warns(UserWarning, match=r"Mismatch in number of tags"):
        result = infill(query, response_str, strict=False)
        # Query tag should remain unfilled
        assert "<|MASKED" in str(result)


def test_infill_repair_fails():
    """Test that if repair fails, the original error is re-raised."""
    query = Query(f"Hello, {g()}")
    # Create a response with nested tags (which cannot be parsed even after repair)
    response_str = (
        f"{RESPONSE_PREFIX}<|MASKED|><|MASKED|>test<|/MASKED|><|/MASKED|>{RESPONSE_SUFFIX}"
    )

    with pytest.raises(InvalidFormatError, match=r"Mismatched or nested masked tags"):
        infill(query, response_str, strict=False)
