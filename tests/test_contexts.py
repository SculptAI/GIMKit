import pytest

from gimkit.contexts import Context, Query, Response
from gimkit.exceptions import InvalidFormatError
from gimkit.guides import guide as g
from gimkit.schemas import QUERY_PREFIX, QUERY_SUFFIX, RESPONSE_PREFIX, RESPONSE_SUFFIX, MaskedTag


def test_context_to_str_valid():
    query = Query(f"Hello, {g(name='obj')}")
    assert str(query) == f'{QUERY_PREFIX}Hello, <|MASKED id="m_0"|><|/MASKED|>{QUERY_SUFFIX}'

    response = Response(f"Hello, {g(name='obj', content='world')}")
    assert str(response) == "Hello, world"

    text = Context("prefix", "suffix", "Hello", g(name="xx", content=", world")).to_string(
        infill_mode=True
    )
    assert text == "Hello, world"

    text = Context("prefix", "suffix", "Hello", g(name="xx", content=", world")).to_string(
        fields="all"
    )
    assert text == 'prefixHello<|MASKED name="xx"|>, world<|/MASKED|>suffix'


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


def test_query_infill_different_types():
    query = Query(f"Hello, {g(name='obj')}")
    assert str(query.infill(Response(g(name="obj", content="world")))) == "Hello, world"
    assert str(query.infill([MaskedTag(id=0, name="obj", content="world")])) == "Hello, world"
    assert str(Response(f"Hello, {g(content='world')}")) == str(
        query.infill("Hello, " + g(name="obj", content="world"))
    )


def test_query_infill_invalid():
    with pytest.warns(UserWarning, match=r"Not enough tags in response to fill the query tags"):
        assert str(Query(g(name="obj")).infill("")) == '<|MASKED id="m_0" name="obj"|><|/MASKED|>'

    with pytest.warns(UserWarning, match=r"There are \d+ unused tags in the response."):
        assert (
            str(
                Query("string").infill(
                    f'{RESPONSE_PREFIX}<|MASKED id="m_0"|>content<|/MASKED|>{RESPONSE_SUFFIX}'
                )
            )
            == "string"
        )

    with pytest.raises(InvalidFormatError, match=r"Mismatched or nested masked tags in .+"):
        Query("<|MASKED|>string").infill(f"{RESPONSE_PREFIX}{RESPONSE_SUFFIX}")

    with pytest.raises(
        InvalidFormatError, match=r"Tag ids should be in order, got \d+ at position \d+"
    ):
        Query(g(), g()).infill('<|MASKED id="m_2"|><|/MASKED|><|MASKED id="m_4"|><|/MASKED|>')

    with pytest.raises(TypeError, match=r"Response must be str, Response, or list of MaskedTag"):
        Query(g()).infill(123)


def test_response_tags():
    tag1 = MaskedTag(id=0, name="obj1", content="world")
    tag2 = MaskedTag(id=1, name="obj2", content="universe")
    rsps = Response(f"Hello, {tag1}", "and ", tag2)
    tags = rsps.tags

    assert len(tags) == 2
    assert tags[0] == tag1
    assert tags["obj2"] == tag2
    assert tags[0:1] == [tag1]
    assert list(tags) == [tag1, tag2]

    with pytest.raises(TypeError, match="Key must be int, slice, or str"):
        tags[None]

    with pytest.raises(KeyError, match=r"Tag name '.+' does not exist."):
        tags["non_exist"]


def test_response_tags_modify():
    tag1 = MaskedTag(id=0, name="obj1", content="world")
    tag2 = MaskedTag(id=1, name="obj2", content="universe")
    rsps = Response(f"Hello, {tag1}", "and ", tag2)
    tags = rsps.tags

    tags[0] = "mars "
    assert len(tags) == 1
    assert str(rsps) == f"Hello, mars and {tag2.content}"

    tags["obj2"].content = "milky way"
    assert str(rsps) == "Hello, mars and milky way"

    tags["obj2"] = MaskedTag(name="obj3", content="galaxy")
    assert len(tags) == 1
    assert str(rsps) == "Hello, mars and galaxy"

    with pytest.raises(TypeError, match="New value must be a str or MaskedTag"):
        tags[0] = 123

    with pytest.raises(KeyError, match=r"Tag name '.+' does not exist."):
        tags["non_exist"] = "new value"

    with pytest.raises(IndexError, match="list index out of range"):
        tags[10] = "new value"

    with pytest.raises(TypeError, match="Key must be int or str"):
        tags[None] = "new value"
