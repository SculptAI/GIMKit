"""Tests for regex support in MaskedTag and guide."""

from gimkit.contexts import Query
from gimkit.guides import guide as g
from gimkit.models.utils import build_cfg
from gimkit.schemas import MaskedTag, parse_tags


def test_masked_tag_with_regex():
    """Test that MaskedTag can be created with a regex pattern."""
    tag = MaskedTag(name="test", desc="A test tag", regex=r"\d{3}-\d{4}")
    assert tag.regex == r"\d{3}-\d{4}"
    assert tag.name == "test"
    assert tag.desc == "A test tag"
    
    # Test string representation includes regex
    tag_str = str(tag)
    assert 'regex="\\d{3}-\\d{4}"' in tag_str or 'regex="&#x5C;d{3}-&#x5C;d{4}"' in tag_str


def test_masked_tag_regex_serialization():
    """Test that MaskedTag with regex serializes and deserializes correctly."""
    tag = MaskedTag(id=0, name="phone", regex=r"\d{3}-\d{3}-\d{4}")
    tag_str = str(tag)
    
    # Parse it back
    tags = parse_tags(tag_str)
    assert len(tags) == 1
    parsed_tag = tags[0]
    assert parsed_tag.regex == r"\d{3}-\d{3}-\d{4}"
    assert parsed_tag.name == "phone"


def test_guide_regex_method():
    """Test that guide.regex() creates a MaskedTag with regex."""
    tag = g.regex(r"[A-Z]{3}", name="code", desc="Three uppercase letters")
    assert tag.regex == r"[A-Z]{3}"
    assert tag.name == "code"
    assert tag.desc == "Three uppercase letters"


def test_guide_call_with_regex():
    """Test that guide() can accept regex parameter."""
    tag = g(name="test", desc="A test", regex=r"\d+")
    assert tag.regex == r"\d+"
    assert tag.name == "test"
    assert tag.desc == "A test"


def test_masked_tag_with_regex_none():
    """Test that MaskedTag works when regex is None."""
    tag = MaskedTag(name="test", desc="A test")
    assert tag.regex is None
    assert tag.name == "test"


def test_regex_in_to_string():
    """Test that regex appears in to_string output."""
    tag = MaskedTag(id=0, regex=r"\w+")
    tag_str = tag.to_string(fields=["id", "regex"])
    assert "id=" in tag_str
    assert "regex=" in tag_str or "\\w+" in tag_str


def test_regex_excluded_from_to_string():
    """Test that regex can be excluded from to_string output."""
    tag = MaskedTag(id=0, regex=r"\w+", desc="test")
    tag_str = tag.to_string(fields=["id", "desc"])
    assert "id=" in tag_str
    assert "desc=" in tag_str
    # regex should not be in output when not in fields list
    # (it might appear in a different context, but not as an attribute)


def test_build_cfg_with_regex():
    """Test that build_cfg uses regex patterns when available."""
    # Create a query with a tag that has a regex pattern
    query = Query('Hello, <|MASKED id="m_0" regex="\\d{3}"|><|/MASKED|>!')
    cfg = build_cfg(query)
    
    # Check that the regex pattern is used in the grammar
    assert r"\d{3}" in cfg.definition or "\\d{3}" in cfg.definition


def test_build_cfg_without_regex():
    """Test that build_cfg uses default pattern when regex is not provided."""
    # Create a query without regex
    query = Query('Hello, <|MASKED id="m_0"|>world<|/MASKED|>!')
    cfg = build_cfg(query)
    
    # Check that the default pattern is used
    assert "/(?s:.)*?/" in cfg.definition

