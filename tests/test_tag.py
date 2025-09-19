import pytest

from gimkit.schemas import MaskedTag, MaskedTags


def test_masked_tag_str():
    assert str(MaskedTag(id=0)) == '<|MASKED id="m_0"|><|/MASKED|>'
    assert (
        str(MaskedTag(id=0, desc="description"))
        == '<|MASKED id="m_0" desc="description"|><|/MASKED|>'
    )
    assert (
        str(MaskedTag(id=0, desc='desc with "quotes"'))
        == '<|MASKED id="m_0" desc="desc with \\"quotes\\""|><|/MASKED|>'
    )
    assert str(MaskedTag(id=0, content="content")) == '<|MASKED id="m_0"|>content<|/MASKED|>'
    assert str(MaskedTag()) == "<|MASKED|><|/MASKED|>"
    assert str(MaskedTag(name="content")) == "<|MASKED|><|/MASKED|>"


def test_masked_tag_invalid_init():
    with pytest.raises(ValueError, match="should be int or None"):
        MaskedTag(id="0")
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(name=123)
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(desc=123)
    with pytest.raises(ValueError, match="should be str or None"):
        MaskedTag(content=object)


def test_masked_tags_collection():
    tag1 = MaskedTag(id=0, name="a")
    tag2 = MaskedTag(id=1, name="b")
    tags = MaskedTags([tag1, tag2])

    assert len(tags) == 2
    assert tags[0] == tag1
    assert tags["b"] == tag2
    assert tags[0:1] == [tag1]
    assert list(tags) == [tag1, tag2]
    with pytest.raises(TypeError, match="Key must be int, slice, or str"):
        tags[None]
