import pytest

from gimkit.models.utils import ensure_str


def test_ensure_str():
    assert ensure_str("hello") == "hello"

    with pytest.raises(TypeError, match="Response is not a string"):
        ensure_str(123)
