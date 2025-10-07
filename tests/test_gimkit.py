import re

import gimkit


def test_version_is_none_or_string():
    v = gimkit.__version__
    assert v is None or isinstance(v, str)

    # If a version string is present, at least ensure it starts with a number
    if isinstance(v, str):
        assert re.match(r"^\d+(?:\.\d+)*", v)
