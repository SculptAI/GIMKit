import re

import gimkit


def test_version_is_none_or_string():
    v = gimkit.__version__
    assert isinstance(v, str)
    assert v == "unknown" or re.match(r"^\d+(?:\.\d+)*", v)
