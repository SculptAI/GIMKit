import re

import gimkit


def test_gimkit_version():
    v = gimkit.__version__
    assert isinstance(v, str)
    if v != "unknown":
        # PEP 440 compliance check
        # Ref: https://peps.python.org/pep-0440/
        assert re.match(
            r"^([1-9][0-9]*!)?(0|[1-9][0-9]*)(\.(0|[1-9][0-9]*))*((a|b|rc)(0|[1-9][0-9]*))?(\.post(0|[1-9][0-9]*))?(\.dev(0|[1-9][0-9]*))?$",
            v,
        )
