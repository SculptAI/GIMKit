import re

import pytest

from gimkit.guides import guide as g


class TestFormMixin:
    def test_single_word(self):
        tag = g.single_word(name="word")
        assert re.fullmatch(tag.regex, "Hello")
        assert not re.fullmatch(tag.regex, "Hello World")

    def test_select(self):
        choices = ["apple", "banana", "cherry", "special|char"]
        tag = g.select(name="fruit", choices=choices)
        assert tag.regex == "apple|banana|cherry|special\\|char"
        assert re.fullmatch(tag.regex, "banana")
        assert not re.fullmatch(tag.regex, "grape")
        assert re.fullmatch(tag.regex, "special|char")
        with pytest.raises(ValueError, match="choices must be a non-empty list of strings"):
            g.select(name="fruit", choices=None)

    def test_datetime(self):
        tag1 = g.datetime(name="dt1", require_date=True, require_time=True)
        assert re.fullmatch(tag1.regex, "2023-10-05 14:30:00")
        assert re.fullmatch(tag1.regex, "2023-10-05T14:30")
        assert not re.fullmatch(tag1.regex, "2023-10-05")
        assert not re.fullmatch(tag1.regex, "14:30")

        tag2 = g.datetime(name="dt2", require_date=True, require_time=False)
        assert re.fullmatch(tag2.regex, "2023-10-05")
        assert not re.fullmatch(tag2.regex, "14:30")

        tag3 = g.datetime(name="dt3", require_date=False, require_time=True)
        assert re.fullmatch(tag3.regex, "14:30:00")
        assert not re.fullmatch(tag3.regex, "2023-10-05")

        with pytest.raises(ValueError, match="At least one of require_date or require_time must be True"):
            g.datetime(name="dt4", require_date=False, require_time=False)

class TestPersonalInfoMixin:
    def test_person_name(self):
        tag = g.person_name(name="name")
        assert tag

    def test_phone_number(self):
        tag = g.phone_number(name="phone")
        assert re.fullmatch(tag.regex, "+1-123-456-7890")
        assert re.fullmatch(tag.regex, "(123) 456-7890")
        assert re.fullmatch(tag.regex, "123-456-7890")
        assert not re.fullmatch(tag.regex, "abc-def-ghij")

    def test_e_mail(self):
        tag = g.e_mail(name="email")
        assert re.fullmatch(tag.regex, "john.doe@example.com")
        assert re.fullmatch(tag.regex, "alice@example.com")
        assert not re.fullmatch(tag.regex, "invalid-email")


def test_guide_call():
    tag1 = g(desc="It's a great day.", name="first")
    assert tag1.id is None
    assert tag1.name == "first"
    assert tag1.desc == "It's a great day."
    assert tag1.content is None
    assert str(tag1) == '<|MASKED name="first" desc="It&#x27;s a great day."|><|/MASKED|>'

    tag2 = g(desc="It&#x27;s a great day.")
    assert tag2.desc == "It's a great day."

    tag3 = g()
    assert str(tag3) == "<|MASKED|><|/MASKED|>"


def test_guide_with_string():
    # Test format string
    prompt = f"""I'm {g(name="sub")}!"""
    assert prompt == """I'm <|MASKED name="sub"|><|/MASKED|>!"""

    # Test __add__ and __radd__
    prompt = g(desc="number") + " + " + g(desc="number") + " = 2"
    assert (
        prompt
        == """<|MASKED desc="number"|><|/MASKED|> + <|MASKED desc="number"|><|/MASKED|> = 2"""
    )

    # Test other types. No error should be raised.
    g() + object
    object + g()
