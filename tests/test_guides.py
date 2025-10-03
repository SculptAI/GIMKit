import pytest

from gimkit.guides import guide as g


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


def test_guide_descriptors():
    """Test the descriptor-based guide methods."""
    # Test person_name
    tag = g.person_name()
    assert tag.desc == "A person's name, e.g., John Doe, Alice, Bob, Charlie Brown, etc."
    assert tag.name is None

    tag = g.person_name(name="user")
    assert tag.name == "user"
    assert tag.desc == "A person's name, e.g., John Doe, Alice, Bob, Charlie Brown, etc."

    # Test phone_number
    tag = g.phone_number()
    assert tag.desc == "A phone number, e.g., +1-123-456-7890, (123) 456-7890, 123-456-7890, etc."

    tag = g.phone_number(name="contact")
    assert tag.name == "contact"

    # Test e_mail
    tag = g.e_mail()
    assert tag.desc == "An email address, e.g., john.doe@example.com, alice@example.com, etc."

    tag = g.e_mail(name="email")
    assert tag.name == "email"

    # Test single_word
    tag = g.single_word()
    assert tag.desc == "A single word without spaces."

    tag = g.single_word(name="word")
    assert tag.name == "word"


def test_guide_options():
    """Test the options guide method."""
    # Test with valid choices
    tag = g.options(choices=["a", "b", "c"])
    assert tag.desc == "Choose one from the following options: a, b, c."
    assert tag.name is None

    tag = g.options(name="choice", choices=["option1", "option2"])
    assert tag.name == "choice"
    assert tag.desc == "Choose one from the following options: option1, option2."

    # Test with empty choices
    with pytest.raises(ValueError, match="choices must be a non-empty list of strings"):
        g.options(choices=[])

    with pytest.raises(ValueError, match="choices must be a non-empty list of strings"):
        g.options(choices=None)
