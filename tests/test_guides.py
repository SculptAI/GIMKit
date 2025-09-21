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
