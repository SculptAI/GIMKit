"""Check that basic features work."""

from gimkit import MaskedTag


tag = MaskedTag(desc="Hello, world!")

if str(tag) == '<|MASKED desc="Hello, world!"|><|/MASKED|>':
    print("Smoke test succeeded")
else:
    raise RuntimeError("Smoke test failed")
