"""Defines the schema for GIM. Some examples are given below.

>>> tag = MaskedTag(id=0, desc="Fill in with appropriate text")
>>> print(tag)
<|MASKED id="m_0" desc="Fill in with appropriate text"|><|/MASKED|>

>>> m_input = '<|M_INPUT|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/M_INPUT|>'
>>> m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|></M_OUTPUT|>'
>>> validate_wrapped_masked_io(m_input, m_output)  # No exception means valid
"""

import re

from dataclasses import dataclass
from typing import overload


@dataclass
class MaskedTag:
    id: int | None = None
    name: str | None = None
    desc: str | None = None
    content: str | None = None

    def __post_init__(self):
        if self.id is not None and not isinstance(self.id, int):
            raise ValueError(f"{type(self.id)=}, {self.id=}, should be int or None")
        if self.name is not None and not isinstance(self.name, str):
            raise ValueError(f"{type(self.name)=}, {self.name=}, should be str or None")
        if self.desc is not None and not isinstance(self.desc, str):
            raise ValueError(f"{type(self.desc)=}, {self.desc=}, should be str or None")
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError(f"{type(self.content)=}, {self.content=}, should be str or None")

    def __str__(self):
        masked_tag_str = "<|MASKED"

        if self.id is not None:
            masked_tag_str += f' id="m_{self.id}"'
        if self.desc is not None:
            escaped_desc = self.escape_in_attr_value(self.desc)
            masked_tag_str += f' desc="{escaped_desc}"'

        masked_tag_str += "|>"

        if self.content is not None:
            masked_tag_str += f"{self.content}"

        masked_tag_str += "<|/MASKED|>"
        return masked_tag_str

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: str):
        if isinstance(other, str):
            return str(self) + other
        return NotImplemented

    def __radd__(self, other: str):
        if isinstance(other, str):
            return other + str(self)
        return NotImplemented

    @staticmethod
    def escape_in_attr_value(value: str) -> str:
        return value.replace('"', '\\"')


def validate_wrapped_masked_io(inp: str | None, outp: str | None):
    """Validate the wrapped masked input and output strings.

    Args:
        input (str): The wrapped masked input string to be validated.
        output (str): The wrapped masked output string to be validated.

    Raises:
        ValueError: If the input or output format is invalid.
    """
    if inp is None or outp is None:
        return  # TODO

    if "<|M_INPUT|>" in inp[11:] or "<|/M_INPUT|>" in inp[:-12]:
        raise ValueError("Invalid input format: Nested <|M_INPUT|> tags are not allowed.")
    if "<|M_OUTPUT|>" in outp[12:] or "<|/M_OUTPUT|>" in outp[:-13]:
        raise ValueError("Invalid output format: Nested <|M_OUTPUT|> tags are not allowed.")

    open_tag_pattern = r'<\|MASKED(?: id="m_(\d+)")?(?: desc=".*?")?\|>'
    end_tag_pattern = r"<\|/MASKED\|>"
    i_open_matches = list(re.finditer(open_tag_pattern, inp))
    i_end_tags = re.findall(end_tag_pattern, inp)
    o_open_matches = list(re.finditer(open_tag_pattern, outp))
    o_end_tags = re.findall(end_tag_pattern, outp)

    if not (len(i_open_matches) == len(i_end_tags) == len(o_open_matches) == len(o_end_tags)):
        raise ValueError("Mismatched number of masked tags between input and output.")

    for idx, (i_open_tag, o_open_tag) in enumerate(
        zip(i_open_matches, o_open_matches, strict=False), start=1
    ):
        i_idx = int(i_open_tag.group(1)) if i_open_tag.group(1) is not None else idx
        o_idx = int(o_open_tag.group(1)) if o_open_tag.group(1) is not None else idx
        if i_idx != o_idx:
            raise ValueError(
                f"Mismatched masked tag ids between input and output at position {idx}: {i_idx} != {o_idx}"
            )


class Guide:
    def __init__(self) -> None:
        self._tags: list[MaskedTag] = []
        self._query: str | None = None
        self._response: str | None = None

    def infill(self, results: list[MaskedTag]) -> str:
        # TODO
        return "A complete string"

    @property
    def query(self) -> str | None:
        return self._query

    @query.setter
    def query(self, value: str):
        if not value.startswith("<|M_INPUT|>") or not value.endswith("<|/M_INPUT|>"):
            value = f"<|M_INPUT|>{value}<|/M_INPUT|>"
        validate_wrapped_masked_io(value, None)
        self._query = value

    @property
    def response(self) -> str | None:
        return self._response

    @response.setter
    def response(self, value: str):
        validate_wrapped_masked_io(self._query, value)
        self._response = value
        # TODO: parse the response and fill in the content of self._tags

    @property
    def results(self) -> list[MaskedTag]:
        return self._tags

    @overload
    def __getitem__(self, key: int | str) -> MaskedTag: ...

    @overload
    def __getitem__(self, key: slice) -> list[MaskedTag]: ...

    def __getitem__(self, key: int | slice | str) -> MaskedTag | list[MaskedTag]:
        if isinstance(key, int | slice):
            return self._tags[key]
        elif isinstance(key, str):
            for tag in self._tags:
                if tag.name == key:
                    return tag
            raise KeyError(f"No tag found with name: {key}")
        else:
            raise TypeError("Key must be an int, slice, or str")

    def _append_tag(self, tag: MaskedTag):
        if tag.id != len(self._tags):
            raise ValueError(f"Tag id should be {len(self._tags)}, got {tag.id}")
        for existing_tag in self._tags:
            if tag.name is not None and existing_tag.name == tag.name:
                raise ValueError(f"Tag name '{tag.name}' already exists.")
        self._tags.append(tag)


class guide(Guide):  # noqa: N801
    def __call__(self, name: str, desc: str | None = None, **kwargs) -> MaskedTag:
        tag = MaskedTag(id=len(self._tags), name=name, desc=desc)
        self._append_tag(tag)
        return tag

    def person_name(self, name: str) -> MaskedTag:
        """A person's name, e.g., John Doe, Alice, Bob, Charlie Brown, etc."""
        return self(name=name, desc=self.person_name.__doc__)

    def phone_number(self, name: str) -> MaskedTag:
        """A phone number, e.g., +1-123-456-7890, (123) 456-7890, 123-456-7890, etc."""
        return self(name=name, desc=self.phone_number.__doc__)

    def e_mail(self, name: str) -> MaskedTag:
        """An email address, e.g., john.doe@example.com, alice@example.com, etc."""
        return self(name=name, desc=self.e_mail.__doc__)

    def single_word(self, name: str) -> MaskedTag:
        """A single word without spaces."""
        return self(name=name, desc=self.single_word.__doc__)
