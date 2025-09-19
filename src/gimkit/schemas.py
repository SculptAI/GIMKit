"""Defines the schema for GIM. Some examples are given below.

>>> tag = MaskedTag(id=0, desc="Fill in with appropriate text")
>>> print(tag)
<|MASKED id="m_0" desc="Fill in with appropriate text"|><|/MASKED|>

>>> m_input = '<|M_INPUT|>This is an <|MASKED id="m_0"|><|/MASKED|> text.<|/M_INPUT|>'
>>> m_output = '<|M_OUTPUT|><|MASKED id="m_0"|>example<|/MASKED|><|/M_OUTPUT|>'
>>> validate_wrapped_masked_io(m_input, m_output)  # No exception means valid
"""

import re

from dataclasses import dataclass
from typing import overload

from gimkit.exceptions import InvalidFormatError


INPUT_PREFIX = "<|M_INPUT|>"
INPUT_SUFFIX = "<|/M_INPUT|>"
OUTPUT_PREFIX = "<|M_OUTPUT|>"
OUTPUT_SUFFIX = "<|/M_OUTPUT|>"
OPEN_TAG_PATTERN = re.compile(
    r'<\|MASKED(?: id="m_(\d+)")?(?: name="(.*?)")?(?: desc="(.*?)")?\|>', re.DOTALL
)
END_TAG_PATTERN = re.compile(r"<\|/MASKED\|>")
FULL_TAG_PATTERN = re.compile(
    r'<\|MASKED(?: id="m_(\d+)")?(?: name="(.*?)")?(?: desc="(.*?)")?\|>(.*?)<\|/MASKED\|>',
    re.DOTALL,
)


def escape_in_attr_val(value: str) -> str:
    return value.replace('"', '\\"')


def unescape_in_attr_val(value: str) -> str:
    return value.replace('\\"', '"')


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
            escaped_desc = escape_in_attr_val(self.desc)
            masked_tag_str += f' desc="{escaped_desc}"'

        masked_tag_str += "|>"

        if self.content is not None:
            masked_tag_str += f"{self.content}"

        masked_tag_str += "<|/MASKED|>"
        return masked_tag_str

    def __repr__(self):  # pragma: no cover
        return self.__str__()

    def __add__(self, other: str) -> str:
        if isinstance(other, str):
            return str(self) + other
        return str(self) + str(other)

    def __radd__(self, other: str) -> str:
        if isinstance(other, str):
            return other + str(self)
        return str(other) + str(self)


class MaskedTags:
    def __init__(self, tags: list[MaskedTag]):
        self._tags = tags
        self._index = {tag.name: tag for tag in tags if tag.name is not None}

    @overload
    def __getitem__(self, key: int | str) -> MaskedTag: ...

    @overload
    def __getitem__(self, key: slice) -> list[MaskedTag]: ...

    def __getitem__(self, key: int | str | slice) -> MaskedTag | list[MaskedTag]:
        if isinstance(key, int | slice):
            return self._tags[key]
        elif isinstance(key, str):
            return self._index[key]
        raise TypeError("Key must be int, slice, or str")

    def __iter__(self):
        return iter(self._tags)

    def __len__(self):
        return len(self._tags)


class ParsedResult:
    def __init__(self, ori_query: str, tags: list[MaskedTag]):
        self._ori_query = ori_query
        self._tags = MaskedTags(tags)

    @property
    def tags(self) -> MaskedTags:
        return self._tags

    def infill(self, query: str | None = None) -> str:
        if query is None:
            query = self._ori_query

        infilled = query.removeprefix(INPUT_PREFIX).removesuffix(INPUT_SUFFIX)
        for tag in self._tags:
            if tag.content is None:
                raise ValueError(f"Tag {tag} has no content to infill.")
            infilled = re.sub(FULL_TAG_PATTERN, tag.content, infilled, count=1)
        return infilled


def parse_inp_or_outp(s: str, prefix: str, suffix: str) -> list[MaskedTag]:
    if not s.startswith(prefix) or not s.endswith(suffix):
        raise InvalidFormatError(f"Missing {prefix} or {suffix} tags.")

    s = s[len(prefix) : -len(suffix)]

    if prefix in s or suffix in s:
        raise InvalidFormatError(f"Nested {prefix} or {suffix} tags are not allowed.")

    open_mathes = list(re.finditer(OPEN_TAG_PATTERN, s))
    end_matches = list(re.finditer(END_TAG_PATTERN, s))
    full_matches = list(re.finditer(FULL_TAG_PATTERN, s))
    if not (len(open_mathes) == len(end_matches) == len(full_matches)):
        raise InvalidFormatError(f"Mismatched or nested masked tags in {prefix}...{suffix}.")

    returned = []
    for idx, match in enumerate(full_matches):
        tag_id = match.group(1)
        tag_name = match.group(2)
        tag_desc = match.group(3)
        tag_content = match.group(4)
        if tag_id is not None:
            tag_id = int(tag_id)
            if tag_id != idx:
                raise InvalidFormatError(
                    f"Tag ids should be in order 0, 1, 2, ..., got {tag_id} at position {idx}."
                )
        if tag_desc is not None:
            tag_desc = unescape_in_attr_val(tag_desc)
        returned.append(MaskedTag(id=tag_id, name=tag_name, desc=tag_desc, content=tag_content))

    return returned


def validate_wrapped_masked_io(inp: str | None, outp: str | None):
    """Validate the wrapped masked input or/and output strings.

    Args:
        inp (str): The wrapped masked input string to be validated.
        outp (str): The wrapped masked output string to be validated.

    Raises:
        ValueError: If both inp and outp are None.
        InvalidFormatError: If the format of inp or outp is invalid, or if the number of masked tags
            or their ids do not match between inp and outp.
    """
    if inp is None and outp is None:
        raise ValueError("At least one of inp or outp must be provided.")
    if inp is not None:
        inp_tags = parse_inp_or_outp(inp, INPUT_PREFIX, INPUT_SUFFIX)
    if outp is not None:
        outp_tags = parse_inp_or_outp(outp, OUTPUT_PREFIX, OUTPUT_SUFFIX)
    if inp is not None and outp is not None and len(inp_tags) != len(outp_tags):
        raise InvalidFormatError("Mismatched number of masked tags between input and output.")


class Guide:
    def __init__(self) -> None:
        self._tags: list[MaskedTag] = []

    def _append_tag(self, tag: MaskedTag):
        for existing_tag in self._tags:
            if tag.name is not None and existing_tag.name == tag.name:
                raise ValueError(f"Tag name '{tag.name}' already exists.")
        self._tags.append(tag)

    def __call__(
        self, name: str | None = None, desc: str | None = None, content: str | None = None, **kwargs
    ) -> MaskedTag:
        tag = MaskedTag(id=len(self._tags), name=name, desc=desc, content=content)
        self._append_tag(tag)
        return tag

    def standardize(self, raw_query: str) -> str:
        query = raw_query
        if not query.startswith(INPUT_PREFIX):
            query = INPUT_PREFIX + query
        if not query.endswith(INPUT_SUFFIX):
            query = query + INPUT_SUFFIX
        validate_wrapped_masked_io(query, None)
        return query

    def parse(self, query: str, response: str) -> ParsedResult:
        validate_wrapped_masked_io(query, response)
        result_tags = parse_inp_or_outp(response, OUTPUT_PREFIX, OUTPUT_SUFFIX)
        for inp_tag, outp_tag in zip(self._tags, result_tags, strict=True):
            outp_tag.name = inp_tag.name
        return ParsedResult(query, result_tags)


class guide(Guide):  # noqa: N801  # pragma: no cover
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

    def options(self, name: str, choices: list[str]) -> MaskedTag:
        """Choose one from the given options."""
        desc = f"Choose one from the following options: {', '.join(choices)}."
        return self(name=name, desc=desc)
