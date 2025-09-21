"""Defines the schema for GIM."""

import html
import re

from dataclasses import dataclass
from typing import Literal

from gimkit.exceptions import InvalidFormatError


QUERY_PREFIX = "<|M_QUERY|>"
QUERY_SUFFIX = "<|/M_QUERY|>"
RESPONSE_PREFIX = "<|M_RESPONSE|>"
RESPONSE_SUFFIX = "<|/M_RESPONSE|>"
OPEN_TAG_PATTERN = re.compile(
    r'<\|MASKED(?: id="m_(\d+)")?(?: name="(.*?)")?(?: desc="(.*?)")?\|>', re.DOTALL
)
END_TAG_PATTERN = re.compile(r"<\|/MASKED\|>")
FULL_TAG_PATTERN = re.compile(
    r'<\|MASKED(?: id="m_(\d+)")?(?: name="(.*?)")?(?: desc="(.*?)")?\|>(.*?)<\|/MASKED\|>',
    re.DOTALL,
)


@dataclass
class MaskedTag:
    id: int | None = None
    name: str | None = None
    desc: str | None = None
    content: str | None = None

    _attrs = ("name", "desc")
    _template = "<|MASKED{}|>{}<|/MASKED|>"

    def to_string(
        self, fields: list[Literal["id", "name", "desc", "content"]] | Literal["all"] = "all"
    ) -> str:
        attr_part = ""
        if fields == "all":
            fields = ["id", "name", "desc", "content"]
        if "id" in fields and self.id is not None:
            attr_part += f' id="m_{self.id}"'
        for attr in self._attrs:
            if attr in fields and getattr(self, attr) is not None:
                escaped_val = self.escape_in_attr_val(getattr(self, attr))
                attr_part += f' {attr}="{escaped_val}"'
        content_part = ""
        if "content" in fields and self.content is not None:
            content_part = f"{self.content}"
        return MaskedTag._template.format(attr_part, content_part)

    @classmethod
    def escape_in_attr_val(cls, value: str) -> str:
        return html.escape(value)

    @classmethod
    def unescape_in_attr_val(cls, value: str) -> str:
        return html.unescape(value)

    def __post_init__(self):
        if not (self.id is None or isinstance(self.id, int)):
            raise ValueError(f"{type(self.id)=}, {self.id=}, should be int or None")

        for attr in self._attrs:
            attr_val = getattr(self, attr)
            if isinstance(attr_val, str):
                setattr(self, attr, MaskedTag.unescape_in_attr_val(attr_val))
            elif attr_val is not None:
                raise ValueError(f"{type(attr_val)=}, {attr_val=}, should be str or None")

        if isinstance(self.content, str):
            special_marks = MaskedTag._template.split("{}")
            if any(special_mark in self.content for special_mark in special_marks):
                raise ValueError(
                    f"content should not contain special marks like {' or '.join(f'`{x}`' for x in special_marks)}"
                )
        elif self.content is not None:
            raise ValueError(f"{type(self.content)=}, {self.content=}, should be str or None")

    def __str__(self):
        return self.to_string()

    def __repr__(self):
        return self.to_string()

    def __add__(self, other: str) -> str:
        if isinstance(other, str):
            return str(self) + other
        return str(self) + str(other)

    def __radd__(self, other: str) -> str:
        if isinstance(other, str):
            return other + str(self)
        return str(other) + str(self)


def parse_parts(s: str) -> list[str | MaskedTag]:
    """Parse a string into a list of strings and MaskedTags.

    Args:
        s (str): The string to be parsed. Note it only contains masked tags or plain texts.
            Tag id may start from any non-negative integer, but must be in order 0, 1, 2, ...

    Returns:
        list[str | MaskedTag]: A list of strings and MaskedTags.

    Example:
        >>> parse_parts('Hello, <|MASKED name="sub">Alice</|MASKED|>!')
        [ "Hello, ", MaskedTag(name="sub", content="Alice"), "!" ]
    """
    open_matches = list(OPEN_TAG_PATTERN.finditer(s))
    end_matches = list(END_TAG_PATTERN.finditer(s))
    full_matches = list(FULL_TAG_PATTERN.finditer(s))
    if not (len(open_matches) == len(end_matches) == len(full_matches)):
        raise InvalidFormatError(f"Mismatched or nested masked tags in {s[:50]}...")

    parts: list[str | MaskedTag] = []
    curr_tag_id = -1
    last_end = 0
    for match in full_matches:
        start, end = match.span()
        if start > last_end:
            parts.append(s[last_end:start])

        tag_id = match.group(1)
        tag_name = match.group(2)
        tag_desc = match.group(3)
        tag_content = match.group(4)
        if tag_id is not None:
            tag_id = int(tag_id)
            if curr_tag_id == -1:
                curr_tag_id = tag_id
            elif tag_id != curr_tag_id:
                raise InvalidFormatError(
                    f"Tag ids should be in order, got {tag_id} at position {curr_tag_id}."
                )
        parts.append(MaskedTag(id=tag_id, name=tag_name, desc=tag_desc, content=tag_content))
        curr_tag_id += 1

        last_end = end
    if last_end < len(s):
        parts.append(s[last_end:])
    return parts


def parse_tags(s: str, prefix: str | None = None, suffix: str | None = None) -> list[MaskedTag]:
    """Parse a string with wrapped masked tags into a list of MaskedTags."""

    if prefix is not None:
        s = s.lstrip()
        if not s.startswith(prefix):
            raise InvalidFormatError(f"String must start with the {prefix} tag.")

        s = s[len(prefix) :]
        if prefix in s:
            raise InvalidFormatError(f"Nested or duplicate {prefix} tag are not allowed.")

    if suffix is not None:
        s = s.rstrip()
        if not s.endswith(suffix):
            raise InvalidFormatError(f"String must end with the {suffix} tag.")

        s = s[: -len(suffix)]
        if suffix in s:
            raise InvalidFormatError(f"Nested or duplicate {suffix} tag are not allowed.")

    parts = parse_parts(s)
    tags = [part for part in parts if isinstance(part, MaskedTag)]

    if prefix is not None:
        expected_ids = list(range(len(tags)))
        actual_ids = [tag.id or idx for idx, tag in enumerate(tags)]
        if expected_ids != actual_ids:
            raise InvalidFormatError(
                f"Tag ids should be in order 0, 1, 2, ..., got {', '.join(map(str, actual_ids))}."
            )

    return tags


def validate_wrapped_masked_qr(query: str | None, response: str | None):
    """Validate the wrapped masked query or/and response strings.

    Args:
        query (str): The wrapped masked query string to be validated.
        response (str): The wrapped masked response string to be validated.

    Raises:
        ValueError: If both query and response are None.
        InvalidFormatError: If the format of query or response is invalid, or if the number of masked tags
            or their ids do not match between query and response.
    """
    if query is None and response is None:
        raise ValueError("At least one of query or response must be provided.")
    if query is not None:
        query_tags = parse_tags(query, QUERY_PREFIX, QUERY_SUFFIX)
    if response is not None:
        response_tags = parse_tags(response, RESPONSE_PREFIX, RESPONSE_SUFFIX)
    if query is not None and response is not None and len(query_tags) != len(response_tags):
        raise InvalidFormatError("Mismatched number of masked tags between query and response.")


def validate_wrapped_masked_io(inp: str | None, outp: str | None):
    """Validate the wrapped masked input or/and output strings (legacy format).

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
        # Check if it's old format (M_INPUT) or new format (M_QUERY)
        if inp.strip().startswith("<|M_INPUT|>"):
            inp_tags = parse_tags(inp, "<|M_INPUT|>", "<|/M_INPUT|>")
        else:
            inp_tags = parse_tags(inp, QUERY_PREFIX, QUERY_SUFFIX)
    if outp is not None:
        # Check if it's old format (M_OUTPUT) or new format (M_RESPONSE)
        if outp.strip().startswith("<|M_OUTPUT|>"):
            outp_tags = parse_tags(outp, "<|M_OUTPUT|>", "<|/M_OUTPUT|>")
        else:
            outp_tags = parse_tags(outp, RESPONSE_PREFIX, RESPONSE_SUFFIX)
    if inp is not None and outp is not None and len(inp_tags) != len(outp_tags):
        raise InvalidFormatError("Mismatched number of masked tags between input and output.")


# Backward compatibility aliases - these point to the old constants for legacy support
INPUT_PREFIX = "<|M_INPUT|>"
INPUT_SUFFIX = "<|/M_INPUT|>"
OUTPUT_PREFIX = "<|M_OUTPUT|>"
OUTPUT_SUFFIX = "<|/M_OUTPUT|>"
