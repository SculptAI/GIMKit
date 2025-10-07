"""Defines the schema for GIM."""

import html
import re

from dataclasses import dataclass
from typing import Literal, TypeAlias

from gimkit.exceptions import InvalidFormatError


QUERY_PREFIX = "<|GIM_QUERY|>"
QUERY_SUFFIX = "<|/GIM_QUERY|>"
RESPONSE_PREFIX = "<|GIM_RESPONSE|>"
RESPONSE_SUFFIX = "<|/GIM_RESPONSE|>"

TAG_OPEN_LEFT = "<|MASKED"
TAG_OPEN_RIGHT = "|>"
TAG_END = "<|/MASKED|>"

MAGIC_STRINGS = [
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_OPEN_LEFT,
    TAG_OPEN_RIGHT,
    TAG_END,
]

_TAG_ATTRS_REGEX = r'(?: id="m_(\d+)")?' + r'(?: name="(.*?)")?' + r'(?: desc="(.*?)")?'
_TAG_CONTENT_REGEX = r"(.*?)"

TAG_OPEN_PATTERN = re.compile(
    re.escape(TAG_OPEN_LEFT) + _TAG_ATTRS_REGEX + re.escape(TAG_OPEN_RIGHT), re.DOTALL
)
TAG_END_PATTERN = re.compile(re.escape(TAG_END))
TAG_FULL_PATTERN = re.compile(
    re.escape(TAG_OPEN_LEFT)
    + _TAG_ATTRS_REGEX
    + re.escape(TAG_OPEN_RIGHT)
    + _TAG_CONTENT_REGEX
    + re.escape(TAG_END),
    re.DOTALL,
)


@dataclass
class MaskedTag:
    id: int | None = None
    name: str | None = None
    desc: str | None = None
    content: str | None = None

    _attrs = ("name", "desc")

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
        return TAG_OPEN_LEFT + attr_part + TAG_OPEN_RIGHT + content_part + TAG_END

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
            # TAG_OPEN_RIGHT is common in text, so we allow it in content.
            # But other magic strings are not allowed.
            special_marks = MAGIC_STRINGS.copy()
            special_marks.remove(TAG_OPEN_RIGHT)
            if any(special_mark in self.content for special_mark in special_marks):
                raise ValueError(
                    "content should not contain special marks like "
                    + " or ".join(f"`{x}`" for x in special_marks)
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


ContextPart: TypeAlias = str | MaskedTag
ContextInput: TypeAlias = ContextPart | list[ContextPart]


def parse_parts(s: str) -> list[ContextPart]:
    """Parse a string into a list of ContextParts (str or MaskedTag).

    Args:
        s (str): The string to be parsed. Note it only contains masked tags or plain texts.
            Tag id may start from any non-negative integer, but must be in order 0, 1, 2, ...

    Returns:
        list[ContextPart]: A list of ContextParts (str or MaskedTag).
    """
    open_matches = list(TAG_OPEN_PATTERN.finditer(s))
    end_matches = list(TAG_END_PATTERN.finditer(s))
    full_matches = list(TAG_FULL_PATTERN.finditer(s))
    if not (len(open_matches) == len(end_matches) == len(full_matches)):
        raise InvalidFormatError(f"Mismatched or nested masked tags in {s}")

    parts: list[ContextPart] = []
    curr_tag_id = None
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
            if curr_tag_id is None:
                curr_tag_id = tag_id
            elif tag_id != curr_tag_id:
                raise InvalidFormatError(
                    f"Tag ids should be in order, got {tag_id} at position {curr_tag_id}."
                )
        if curr_tag_id is not None:
            curr_tag_id += 1
        parts.append(MaskedTag(id=tag_id, name=tag_name, desc=tag_desc, content=tag_content))

        last_end = end
    if last_end < len(s):
        parts.append(s[last_end:])
    return parts


def parse_tags(s: str, prefix: str | None = None, suffix: str | None = None) -> list[MaskedTag]:
    """Parse a string into a list of MaskedTags.

    Args:
        s (str): The string to be parsed. It may be wrapped with a prefix and suffix.
            Tag id may start from any non-negative integer, but must be in order 0, 1, 2, ...
        prefix (str | None): The prefix tag that the string should start with. Default is None.
        suffix (str | None): The suffix tag that the string should end with. Default is None.

    Returns:
        list[MaskedTag]: A list of MaskedTags.
    """

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


def validate(query: str | None, response: str | None):
    """Validate the GIM query or/and GIM response.

    Args:
        query (str): Wrapped with query prefix and suffix.
        response (str): Wrapped with response prefix and suffix.

    Raises:
        ValueError: If both query and response are None.
        InvalidFormatError: If the format of query or response is invalid,
            or if the number of masked tags or their ids do not match
            between query and response.
    """
    if query is None and response is None:
        raise ValueError("At least one of query or response must be provided.")
    if query is not None:
        query_tags = parse_tags(query, QUERY_PREFIX, QUERY_SUFFIX)
    if response is not None:
        response_tags = parse_tags(response, RESPONSE_PREFIX, RESPONSE_SUFFIX)
    if query is not None and response is not None and len(query_tags) != len(response_tags):
        raise InvalidFormatError("Mismatched number of masked tags between query and response.")
