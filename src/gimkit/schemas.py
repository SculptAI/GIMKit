"""Defines the schema for GIM."""

import html
import re

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar, Literal, TypeAlias, cast

from gimkit.exceptions import InvalidFormatError


# ─── Gim Query, Response And Tag Definitions ──────────────────────────────────

QUERY_PREFIX = "<|GIM_QUERY|>"
QUERY_SUFFIX = "<|/GIM_QUERY|>"
RESPONSE_PREFIX = "<|GIM_RESPONSE|>"
RESPONSE_SUFFIX = "<|/GIM_RESPONSE|>"

TAG_OPEN_LEFT = "<|MASKED"
TAG_OPEN_RIGHT = "|>"
TAG_END = "<|/MASKED|>"

MAGIC_STRINGS = (
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_OPEN_LEFT,
    TAG_OPEN_RIGHT,
    TAG_END,
)


# ─── Tag Fields Definitions ───────────────────────────────────────────────────

COMMON_ATTRS = ("name", "desc", "regex")
ALL_ATTRS = ("id", *COMMON_ATTRS)
ALL_FIELDS = ("id", *COMMON_ATTRS, "content")

TagField: TypeAlias = Literal["id", "name", "desc", "regex", "content"]


# ─── Regex Patterns For Tag Parsing ───────────────────────────────────────────

_TAG_ATTRS_REGEX = r'(?: id="m_(?P<id>\d+)")?' + "".join(
    rf'(?: {field}="(?P<{field}>.*?)")?' for field in COMMON_ATTRS
)
_TAG_CONTENT_REGEX = r"(?P<content>.*?)"

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

# !NOTE:
# Some edge-case masked-tag strings (e.g., with embedded escaped quotes in attributes)
# may not be matched by TAG_FULL_PATTERN. Example:
# `<|MASKED id="m_1" desc="sample\"|>" regex="[a-zA-Z]+"|>content here<|/MASKED|>`
# GIM expects tags to be generated via guide() helpers rather than hand-written,
# so parsing favors simplicity and performance over covering every exotic edge case.


# ─── MaskedTag Definition ─────────────────────────────────────────────────────


@dataclass
class MaskedTag:
    """Represents a masked tag in the GIM schema.

    A masked tag consists of three main types of components:
    1. **Tag ID**: An integer identifier for the tag, represented as `m_{id}` in the tag attributes.
    2. **Tag content**: The content located between the opening and closing masked tag markers.
    3. **Tag common attributes**: All other tag attributes aside from the ID (e.g., name, desc, regex).

    Example of a masked tag:
        `<|MASKED id="m_0" name="xxx" desc="xxx" regex="xxx"|>content here<|/MASKED|>`
    """

    id: int | str | None = None
    name: str | None = None
    desc: str | None = None
    regex: str | None = None
    content: str | None = None

    # Read-only class variable for additional attribute escapes. These
    # characters may appear in tag attributes such as `desc` or `grammar`.
    # Hexadecimal numeric character references are used for consistency and
    # compatibility with Python's built-in `html.escape` conventions.
    # Ref: https://www.w3.org/MarkUp/html-spec/html-spec_13.html
    _ADDITIONAL_ATTR_ESCAPES: ClassVar[Mapping[str, str]] = MappingProxyType(
        {
            "\t": "&#x09;",  # Tab
            "\n": "&#x0a;",  # Line Feed
            "\r": "&#x0d;",  # Carriage Return
        }
    )

    @classmethod
    def attr_escape(cls, text: str) -> str:
        escaped_text = html.escape(text, quote=True)
        for char, escape_seq in cls._ADDITIONAL_ATTR_ESCAPES.items():
            escaped_text = escaped_text.replace(char, escape_seq)
        return escaped_text

    @classmethod
    def attr_unescape(cls, text: str) -> str:
        return html.unescape(text)

    def __post_init__(self):
        # 1. Validate id
        if not (
            self.id is None
            or isinstance(self.id, int)
            or (isinstance(self.id, str) and self.id.isdigit())
        ):
            raise ValueError(f"{type(self.id)=}, {self.id=}, should be int, str of digits, or None")
        if isinstance(self.id, str):
            self.id = int(self.id)

        # 2. Validate common attributes
        for attr in COMMON_ATTRS:
            attr_val = getattr(self, attr)
            if isinstance(attr_val, str):
                setattr(self, attr, MaskedTag.attr_unescape(attr_val))
            elif attr_val is not None:
                raise ValueError(f"{type(attr_val)=}, {attr_val=}, should be str or None")

        # 3. Validate content
        if isinstance(self.content, str):
            # TAG_OPEN_RIGHT is common in text, so we allow it in content.
            # But other magic strings are not allowed.
            special_marks = [s for s in MAGIC_STRINGS if s != TAG_OPEN_RIGHT]
            if any(special_mark in self.content for special_mark in special_marks):
                raise ValueError(
                    "content should not contain special marks like "
                    + " or ".join(f"`{x}`" for x in special_marks)
                )
        elif self.content is not None:
            raise ValueError(f"{type(self.content)=}, {self.content=}, should be str or None")

        # 4. Validate regex if provided
        if isinstance(self.regex, str):
            if self.regex.startswith("^") or self.regex.endswith("$"):
                raise ValueError(
                    "regex should not start with ^ or end with $, "
                    "as it will be used within a larger regex pattern."
                )
            if self.regex.startswith("/") or self.regex.endswith("/"):
                raise ValueError(
                    "regex should not start or end with /, "
                    "as it will be wrapped with /.../ in CFG grammar."
                )
            if self.regex == "":
                raise ValueError("regex should not be an empty string.")
            try:
                re.compile(self.regex)
            except re.error as e:
                raise ValueError(f"Invalid regex pattern: {self.regex}") from e

    def to_string(
        self,
        fields: list[TagField] | Literal["all"] = "all",
    ) -> str:
        attr_part = ""
        if fields == "all":
            fields = cast("list[TagField]", list(ALL_FIELDS))
        if "id" in fields and self.id is not None:
            attr_part += f' id="m_{self.id}"'
        for attr in COMMON_ATTRS:
            if attr in fields and getattr(self, attr) is not None:
                escaped_val = self.attr_escape(getattr(self, attr))
                attr_part += f' {attr}="{escaped_val}"'
        content_part = ""
        if "content" in fields and self.content is not None:
            content_part = f"{self.content}"
        return TAG_OPEN_LEFT + attr_part + TAG_OPEN_RIGHT + content_part + TAG_END

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


# ─── Schema Parsing And Validation ────────────────────────────────────────────


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

        fields = match.groupdict()
        tag_id = fields.get("id")
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
        parts.append(MaskedTag(**fields))

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
