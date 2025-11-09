"""Defines the schema for GIM."""

import html
import re

from dataclasses import dataclass
from typing import Literal, TypeAlias


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

_TAG_ATTRS_REGEX = (
    r'(?: id="m_(\d+)")?' + r'(?: name="(.*?)")?' + r'(?: desc="(.*?)")?' + r'(?: regex="(.*?)")?'
)
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
    """Represents a masked tag in the GIM schema. A tag consists of a tag
    id, tag content and some other related attributes. It looks like:

    `<|MASKED id="m_0" name="xxx" desc="xxx" regex="xxx"|>xxx<|/MASKED|>`
    """

    id: int | None = None
    name: str | None = None
    desc: str | None = None
    regex: str | None = None
    content: str | None = None

    _attrs = ("name", "desc", "regex")

    def to_string(
        self,
        fields: list[Literal["id", "name", "desc", "regex", "content"]] | Literal["all"] = "all",
    ) -> str:
        attr_part = ""
        if fields == "all":
            fields = ["id", "name", "desc", "regex", "content"]
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
        # 1. Validate id
        if not (self.id is None or isinstance(self.id, int)):
            raise ValueError(f"{type(self.id)=}, {self.id=}, should be int or None")

        # 2. Validate all attrs
        for attr in self._attrs:
            attr_val = getattr(self, attr)
            if isinstance(attr_val, str):
                setattr(self, attr, MaskedTag.unescape_in_attr_val(attr_val))
            elif attr_val is not None:
                raise ValueError(f"{type(attr_val)=}, {attr_val=}, should be str or None")

        # 3. Validate content
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

        # 4. Validate regex if provided
        if isinstance(self.regex, str):
            if "^" in self.regex or "$" in self.regex:
                raise ValueError(
                    "regex should not contain ^ or $, "
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
