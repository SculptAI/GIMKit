from __future__ import annotations

import re
import warnings

from typing import TYPE_CHECKING, Literal, cast, overload

from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import (
    ALL_FIELDS,
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_END,
    TAG_OPEN_LEFT,
    TAG_OPEN_RIGHT,
    ContextInput,
    ContextPart,
    MaskedTag,
    TagField,
    parse_parts,
)


if TYPE_CHECKING:
    from collections.abc import Iterator


class Context:
    class TagsView:
        def __init__(self, parts: list[ContextPart]):
            self._parts = parts

        @property
        def _tags_by_index(self) -> list[int]:
            return [i for i, part in enumerate(self._parts) if isinstance(part, MaskedTag)]

        @property
        def _tags_by_name(self) -> dict[str, int]:
            return {
                part.name: i
                for i, part in enumerate(self._parts)
                if isinstance(part, MaskedTag) and part.name is not None
            }

        def __setitem__(self, key: int | str, value: ContextPart) -> None:
            if not isinstance(value, ContextPart):
                raise TypeError("New value must be a ContextPart (str or MaskedTag)")
            if isinstance(key, int):
                self._parts[self._tags_by_index[key]] = value
            elif isinstance(key, str):
                if key in self._tags_by_name:
                    self._parts[self._tags_by_name[key]] = value
                else:
                    raise KeyError(f"Tag name '{key}' does not exist.")
            else:
                raise TypeError("Key must be int or str")

        @overload
        def __getitem__(self, key: int | str) -> MaskedTag: ...

        @overload
        def __getitem__(self, key: slice) -> list[MaskedTag]: ...

        def __getitem__(self, key: int | str | slice) -> MaskedTag | list[MaskedTag]:
            if isinstance(key, int):
                return cast("MaskedTag", self._parts[self._tags_by_index[key]])
            elif isinstance(key, slice):
                return [cast("MaskedTag", self._parts[i]) for i in self._tags_by_index[key]]
            elif isinstance(key, str):
                if key in self._tags_by_name:
                    return cast("MaskedTag", self._parts[self._tags_by_name[key]])
                else:
                    raise KeyError(f"Tag name '{key}' does not exist.")
            raise TypeError("Key must be int, slice, or str")

        def __len__(self) -> int:
            return len(self._tags_by_index)

        def __iter__(self) -> Iterator[MaskedTag]:
            for i in self._tags_by_index:
                yield cast("MaskedTag", self._parts[i])

    def __init__(self, prefix: str, suffix: str, *args: ContextInput) -> None:
        _inner_parts = self._process_context_inputs(*args)

        # Remove prefix and suffix from the inner parts if present
        if (
            prefix
            and _inner_parts
            and isinstance(_inner_parts[0], str)
            and _inner_parts[0].startswith(prefix)
        ):
            _inner_parts[0] = _inner_parts[0].removeprefix(prefix)
        if (
            suffix
            and _inner_parts
            and isinstance(_inner_parts[-1], str)
            and _inner_parts[-1].endswith(suffix)
        ):
            _inner_parts[-1] = _inner_parts[-1].removesuffix(suffix)

        _str_inner_parts = "".join(str(part) for part in _inner_parts)
        if prefix and prefix in _str_inner_parts:
            raise InvalidFormatError(f"Nested or duplicate {prefix} tags are not allowed.")
        if suffix and suffix in _str_inner_parts:
            raise InvalidFormatError(f"Nested or duplicate {suffix} tags are not allowed.")

        self._prefix = prefix
        self._suffix = suffix
        self._parts = [prefix, *_inner_parts, suffix]

    @property
    def parts(self) -> list[ContextPart]:
        return self._parts

    @property
    def tags(self) -> TagsView:
        return Context.TagsView(self._parts)

    def to_string(
        self,
        fields: list[TagField] | Literal["all"] | None = None,
        infill_mode: Literal[True] | None = None,
    ) -> str:
        if not ((fields is None) ^ (infill_mode is None)):
            raise ValueError("Exactly one of fields or infill_mode must be specified.")
        content = ""
        if fields is not None:
            if fields == "all":
                fields = cast("list[TagField]", list(ALL_FIELDS))

            for part in self._parts:
                if isinstance(part, MaskedTag):
                    content += part.to_string(fields=fields)
                else:
                    content += str(part)
        if infill_mode is not None:
            content = ""
            for part in self._parts:
                if isinstance(part, MaskedTag):
                    if part.content is not None:
                        content += part.content
                    else:
                        content += part.to_string(fields="all")
                else:
                    content += part
            content = content[len(self._prefix) : len(content) - len(self._suffix)]
        return content

    def __repr__(self):
        return self.to_string(fields="all")

    @staticmethod
    def _process_context_inputs(*args: ContextInput) -> list[ContextPart]:
        parts = []
        for arg in args:
            if isinstance(arg, str):
                parts.extend(parse_parts(arg))
            elif isinstance(arg, MaskedTag):
                parts.append(arg)
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, str):
                        parts.extend(parse_parts(item))
                    elif isinstance(item, MaskedTag):
                        parts.append(item)
                    else:
                        raise TypeError("List items must be str or MaskedTag")
            else:
                raise TypeError(
                    f"Arguments must be str, MaskedTag, or list of str/MaskedTag. Got {type(arg)}"
                )
        return parts


class Query(Context):
    def __init__(self, *args: ContextInput) -> None:
        super().__init__(QUERY_PREFIX, QUERY_SUFFIX, *args)

        # Validate and standardize the tags
        tag_count = 0
        tag_names = set()
        for part in self._parts:
            if isinstance(part, MaskedTag):
                if part.id is not None and part.id != tag_count:
                    raise InvalidFormatError("Tag ids must be sequential starting from 0.")
                part.id = tag_count
                tag_count += 1

                if part.name is not None:
                    if part.name in tag_names:
                        raise InvalidFormatError(f"Tag name '{part.name}' already exists.")
                    tag_names.add(part.name)

                if part.content == "":
                    part.content = None

    def infill(self, response: Response | ContextInput) -> Result:
        """Fills tags in this query (self) with content from the provided response."""
        return infill(self, response)

    def __str__(self) -> str:
        return self.to_string(fields=["id", "desc", "content"])


class Response(Context):
    def __init__(self, *args: ContextInput) -> None:
        super().__init__(RESPONSE_PREFIX, RESPONSE_SUFFIX, *args)

    def infill(self, query: Query | ContextInput) -> Result:
        """Fills the tags in the provided query with content from this response (self)."""
        return infill(query, self)

    def __str__(self) -> str:
        return self.to_string(fields=["id", "content"])


class Result(Context):
    def __init__(self, *args: ContextInput) -> None:
        super().__init__("", "", *args)

    def __str__(self) -> str:
        return self.to_string(infill_mode=True)


def _repair_response_string(response_str: str) -> str:
    """Repair a malformed response string to make it parseable.

    This function attempts to fix common issues in response strings:
    1. Renumbers tag IDs to be sequential (0, 1, 2, ...)
    2. Handles responses with missing or incorrect tag IDs

    Args:
        response_str: The response string to repair

    Returns:
        A repaired response string with sequential tag IDs
    """
    # Pattern to match masked tags with optional well-formed ID
    # Note: This only matches IDs in correct format (m_\d+) or no ID at all.
    # Malformed IDs that don't match this pattern won't be captured, but that's
    # intentional - we want to renumber all tags regardless of their original IDs.
    tag_pattern = re.compile(
        re.escape(TAG_OPEN_LEFT)
        + r'(?:\s+id="m_\d+")?'  # Optional well-formed ID
        + r'((?:\s+\w+="[^"]*")*)'  # Other attributes
        + re.escape(TAG_OPEN_RIGHT)
        + r"(.*?)"  # Content
        + re.escape(TAG_END),
        re.DOTALL,
    )

    # Find all tags in the response
    matches = list(tag_pattern.finditer(response_str))

    if not matches:
        return response_str

    # Build repaired string by renumbering tags sequentially
    repaired = response_str
    offset = 0
    for idx, match in enumerate(matches):
        old_tag = match.group(0)
        attrs = match.group(1)  # Other attributes
        content = match.group(2)  # Content

        # Build new tag with sequential ID
        new_tag = f'{TAG_OPEN_LEFT} id="m_{idx}"{attrs}{TAG_OPEN_RIGHT}{content}{TAG_END}'

        # Calculate positions with offset
        start_pos = match.start() + offset
        end_pos = match.end() + offset

        # Replace in repaired string
        repaired = repaired[:start_pos] + new_tag + repaired[end_pos:]
        offset += len(new_tag) - len(old_tag)

    return repaired


def infill(
    query: Query | ContextInput, response: Response | ContextInput, strict: bool = False
) -> Result:
    """Combines query and response by infilling missing content.

    Args:
        query: The query containing masked tags to be filled
        response: The response containing content to fill the tags
        strict: If True, raises errors on format mismatches. If False, attempts to repair
                and merge responses in a best-effort manner.

    Returns:
        A Result object with tags filled from the response

    Raises:
        InvalidFormatError: If strict=True and there are format mismatches
    """
    if not isinstance(query, Query):
        query = Query(query)

    # When strict=False, try to repair the response string before parsing
    if not strict and isinstance(response, str):
        response_str = response
        try:
            # First attempt: try to parse as-is
            response = Response(response_str)
        except InvalidFormatError as e:
            # If parsing fails, try to repair the response string
            # Note: We check the error message to identify tag ID ordering errors.
            # This is somewhat fragile, but adding a new exception type would be a
            # breaking change. The error message comes from schemas.parse_parts()
            # which raises "Tag ids should be in order, got X at position Y."
            if "Tag ids should be in order" in str(e):
                warnings.warn(
                    "Response has malformed or out-of-order tag IDs. Attempting automatic repair.",
                    stacklevel=2,
                )
                repaired_response = _repair_response_string(response_str)
                try:
                    response = Response(repaired_response)
                except InvalidFormatError:
                    # If repair also fails, re-raise the original error
                    raise e from None
            else:
                # For other errors, just re-raise
                raise
    elif not isinstance(response, Response):
        response = Response(response)

    query_tags = list(query.tags)
    response_tags = list(response.tags)
    if len(query_tags) != len(response_tags):
        msg = (
            "Mismatch in number of tags between query and response. "
            f"Query has {len(query_tags)} tag(s), response has {len(response_tags)} tag(s)."
        )
        if strict:
            raise InvalidFormatError(msg)
        else:
            warnings.warn(msg + " Will merge as many as possible.", stacklevel=2)

    result_parts: list[ContextPart] = []
    for part in query.parts[1:-1]:  # Exclude prefix and suffix
        if isinstance(part, MaskedTag) and query_tags and response_tags:
            q_tag = query_tags.pop(0)
            r_tag = response_tags.pop(0)
            part = MaskedTag(
                id=q_tag.id,
                name=q_tag.name,
                desc=q_tag.desc,
                regex=q_tag.regex,
                content=r_tag.content if r_tag.content is not None else q_tag.content,
            )
        result_parts.append(part)

    return Result(result_parts)
