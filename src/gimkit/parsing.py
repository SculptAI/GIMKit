"""Parsing and validation functions for GIM schemas."""

from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import (
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    TAG_END_PATTERN,
    TAG_FULL_PATTERN,
    TAG_OPEN_PATTERN,
    ContextPart,
    MaskedTag,
)


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
        tag_regex = match.group(4)
        tag_content = match.group(5)
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
        parts.append(
            MaskedTag(id=tag_id, name=tag_name, desc=tag_desc, regex=tag_regex, content=tag_content)
        )

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
