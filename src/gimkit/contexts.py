import warnings

from copy import deepcopy
from typing import Literal, cast, overload

from gimkit.exceptions import InvalidFormatError
from gimkit.schemas import (
    QUERY_PREFIX,
    QUERY_SUFFIX,
    RESPONSE_PREFIX,
    RESPONSE_SUFFIX,
    ContextInput,
    ContextPart,
    MaskedTag,
    parse_parts,
)


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

        def __len__(self):
            return len(self._tags_by_index)

        def __iter__(self):
            for i in self._tags_by_index:
                yield self._parts[i]

    def __init__(self, prefix: str, suffix: str, *args: ContextInput) -> None:
        self._prefix = prefix
        self._suffix = suffix

        self._parts: list[ContextPart] = [prefix]
        for arg in args:
            if isinstance(arg, str):
                self._parts.extend(parse_parts(arg))
            elif isinstance(arg, MaskedTag):
                self._parts.append(arg)
            elif isinstance(arg, list):
                for item in arg:
                    if isinstance(item, str):
                        self._parts.extend(parse_parts(item))
                    elif isinstance(item, MaskedTag):
                        self._parts.append(item)
                    else:
                        raise TypeError("List items must be str or MaskedTag")
            else:
                raise TypeError("Arguments must be str, MaskedTag, or list of str/MaskedTag")
        self._parts.append(suffix)

    @property
    def parts(self) -> list[ContextPart]:
        return self._parts

    @property
    def tags(self) -> TagsView:
        return Context.TagsView(self._parts)

    def to_string(
        self,
        fields: list[Literal["id", "name", "desc", "content"]] | Literal["all"] | None = None,
        infill_mode: Literal[True] | None = None,
    ) -> str:
        if not ((fields is None) ^ (infill_mode is None)):
            raise ValueError("Exactly one of fields or infill_mode must be specified.")
        content = ""
        if fields is not None:
            if fields == "all":
                fields = ["id", "name", "desc", "content"]

            for part in self._parts:
                if isinstance(part, MaskedTag):
                    content += part.to_string(fields=fields)
                else:
                    content += str(part)
        if infill_mode is not None:
            content = ""
            for part in self._parts:
                if isinstance(part, MaskedTag) and part.content is not None:
                    content += part.content
                else:
                    content += str(part)
            content = content[len(self._prefix) : -len(self._suffix)]
        return content


class Response(Context):
    def __init__(self, *args: ContextInput) -> None:
        super().__init__(RESPONSE_PREFIX, RESPONSE_SUFFIX, *args)

    def __str__(self) -> str:
        return self.to_string(infill_mode=True)


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

    def infill(self, response: str | Response | list[MaskedTag]) -> Response:
        if isinstance(response, str):
            tmp_parts = parse_parts(response)
            response_tags = [part for part in tmp_parts if isinstance(part, MaskedTag)]
        elif isinstance(response, Response):
            response_tags = list(response.tags)
        elif isinstance(response, list) and all(isinstance(part, MaskedTag) for part in response):
            response_tags = response
        else:
            raise TypeError("Response must be str, Response, or list of MaskedTag")

        # Create a new parts list without the prefix/suffix
        new_parts = deepcopy(self._parts[1:-1])
        for part in new_parts:
            if isinstance(part, MaskedTag):
                if response_tags:
                    part.content = response_tags.pop(0).content
                else:
                    warnings.warn(
                        "Not enough tags in response to fill the query tags.", stacklevel=2
                    )
                    break

        if len(response_tags) > 0:
            warnings.warn(
                f"There are {len(response_tags)} unused tags in the response.", stacklevel=2
            )

        return Response(new_parts)

    def __str__(self) -> str:
        return self.to_string(fields=["id", "desc", "content"])
