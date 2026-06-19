from dataclasses import dataclass
from typing import Literal, TypeAlias

from gimkit.contexts import Result


ErrorMode: TypeAlias = Literal["raise", "collect"]


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """The parsed result or parsing error for one raw model generation."""

    raw_response: str
    result: Result | None = None
    error_type: str | None = None
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        """Whether the raw generation was parsed and infilled successfully."""
        return self.result is not None
