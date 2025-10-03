from __future__ import annotations

from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from collections.abc import Callable

from gimkit.schemas import MaskedTag


class GuideDescriptor:
    """A descriptor that creates guide methods with descriptions."""

    def __init__(self, desc: str, validator: Callable[..., str] | None = None):
        """Initialize a guide descriptor.

        Args:
            desc: The description to use for the MaskedTag.
            validator: Optional function that takes kwargs and returns a description.
                      If provided, it overrides the default desc.
        """
        self.desc = desc
        self.validator = validator

    def __get__(self, obj: object, objtype: type | None = None) -> Callable[..., MaskedTag]:
        """Return a function that creates a MaskedTag."""

        def guide_method(name: str | None = None, **kwargs) -> MaskedTag:
            desc = self.validator(**kwargs) if self.validator else self.desc
            return MaskedTag(name=name, desc=desc)

        guide_method.__doc__ = self.desc
        return guide_method


class Guide:
    """Guide system for creating masked tags with common patterns."""

    # Form guides
    single_word = GuideDescriptor("A single word without spaces.")

    # Personal information guides
    person_name = GuideDescriptor(
        "A person's name, e.g., John Doe, Alice, Bob, Charlie Brown, etc."
    )
    phone_number = GuideDescriptor(
        "A phone number, e.g., +1-123-456-7890, (123) 456-7890, 123-456-7890, etc."
    )
    e_mail = GuideDescriptor(
        "An email address, e.g., john.doe@example.com, alice@example.com, etc."
    )

    def __call__(
        self, name: str | None = None, desc: str | None = None, content: str | None = None
    ) -> MaskedTag:
        """Create a MaskedTag with custom parameters.

        Args:
            name: Optional name for the tag.
            desc: Optional description for the tag.
            content: Optional content for the tag.

        Returns:
            A MaskedTag instance.
        """
        return MaskedTag(name=name, desc=desc, content=content)

    def options(self, name: str | None = None, choices: list[str] | None = None) -> MaskedTag:
        """Create a MaskedTag for choosing from options.

        Args:
            name: Optional name for the tag.
            choices: List of choices to select from.

        Returns:
            A MaskedTag instance.

        Raises:
            ValueError: If choices is empty or None.
        """
        if not choices:
            raise ValueError("choices must be a non-empty list of strings.")
        desc = f"Choose one from the following options: {', '.join(choices)}."
        return MaskedTag(name=name, desc=desc)


guide = Guide()
