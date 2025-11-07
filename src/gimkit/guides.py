import re

from gimkit.schemas import MaskedTag


class BaseMixin:
    def __call__(
        self,
        name: str | None = None,
        desc: str | None = None,
        regex: str | None = None,
        content: str | None = None,
    ) -> MaskedTag:
        return MaskedTag(name=name, desc=desc, regex=regex, content=content)


class FormMixin:
    def single_word(self, name: str | None = None) -> MaskedTag:
        """A single word without spaces."""
        return MaskedTag(name=name, desc=self.single_word.__doc__, regex=r"\S+")

    def select(self, name: str | None = None, choices: list[str] | None = None) -> MaskedTag:
        """Choose one from the given options."""
        if not choices:
            raise ValueError("choices must be a non-empty list of strings.")
        desc = f"Choose one from the following options: {', '.join(choices)}."
        regex = "|".join(re.escape(choice) for choice in choices)
        return MaskedTag(name=name, desc=desc, regex=regex)


class PersonalInfoMixin:
    def person_name(self, name: str | None = None) -> MaskedTag:
        """A person's name, e.g., John Doe, Alice, Bob, Charlie Brown, 张三, etc."""
        return MaskedTag(name=name, desc=self.person_name.__doc__)

    def phone_number(self, name: str | None = None) -> MaskedTag:
        """A phone number, e.g., +1-123-456-7890, (123) 456-7890, 123-456-7890, etc."""

        # Adapted from https://regexr.com/38pvb
        regex = (
            r"(?:\+?(\d{1,3}))?([-. (]*(\d{3})[-. )]*)?((\d{3})[-. ]*(\d{2,4})(?:[-.x ]*(\d+))?)"
        )
        return MaskedTag(name=name, desc=self.phone_number.__doc__, regex=regex)

    def e_mail(self, name: str | None = None) -> MaskedTag:
        """An email address, e.g., john.doe@example.com, alice@example.com, etc."""

        # Adapted from https://regexr.com/3a2i5
        regex = r"([\w\.]+)@([\w\.]+)\.(\w+)"
        return MaskedTag(name=name, desc=self.e_mail.__doc__, regex=regex)


class Guide(BaseMixin, FormMixin, PersonalInfoMixin): ...


guide = Guide()
