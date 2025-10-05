from gimkit.schemas import MaskedTag


class BaseMixin:
    def __call__(
        self, name: str | None = None, desc: str | None = None, content: str | None = None
    ) -> MaskedTag:
        return MaskedTag(name=name, desc=desc, content=content)


class FormMixin:  # pragma: no cover
    def single_word(self, name: str | None = None) -> MaskedTag:
        """A single word without spaces."""
        return MaskedTag(name=name, desc=self.single_word.__doc__)

    def select(self, name: str | None = None, choices: list[str] | None = None) -> MaskedTag:
        """Choose one from the given options."""
        if not choices:
            raise ValueError("choices must be a non-empty list of strings.")
        desc = f"Choose one from the following options: {', '.join(choices)}."
        return MaskedTag(name=name, desc=desc)


class PersonalInfoMixin:  # pragma: no cover
    def person_name(self, name: str | None = None) -> MaskedTag:
        """A person's name, e.g., John Doe, Alice, Bob, Charlie Brown, etc."""
        return MaskedTag(name=name, desc=self.person_name.__doc__)

    def phone_number(self, name: str | None = None) -> MaskedTag:
        """A phone number, e.g., +1-123-456-7890, (123) 456-7890, 123-456-7890, etc."""
        return MaskedTag(name=name, desc=self.phone_number.__doc__)

    def e_mail(self, name: str | None = None) -> MaskedTag:
        """An email address, e.g., john.doe@example.com, alice@example.com, etc."""
        return MaskedTag(name=name, desc=self.e_mail.__doc__)


class Guide(BaseMixin, FormMixin, PersonalInfoMixin): ...


guide = Guide()
