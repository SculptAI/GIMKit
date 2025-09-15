"""Defines the schema for GIM. Some examples are given below.

>>> tag = MaskedTag(id=1, desc="Fill in with appropriate text")
>>> print(tag)
<|MASKED id="m_1" desc="Fill in with appropriate text"|><|/MASKED|>

>>> m_input = '<|M_INPUT|>This is an <|MASKED id="m_1"|><|/MASKED|> text.<|/M_INPUT|>'
>>> m_output = '<|M_OUTPUT|><|MASKED id="m_1"|>example<|/MASKED|></M_OUTPUT|>'
>>> validate_wrapped_masked_io(m_input, m_output)  # No exception means valid
"""

import re

from dataclasses import dataclass


@dataclass
class MaskedTag:
    id: int = None
    desc: str = None
    content: str = None

    def __post_init__(self):
        if self.id is not None and not isinstance(self.id, int):
            raise ValueError(f"{type(self.id)=}, {self.id=}, should be int or None")
        if self.desc is not None and not isinstance(self.desc, str):
            raise ValueError(f"{type(self.desc)=}, {self.desc=}, should be str or None")
        if self.content is not None and not isinstance(self.content, str):
            raise ValueError(f"{type(self.content)=}, {self.content=}, should be str or None")

    def __str__(self):
        masked_tag_str = "<|MASKED"

        if self.id is not None:
            masked_tag_str += f' id="m_{self.id}"'
        if self.desc is not None:
            escaped_desc = self.escape_in_attr_value(self.desc)
            masked_tag_str += f' desc="{escaped_desc}"'

        masked_tag_str += "|>"

        if self.content is not None:
            masked_tag_str += f"{self.content}"

        masked_tag_str += "<|/MASKED|>"
        return masked_tag_str

    def __repr__(self):
        return self.__str__()

    def __add__(self, other: str):
        if isinstance(other, str):
            return str(self) + other
        return NotImplemented

    def __radd__(self, other: str):
        if isinstance(other, str):
            return other + str(self)
        return NotImplemented

    @staticmethod
    def escape_in_attr_value(value: str) -> str:
        return value.replace('"', '\\"')


def validate_wrapped_masked_io(input: str, output: str):
    """Validate the wrapped masked input and output strings.

    Args:
        input (str): The wrapped masked input string to be validated.
        output (str): The wrapped masked output string to be validated.

    Raises:
        ValueError: If the input or output format is invalid.
    """
    if "<|M_INPUT|>" in input[11:] or "<|/M_INPUT|>" in input[:-12]:
        raise ValueError("Invalid input format: Nested <|M_INPUT|> tags are not allowed.")
    if "<|M_OUTPUT|>" in output[12:] or "<|/M_OUTPUT|>" in output[:-13]:
        raise ValueError("Invalid output format: Nested <|M_OUTPUT|> tags are not allowed.")

    open_tag_pattern = r'<\|MASKED(?: id="m_(\d+)")?(?: desc=".*?")?\|>'
    end_tag_pattern = r"<\|/MASKED\|>"
    i_open_matches = list(re.finditer(open_tag_pattern, input))
    i_end_tags = re.findall(end_tag_pattern, input)
    o_open_matches = list(re.finditer(open_tag_pattern, output))
    o_end_tags = re.findall(end_tag_pattern, output)

    if not (len(i_open_matches) == len(i_end_tags) == len(o_open_matches) == len(o_end_tags)):
        raise ValueError("Mismatched number of masked tags between input and output.")

    for idx, (i_open_tag, o_open_tag) in enumerate(
        zip(i_open_matches, o_open_matches, strict=False), start=1
    ):
        i_idx = int(i_open_tag.group(1)) if i_open_tag.group(1) is not None else idx
        o_idx = int(o_open_tag.group(1)) if o_open_tag.group(1) is not None else idx
        if i_idx != o_idx:
            raise ValueError(
                f"Mismatched masked tag ids between input and output at position {idx}: {i_idx} != {o_idx}"
            )
