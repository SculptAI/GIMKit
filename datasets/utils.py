import random
import re

from dataclasses import dataclass

import numpy as np


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


def wrap_masked_io(masked_input: str, masked_output: str) -> tuple[str, str]:
    """Wrap the masked output with <|M_INPUT|> and <|M_OUTPUT|> tags.

    Args:
        masked_input (str): The masked input string to be wrapped.
        masked_output (str): The masked output string to be wrapped.

    Returns:
        tuple: The wrapped masked input and output strings.
    """
    return f"<|M_INPUT|>{masked_input}<|/M_INPUT|>", f"<|M_OUTPUT|>{masked_output}<|/M_OUTPUT|>"


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

    for idx, (i_open_tag, o_open_tag) in enumerate(zip(i_open_matches, o_open_matches, strict=False), start=1):
        i_idx = int(i_open_tag.group(1)) if i_open_tag.group(1) is not None else idx
        o_idx = int(o_open_tag.group(1)) if o_open_tag.group(1) is not None else idx
        if i_idx != o_idx:
            raise ValueError(
                f"Mismatched masked tag ids between input and output at position {idx}: {i_idx} != {o_idx}"
            )


def gen_possion_masked(text: str, lam: int) -> tuple[str, str]:
    """Generate masked input and output using Poisson distribution.

    Args:
        text (str): The original text to be masked.
        lam (int): The lambda parameter for the Poisson distribution, controlling the average number of masks.

    Example:
        >>> m_input, m_output = gen_possion_masked("This is an example text for masking.", 2)
        >>> print(m_input)
        This is an <|MASKED id="m_1"|><|/MASKED|> text fo<|MASKED id="m_2"|><|/MASKED|>.
        >>> print(m_output)
        <|MASKED id="m_1"|>example<|/MASKED|><|MASKED id="m_2"|>r masking<|/MASKED|>
    """

    def gen_mask_ranges(text: str, mask_num: int) -> list[tuple[int]]:
        indices = random.sample(range(len(text)), mask_num * 2)
        indices.sort()
        ranges = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
        return ranges

    def gen_mask_io(text: str, ranges: list[tuple[int]]) -> tuple[str, str]:
        input, output = "", ""
        last_end = 0
        for idx, (start, end) in enumerate(ranges):
            input += text[last_end:start] + random.choice(
                [MaskedTag(id=idx + 1), MaskedTag(), MaskedTag(desc="Fill in with appropriate text")]
            )
            output += MaskedTag(id=idx + 1, content=text[start:end])
            last_end = end
        input += text[last_end:]
        return input, output

    mask_nums = np.random.poisson(lam=lam)
    ranges = gen_mask_ranges(text, mask_nums)
    m_input, m_output = gen_mask_io(text, ranges)
    return m_input, m_output
