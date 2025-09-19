import random

import numpy as np

from gimkit import MaskedTag
from gimkit.schemas import INPUT_PREFIX, INPUT_SUFFIX, OUTPUT_PREFIX, OUTPUT_SUFFIX


def wrap_masked_io(masked_input: str, masked_output: str) -> tuple[str, str]:
    """Wrap the masked output with surrounding tags.

    Args:
        masked_input (str): The masked input string to be wrapped.
        masked_output (str): The masked output string to be wrapped.

    Returns:
        tuple: The wrapped masked input and output strings.
    """
    return INPUT_PREFIX + masked_input + INPUT_SUFFIX, OUTPUT_PREFIX + masked_output + OUTPUT_SUFFIX


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

    def gen_mask_ranges(text: str, mask_num: int) -> list[tuple[int, int]]:
        indices = random.sample(range(len(text)), mask_num * 2)
        indices.sort()
        ranges = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
        return ranges

    def gen_mask_io(text: str, ranges: list[tuple[int, int]]) -> tuple[str, str]:
        input, output = "", ""
        last_end = 0
        for idx, (start, end) in enumerate(ranges):
            input += text[last_end:start] + random.choice(
                [
                    MaskedTag(id=idx),
                    MaskedTag(),
                    MaskedTag(desc="Fill in with appropriate text"),
                ]
            )
            output += MaskedTag(id=idx, content=text[start:end])
            last_end = end
        input += text[last_end:]
        return input, output

    mask_nums = np.random.poisson(lam=lam)
    ranges = gen_mask_ranges(text, mask_nums)
    m_input, m_output = gen_mask_io(text, ranges)
    return m_input, m_output
