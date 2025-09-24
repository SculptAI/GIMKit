import random

from pathlib import Path

import numpy as np

from datasets import Dataset

from gimkit import MaskedTag, validate
from gimkit.schemas import QUERY_PREFIX, QUERY_SUFFIX, RESPONSE_PREFIX, RESPONSE_SUFFIX


QUERY_COLUMN = "gim_query"
RESPONSE_COLUMN = "gim_response"

COLUMNS = [QUERY_COLUMN, RESPONSE_COLUMN]


def gen_possion_masked(text: str, lam: int) -> tuple[str, str]:
    """Generate GIM query and response using Poisson distribution.
    Note: query and response do not contain prefix and suffix!

    Args:
        text (str): The original text to be masked.
        lam (int): The lambda parameter for the Poisson distribution, controlling the average number of masks.
    """

    def gen_mask_ranges(text: str, mask_num: int) -> list[tuple[int, int]]:
        indices = random.sample(range(len(text)), mask_num * 2)
        indices.sort()
        ranges = [(indices[i], indices[i + 1]) for i in range(0, len(indices), 2)]
        return ranges

    def gen_gim_query_response(text: str, ranges: list[tuple[int, int]]) -> tuple[str, str]:
        query, response = "", ""
        last_end = 0
        for idx, (start, end) in enumerate(ranges):
            query += text[last_end:start] + random.choice(
                [
                    MaskedTag(id=idx),
                    MaskedTag(),
                    MaskedTag(desc="Fill in with appropriate text"),
                    MaskedTag(id=idx, desc="Please provide the missing text"),
                ]
            )
            response += MaskedTag(id=idx, content=text[start:end])
            last_end = end
        query += text[last_end:]
        return query, response

    mask_nums = np.random.poisson(lam=lam)
    ranges = gen_mask_ranges(text, mask_nums)
    raw_query, raw_response = gen_gim_query_response(text, ranges)
    return raw_query, raw_response


def to_gim_format(raw_query: str, raw_response: str) -> dict[str, str]:
    query = QUERY_PREFIX + raw_query + QUERY_SUFFIX
    response = RESPONSE_PREFIX + raw_response + RESPONSE_SUFFIX
    validate(query, response)
    return {QUERY_COLUMN: query, RESPONSE_COLUMN: response}


def save_dataset(
    ds: Dataset, script_path: str, save_dir: str = "data", dataset_name: str = "GIM-SFT"
):
    # Ensure the ds only has the required columns
    assert set(ds.column_names) == set(COLUMNS), (
        f"Dataset columns should be {COLUMNS}, got {ds.column_names}"
    )

    # Ensure the first row is valid
    validate(ds[0][QUERY_COLUMN], ds[0][RESPONSE_COLUMN])

    subset_name = Path(script_path).stem.removeprefix("mask_")
    output_path = Path(save_dir) / dataset_name / subset_name / "train.jsonl"
    ds.to_json(output_path.as_posix(), force_ascii=False)
