import os
import re

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


TAG2DESC = {
    "thinking": "Your initial thought process goes here.",
    "reasoning": "Your step-by-step reasoning goes here. This is your internal thought process, not the final answer. You can create as many reasoning steps as necessary in your process.",
    "reflection": "Your reflection on your reasoning, checking for errors or improvements. You can create as many reflection steps as necessary in your process.",
    "adjustment": "Any adjustments to your thinking based on your reflection.",
    "output": "Your final, concise answer to the query. This is the only part that will be shown to the user.",
}
TAGS = ["thinking", "reasoning", "reflection", "adjustment", "output"]


def _extract_tags_content(example: dict) -> dict:
    """Extracts tags and their content from the generation field of the example.

    Example:
    `‹thinking>Thinking goes here.</thinking><reasoning>Reasoning goes here.</reasoning>` ->
    `{'thinking': 'Thinking goes here.', 'reasoning': 'Reasoning goes here.'}`
    """  # noqa: RUF002
    if example["generation"] is None:
        example["generation"] = ""

    # Regular expression to match tags and their content with various brackets
    pattern = r"(?:<|\‹)(\w+)(?:>|\›)(.*?)(?:</|\</|\‹/|</\›)\1(?:>|\›)"

    matches = re.findall(pattern, example["generation"], re.DOTALL)
    result = {}
    for tag, content in matches:
        result[tag] = content.strip()
    example["generation"] = {tag: result.get(tag, "") for tag in TAGS}
    return example


def _is_valid_example(example: dict) -> bool:
    """All tags must be present and non-empty."""
    return all(value not in ("", None) for value in example["generation"].values())


def _mask_tags_content(example: dict) -> dict:
    query = example["question"].strip() + "".join(
        [f"\n\n{MaskedTag(desc=TAG2DESC[tag])}" for tag in TAGS]
    )
    response = "".join(
        [str(MaskedTag(id=idx, content=example["generation"][tag])) for idx, tag in enumerate(TAGS)]
    )
    return to_gim_format(query, response)


ds = load_dataset("thesven/gsm8k-reasoning", split="train")
ds = (
    ds.map(_extract_tags_content, num_proc=os.cpu_count())
    .filter(_is_valid_example, num_proc=os.cpu_count())
    .map(_mask_tags_content, num_proc=os.cpu_count())
    .select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
)
save_dataset(ds, __file__)
