import random

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


random.seed(0)


def _mask_rationale_and_target(example: dict) -> dict:
    query = (
        example["source"].strip()
        + random.choice(
            [
                "\n\nSolution: " + MaskedTag(desc="rationale for the answer"),
                "\n\nResponse: " + MaskedTag(desc="Let's think step by step."),
                "\n\n" + MaskedTag(desc="rationale for the answer"),
                "\n\n" + MaskedTag(desc="reasoning process"),
            ]
        )
        + random.choice(
            [
                "\n\nAnswer: " + MaskedTag(desc="the precise target answer"),
                "\n\nThe key answer: " + MaskedTag(),
                "\n\n" + MaskedTag(desc="A short target answer"),
            ]
        )
    )
    response = str(MaskedTag(id=0, content=example["rationale"])) + str(
        MaskedTag(id=1, content=example["target"])
    )
    return to_gim_format(query, response)


ds = load_dataset("kaist-ai/CoT-Collection", split="train", trust_remote_code=True)
ds = ds.map(_mask_rationale_and_target).select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
save_dataset(ds, __file__)
