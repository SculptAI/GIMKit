import random

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


random.seed(0)


def _mask_cot_and_answer(example: dict) -> dict:
    question = example["question"].strip()
    long_cot = example["longCOT"].split("####")[0].strip()
    answer = example["answer"].strip()
    long_cot_desc = random.choice(
        [
            "Step-by-step thinking and the answer",
            "Chain-of-thought reasoning and the answer",
            "Please provide a step-by-step explanation",
            "Please provide a chain-of-thought reasoning",
        ]
    )
    answer_desc = random.choice(
        [
            "The exact answer",
            "The final exact answer",
            "The short answer to the question",
        ]
    )
    query = question + "\n\n" + MaskedTag(desc=long_cot_desc) + "\n\n" + MaskedTag(desc=answer_desc)
    response = str(MaskedTag(id=0, content=long_cot)) + str(MaskedTag(id=1, content=answer))
    return to_gim_format(query, response)


ds = load_dataset("GAIR/o1-journey", split="train")
ds = ds.map(_mask_cot_and_answer).select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
save_dataset(ds, __file__)
