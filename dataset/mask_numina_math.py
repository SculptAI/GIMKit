import os
import random

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


random.seed(0)


def _mask_solution(example: dict) -> dict:
    problem = example["problem"].strip()
    desc = random.choice(
        [
            "Step-by-step thinking and the answer",
            "Chain-of-thought reasoning and the answer",
            "Please provide a step-by-step explanation",
            "Please provide a chain-of-thought reasoning",
            "Solution to the above problem",
            "Solution to the problem",
            "Solution to the question",
            "Give a step-by-step explanation",
        ]
    )
    query = random.choice(
        [
            f"{problem}\n\n{MaskedTag(desc=desc)}",
            f"{problem}\n\n---\n\n{MaskedTag(desc=desc)}",
            f"<question>{problem}</question>\n<answer>{MaskedTag(desc=desc)}</answer>",
            f"## Question\n\n{problem}\n\n## Answer\n\n{MaskedTag(desc=desc)}",
            f"## Question\n\n{problem}\n\n## CoT Reasoning and the Answer\n\n{MaskedTag(desc=desc)}",
            f"Question: {problem}\n\nAnswer: {MaskedTag(desc=desc)}",
        ]
    )
    response = str(MaskedTag(id=0, content=example["solution"]))
    return to_gim_format(query, response)


ds = load_dataset("AI-MO/NuminaMath-CoT", split="train", num_proc=os.cpu_count())
ds = ds.map(_mask_solution, num_proc=os.cpu_count()).select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
save_dataset(ds, __file__)
