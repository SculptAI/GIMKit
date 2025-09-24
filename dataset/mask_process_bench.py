import os
import random

from datasets import concatenate_datasets, load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


random.seed(0)


def _is_correct_reasoning(example: dict) -> bool:
    """Check if the example is a correct process.

    Note:
    - label == -1 means the process has no mistakes.
    - final_answer_correct == True means the final answer is correct.
    """
    return example["label"] == -1 and example["final_answer_correct"]


def _mask_process(example: dict) -> dict:
    desc = random.choice(
        [
            "One single solution step to the question above",
            "A step in the solution to the question above",
            "Single reasoning step",
            "A step in the solution",
            "Write a step in the solution",
        ]
    )
    query = (
        f"## Question\n\n{example['problem'].strip()}"
        + "\n\n## CoT Reasoning"
        + "".join(
            [
                random.choice(
                    [
                        "\n\n" + MaskedTag(id=idx, desc=desc),
                        "\n\n" + MaskedTag(desc=desc),
                        "\n\n" + MaskedTag(),
                    ]
                )
                for idx in range(len(example["steps"]))
            ]
        )
    )
    response = "".join(
        [str(MaskedTag(id=idx, content=step)) for idx, step in enumerate(example["steps"])]
    )
    return to_gim_format(query, response)


ds = load_dataset("Qwen/ProcessBench", num_proc=os.cpu_count())
ds = concatenate_datasets([ds[split] for split in ds])
ds = (
    ds.filter(_is_correct_reasoning, num_proc=os.cpu_count())
    .map(_mask_process, num_proc=os.cpu_count())
    .select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
)
save_dataset(ds, __file__)
