import random

from utils import MaskedTag, validate_wrapped_masked_io, wrap_masked_io

from datasets import concatenate_datasets, load_dataset


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
    m_input = (
        f"## Question\n\n{example['problem'].strip()}"
        + "\n\n## CoT Reasoning"
        + "".join(
            [
                random.choice(
                    [
                        "\n\n" + MaskedTag(id=idx + 1, desc=desc),
                        "\n\n" + MaskedTag(desc=desc),
                        "\n\n" + MaskedTag(),
                    ]
                )
                for idx in range(len(example["steps"]))
            ]
        )
    )
    m_output = "".join([str(MaskedTag(id=idx + 1, content=step)) for idx, step in enumerate(example["steps"])])
    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("Qwen/ProcessBench")
ds = concatenate_datasets([ds[split] for split in ds])
ds = ds.filter(_is_correct_reasoning).map(_mask_process).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
