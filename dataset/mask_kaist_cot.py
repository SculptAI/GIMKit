import random

from datasets import load_dataset
from utils import wrap_masked_io

from simfill import MaskedTag, validate_wrapped_masked_io


random.seed(0)


def _mask_rationale_and_target(example: dict) -> dict:
    m_input = (
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
    m_output = str(MaskedTag(id=1, content=example["rationale"])) + str(
        MaskedTag(id=2, content=example["target"])
    )
    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("kaist-ai/CoT-Collection", split="train")
ds = ds.map(_mask_rationale_and_target).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
