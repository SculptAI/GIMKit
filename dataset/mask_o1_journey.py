import random

from datasets import load_dataset
from utils import wrap_masked_io

from gimkit import MaskedTag, validate_wrapped_masked_io


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
    m_input = (
        question + "\n\n" + MaskedTag(desc=long_cot_desc) + "\n\n" + MaskedTag(desc=answer_desc)
    )
    m_output = str(MaskedTag(id=1, content=long_cot)) + str(MaskedTag(id=2, content=answer))
    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("GAIR/o1-journey", split="train")
ds = ds.map(_mask_cot_and_answer).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
