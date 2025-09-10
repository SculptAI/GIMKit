import random

from datasets import load_dataset
from utils import wrap_masked_io

from simfill import MaskedTag, validate_wrapped_masked_io


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
    m_input = random.choice(
        [
            f"{problem}\n\n{MaskedTag(desc=desc)}",
            f"{problem}\n\n---\n\n{MaskedTag(desc=desc)}",
            f"<question>{problem}</question>\n<answer>{MaskedTag(desc=desc)}</answer>",
            f"## Question\n\n{problem}\n\n## Answer\n\n{MaskedTag(desc=desc)}",
            f"## Question\n\n{problem}\n\n## CoT Reasoning and the Answer\n\n{MaskedTag(desc=desc)}",
            f"Question: {problem}\n\nAnswer: {MaskedTag(desc=desc)}",
        ]
    )
    m_output = str(MaskedTag(id=1, content=example["solution"]))
    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("AI-MO/NuminaMath-CoT", split="train")
ds = ds.map(_mask_solution).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
