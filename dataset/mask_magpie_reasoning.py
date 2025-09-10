import random

from datasets import load_dataset
from utils import wrap_masked_io

from simfill import MaskedTag, validate_wrapped_masked_io


random.seed(0)

TEMPLATE = """Fill in the missing parts with appropriate values.

instruction: {instruction}\n
response: {response}\n
intent: {intent}\n
knowledge: {knowledge}\n
difficulty: {difficulty}\n
input_quality: {input_quality}\n
quality_explanation: {quality_explanation}\n
task_category: {task_category}\n
"""

FIELD2DESC = {
    "instruction": "A description of the user query or task.",
    "response": "The response or solution to the user query.",
    "intent": "What is the user trying to achieve with the query.",
    "knowledge": "What knowledge is needed to respond to the user query.",
    "difficulty": "Difficulty level (very easy/easy/medium/hard/very hard) of the user query",
    "input_quality": "A rating from very poor/poor/average/good/excellent of the user query.",
    "quality_explanation": "Provide an assessment highlighting the strengths and/or weaknesses of the user's query.",
    "task_category": (
        'One task category tag from "Information seeking", '
        '"Reasoning", "Planning", "Editing", '
        '"Coding & Debugging", "Math", "Role playing", '
        '"Data analysis", "Creative writing", "Advice seeking", '
        '"Brainstorming", and "Others" that best describes the user query.'
    ),
}


def _mask_some_fields(example: dict) -> dict:
    m_input_dict, m_output = {}, ""
    idx = 1
    for field in FIELD2DESC:
        if random.random() < 0.5:
            m_input_dict[field] = str(MaskedTag(id=idx, desc=FIELD2DESC[field]))
            m_output += str(MaskedTag(id=idx, content=example[field]))
            idx += 1
        else:
            m_input_dict[field] = example[field]
    m_input = TEMPLATE.format(**m_input_dict)
    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("Magpie-Align/Magpie-Reasoning-150K", split="train")
ds = ds.map(_mask_some_fields).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
