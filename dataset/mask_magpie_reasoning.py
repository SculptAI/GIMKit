import os
import random

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


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
    query_dict, response = {}, ""
    idx = 0
    for field in FIELD2DESC:
        if random.random() < 0.5:
            query_dict[field] = str(MaskedTag(id=idx, desc=FIELD2DESC[field]))
            response += str(MaskedTag(id=idx, content=example[field]))
            idx += 1
        else:
            query_dict[field] = example[field]
    query = TEMPLATE.format(**query_dict)
    return to_gim_format(query, response)


ds = load_dataset("Magpie-Align/Magpie-Reasoning-150K", split="train", num_proc=os.cpu_count())
ds = ds.map(_mask_some_fields, num_proc=os.cpu_count()).select_columns(
    [QUERY_COLUMN, RESPONSE_COLUMN]
)
save_dataset(ds, __file__)
