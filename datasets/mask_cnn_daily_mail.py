import random

import numpy as np

from utils import gen_possion_masked, validate_wrapped_masked_io, wrap_masked_io

from datasets import load_dataset


random.seed(0)
np.random.seed(0)


def _mask_possion_4(example: dict) -> dict:
    text = example["article"]
    m_input, m_output = gen_possion_masked(text, lam=4)
    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("abisee/cnn_dailymail", name="3.0.0", split="train")
ds = ds.map(_mask_possion_4).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
