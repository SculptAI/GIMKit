import os
import random

import numpy as np

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, gen_possion_masked, save_dataset, to_gim_format


random.seed(0)
np.random.seed(0)


def _mask_possion_4(example: dict) -> dict:
    text = example["article"]
    query, response = gen_possion_masked(text, lam=4)
    return to_gim_format(query, response)


ds = load_dataset("abisee/cnn_dailymail", name="3.0.0", split="train", num_proc=os.cpu_count())
ds = ds.map(_mask_possion_4, num_proc=os.cpu_count()).select_columns(
    [QUERY_COLUMN, RESPONSE_COLUMN]
)
save_dataset(ds, __file__)
