import random

import numpy as np

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, gen_possion_masked, save_dataset, to_gim_format


random.seed(0)
np.random.seed(0)


def _mask_possion_4(example: dict) -> dict:
    text = f"{example['headLine']}\n{example['broadcastDate']}\n{example['newsBeginning']}\n{example['newsRemainder']}"
    query, response = gen_possion_masked(text, lam=4)
    return to_gim_format(query, response)


ds = load_dataset("Ki-Seki/UHGEvalDataset", name="full", split="validation")
ds = ds.map(_mask_possion_4).select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
save_dataset(ds, __file__)
