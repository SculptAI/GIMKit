import random

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


random.seed(0)


def _mask_internal_thinking(example: dict) -> dict:
    desc = random.choice(
        [
            "模仿人类，进行细致的分析和思考，写出来完整的推理步骤（援引香港的法条）",
            "请进行完整的思考，以得出最后的结论",
            "如果你是一名香港律师，怎么从问题得到答案的，写出中间的想法",
        ]
    )
    query = random.choice(
        [
            (
                f"问题：{example['prompt'].strip()}\n"
                f"思考：{MaskedTag(desc=desc)}\n"
                f"回答：{example['answer']}"
            ),
            (
                f"## 问题\n\n{example['prompt'].strip()}\n\n"
                f"## 思考\n\n{MaskedTag(desc=desc)}\n\n"
                f"## 回答\n\n{example['answer']}"
            ),
            (
                f"{example['prompt'].strip()}\n\n---\n\n"
                f"{MaskedTag(desc=desc)}\n\n---\n\n"
                f"{example['answer']}"
            ),
        ]
    )
    response = str(MaskedTag(id=0, content=example["thinking"]))
    return to_gim_format(query, response)


ds = load_dataset("HKAIR-Lab/HK-O1aw-SFT-16K", split="train")
ds = ds.map(_mask_internal_thinking).select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
save_dataset(ds, __file__)
