import random

from datasets import load_dataset
from utils import wrap_masked_io

from gimkit import MaskedTag, validate_wrapped_masked_io


random.seed(0)


def _mask_internal_thinking(example: dict) -> dict:
    desc = random.choice(
        [
            "模仿人类，进行细致的分析和思考，写出来完整的推理步骤（援引香港的法条）",
            "请进行完整的思考，以得出最后的结论",
            "如果你是一名香港律师，怎么从问题得到答案的，写出中间的想法",
        ]
    )
    m_input = random.choice(
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
    m_output = str(MaskedTag(id=0, content=example["thinking"]))
    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("HKAIR-Lab/HK-O1aw-SFT-16K", split="train")
ds = ds.map(_mask_internal_thinking).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
