import random

from datasets import load_dataset
from utils import wrap_masked_io

from gimkit import MaskedTag, validate_wrapped_masked_io


random.seed(0)


def _mask_one_remark_in_conversation(example: dict) -> dict:
    template = random.choice(
        [
            "Character A: {}\n\nCharacter B: {}\n\n",
            "## Character A\n\n{}\n\n## Character B\n\n{}\n\n",
            "<CharacterA>{}</CharacterA>\n<CharacterB>{}</CharacterB>\n",
            "> **Character A:** {}\n> **Character B:** {}\n",
            "[Character A] {}\n[Character B] {}\n",
            "-> Character A: {}\n<- Character B: {}\n",
            "### Character A\n{}\n\n### Character B\n{}\n\n",
            'Character A said: "{}"\nCharacter B replied: "{}"\n',
        ]
    )
    conversations = example["conversations"]

    # Leave out the last message if odd
    conv_len = len(conversations) - (len(conversations) % 2)

    # Randomly mask one remark
    masked_id = random.randint(0, conv_len - 1)
    m_output = str(MaskedTag(id=0, content=conversations[masked_id]))
    conversations[masked_id] = str(MaskedTag())

    # Format the conversations
    m_input = ""
    for i in range(0, conv_len, 2):
        m_input += template.format(conversations[i], conversations[i + 1])

    m_input, m_output = wrap_masked_io(m_input, m_output)
    validate_wrapped_masked_io(m_input, m_output)
    return {"m_input": m_input, "m_output": m_output}


ds = load_dataset("Ki-Seki/GAIR_lima", split="train")
ds = ds.map(_mask_one_remark_in_conversation).select_columns(["m_input", "m_output"])
ds.to_json("data/" + __file__.split("/")[-1].replace(".py", ".jsonl"), force_ascii=False)
