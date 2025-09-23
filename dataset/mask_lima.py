import random

from datasets import load_dataset
from utils import QUERY_COLUMN, RESPONSE_COLUMN, save_dataset, to_gim_format

from gimkit import MaskedTag


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
    response = str(MaskedTag(id=0, content=conversations[masked_id]))
    conversations[masked_id] = str(MaskedTag())

    # Format the conversations
    query = ""
    for i in range(0, conv_len, 2):
        query += template.format(conversations[i], conversations[i + 1])

    return to_gim_format(query, response)


ds = load_dataset("Ki-Seki/GAIR_lima", split="train")
ds = ds.map(_mask_one_remark_in_conversation).select_columns([QUERY_COLUMN, RESPONSE_COLUMN])
save_dataset(ds, __file__)
