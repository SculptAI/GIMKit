from gimkit.parsing import validate
from gimkit.prompts import DEMO_CONVERSATION_MSGS


def test_demo_conversation_validity():
    for idx in range(0, len(DEMO_CONVERSATION_MSGS), 2):
        user_msg = DEMO_CONVERSATION_MSGS[idx]["content"]
        assistant_msg = DEMO_CONVERSATION_MSGS[idx + 1]["content"]
        validate(user_msg, assistant_msg)
