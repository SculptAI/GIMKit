from gimkit.models.utils import json_responses_to_gim_response
from gimkit.prompts import DEMO_CONVERSATION_MSGS, DEMO_CONVERSATION_MSGS_JSON
from gimkit.schemas import validate


def test_demo_conversation_validity():
    for idx in range(0, len(DEMO_CONVERSATION_MSGS), 2):
        user_msg = DEMO_CONVERSATION_MSGS[idx]["content"]
        assistant_msg = DEMO_CONVERSATION_MSGS[idx + 1]["content"]
        validate(user_msg, assistant_msg)


def test_demo_conversation_json_validity():
    for idx in range(0, len(DEMO_CONVERSATION_MSGS_JSON), 2):
        user_msg = DEMO_CONVERSATION_MSGS_JSON[idx]["content"]
        json_response = DEMO_CONVERSATION_MSGS_JSON[idx + 1]["content"]
        # Convert JSON response to GIM response format
        gim_response = json_responses_to_gim_response(json_response)
        # Validate the converted response
        validate(user_msg, gim_response)


def test_two_prompts_equivalence():
    for idx in range(0, len(DEMO_CONVERSATION_MSGS), 2):
        json_response = DEMO_CONVERSATION_MSGS_JSON[idx + 1]["content"]
        gim_response = json_responses_to_gim_response(json_response)
        assistant_msg = DEMO_CONVERSATION_MSGS[idx + 1]["content"]
        assert gim_response == assistant_msg
