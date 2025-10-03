"""
Example demonstrating how to use few-shot prompts with non-GIM LLMs.

This example shows how to use the built-in system prompt and few-shot examples
to enable non-GIM LLMs (like GPT-4, Claude, etc.) to generate responses in the
Guided Infilling Modeling (GIM) format.
"""

from gimkit import Query, build_few_shot_messages, build_few_shot_prompt
from gimkit import guide as g


# ─── 1. Create a Query ─────────────────────────────────────────────────────────

query = f"""I'm {g.person_name(name="pred")}. Hello, {g.single_word(name="obj")}!

My favorite hobby is {g.options(name="hobby", choices=["reading", "traveling", "cooking", "swimming"])}.

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""

print("Query with masked tags:")
print(query)
print("=" * 80)


# ─── 2. Build Few-Shot Messages for OpenAI/Anthropic APIs ─────────────────────

# For OpenAI's chat completion API
messages_openai = build_few_shot_messages(
    str(Query(query)),  # Convert to GIM query format
    num_examples=3,  # Include 3 few-shot examples
    message_format="openai",
)

print("Messages for OpenAI API (first 3):")
for i, msg in enumerate(messages_openai[:3]):
    print(f"\n{i+1}. Role: {msg['role']}")
    print(f"   Content (truncated): {msg['content'][:100]}...")
print(f"\nTotal messages: {len(messages_openai)}")
print("=" * 80)


# ─── 3. Build Single Prompt String (for non-chat models) ──────────────────────

# For models that don't support chat format (e.g., completion APIs)
prompt_string = build_few_shot_prompt(
    str(Query(query)),  # Convert to GIM query format
    num_examples=3,  # Include 3 few-shot examples
)

print("Single prompt string (first 500 chars):")
print(prompt_string[:500])
print("...")
print("=" * 80)


# ─── 4. Example Usage with OpenAI (commented out) ──────────────────────────────

"""
# Uncomment to use with actual OpenAI API:

from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# For a non-GIM model like GPT-4
response = client.chat.completions.create(
    model="gpt-4",
    messages=messages_openai,
    temperature=0.7,
)

# Extract the response
response_text = response.choices[0].message.content

# Parse the response back into a Result
from gimkit.contexts import Response, infill

result = infill(Query(query), Response(response_text))
print("Result:")
print(result)

# Access individual tags
for tag in result.tags:
    print(f"{tag.name or tag.id}: {tag.content}")
"""


# ─── 5. Benefits for Non-GIM LLMs ──────────────────────────────────────────────

print("\nBenefits of using few-shot prompts with non-GIM LLMs:")
print("1. System prompt explains the GIM format and task clearly")
print("2. Few-shot examples demonstrate correct response formatting")
print("3. Enables comparison between GIM-trained and non-GIM LLMs")
print("4. Works with any LLM that supports text generation")
print("5. Flexible number of examples (0-5) for different trade-offs")
print("=" * 80)
