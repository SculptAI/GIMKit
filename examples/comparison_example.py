"""
Comparison example: GIM-trained vs non-GIM LLMs.

This example demonstrates the three approaches mentioned in the issue:
1. GIM-trained LLMs (baseline)
2. Non-GIM LLMs with few-shot demos and system prompts
3. Non-GIM LLMs with JSON schema decoding support

This enables comparing the effectiveness of each approach.
"""

import json

from gimkit import (
    Query,
    build_few_shot_messages,
    build_json_schema,
)
from gimkit import guide as g


# ─── Create a Test Query ───────────────────────────────────────────────────────

query_template = f"""Generate a character profile:

Name: {g.person_name(name="name")}
Age: {g(name="age", desc="A number between 18 and 80")}
Occupation: {g(name="occupation", desc="A realistic job title")}
Personality: {g(name="personality", desc="3-5 personality traits, comma-separated")}
Background: {g(name="background", desc="A brief 2-3 sentence background story")}
"""

query = Query(query_template)
print("Test Query:")
print(query)
print("=" * 80)


# ─── Approach 1: GIM-trained LLM ───────────────────────────────────────────────

print("\n🔹 Approach 1: GIM-trained LLM")
print("=" * 80)
print("Description:")
print("  - Model is trained with the GIM paradigm")
print("  - Directly outputs GIM response format")
print("  - No additional prompting needed")
print()
print("Example usage:")
print("""
from openai import OpenAI
from gimkit import from_openai

client = OpenAI(api_key="", base_url="http://localhost:8000/v1")
model = from_openai(client, model_name="your-gim-model")

# Simple call with the query
result = model(query)
print(result)
""")
print("Advantages:")
print("  ✓ Most efficient (no extra tokens for prompting)")
print("  ✓ Best format compliance (trained specifically for this)")
print("  ✓ Fastest inference (no few-shot examples to process)")
print("=" * 80)


# ─── Approach 2: Non-GIM LLM with Few-Shot Prompts ─────────────────────────────

print("\n🔹 Approach 2: Non-GIM LLM with Few-Shot Prompts")
print("=" * 80)
print("Description:")
print("  - Use standard LLM (GPT-4, Claude, etc.)")
print("  - Provide system prompt explaining GIM format")
print("  - Include few-shot examples demonstrating the format")
print()

# Build few-shot messages
messages = build_few_shot_messages(str(query), num_examples=3, message_format="openai")
print(f"Messages generated: {len(messages)} total")
print("  - 1 system message (explains GIM format)")
print("  - 6 example messages (3 user-assistant pairs)")
print("  - 1 user message (actual query)")
print()
print("Example usage:")
print("""
from openai import OpenAI
from gimkit import Query, build_few_shot_messages
from gimkit.contexts import Response, infill

client = OpenAI(api_key="your-key")
messages = build_few_shot_messages(str(query), num_examples=3)

response = client.chat.completions.create(
    model="gpt-4",
    messages=messages,
    temperature=0.7
)

# Parse response back into Result
result = infill(query, Response(response.choices[0].message.content))
print(result)
""")
print("Advantages:")
print("  ✓ Works with any standard LLM")
print("  ✓ No special training required")
print("  ✓ Flexible (can adjust number of examples)")
print("  ✓ System prompt provides clear instructions")
print()
print("Trade-offs:")
print("  ⚠ Uses more tokens (system prompt + examples)")
print("  ⚠ May require tuning number of examples")
print("  ⚠ Format compliance depends on model's instruction-following")
print("=" * 80)


# ─── Approach 3: Non-GIM LLM with JSON Schema ──────────────────────────────────

print("\n🔹 Approach 3: Non-GIM LLM with JSON Schema")
print("=" * 80)
print("Description:")
print("  - Use LLM with JSON mode support")
print("  - Provide JSON schema specifying output structure")
print("  - Constrained decoding ensures valid JSON")
print()

# Build JSON schema
json_schema = build_json_schema(query)
print("JSON Schema generated with:")
print(f"  - {len(query.tags)} required fields")
print("  - Type constraints (all strings)")
print("  - Descriptions from query tags")
print()
print("Example schema structure:")
schema_dict = json.loads(json_schema.schema)
print(json.dumps({
    "type": "object",
    "properties": {
        "tags": {
            "m_0": {"type": "string", "description": "..."},
            "m_1": {"type": "string", "description": "..."},
            "...": "..."
        }
    }
}, indent=2))
print()
print("Example usage:")
print("""
from openai import OpenAI
from gimkit import Query, build_json_schema
from gimkit.models.utils import infill_responses

client = OpenAI(api_key="your-key")
schema = build_json_schema(query)

response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {"role": "system", "content": "Generate JSON matching the schema."},
        {"role": "user", "content": f"Schema: {schema.schema}\\nQuery: {query}"}
    ],
    response_format={"type": "json_object"}
)

# Parse JSON response back into Result
result = infill_responses(query, response.choices[0].message.content, output_type="json")
print(result)
""")
print("Advantages:")
print("  ✓ Guaranteed valid JSON structure")
print("  ✓ Works with models supporting JSON mode")
print("  ✓ No need for examples (schema is self-documenting)")
print("  ✓ Can use with outlines for constrained generation")
print()
print("Trade-offs:")
print("  ⚠ Requires JSON mode support")
print("  ⚠ Response needs JSON-to-GIM conversion")
print("  ⚠ Less natural than text-based GIM format")
print("=" * 80)


# ─── Comparison Summary ────────────────────────────────────────────────────────

print("\n📊 Comparison Summary")
print("=" * 80)
print("Choose the approach based on your needs:\n")
print("┌─────────────────────┬───────────┬────────────┬──────────────┬──────────────┐")
print("│ Approach            │ Tokens    │ Setup      │ Compliance   │ Use Case     │")
print("├─────────────────────┼───────────┼────────────┼──────────────┼──────────────┤")
print("│ GIM-trained         │ Minimal   │ Training   │ Excellent    │ Production   │")
print("│ Few-shot prompts    │ High      │ None       │ Good         │ Testing      │")
print("│ JSON schema         │ Medium    │ None       │ Very Good    │ Structured   │")
print("└─────────────────────┴───────────┴────────────┴──────────────┴──────────────┘")
print()
print("Recommendations:")
print("  • Production systems: Use GIM-trained models (Approach 1)")
print("  • Quick experiments: Use few-shot prompts (Approach 2)")
print("  • Strict structure needs: Use JSON schema (Approach 3)")
print("  • Comparison studies: Test all three approaches!")
print("=" * 80)
