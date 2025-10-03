"""
Example demonstrating JSON schema support for non-GIM LLMs.

This example shows how to use JSON schema mode with non-GIM LLMs
to enforce structured output format, enabling comparison with GIM-trained models.
"""

import json

from gimkit import Query, build_json_schema
from gimkit import guide as g
from gimkit.models.utils import infill_responses


# ─── 1. Create a Query ─────────────────────────────────────────────────────────

query = f"""Write a product description:

Product Name: {g(name="name", desc="A creative product name")}

Category: {g.options(name="category", choices=["Electronics", "Clothing", "Food", "Books"])}

Price: ${g(name="price", desc="A number between 10 and 500")}

Description: {g(name="description", desc="A compelling 2-3 sentence product description")}

Rating: {g(name="rating", desc="A rating from 1 to 5 stars")}
"""

print("Query with masked tags:")
print(query)
print("=" * 80)


# ─── 2. Build JSON Schema ──────────────────────────────────────────────────────

query_obj = Query(query)
json_schema = build_json_schema(query_obj)

print("\nJSON Schema:")
schema_dict = json.loads(json_schema.schema)
print(json.dumps(schema_dict, indent=2))
print("=" * 80)


# ─── 3. Example Usage with OpenAI (JSON mode) ──────────────────────────────────

"""
# Uncomment to use with actual OpenAI API with JSON mode:

from openai import OpenAI

client = OpenAI(api_key="your-api-key")

# For models that support JSON mode (like GPT-4 with response_format)
response = client.chat.completions.create(
    model="gpt-4-turbo-preview",
    messages=[
        {
            "role": "system", 
            "content": "You are a helpful assistant that generates structured JSON responses."
        },
        {
            "role": "user",
            "content": f"Generate content for the following query and return it as JSON matching this schema: {json_schema.schema}\\n\\nQuery: {str(query_obj)}"
        }
    ],
    response_format={"type": "json_object"},
    temperature=0.7,
)

# Extract the response
response_json = response.choices[0].message.content

# Parse the JSON response back into a Result
from gimkit.models.utils import infill_responses

result = infill_responses(query_obj, response_json, output_type="json")
print("Result:")
print(result)

# Access individual tags
for tag in result.tags:
    print(f"{tag.name or tag.id}: {tag.content}")
"""


# ─── 4. Example JSON Response Format ───────────────────────────────────────────

print("\nExample JSON response that would be generated:")
example_json = {
    "tags": {
        "m_0": "SmartHome Echo Plus",
        "m_1": "Electronics",
        "m_2": "199.99",
        "m_3": "The SmartHome Echo Plus transforms your living space with voice-controlled automation and premium sound quality. Connect all your smart devices and enjoy hands-free control with advanced AI.",
        "m_4": "4"
    }
}
print(json.dumps(example_json, indent=2))
print("=" * 80)


# ─── 5. Parse Example JSON Response ────────────────────────────────────────────

result = infill_responses(query_obj, json.dumps(example_json), output_type="json")
print("\nParsed Result:")
print(result)
print("=" * 80)

print("\nIndividual tags:")
for tag in result.tags:
    print(f"  {tag.name or f'Tag {tag.id}'}: {tag.content}")
print("=" * 80)


# ─── 6. Benefits of JSON Schema for Non-GIM LLMs ──────────────────────────────

print("\nBenefits of using JSON schema with non-GIM LLMs:")
print("1. Enforces structured output format automatically")
print("2. Works with any LLM that supports JSON mode (GPT-4, Claude, etc.)")
print("3. Enables direct comparison with GIM-trained models")
print("4. Provides clear schema for what the model should generate")
print("5. Can be used with outlines library for constrained generation")
print("6. More reliable than prompting alone for structured outputs")
print("=" * 80)
