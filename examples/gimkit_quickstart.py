from gimkit import Query
from gimkit import guide as g


# ─── 1. Construct ─────────────────────────────────────────────────────────────

# Define the query with guides
raw_query = f"""I'm {g.person_name(name="sub")}. Hello, {g.single_word(name="obj")}!

My favorite hobby is {g.options(name="hobby", choices=["reading", "traveling", "cooking", "swimming"])}.

## Bio

{g(name="bio", desc="No more than four sentences.")}

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""

# Query() adds necessary tags and standardizes format
query = Query(raw_query)
print(query)
print("=" * 80)

# ─── 2. Request ───────────────────────────────────────────────────────────────


# A mock LLM request function
def llm_request(query: str) -> str:
    return (
        "<|GIM_QUERY|>"
        '<|MASKED id="m_0"|>Alice<|/MASKED|>'
        '<|MASKED id="m_1"|>World<|/MASKED|>'
        '<|MASKED id="m_2"|>reading<|/MASKED|>'
        '<|MASKED id="m_3"|>Alice is a software engineer with 5 years of experience. She loves hiking and photography. She graduated from MIT with a degree in Computer Science. In her free time, she volunteers at local animal shelters.<|/MASKED|>'
        '<|MASKED id="m_4"|>123-456-7890<|/MASKED|>'
        '<|MASKED id="m_5"|>alice@example.com<|/MASKED|>'
        "<|/GIM_RESPONSE|>"
    )


raw_response = llm_request(str(query))

# ─── 3. Infill ────────────────────────────────────────────────────────────────

# Infill predicted tags back to the original query
response = query.infill(raw_response)
print(response)
print("=" * 80)

# You can also visit the tags
for tag in response.tags:
    print(tag)
print("=" * 80)

# Or visit tags by id/name
assert response.tags[0] == response.tags["sub"]

# Change the content of a tag
response.tags["email"].content = "PRIVATE"
print(str(response)[-40:])
