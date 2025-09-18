from gimkit import guide


# ─── 1. Construct ─────────────────────────────────────────────────────────────

g = guide()

# Define the query with masked tags
raw_query = f"""I'm {g.person_name(name="sub")}. Hello, {g.single_word(name="obj")}!

My favorite hobby is {g.options(name="hobby", choices=["reading", "traveling", "cooking", "swimming"])}.

## Bio

{g(name="bio", desc="No more than four sentences.")}

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""

# Add extra prefix/suffix
query = g.standardize(raw_query)
print(query)

# ─── 2. Request ───────────────────────────────────────────────────────────────


# A mock LLM request function
def llm_request(query: str) -> str:
    return (
        "<|M_OUTPUT|>"
        '<|MASKED id="m_0"|>Alice<|/MASKED|>'
        '<|MASKED id="m_1"|>World<|/MASKED|>'
        '<|MASKED id="m_2"|>reading<|/MASKED|>'
        '<|MASKED id="m_3"|>Alice is a software engineer with 5 years of experience. She loves hiking and photography. She graduated from MIT with a degree in Computer Science. In her free time, she volunteers at local animal shelters.<|/MASKED|>'
        '<|MASKED id="m_4"|>123-456-7890<|/MASKED|>'
        '<|MASKED id="m_5"|>alice@example.com<|/MASKED|>'
        "<|/M_OUTPUT|>"
    )


response = llm_request(query)

# ─── 3. Parse ─────────────────────────────────────────────────────────────────

# Parse the query and response to get the predicated tags
result = g.parse(query, response)

# Visit results by iteration
for tag in result.tags:
    print(tag)

# Or visit results by id/name
assert result.tags[0] == result.tags["sub"]

# Change the content of a tag
result.tags["phone"].content = "PRIVATE"

# Infill the original query with the predicted contents
infilled = result.infill()
print(infilled)
