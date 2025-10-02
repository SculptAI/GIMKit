from openai import OpenAI

from gimkit import from_openai
from gimkit import guide as g


# ─── 1. Use A Model ───────────────────────────────────────────────────────────


openai_client = OpenAI(api_key="", base_url="http://localhost:8000/v1")
model = from_openai(openai_client, model_name="artifacts/09251-gim-sft-tmp/sft-gim")


# ─── 2. Define A Query With Guide ─────────────────────────────────────────────

query = f"""I'm {g.person_name(name="pred")}. Hello, {g.single_word(name="obj")}!

My favorite hobby is {g.options(name="hobby", choices=["reading", "traveling", "cooking", "swimming"])}.

## Bio

{g(name="bio", desc="No more than four sentences.")}

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""
print(query)
print("=" * 80)


# ─── 3. Get The Result ────────────────────────────────────────────────────────

result = model(query, output_type=None)
print(result)
print("=" * 80)

# You can also visit the tags in the result
for tag in result.tags:
    print(tag)
print("=" * 80)

# Or visit tags by id/name
assert result.tags[0] == result.tags["pred"]

# Change the content of a tag
result.tags["email"].content = "PRIVATE"
print(str(result)[-40:])
