from vllm import LLM, SamplingParams

from gimkit import from_vllm_offline
from gimkit import guide as g


# ─── 1. Use A Model ───────────────────────────────────────────────────────────

llm = LLM(model="Sculpt-AI/GIM-4B", max_model_len=8192)
model = from_vllm_offline(llm)


# ─── 2. Define A Query With Guide ─────────────────────────────────────────────

query = f"""I'm {g.person_name(name="pred")}. Hello, {g.single_word(name="obj")}!

My favorite hobby is {g.select(name="hobby", choices=["reading", "traveling", "cooking", "swimming"])}.

## Bio

{g(name="bio", desc="Four sentences.", regex=r"(?:[^.!?]+[.!?]){4}")}

## Contact

* Phone number: {g.phone_number(name="phone")}
* E-mail: {g.e_mail(name="email")}
"""
print(query)
print("=" * 80)


# ─── 3. Get The Result ────────────────────────────────────────────────────────

sampling_params = SamplingParams(temperature=0.0, max_tokens=8192, presence_penalty=1, seed=0)
result = model(query, output_type="cfg", sampling_params=sampling_params)
result = result if not isinstance(result, list) else result[0]
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
