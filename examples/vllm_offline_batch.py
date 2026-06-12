from vllm import LLM, SamplingParams

from gimkit import from_vllm_offline
from gimkit import guide as g


llm = LLM(model="Sculpt-AI/GIM-1.7B", max_model_len=8192)
model = from_vllm_offline(llm)

queries = [
    f"Extract the person's name: Alice Zhang -> {g.person_name(name='name')}",
    (
        "Extract contact fields from: Bob Chen, bob@example.com, +1-212-555-0101\n"
        f"Name: {g.person_name(name='name')}\n"
        f"Email: {g.e_mail(name='email')}\n"
        f"Phone: {g.phone_number(name='phone')}"
    ),
]

sampling_params = [
    SamplingParams(temperature=0.0, max_tokens=256, seed=0),
    SamplingParams(temperature=0.0, max_tokens=512, seed=1),
]

batch_results = model.batch(queries, output_type="cfg", sampling_params=sampling_params)

# batch_results keeps two dimensions: inputs, then completions per input.
for input_index, completions in enumerate(batch_results):
    print(f"Input {input_index}")
    for completion_index, result in enumerate(completions):
        print(f"Completion {completion_index}: {result}")
