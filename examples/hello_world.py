from openai import OpenAI

from gimkit import from_openai
from gimkit import guide as g


# Make sure you have set the OPENAI_API_KEY environment variable before running this code.
client = OpenAI(base_url="https://openrouter.ai/api/v1")
model = from_openai(client, model_name="openai/gpt-5")

result = model(f"Hello, {g(desc='a single word')}!", output_type="json", use_gim_prompt=True)
print(result)  # This probably prints: Hello, world!
