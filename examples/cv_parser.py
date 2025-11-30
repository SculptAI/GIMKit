from openai import OpenAI

from gimkit import from_openai
from gimkit import guide as g


client = OpenAI(
    api_key="***",
    base_url="https://openrouter.ai/api/v1",
)
model = from_openai(client, model_name="qwen/qwen3-235b-a22b")

cv_content = ""
extraction_fields = f'''
{{
  "name": "{g.person_name()}",
  "birthDay": "{g.datetime(require_time=False)}",
  "phone": "{g.phone_number()}",
  "email": "{g.e_mail()}",
  "paperCount": "{g()}",
  "ccfACount": "{g()}",
  "scholarship": "{g()}",
  "competitionAward": "{g()}",
  "honors": "{g()}",
  "homepageUrl": "{g()}",
  "googleScholarUrl": "{g()}",
  "githubUrl": "{g()}",
  "citationCount": "{g()}",
  "hIndex": "{g()}",
  "talentEducationalParamList": [
    {{
      "degreeLevel": "{g.select(choices=["BACHELOR", "MASTER", "DOCTOR"])}",
      "school": "{g()}",
      "department": "{g()}",
      "major": "{g()}",
      "advisor": "{g()}",
      "advisorTitles": "{g()}",
      "lab": "{g()}",
      "researchDirection": "{g()}",
      "startDate": "{g.datetime(require_time=False)}",
      "endDate": "{g.datetime(require_time=False)}",
    }},
    {{
      "degreeLevel": "{g.select(choices=["BACHELOR", "MASTER", "DOCTOR"])}",
      "school": "{g()}",
      "department": "{g()}",
      "major": "{g()}",
      "advisor": "{g()}",
      "advisorTitles": "{g()}",
      "lab": "{g()}",
      "researchDirection": "{g()}",
      "startDate": "{g.datetime(require_time=False)}",
      "endDate": "{g.datetime(require_time=False)}",
    }},
    {{
      "degreeLevel": "{g.select(choices=["BACHELOR", "MASTER", "DOCTOR"])}",
      "school": "{g()}",
      "department": "{g()}",
      "major": "{g()}",
      "advisor": "{g()}",
      "advisorTitles": "{g()}",
      "lab": "{g()}",
      "researchDirection": "{g()}",
      "startDate": "{g.datetime(require_time=False)}",
      "endDate": "{g.datetime(require_time=False)}",
    }}
  ]
}}'''

result = model(cv_content + extraction_fields, output_type="json", use_gim_prompt=True)
print(result)
