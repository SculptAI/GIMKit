# Privacy & PII Protection

With the rise of AI agents and memory systems, protecting personally identifiable information (PII) has become critical. GIMKit can extract, classify, redact, and filter PII from text — all through simple templates.

## PII Extraction from Chat Logs

Extract all personally identifiable information from a conversation in one pass.

```python
chat_log = """
User: Hi, my name is Zhang Wei. I live at 123 Nanjing Road, Shanghai.
Agent: Hello Zhang Wei, how can I help you?
User: I need to update my payment info. My card is 4532-1234-5678-9010, exp 12/27.
Agent: Got it. And your email for the receipt?
User: Sure, it's zhangwei@outlook.com. My phone is 138-1234-5678.
"""

query = f"""Extract all personally identifiable information from this chat log:

Chat log: "{chat_log}"

Person name: {g(name="name", desc="full name of the person")}
Address: {g(name="address", desc="physical address")}
Email: {g.e_mail(name="email")}
Phone: {g.phone_number(name="phone")}
Credit card: {g(name="card", desc="credit card number", regex=r"[\d-]{13,19}")}"""

result = model(query, use_gim_prompt=True)
# result.tags["name"].content  → "Zhang Wei"
# result.tags["address"].content → "123 Nanjing Road, Shanghai"
# result.tags["email"].content → "zhangwei@outlook.com"
# result.tags["phone"].content → "138-1234-5678"
# result.tags["card"].content  → "4532-1234-5678-9010"
```

---

## PII Redaction

Replace sensitive data with generic placeholders for safe storage or sharing.

```python
text = "Patient John Doe, SSN 123-45-6789, was admitted on 2024-03-15. Contact: john.doe@hospital.com, 555-0123."

query = f"""Redact all personal information in the following text. Replace each piece of PII with a placeholder like [NAME], [SSN], [EMAIL], [PHONE], [DATE].

Text: "{text}"

Redacted text: {g(name="redacted", desc="the full text with PII replaced by placeholders")}"""

result = model(query, use_gim_prompt=True)
# result.tags["redacted"].content
# → "Patient [NAME], SSN [SSN], was admitted on [DATE]. Contact: [EMAIL], [PHONE]."
```

---

## Privacy Risk Classification

Classify text by privacy risk level before processing or storing.

```python
texts = [
    "The weather in Beijing is nice today.",
    "My SSN is 123-45-6789 and I live at 456 Elm Street.",
    "We met at the conference last week.",
    "Send the payment to account 8888-1234-5678, holder: Li Ming.",
]

# For each text:
query = f"""Classify the privacy risk level of the following text:

Text: "{text}"

Risk level: {g.select(name="risk", choices=["none", "low", "medium", "high"])}
Reason: {g(name="reason", desc="brief reason for the risk level")}"""

result = model(query, use_gim_prompt=True)
```

| Text | Risk | Reason |
|------|------|--------|
| The weather in Beijing is nice today. | low | mentions a location but no PII |
| My SSN is 123-45-6789 and I live at 456 Elm Street. | **high** | SSN and home address |
| We met at the conference last week. | low | no PII |
| Send the payment to account 8888-1234-5678, holder: Li Ming. | **high** | bank account and name |

---

## Safe Upload Filter for Agent Memory

Extract useful context for agent memory systems while stripping all PII.

```python
user_message = """
Hey, I'm Li Na, born 1992-05-14. I work at ByteDance as a product manager.
My employee ID is BD-20231456. I prefer working on user growth projects.
You can reach me at lina@bytedance.com or WeChat: lina_pm.
"""

query = f"""From the following user message, extract ONLY information that is safe to store in an agent memory system. Do NOT include any PII (name, email, phone, birthday, employee ID, social media handles).

Text: "{user_message}"

Company: {g(name="company", desc="company name")}
Role: {g(name="role", desc="job title or role")}
Interests: {g(name="interests", desc="work interests or preferences")}"""

result = model(query, use_gim_prompt=True)
# result.tags["company"].content    → "ByteDance"
# result.tags["role"].content       → "product manager"
# result.tags["interests"].content  → "user growth projects"
# All PII (name, birthday, employee ID, email, WeChat) excluded.
```

---

## Mixed Content Separation

Split text into safe and PII-containing parts for selective processing.

```python
text = "Dr. Wang from Tsinghua University published a paper on LLM alignment. Contact: wang@tsinghua.edu.cn, phone 010-12345678."

query = f"""Separate the following text into safe and unsafe (PII-containing) parts:

Text: "{text}"

Safe content: {g(name="safe", desc="the parts of the text that contain no personal information")}
Unsafe content: {g(name="unsafe", desc="the parts that contain personal information")}"""

result = model(query, use_gim_prompt=True)
# result.tags["safe"].content
# → "from Tsinghua University published a paper on LLM alignment."
# result.tags["unsafe"].content
# → "Dr. Wang, wang@tsinghua.edu.cn, 010-12345678."
```

---

## Why Use GIMKit for Privacy?

- **No training data needed** — define what counts as PII through natural language descriptions
- **Flexible PII definitions** — adapt to different regulations (GDPR, CCPA, etc.) by changing the template
- **Combined extraction + redaction** — extract structured PII and produce redacted text in the same pipeline
- **Agent memory ready** — filter sensitive data before it enters long-term memory systems
