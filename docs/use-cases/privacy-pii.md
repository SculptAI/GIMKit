# Privacy and PII Use Cases

This chapter shows privacy-oriented extraction workflows. For a deeper walkthrough, see [Privacy & PII Protection](../privacy.md).

## PII Extraction from Chat Logs

```python
query = f"""Extract all personally identifiable information from this chat log:

Chat log: \"{chat_log}\"

Person name: {g(name="name", desc="full name of the person")}
Address: {g(name="address", desc="physical address")}
Email: {g.e_mail(name="email")}
Phone: {g.phone_number(name="phone")}
Credit card: {g(name="card", desc="credit card number", regex=r"[\d-]{13,19}")}"""

result = model(query, use_gim_prompt=True)
```

## PII Redaction

```python
query = f"""Redact all personal information in the following text. Replace each piece of PII with placeholders.

Text: \"{text}\"

Redacted text: {g(name="redacted", desc="the full text with PII replaced by placeholders")}"""

result = model(query, use_gim_prompt=True)
```

## Privacy Risk Classification

```python
query = f"""Classify the privacy risk level of the following text:

Text: \"{text}\"

Risk level: {g.select(name="risk", choices=["none", "low", "medium", "high"])}
Reason: {g(name="reason", desc="brief reason for the risk level")}"""

result = model(query, use_gim_prompt=True)
```

## Safe Upload Filter for Agent Memory

```python
query = f"""From the following user message, extract ONLY information safe for agent memory. Do NOT include PII.

Text: \"{user_message}\"

Company: {g(name="company", desc="company name")}
Role: {g(name="role", desc="job title or role")}
Interests: {g(name="interests", desc="work interests or preferences")}"""

result = model(query, use_gim_prompt=True)
```
