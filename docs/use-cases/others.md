# Other Use Cases

These examples are practical domain use cases beyond classic IE tasks.

## Resume and CV Parsing

```python
text = "Dr. Sarah Chen, PhD in Computer Science from MIT. 10 years of experience in machine learning."

query = f"""Parse the resume information:

Text: \"{text}\"

Name: {g.person_name(name="name")}
Title: {g(name="title", desc="current job title")}
Organization: {g(name="org", desc="current organization")}
Education: {g(name="education", desc="highest degree and field")}
Experience: {g(name="experience", desc="years of experience", regex=r"\d+ years?")}"""

result = model(query, use_gim_prompt=True)
```

## Product Review Extraction

```python
text = "The iPhone 15 Pro costs $999. It has an amazing camera but the battery life could be better."

query = f"""Extract product review information:

Text: \"{text}\"

Product: {g(name="product", desc="product name")}
Price: {g(name="price", desc="price with currency symbol", regex=r"\$\d+")}
Rating: {g(name="rating", desc="rating out of 5", regex=r"[1-5](\.\d)?")}
Positive: {g(name="positive", desc="positive aspect mentioned")}
Negative: {g(name="negative", desc="negative aspect mentioned")}"""

result = model(query, use_gim_prompt=True)
```

## Mixed Content Separation

```python
text = "Dr. Wang from Tsinghua University published a paper. Contact: wang@tsinghua.edu.cn."

query = f"""Separate the text into safe and unsafe (PII-containing) parts:

Text: \"{text}\"

Safe content: {g(name="safe", desc="parts with no personal information")}
Unsafe content: {g(name="unsafe", desc="parts containing personal information")}"""

result = model(query, use_gim_prompt=True)
```
