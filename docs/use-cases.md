# Use Cases

GIMKit is more than a text generation tool — it's a **general-purpose information extraction framework**. By writing a natural-language template with embedded masked tags, you can extract structured data from any unstructured text. No label lists, no model training, no complex pipelines.

## How It Works

The pattern is always the same:

1. Write a template describing what to extract
2. Embed typed masked tags for each field
3. The LLM fills in the blanks — constrained by your tags

```python
from gimkit import guide as g

query = f"""Extract from: "{text}"

Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}"""

result = model(query, use_gim_prompt=True)
result.tags["name"].content   # → "John Smith"
result.tags["email"].content  # → "john@example.com"
```

---

## Contact Information Extraction

Extract names, emails, phone numbers, and other contact details from free-form text.

```python
text = "Hi, I'm John Smith. You can reach me at john.smith@gmail.com or call 555-123-4567."

query = f"""Extract contact information from the following text:

Text: "{text}"

Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
Phone: {g.phone_number(name="phone")}"""

result = model(query, use_gim_prompt=True)
# result.tags["name"].content  → "John Smith"
# result.tags["email"].content → "john.smith@gmail.com"
# result.tags["phone"].content → "555-123-4567"
```

---

## Named Entity Recognition (NER)

Extract entities like organizations, people, locations, and dates from any text.

```python
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."

query = f"""Extract entities from the following text:

Text: "{text}"

Organization: {g(name="org", desc="organization name")}
Person: {g(name="person", desc="person name")}
Location: {g(name="location", desc="location name")}
Year: {g(name="year", desc="year", regex=r"\d{4}")}"""

result = model(query, use_gim_prompt=True)
# result.tags["org"].content      → "Apple Inc."
# result.tags["person"].content   → "Steve Jobs"
# result.tags["location"].content → "Cupertino, California"
# result.tags["year"].content     → "1976"
```

---

## Text Classification

Classify text into categories or assign sentiment labels using `g.select()`.

```python
text = "SpaceX successfully launched a new rocket into orbit yesterday."

query = f"""Classify the following text:

Text: "{text}"

Category: {g.select(name="category", choices=["science", "technology", "business", "sports", "politics"])}
Sentiment: {g.select(name="sentiment", choices=["positive", "negative", "neutral"])}"""

result = model(query, use_gim_prompt=True)
# result.tags["category"].content  → "technology"
# result.tags["sentiment"].content → "positive"
```

---

## Event Extraction

Pull structured event information — what happened, where, when, and the impact.

```python
text = "The earthquake struck Nepal on April 25, 2015, killing nearly 9,000 people."

query = f"""Extract event information from the following text:

Text: "{text}"

Event type: {g(name="event_type", desc="type of event")}
Location: {g(name="location", desc="where the event happened")}
Date: {g(name="date", desc="when the event happened")}
Impact: {g(name="impact", desc="consequence or impact")}"""

result = model(query, use_gim_prompt=True)
# result.tags["event_type"].content → "earthquake"
# result.tags["location"].content   → "Nepal"
# result.tags["date"].content       → "April 25, 2015"
# result.tags["impact"].content     → "killing nearly 9,000 people"
```

---

## Relation Extraction

Extract entities and the relationships between them in a single pass.

```python
text = "Bill Gates founded Microsoft in 1975. The company is headquartered in Redmond."

query = f"""Extract entities and relationships from the following text:

Text: "{text}"

Person: {g(name="person", desc="person name")}
Organization: {g(name="org", desc="organization name")}
Location: {g(name="location", desc="location name")}
Year: {g(name="year", desc="year", regex=r"\d{4}")}
Relationship: {g(name="relation", desc="relationship between person and organization")}"""

result = model(query, use_gim_prompt=True)
# result.tags["person"].content → "Bill Gates"
# result.tags["org"].content    → "Microsoft"
# result.tags["relation"].content → "founded"
```

---

## Resume / CV Parsing

Extract structured candidate information from resume text.

```python
text = "Dr. Sarah Chen, PhD in Computer Science from MIT. 10 years of experience in machine learning. Currently Senior Research Scientist at Google DeepMind."

query = f"""Parse the resume information:

Text: "{text}"

Name: {g.person_name(name="name")}
Title: {g(name="title", desc="current job title")}
Organization: {g(name="org", desc="current organization")}
Education: {g(name="education", desc="highest degree and field")}
Experience: {g(name="experience", desc="years of experience", regex=r"\d+ years?")}"""

result = model(query, use_gim_prompt=True)
# result.tags["name"].content       → "Dr. Sarah Chen"
# result.tags["title"].content      → "Senior Research Scientist"
# result.tags["org"].content        → "Google DeepMind"
# result.tags["education"].content  → "PhD in Computer Science from MIT"
# result.tags["experience"].content → "10 years"
```

---

## Product Review Extraction

Parse product reviews into structured fields — product name, price, rating, and pros/cons.

```python
text = "The iPhone 15 Pro costs $999. It has an amazing camera but the battery life could be better. I'd rate it 4 out of 5."

query = f"""Extract product review information:

Text: "{text}"

Product: {g(name="product", desc="product name")}
Price: {g(name="price", desc="price with currency symbol", regex=r"\$\d+")}
Rating: {g(name="rating", desc="rating out of 5", regex=r"[1-5](\.\d)?")}
Positive: {g(name="positive", desc="positive aspect mentioned")}
Negative: {g(name="negative", desc="negative aspect mentioned")}"""

result = model(query, use_gim_prompt=True)
# result.tags["product"].content  → "iPhone 15 Pro"
# result.tags["price"].content    → "$999"
# result.tags["rating"].content   → "4"
# result.tags["positive"].content → "amazing camera"
# result.tags["negative"].content → "battery life"
```

---

## Why GIMKit for Information Extraction?

| Feature | GIMKit | Traditional NER / IE |
|---------|--------|---------------------|
| **Setup** | Write a template, done | Train models or configure label sets |
| **Flexibility** | Any field, any format | Fixed entity types |
| **Format control** | Regex constraints, choices | Post-processing needed |
| **Output** | Named fields, direct access | Token-level labels to parse |
| **Relation extraction** | Same template, single pass | Separate model or pipeline |
| **Model requirement** | Any LLM (even 4B params) | Task-specific fine-tuned model |

GIMKit treats information extraction as a **fill-in-the-blank** problem. You describe what you want in natural language, embed typed placeholders, and the model does the rest. The result is structured, named, and ready to use — no post-processing required.
