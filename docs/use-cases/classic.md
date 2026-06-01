# Classic Information Extraction Use Cases

This chapter focuses on core IE tasks that are commonly used in NLP pipelines.

## Contact Information Extraction

```python
text = "Hi, I'm John Smith. You can reach me at john.smith@gmail.com or call 555-123-4567."

query = f"""Extract contact information from the following text:

Text: \"{text}\"

Name: {g.person_name(name="name")}
Email: {g.e_mail(name="email")}
Phone: {g.phone_number(name="phone")}"""

result = model(query, use_gim_prompt=True)
```

## Named Entity Recognition (NER)

```python
text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."

query = f"""Extract entities from the following text:

Text: \"{text}\"

Organization: {g(name="org", desc="organization name")}
Person: {g(name="person", desc="person name")}
Location: {g(name="location", desc="location name")}
Year: {g(name="year", desc="year", regex=r"\d{4}")}"""

result = model(query, use_gim_prompt=True)
```

## Text Classification

```python
text = "SpaceX successfully launched a new rocket into orbit yesterday."

query = f"""Classify the following text:

Text: \"{text}\"

Category: {g.select(name="category", choices=["science", "technology", "business", "sports", "politics"])}
Sentiment: {g.select(name="sentiment", choices=["positive", "negative", "neutral"])}"""

result = model(query, use_gim_prompt=True)
```

## Event Extraction

```python
text = "The earthquake struck Nepal on April 25, 2015, killing nearly 9,000 people."

query = f"""Extract event information from the following text:

Text: \"{text}\"

Event type: {g(name="event_type", desc="type of event")}
Location: {g(name="location", desc="where the event happened")}
Date: {g(name="date", desc="when the event happened")}
Impact: {g(name="impact", desc="consequence or impact")}"""

result = model(query, use_gim_prompt=True)
```

## Relation Extraction

```python
text = "Bill Gates founded Microsoft in 1975. The company is headquartered in Redmond."

query = f"""Extract entities and relationships from the following text:

Text: \"{text}\"

Person: {g(name="person", desc="person name")}
Organization: {g(name="org", desc="organization name")}
Location: {g(name="location", desc="location name")}
Year: {g(name="year", desc="year", regex=r"\d{4}")}
Relationship: {g(name="relation", desc="relationship between person and organization")}"""

result = model(query, use_gim_prompt=True)
```

## Table Completion

```python
question = f"""
Please fill in the missing values in the following table based on the existing values.
| Week | Milk (liters)      | Bread (loaves)     | Eggs (dozen) |
| ---- | ------------------ | ------------------ | ------------ |
| 1    | 6                  | 4                  | 2            |
| 2    | 6                  | 5                  | 3            |
| 3    | {g(desc="number")} | 4                  | 2            |
| 4    | 6                  | 5                  | 3            |
"""

result = model(question, use_gim_prompt=True)
```

## Knowledge Graph Extraction

```python
question = f"""## Content

This small ebook is here to teach you a programming language called Forth.

## Extraction

Extract knowledge graph triplets (head, relation, tail) from the content.

1. ({g()}, {g()}, {g()})
2. ({g()}, {g()}, {g()})
3. ({g()}, {g()}, {g()})
4. ({g()}, {g()}, {g()})
5. ({g()}, {g()}, {g()})
6. ({g()}, {g()}, {g()})
"""

result = model(question, use_gim_prompt=True)
```
