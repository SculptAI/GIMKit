# Other Use Cases

These examples are practical domain use cases beyond classic IE tasks.

## Resume and CV Parsing

```python
text = (
    "Dr. Sarah Chen, PhD in Computer Science from MIT. 10 years of experience in machine learning."
)

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
text = (
    "The iPhone 15 Pro costs $999. It has an amazing camera but the battery life could be better."
)

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

## 24 Puzzle QA

```python
question = """24 = 24 {} 24 {} 24 {} 0""".format(
    *[g(desc="a single math operator") for _ in range(3)]
)

result = model(question, use_gim_prompt=True)
```

## Chinese Chain-of-Thought QA

```python
question = """对于淀粉和纤维素两种物质，下列说法正确的是____
A. 两者都能水解，且水解的最终产物相同
B. 两者含C、H、O三种元素的质量分数相同，且互为同分异构体
C. 它们都属于糖类，且都是溶于水的高分子化合物
D. 都可用$\\left ( C_6H_6O_5\\right )_n$表示，但淀粉能发生银镜反应，而纤维素不能

让我们一步步思考
1. {}
2. {}
3. {}
4. {}
5. {}
6. {}
所以答案是：{}
""".format(*([g(desc="一个思考步骤") for _ in range(6)] + [g.select(choices=["A", "B", "C", "D"])]))

result = model(question, use_gim_prompt=True)
```

## Data Synthesis (Dependent QA)

```python
question = f"""# A short article

This small ebook is here to teach you a programming language called Forth.

# Dependent Questions and Answers

Q: {g("An easy question about the article.")}
A: {g("A concise answer to the question.")}

---

Q: {g("A hard question about the article.")}
A: {g("A detailed answer to the question.")}

---

Q: {g("A very hard question about the article.")}
A: {g("A very detailed answer to the question.")}
"""

result = model(question, use_gim_prompt=True)
```

## Controllable Text Generation

```python
question = "On a screen, mixing #{} and #{} results in the color #{}.".format(
    g(desc="Hex color of Red", regex=r"[0-9a-fA-F]{6}"),
    g(desc="Hex color of Green", regex=r"[0-9a-fA-F]{6}"),
    g(desc="Hex color after mixing", regex=r"[0-9a-fA-F]{6}"),
)

result = model(question, use_gim_prompt=True)
```

## Structured Text Generation

```python
question = f"""Complete the algorithm profile in TOML:

[algorithm_profile]
name = {g(desc="any algorithm name")}
type = {g(desc="list[str] type; at most four items")}
performance.time_complexity: {g(desc="str type; use LaTeX expression")},
performance.space_complexity: {g(desc="str type; use LaTeX expression")}"""

result = model(question, use_gim_prompt=True)
```

## Code Completion

```python
question = f'''def count_words(filename):
	"""{g(desc="explain what the function does in one sentence")}"""
	counts = Counter()
	with open(filename) as file:
		for line in file:
			words = line.split(' ')
			counts.update(words)
	return counts
'''

result = model(question, use_gim_prompt=True)
```
