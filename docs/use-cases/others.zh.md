# 其他应用案例

本章展示经典 IE 之外、面向业务落地的实用案例。

## 简历解析

```python
text = "陈博士，麻省理工学院计算机科学博士，10年机器学习经验。"

query = f"""解析简历信息：

文本: \"{text}\"

姓名: {g.person_name(name="name")}
职位: {g(name="title", desc="当前职位")}
单位: {g(name="org", desc="当前工作单位")}
学历: {g(name="education", desc="最高学历及专业")}
经验: {g(name="experience", desc="工作年限", regex=r"\d+年")}"""

result = model(query, use_gim_prompt=True)
```

## 产品评论分析

```python
text = "iPhone 15 Pro 售价 7999 元。摄像头非常出色，但续航还有提升空间。"

query = f"""提取产品评论信息：

文本: \"{text}\"

产品: {g(name="product", desc="产品名称")}
价格: {g(name="price", desc="含货币符号的价格", regex=r"\d+元")}
评分: {g(name="rating", desc="5分制评分", regex=r"[1-5](\.\d)?")}
优点: {g(name="positive", desc="提到的优点")}
缺点: {g(name="negative", desc="提到的缺点")}"""

result = model(query, use_gim_prompt=True)
```

## 混合内容分离

```python
text = "清华大学的王博士发表了一篇论文。联系方式：wang@tsinghua.edu.cn。"

query = f"""将文本拆分为安全部分和非安全部分（含个人信息）：

文本: \"{text}\"

安全内容: {g(name="safe", desc="不含个人信息的部分")}
非安全内容: {g(name="unsafe", desc="含个人信息的部分")}"""

result = model(query, use_gim_prompt=True)
```

## 24 点问答

```python
question = """24 = 24 {} 24 {} 24 {} 0""".format(
	*[g(desc="一个数学运算符") for _ in range(3)]
)

result = model(question, use_gim_prompt=True)
```

## 中文链式思考问答

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

## 数据合成（依赖式问答）

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

## 可控文本生成

```python
question = "On a screen, mixing #{} and #{} results in the color #{}.".format(
	g(desc="红色十六进制色值", regex=r"[0-9a-fA-F]{6}"),
	g(desc="绿色十六进制色值", regex=r"[0-9a-fA-F]{6}"),
	g(desc="混合后的十六进制色值", regex=r"[0-9a-fA-F]{6}"),
)

result = model(question, use_gim_prompt=True)
```

## 结构化文本生成

```python
question = f"""补全以下 TOML 算法档案：

[algorithm_profile]
name = {g(desc="任意算法名称")}
type = {g(desc="list[str] 类型；最多四项")}
performance.time_complexity: {g(desc="str 类型；使用 LaTeX 表达式")},
performance.space_complexity: {g(desc="str 类型；使用 LaTeX 表达式")}"""

result = model(question, use_gim_prompt=True)
```

## 代码补全

```python
question = f'''def count_words(filename):
	"""{g(desc="用一句话解释函数功能")}"""
	counts = Counter()
	with open(filename) as file:
		for line in file:
			words = line.split(' ')
			counts.update(words)
	return counts
'''

result = model(question, use_gim_prompt=True)
```
