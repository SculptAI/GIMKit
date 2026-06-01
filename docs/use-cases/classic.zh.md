# 经典信息抽取案例

本章聚焦 NLP 管线中最常见的核心信息抽取任务。

## 联系人信息提取

```python
text = "你好，我叫张伟。可以通过 zhangwei@gmail.com 联系我，电话是 138-1234-5678。"

query = f"""从以下文本中提取联系信息：

文本: \"{text}\"

姓名: {g.person_name(name="name")}
邮箱: {g.e_mail(name="email")}
电话: {g.phone_number(name="phone")}"""

result = model(query, use_gim_prompt=True)
```

## 命名实体识别（NER）

```python
text = "苹果公司由史蒂夫·乔布斯于1976年在加利福尼亚州库比蒂诺创立。"

query = f"""从以下文本中提取实体：

文本: \"{text}\"

组织: {g(name="org", desc="组织名称")}
人物: {g(name="person", desc="人物姓名")}
地点: {g(name="location", desc="地点名称")}
年份: {g(name="year", desc="年份", regex=r"\d{4}")}"""

result = model(query, use_gim_prompt=True)
```

## 文本分类

```python
text = "SpaceX 昨天成功将一枚新火箭送入轨道。"

query = f"""对以下文本进行分类：

文本: \"{text}\"

类别: {g.select(name="category", choices=["科技", "商业", "体育", "政治", "娱乐"])}
情感: {g.select(name="sentiment", choices=["正面", "负面", "中性"])}"""

result = model(query, use_gim_prompt=True)
```

## 事件抽取

```python
text = "2015年4月25日，尼泊尔发生地震，造成近9000人死亡。"

query = f"""从以下文本中提取事件信息：

文本: \"{text}\"

事件类型: {g(name="event_type", desc="事件类型")}
地点: {g(name="location", desc="事件发生地点")}
日期: {g(name="date", desc="事件发生日期")}
影响: {g(name="impact", desc="后果或影响")}"""

result = model(query, use_gim_prompt=True)
```

## 关系抽取

```python
text = "比尔·盖茨于1975年创立了微软。公司总部位于雷德蒙德。"

query = f"""从以下文本中提取实体和关系：

文本: \"{text}\"

人物: {g(name="person", desc="人物姓名")}
组织: {g(name="org", desc="组织名称")}
地点: {g(name="location", desc="地点名称")}
年份: {g(name="year", desc="年份", regex=r"\d{4}")}
关系: {g(name="relation", desc="人物与组织之间的关系")}"""

result = model(query, use_gim_prompt=True)
```

## 表格补全

```python
question = f"""
请根据已有值补全下列表格中的缺失项。
| 周次 | 牛奶（升）          | 面包（条）          | 鸡蛋（打）     |
| ---- | ------------------ | ------------------ | ------------ |
| 1    | 6                  | 4                  | 2            |
| 2    | 6                  | 5                  | 3            |
| 3    | {g(desc="number")} | 4                  | 2            |
| 4    | 6                  | 5                  | 3            |
"""

result = model(question, use_gim_prompt=True)
```

## 知识图谱三元组抽取

```python
question = f"""## 内容

This small ebook is here to teach you a programming language called Forth.

## 抽取任务

从上述内容中抽取知识图谱三元组（头实体、关系、尾实体）。

1. ({g()}, {g()}, {g()})
2. ({g()}, {g()}, {g()})
3. ({g()}, {g()}, {g()})
4. ({g()}, {g()}, {g()})
5. ({g()}, {g()}, {g()})
6. ({g()}, {g()}, {g()})
"""

result = model(question, use_gim_prompt=True)
```
