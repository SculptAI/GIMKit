# 应用案例

GIMKit 不仅仅是一个文本生成工具——它是一个**通用信息抽取框架**。只需用自然语言写一个模板，嵌入类型化的标签，就能从任意非结构化文本中提取结构化数据。无需标签列表，无需模型训练，无需复杂流水线。

## 工作原理

模式始终一致：

1. 写一个描述提取目标的模板
2. 为每个字段嵌入类型化标签
3. LLM 填充空白——受标签约束

```python
from gimkit import guide as g

query = f"""从以下文本中提取："{text}"

姓名: {g.person_name(name="name")}
邮箱: {g.e_mail(name="email")}"""

result = model(query, use_gim_prompt=True)
result.tags["name"].content   # → "张伟"
result.tags["email"].content  # → "zhangwei@example.com"
```

---

## 联系人信息提取

从自由文本中提取姓名、邮箱、电话等联系信息。

```python
text = "你好，我叫张伟。可以通过 zhangwei@gmail.com 联系我，电话是 138-1234-5678。"

query = f"""从以下文本中提取联系信息：

文本: "{text}"

姓名: {g.person_name(name="name")}
邮箱: {g.e_mail(name="email")}
电话: {g.phone_number(name="phone")}"""

result = model(query, use_gim_prompt=True)
# result.tags["name"].content  → "张伟"
# result.tags["email"].content → "zhangwei@gmail.com"
# result.tags["phone"].content → "138-1234-5678"
```

---

## 命名实体识别（NER）

从任意文本中提取组织、人物、地点、日期等实体。

```python
text = "苹果公司由史蒂夫·乔布斯于1976年在加利福尼亚州库比蒂诺创立。"

query = f"""从以下文本中提取实体：

文本: "{text}"

组织: {g(name="org", desc="组织名称")}
人物: {g(name="person", desc="人物姓名")}
地点: {g(name="location", desc="地点名称")}
年份: {g(name="year", desc="年份", regex=r"\d{4}")}"""

result = model(query, use_gim_prompt=True)
# result.tags["org"].content      → "苹果公司"
# result.tags["person"].content   → "史蒂夫·乔布斯"
# result.tags["location"].content → "加利福尼亚州库比蒂诺"
# result.tags["year"].content     → "1976"
```

---

## 文本分类

使用 `g.select()` 将文本分类到指定类别或标注情感。

```python
text = "SpaceX 昨天成功将一枚新火箭送入轨道。"

query = f"""对以下文本进行分类：

文本: "{text}"

类别: {g.select(name="category", choices=["科技", "商业", "体育", "政治", "娱乐"])}
情感: {g.select(name="sentiment", choices=["正面", "负面", "中性"])}"""

result = model(query, use_gim_prompt=True)
# result.tags["category"].content  → "科技"
# result.tags["sentiment"].content → "正面"
```

---

## 事件抽取

提取结构化的事件信息——发生了什么、在哪里、什么时候、有什么影响。

```python
text = "2015年4月25日，尼泊尔发生地震，造成近9000人死亡。"

query = f"""从以下文本中提取事件信息：

文本: "{text}"

事件类型: {g(name="event_type", desc="事件类型")}
地点: {g(name="location", desc="事件发生地点")}
日期: {g(name="date", desc="事件发生日期")}
影响: {g(name="impact", desc="后果或影响")}"""

result = model(query, use_gim_prompt=True)
# result.tags["event_type"].content → "地震"
# result.tags["location"].content   → "尼泊尔"
# result.tags["date"].content       → "2015年4月25日"
# result.tags["impact"].content     → "造成近9000人死亡"
```

---

## 关系抽取

在一次提取中同时获取实体及其之间的关系。

```python
text = "比尔·盖茨于1975年创立了微软。公司总部位于雷德蒙德。"

query = f"""从以下文本中提取实体和关系：

文本: "{text}"

人物: {g(name="person", desc="人物姓名")}
组织: {g(name="org", desc="组织名称")}
地点: {g(name="location", desc="地点名称")}
年份: {g(name="year", desc="年份", regex=r"\d{4}")}
关系: {g(name="relation", desc="人物与组织之间的关系")}"""

result = model(query, use_gim_prompt=True)
# result.tags["person"].content   → "比尔·盖茨"
# result.tags["org"].content      → "微软"
# result.tags["relation"].content → "创立"
```

---

## 简历解析

从简历文本中提取结构化的候选人信息。

```python
text = "陈博士，麻省理工学院计算机科学博士，10年机器学习经验，现任 Google DeepMind 高级研究科学家。"

query = f"""解析简历信息：

文本: "{text}"

姓名: {g.person_name(name="name")}
职位: {g(name="title", desc="当前职位")}
单位: {g(name="org", desc="当前工作单位")}
学历: {g(name="education", desc="最高学历及专业")}
经验: {g(name="experience", desc="工作年限", regex=r"\d+年")}"""

result = model(query, use_gim_prompt=True)
# result.tags["name"].content       → "陈博士"
# result.tags["title"].content      → "高级研究科学家"
# result.tags["org"].content        → "Google DeepMind"
# result.tags["education"].content  → "麻省理工学院计算机科学博士"
# result.tags["experience"].content → "10年"
```

---

## 产品评论分析

将产品评论解析为结构化字段——产品名、价格、评分、优缺点。

```python
text = "iPhone 15 Pro 售价 7999 元。摄像头非常出色，但续航还有提升空间。给 4 分（满分5分）。"

query = f"""提取产品评论信息：

文本: "{text}"

产品: {g(name="product", desc="产品名称")}
价格: {g(name="price", desc="含货币符号的价格", regex=r"\d+元")}
评分: {g(name="rating", desc="5分制评分", regex=r"[1-5](\.\d)?")}
优点: {g(name="positive", desc="提到的优点")}
缺点: {g(name="negative", desc="提到的缺点")}"""

result = model(query, use_gim_prompt=True)
# result.tags["product"].content  → "iPhone 15 Pro"
# result.tags["price"].content    → "7999元"
# result.tags["rating"].content   → "4"
# result.tags["positive"].content → "摄像头非常出色"
# result.tags["negative"].content → "续航"
```

---

## 为什么选择 GIMKit 做信息抽取？

| 特性 | GIMKit | 传统 NER / IE |
|------|--------|---------------|
| **上手** | 写个模板就完事 | 训练模型或配置标签集 |
| **灵活性** | 任意字段、任意格式 | 固定实体类型 |
| **格式控制** | 正则约束、选项限制 | 需要后处理 |
| **输出** | 按字段命名，直接访问 | 需要解析 token 级标签 |
| **关系抽取** | 同一模板，单次完成 | 需要独立模型或流水线 |
| **模型要求** | 任意 LLM（低至 4B 参数） | 需要特定任务的微调模型 |

GIMKit 将信息抽取视为一个**填空题**。你用自然语言描述想要什么，嵌入类型化占位符，模型完成剩下的工作。结果是结构化的、按字段命名的、开箱即用的——无需后处理。
