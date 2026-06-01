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
