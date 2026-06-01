# 隐私与 PII 案例

本章聚焦隐私场景下的信息抽取与过滤。

## 从聊天记录中提取 PII

```python
query = f"""从以下聊天记录中提取所有个人身份信息：

聊天记录: \"{chat_log}\"

姓名: {g(name="name", desc="full name of the person")}
地址: {g(name="address", desc="physical address")}
邮箱: {g.e_mail(name="email")}
电话: {g.phone_number(name="phone")}
银行卡号: {g(name="card", desc="credit card number", regex=r"[\d-]{13,19}")}"""

result = model(query, use_gim_prompt=True)
```

## PII 脱敏

```python
query = f"""对以下文本中的个人信息进行脱敏，用占位符替换：

文本: \"{text}\"

脱敏后: {g(name="redacted", desc="the full text with PII replaced by placeholders")}"""

result = model(query, use_gim_prompt=True)
```

## 隐私风险分类

```python
query = f"""对以下文本的隐私风险等级进行分类：

文本: \"{text}\"

风险等级: {g.select(name="risk", choices=["无", "低", "中", "高"])}
原因: {g(name="reason", desc="brief reason for the risk level")}"""

result = model(query, use_gim_prompt=True)
```

## Agent Memory 安全上传过滤

```python
query = f"""从以下用户消息中，仅提取可安全存储到 Agent 记忆的信息。不要包含 PII。

文本: \"{user_message}\"

公司: {g(name="company", desc="company name")}
职位: {g(name="role", desc="job title or role")}
兴趣: {g(name="interests", desc="work interests or preferences")}"""

result = model(query, use_gim_prompt=True)
```
