# 隐私与 PII 保护

随着 AI Agent 和记忆系统的兴起，保护个人身份信息（PII）变得至关重要。GIMKit 可以通过简单的模板，从文本中提取、分类、脱敏和过滤 PII。

## 从聊天记录中提取 PII

一次提取对话中的所有个人身份信息。

```python
chat_log = """
用户：你好，我叫张伟，住在上海南京路123号。
助手：你好张伟，有什么可以帮你的？
用户：我需要更新支付信息，我的卡号是 4532-1234-5678-9010，有效期 12/27。
助手：好的，收据发到哪个邮箱？
用户：zhangwei@outlook.com，电话是 138-1234-5678。
"""

query = f"""从以下聊天记录中提取所有个人身份信息：

聊天记录: "{chat_log}"

姓名: {g(name="name", desc="full name of the person")}
地址: {g(name="address", desc="physical address")}
邮箱: {g.e_mail(name="email")}
电话: {g.phone_number(name="phone")}
银行卡号: {g(name="card", desc="credit card number", regex=r"[\d-]{13,19}")}"""

result = model(query, use_gim_prompt=True)
# result.tags["name"].content    → "张伟"
# result.tags["address"].content → "上海南京路123号"
# result.tags["email"].content   → "zhangwei@outlook.com"
# result.tags["phone"].content   → "138-1234-5678"
# result.tags["card"].content    → "4532-1234-5678-9010"
```

---

## PII 脱敏

将敏感数据替换为通用占位符，用于安全存储或分享。

```python
text = "患者张三，身份证号 110101199001011234，于2024年3月15日入院。联系方式：zhangsan@hospital.com，138-0000-1234。"

query = f"""对以下文本中的所有个人信息进行脱敏处理，用 [姓名]、[身份证]、[邮箱]、[电话]、[日期] 等占位符替换。

文本: "{text}"

脱敏后: {g(name="redacted", desc="the full text with PII replaced by placeholders")}"""

result = model(query, use_gim_prompt=True)
# result.tags["redacted"].content
# → "患者[姓名]，身份证号[身份证]，于[日期]入院。联系方式：[邮箱]，[电话]。"
```

---

## 隐私风险分类

在处理或存储之前，按隐私风险等级对文本进行分类。

```python
texts = [
    "今天北京天气不错。",
    "我的身份证号是 110101199001011234，住在海淀区中关村大街1号。",
    "我们上周在会议上认识的。",
    "请将款项汇到账户 8888-1234-5678，户主：李明。",
]

# 对每条文本：
query = f"""对以下文本的隐私风险等级进行分类：

文本: "{text}"

风险等级: {g.select(name="risk", choices=["无", "低", "中", "高"])}
原因: {g(name="reason", desc="brief reason for the risk level")}"""

result = model(query, use_gim_prompt=True)
```

| 文本 | 风险 | 原因 |
|------|------|------|
| 今天北京天气不错。 | 低 | 提到地点但不含 PII |
| 我的身份证号是...住在... | **高** | 包含身份证号和住址 |
| 我们上周在会议上认识的。 | 低 | 不含 PII |
| 请将款项汇到账户...户主... | **高** | 包含银行账号和姓名 |

---

## Agent Memory 安全上传过滤

为 Agent 记忆系统提取有用上下文，同时去除所有 PII。

```python
user_message = """
嗨，我叫李娜，1992年5月14日出生。我在字节跳动做产品经理，
工号 BD-20231456。我喜欢做用户增长相关的项目。
可以通过 lina@bytedance.com 或微信 lina_pm 联系我。
"""

query = f"""从以下用户消息中，仅提取适合存储在 Agent 记忆系统中的信息。不要包含任何 PII（姓名、邮箱、电话、生日、工号、社交媒体账号）。

文本: "{user_message}"

公司: {g(name="company", desc="company name")}
职位: {g(name="role", desc="job title or role")}
兴趣: {g(name="interests", desc="work interests or preferences")}"""

result = model(query, use_gim_prompt=True)
# result.tags["company"].content    → "字节跳动"
# result.tags["role"].content       → "产品经理"
# result.tags["interests"].content  → "用户增长"
# 所有 PII（姓名、生日、工号、邮箱、微信）均被排除。
```

---

## 混合内容分离

将文本拆分为安全部分和含 PII 的非安全部分，用于选择性处理。

```python
text = "清华大学的王博士发表了一篇关于 LLM 对齐的论文。联系方式：wang@tsinghua.edu.cn，电话 010-12345678。"

query = f"""将以下文本分为安全部分和非安全部分（含个人信息的部分）：

文本: "{text}"

安全内容: {g(name="safe", desc="the parts of the text that contain no personal information")}
非安全内容: {g(name="unsafe", desc="the parts that contain personal information")}"""

result = model(query, use_gim_prompt=True)
# result.tags["safe"].content
# → "清华大学的王博士发表了一篇关于 LLM 对齐的论文。"
# result.tags["unsafe"].content
# → "wang@tsinghua.edu.cn，电话 010-12345678。"
```

---

## 为什么用 GIMKit 做隐私保护？

- **无需训练数据** — 通过自然语言描述定义什么算 PII
- **灵活的 PII 定义** — 通过修改模板适配不同法规（GDPR、个保法等）
- **提取 + 脱敏一体化** — 在同一流水线中提取结构化 PII 并生成脱敏文本
- **Agent Memory 就绪** — 在敏感数据进入长期记忆系统之前进行过滤
