# GIMKit

**Guided Infilling Modeling Toolkit** — 基于语言模型的结构化文本生成与信息抽取工具。

GIMKit 允许你在文本中定义占位符（masked tags），由语言模型来填充。通过类型化的标签系统和可选的正则约束，实现对模型输出的精细控制。

[![PyPI Version](https://img.shields.io/pypi/v/gimkit?label=pypi%20package)](https://pypi.org/project/gimkit)
[![Python Versions](https://img.shields.io/pypi/pyversions/gimkit.svg)](https://pypi.org/project/gimkit)
[![Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20macOS%20%7C%20Windows-lightgrey)](https://pypi.org/project/gimkit)

---

## GIMKit 能做什么？

GIMKit 是一个**通用信息抽取框架**。用自然语言写一个模板，嵌入类型化的占位符，模型就能从任意非结构化文本中提取结构化数据。

| 应用场景 | 说明 |
|----------|------|
| **联系人提取** | 从自由文本中解析姓名、邮箱、电话 |
| **命名实体识别** | 提取组织、人物、地点、日期 |
| **文本分类** | 对文本进行分类、情感标注 |
| **事件抽取** | 提取结构化事件信息（何事/何地/何时/影响） |
| **关系抽取** | 发现实体及其之间的关系 |
| **简历解析** | 提取候选人姓名、职位、学历、经验 |
| **评论分析** | 解析产品名、价格、评分、优缺点 |
| **隐私与 PII 保护** | 提取、分类、脱敏和过滤个人信息 |

完整代码示例见 [应用案例](use-cases.zh.md) 和 [隐私与 PII 保护](privacy.zh.md) 页面。

---

## 特性

- **标签系统** — 直接在 f-string 中嵌入类型化占位符。
- **正则约束** — 将模型输出限制为特定模式。
- **按名访问** — 通过标签名或索引获取结果。
- **多后端支持** — OpenAI、vLLM（服务端和离线模式）。
- **小模型友好** — 专为小型开源模型设计。

## 设计理念

- **稳定优先** — 可靠性和正确性优先于新功能。
- **小开源模型优先** — 专为小型、可自由获取的语言模型设计。
