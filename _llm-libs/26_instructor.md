---
title: "instructor 结构化输出"
excerpt: "response_model、流式模式、Mode(TOOLS/JSON/MD_JSON)、Pydantic验证"
collection: llm-libs
permalink: /llm-libs/26-instructor
category: agent
toc: true
---


## 1. 库的简介和在LLM开发中的作用

Instructor 是目前最流行的 Python 结构化LLM输出库，月下载量超过 100 万次。它的核心使命是**让 LLM 返回可靠的结构化、类型安全、经过验证的数据**，而非自由文本。

在 LLM 开发中，Instructor 解决了一个根本性问题：LLM 原生输出是自由文本，而应用程序通常需要结构化数据（如 JSON 对象、列表、枚举值等）。Instructor 通过将 **Pydantic 的数据验证能力**与 **LLM 的生成能力**结合，让开发者只需定义 Pydantic 模型，即可从 LLM 获得完全类型安全、自动验证的结构化输出。

**核心价值：**
- 结构化输出：基于 Pydantic 模型定义输出格式，LLM 自动遵循
- 自动重试：验证失败时自动重新请求 LLM，附带错误信息
- 多提供商支持：兼容 15+ 种 LLM 提供商（OpenAI、Anthropic、Google、Mistral、Ollama 等）
- 流式处理：支持部分流式（partial streaming）和可迭代流式（iterable streaming）
- 类型安全：完整的 IDE 类型推断和自动补全支持

```bash
pip install instructor
```

## 2. 安装方式

```bash
# 基础安装
pip install instructor

# 使用 uv
uv add instructor

# 使用 poetry
poetry add instructor

# 带特定提供商依赖
pip install instructor[anthropic]   # Anthropic 支持
pip install instructor[google-generativeai]  # Google Gemini 支持

# 从源码安装
pip install git+https://github.com/567-labs/instructor.git
```

## 3. 核心类/函数/工具的详细说明

### 3.1 客户端创建：from_provider 与 from_openai / from_anthropic

#### from_provider（推荐，统一接口）

`from_provider` 是 v1.x 推荐的统一客户端创建方式，使用 `"provider/model"` 字符串格式：

```python
import instructor

# 同步客户端（默认）
client = instructor.from_provider("openai/gpt-4o")
client = instructor.from_provider("anthropic/claude-3-5-sonnet-20240620")
client = instructor.from_provider("google/gemini-2.5-flash")
client = instructor.from_provider("ollama/llama3")       # 本地模型
client = instructor.from_provider("deepseek/deepseek-chat")

# 异步客户端
client = instructor.from_provider("openai/gpt-4o", async_client=True)

# 指定模式
client = instructor.from_provider(
    "openai/gpt-4o",
    mode=instructor.Mode.TOOLS,
    api_key="sk-...",  # 也可通过环境变量设置
)
```

**关键参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `provider_model` | str | 提供商/模型字符串，如 "openai/gpt-4o" |
| `async_client` | bool | 是否创建异步客户端，默认 False |
| `mode` | Mode | 输出模式，默认根据提供商自动选择 |
| `api_key` | str | API 密钥 |

#### from_openai（经典方式）

```python
import instructor
from openai import OpenAI, AsyncOpenAI

# 同步客户端
client = instructor.from_openai(OpenAI(api_key="sk-..."))

# 异步客户端
client = instructor.from_openai(AsyncOpenAI(api_key="sk-..."))

# 指定模式
client = instructor.from_openai(
    OpenAI(),
    mode=instructor.Mode.JSON,
)
```

#### from_anthropic

```python
import instructor
from anthropic import Anthropic, AsyncAnthropic

# 同步客户端
client = instructor.from_anthropic(Anthropic(api_key="sk-ant-..."))

# 异步客户端
client = instructor.from_anthropic(AsyncAnthropic(api_key="sk-ant-..."))
```

#### patch（旧版兼容方式）

```python
import instructor
import openai

# 补丁方式：修改 openai.Client 使其支持 response_model
client = instructor.patch(openai.OpenAI())
```

### 3.2 Mode 模式

Mode 决定了 Instructor 如何将 Pydantic 模型的 Schema 传达给 LLM，以及如何解析返回结果。

```python
from instructor import Mode

# 常用模式
Mode.TOOLS              # 使用 Function Calling / Tool Calling API（推荐，最可靠）
Mode.JSON               # 使用 JSON Mode，强制返回 JSON
Mode.MD_JSON            # 普通 Chat Completion，从 Markdown 代码块中解析 JSON
Mode.TOOLS_STRICT       # 使用 OpenAI Structured Outputs（受限 JSON Schema 采样）
Mode.PARALLEL_TOOLS     # 并行工具调用，单次返回多个结构化对象
Mode.FUNCTIONS          # 旧版 Function Calling（已弃用）
Mode.RESPONSES_TOOLS    # 使用 OpenAI Responses API（新一代 API）
Mode.JSON_O1            # O1 模型专用（不支持 system message 和 tool calling）

# Anthropic 专用
Mode.ANTHROPIC_TOOLS    # Anthropic Tool Calling
Mode.ANTHROPIC_JSON     # Anthropic JSON Mode
```

**模式选择指南：**
| 场景 | 推荐模式 | 说明 |
|------|---------|------|
| OpenAI 标准使用 | `Mode.TOOLS` | 利用 Function Calling，最稳定 |
| OpenAI 新项目 | `Mode.RESPONSES_TOOLS` | 更低延迟，支持缓存和有状态上下文 |
| 不支持 Tool Calling 的模型 | `Mode.MD_JSON` | 从纯文本中提取 JSON |
| 需要严格 JSON Schema 约束 | `Mode.TOOLS_STRICT` | OpenAI 的受限采样 |
| 提取多个同类对象 | `Mode.PARALLEL_TOOLS` | 并行返回 |
| Anthropic 模型 | `Mode.ANTHROPIC_TOOLS` | Anthropic 的工具调用 |

### 3.3 create() — 核心提取方法

`create()` 是 Instructor 最核心的方法，返回一个完全填充的 Pydantic 模型实例。

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import instructor

client = instructor.from_provider("openai/gpt-4o")

# 基本使用
class User(BaseModel):
    name: str
    age: int

user = client.create(
    response_model=User,
    messages=[
        {"role": "user", "content": "提取信息：张三今年25岁"},
    ],
)
print(user)  # User(name='张三', age=25)

# 带字段描述和约束
class Movie(BaseModel):
    title: str = Field(description="电影标题")
    year: int = Field(description="上映年份", ge=1900, le=2030)
    genre: str = Field(description="电影类型")
    rating: Optional[float] = Field(None, description="评分（0-10）", ge=0, le=10)

movie = client.create(
    response_model=Movie,
    messages=[{"role": "user", "content": "电影《盗梦空间》是2010年上映的科幻片，评分8.8"}],
)
print(movie)  # Movie(title='盗梦空间', year=2010, genre='科幻', rating=8.8)

# 嵌套模型
class Address(BaseModel):
    street: str
    city: str
    country: str

class Person(BaseModel):
    name: str
    age: int
    addresses: List[Address]

person = client.create(
    response_model=Person,
    messages=[{"role": "user", "content": "李明30岁，住在北京朝阳区建国路88号，中国"}],
)

# 枚举类型
class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"

class SentimentResult(BaseModel):
    text: str
    sentiment: Sentiment
    confidence: float = Field(ge=0, le=1)

result = client.create(
    response_model=SentimentResult,
    messages=[{"role": "user", "content": "这个产品非常好用，我非常满意！"}],
)
```

**create() 关键参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `response_model` | Type[BaseModel] | Pydantic 模型类，定义输出结构 |
| `messages` | List[dict] | 对话消息列表 |
| `max_retries` | int | 验证失败时的最大重试次数，默认 0 |
| `validation_context` | dict | 传递给 Pydantic 验证器的上下文 |
| `mode` | Mode | 覆盖客户端默认模式 |
| `stream` | bool | 是否流式输出 |

### 3.4 create_partial() — 部分流式

`create_partial()` 流式返回单个对象的渐进构建过程，适用于实时 UI 更新。

```python
class UserProfile(BaseModel):
    name: str
    age: int
    bio: str
    interests: List[str]

for partial_user in client.create_partial(
    response_model=UserProfile,
    messages=[{"role": "user", "content": "为一位28岁的软件工程师创建用户档案，名字叫小王"}],
):
    # 每次迭代返回一个更完整的 UserProfile 实例
    # 尚未填充的字段为 None
    print(partial_user)

# 输出示例：
# UserProfile(name='小', age=None, bio=None, interests=None)
# UserProfile(name='小王', age=28, bio='一位', interests=None)
# UserProfile(name='小王', age=28, bio='一位热爱编程的软件工程师', interests=['编程'])
# UserProfile(name='小王', age=28, bio='一位热爱编程的软件工程师', interests=['编程', '阅读', '游戏'])
```

### 3.5 create_iterable() — 可迭代流式

`create_iterable()` 从单次 LLM 响应中流式提取多个同类对象。

```python
class Tag(BaseModel):
    name: str
    confidence: float

for tag in client.create_iterable(
    response_model=Tag,
    messages=[{"role": "user", "content": "从以下文本中提取标签：Python是数据科学和机器学习领域最流行的编程语言，也被广泛用于Web开发"}],
):
    print(tag)

# 输出示例：
# Tag(name='Python', confidence=0.99)
# Tag(name='数据科学', confidence=0.95)
# Tag(name='机器学习', confidence=0.93)
# Tag(name='Web开发', confidence=0.85)
```

### 3.6 create_with_completion() — 获取原始完成

```python
user, completion = client.create_with_completion(
    response_model=User,
    messages=[{"role": "user", "content": "提取：李四今年35岁"}],
)

print(user)         # User(name='李四', age=35)
print(completion)   # 原始 OpenAI ChatCompletion 对象
print(completion.usage)  # token 使用信息
```

### 3.7 验证与重试

#### 自动重试

```python
from pydantic import BaseModel, Field, field_validator

class EmailContact(BaseModel):
    name: str = Field(min_length=2, max_length=50)
    email: str

    @field_validator('email')
    @classmethod
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('邮箱格式不正确，必须包含@符号')
        return v

# 当LLM返回的email不合法时，Instructor会自动重试
# 重试时会把验证错误信息附加到消息中，帮助LLM修正
contact = client.create(
    response_model=EmailContact,
    messages=[{"role": "user", "content": "联系人是张三，邮箱是zhangsan-at-gmail.com"}],
    max_retries=3,  # 最多重试3次
)
```

**重试流程：**
```
第1次请求 → LLM返回 {"name": "张三", "email": "zhangsan-at-gmail.com"}
    ↓ 验证失败：邮箱格式不正确
第2次请求（附带错误信息）→ LLM返回 {"name": "张三", "email": "zhangsan@gmail.com"}
    ↓ 验证通过
返回 EmailContact(name='张三', email='zhangsan@gmail.com')
```

#### LLM 驱动的验证（llm_validator）

Instructor 提供了 `llm_validator`，使用 LLM 本身来验证字段内容：

```python
from typing_extensions import Annotated
from pydantic import BeforeValidator
from instructor import llm_validator

class QuestionAnswer(BaseModel):
    question: str
    answer: Annotated[
        str,
        BeforeValidator(
            llm_validator("回答不能包含有害、违法或不道德的内容", client=client)
        ),
    ]

qa = client.create(
    response_model=QuestionAnswer,
    messages=[{"role": "user", "content": "问题：如何学习Python？"}],
)
```

### 3.8 Maybe 类型

`Maybe` 是 Instructor 提供的一个泛型包装器，用于处理 LLM 可能无法提取到完整数据的情况：

```python
from instructor import Maybe

# Maybe[User] 意味着：要么返回完整的User，要么返回部分数据+错误信息
result = client.create(
    response_model=Maybe[User],
    messages=[{"role": "user", "content": "提取用户信息：有一只猫叫小花"}],
)

if result.is_valid():
    print(result.value)  # 完整的 User 对象
else:
    print(result.error)  # 验证失败的原因
    print(result.value)  # 部分提取的数据
```

### 3.9 Hooks 系统

Hooks 允许在 LLM 交互过程中拦截事件，用于日志记录、监控和错误处理：

```python
client = instructor.from_provider("openai/gpt-4o")

# 请求前日志
client.on("completion:kwargs", lambda **kw: print(f"请求参数: model={kw.get('model')}"))

# 错误处理
client.on("completion:error", lambda e: print(f"请求错误: {e}"))

# 成功回调
client.on("completion:success", lambda response: print(f"成功获取响应"))
```

### 3.10 Jinja2 模板支持

Instructor 支持在消息中使用 Jinja2 模板，通过 `context` 参数填充变量：

```python
class ExtractionResult(BaseModel):
    entities: List[str]
    relationships: List[str]

result = client.create(
    response_model=ExtractionResult,
    messages=[
        {
            "role": "user",
            "content": """从以下文本中提取实体和关系：
            {{ text }}
            
            请使用{{ language }}回答。""",
        },
    ],
    context={"text": "苹果公司由史蒂夫·乔布斯创立，总部位于加利福尼亚", "language": "中文"},
)
```

### 3.11 多模态支持

Instructor 提供了统一的多模态接口，自动处理不同提供商的格式差异：

```python
from instructor.processing.multimodal import Image, PDF, Audio

# 图片分析
class ImageDescription(BaseModel):
    objects: List[str]
    scene: str
    mood: str

response = client.create(
    response_model=ImageDescription,
    messages=[
        {
            "role": "user",
            "content": [
                "描述这张图片",
                Image.from_url("https://example.com/photo.jpg"),
            ],
        },
    ],
)

# 从 PDF 提取结构化数据
class InvoiceInfo(BaseModel):
    invoice_number: str
    total_amount: float
    date: str

response = client.create(
    response_model=InvoiceInfo,
    messages=[
        {
            "role": "user",
            "content": [
                "提取发票信息",
                PDF.from_path("./invoice.pdf"),
            ],
        },
    ],
)

# 音频转录
class TranscriptSummary(BaseModel):
    summary: str
    key_points: List[str]

response = client.create(
    response_model=TranscriptSummary,
    messages=[
        {
            "role": "user",
            "content": [
                "总结这段音频",
                Audio.from_path("./meeting.mp3"),
            ],
        },
    ],
)
```

**多模态加载方式：**
| 类 | 加载方法 | 说明 |
|----|---------|------|
| `Image` | `from_url()`, `from_path()`, `from_base64()`, `autodetect()` | 图片分析 |
| `PDF` | `from_url()`, `from_path()`, `from_base64()`, `autodetect()` | PDF 提取 |
| `Audio` | `from_url()`, `from_path()` | 音频转录分析 |

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景1：智能文档信息提取

```python
from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
import instructor

client = instructor.from_provider("openai/gpt-4o")

class DocumentType(str, Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    REPORT = "report"
    EMAIL = "email"
    OTHER = "other"

class Entity(BaseModel):
    name: str = Field(description="实体名称")
    entity_type: str = Field(description="实体类型：人名/公司/地点/日期/金额等")

class DocumentAnalysis(BaseModel):
    doc_type: DocumentType = Field(description="文档类型")
    title: str = Field(description="文档标题/主题")
    summary: str = Field(description="100字以内的摘要")
    entities: List[Entity] = Field(description="提取的命名实体")
    key_dates: List[str] = Field(description="关键日期列表")
    confidence: float = Field(description="分析置信度(0-1)", ge=0, le=1)

text = """
合同编号：HT-2024-0089
甲方：北京科技有限公司
乙方：上海创新软件有限公司
签署日期：2024年3月15日
合同金额：人民币580,000元
项目内容：AI智能客服系统开发
交付日期：2024年9月30日
"""

analysis = client.create(
    response_model=DocumentAnalysis,
    messages=[{"role": "user", "content": f"分析以下文档：\n{text}"}],
    max_retries=2,
)

print(f"类型: {analysis.doc_type}")
print(f"标题: {analysis.title}")
print(f"摘要: {analysis.summary}")
for entity in analysis.entities:
    print(f"  实体: {entity.name} ({entity.entity_type})")
```

### 场景2：多步骤数据管道（链式提取）

```python
import instructor
from pydantic import BaseModel, Field
from typing import List

client = instructor.from_provider("openai/gpt-4o")

# 第1步：提取关键词
class KeywordExtraction(BaseModel):
    keywords: List[str] = Field(description="关键词列表")
    main_topic: str = Field(description="主主题")

# 第2步：生成分类
class Classification(BaseModel):
    category: str = Field(description="分类类别")
    subcategory: str = Field(description="子类别")
    relevance_score: float = Field(description="相关度分数", ge=0, le=1)

# 第3步：生成摘要
class Summary(BaseModel):
    brief: str = Field(description="一句话摘要")
    detailed: str = Field(description="详细摘要（200字以内）")

# 链式调用
article = "量子计算是一种利用量子力学原理进行计算的技术..."

# Step 1
keywords = client.create(
    response_model=KeywordExtraction,
    messages=[{"role": "user", "content": f"提取关键词：{article}"}],
)
print(f"关键词: {keywords.keywords}")

# Step 2（使用 Step 1 的结果）
classification = client.create(
    response_model=Classification,
    messages=[{"role": "user", "content": f"对以下主题进行分类：{keywords.main_topic}"}],
)
print(f"分类: {classification.category} > {classification.subcategory}")

# Step 3
summary = client.create(
    response_model=Summary,
    messages=[{"role": "user", "content": f"总结：{article}"}],
)
print(f"摘要: {summary.brief}")
```

### 场景3：异步批量处理

```python
import asyncio
import instructor
from pydantic import BaseModel
from typing import List

async_client = instructor.from_provider("openai/gpt-4o-mini", async_client=True)

class SentimentResult(BaseModel):
    text: str
    sentiment: str
    score: float

async def analyze_sentiment(text: str) -> SentimentResult:
    return await async_client.create(
        response_model=SentimentResult,
        messages=[{"role": "user", "content": f"分析情感：{text}"}],
    )

async def batch_analyze(texts: List[str]) -> List[SentimentResult]:
    # 并发处理所有文本
    tasks = [analyze_sentiment(text) for text in texts]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]

# 运行
texts = [
    "这个产品太棒了！",
    "服务态度很差，再也不来了",
    "还行吧，中规中矩",
    "性价比超高，推荐购买",
]

results = asyncio.run(batch_analyze(texts))
for r in results:
    print(f"[{r.sentiment}] {r.text} → 得分: {r.score}")
```

## 5. 数学原理

### 5.1 结构化输出的概率模型

Instructor 的核心工作是在 LLM 的条件概率分布上施加结构约束。给定提示 $x$，LLM 本身生成的是词元序列 $y = (y_1, y_2, ..., y_n)$ 的概率分布：

$$P(y \mid x) = \prod_{t=1}^{n} P(y_t \mid y_{<t}, x)$$

Instructor 通过不同 Mode 对此分布施加约束：

- **Mode.TOOLS / FUNCTIONS**：利用 LLM 的 Function Calling 接口，LLM 被约束为生成符合工具参数 Schema 的 JSON。此时生成概率被限制在 Schema 定义的子空间：

$$P_{\text{constrained}}(y \mid x) = P(y \mid x, y \in \text{Schema})$$

- **Mode.JSON**：激活 LLM 的 JSON Mode，在采样时强制生成合法 JSON（每个 token 位置只允许符合 JSON 语法的 token）。

- **Mode.MD_JSON**：无采样约束，LLM 自由生成，事后解析。相当于从 $P(y \mid x)$ 中采样后验证是否属于 Schema。

### 5.2 重试的期望次数

设单次 LLM 调用通过验证的概率为 $p$，则使用 `max_retries=N` 时成功获取有效输出的概率为：

$$P_{\text{success}} = 1 - (1 - p)^{N+1}$$

期望调用次数为：

$$E[\text{calls}] = \sum_{k=1}^{N+1} k \cdot p \cdot (1-p)^{k-1} + (N+1) \cdot (1-p)^{N+1}$$

当 $p = 0.9$，`max_retries=3` 时：$P_{\text{success}} = 1 - 0.1^4 = 99.99\%$。

### 5.3 Pydantic 验证的代数结构

Pydantic 的验证链本质上是一个函数组合（function composition）。给定字段 $f$ 上的验证器序列 $v_1, v_2, ..., v_k$，最终值通过依次应用验证器得到：

$$f_{\text{final}} = v_k \circ v_{k-1} \circ ... \circ v_1(f_{\text{raw}})$$

任何验证器返回 `ValidationError` 即中断链，触发重试。

## 6. 代码原理/架构原理

### 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    用户代码层                         │
│  client.create(response_model=MyModel, messages=...) │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Instructor 核心层                    │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐ │
│  │ Mode Handler│  │ Retry Engine │  │  Validator  │ │
│  │             │  │              │  │             │ │
│  │ TOOLS:      │  │ 验证失败 →   │  │ Pydantic    │ │
│  │ 构造Tool    │  │ 附加错误信息 │  │ 模型验证    │ │
│  │ Calling请求 │  │ → 重新请求   │  │ + 自定义    │ │
│  │             │  │              │  │   验证器    │ │
│  │ JSON:       │  │ max_retries  │  │             │ │
│  │ 构造JSON    │  │ 次数控制     │  │ llm_        │ │
│  │ Mode请求    │  │              │  │ validator   │ │
│  │             │  │              │  │             │ │
│  │ MD_JSON:    │  │              │  │             │ │
│  │ 普通请求+   │  │              │  │             │ │
│  │ 解析提取    │  │              │  │             │ │
│  └──────┬──────┘  └──────┬───────┘  └────────────┘ │
└─────────┼────────────────┼──────────────────────────┘
          │                │
┌─────────▼────────────────▼──────────────────────────┐
│                   LLM Provider 层                    │
│  ┌──────────┐ ┌───────────┐ ┌──────┐ ┌───────────┐ │
│  │  OpenAI  │ │ Anthropic │ │Google│ │   Ollama  │ │
│  │          │ │           │ │      │ │           │ │
│  │ ChatML   │ │ Messages  │ │  ... │ │  OpenAI   │ │
│  │ + Tools  │ │ + Tools   │ │      │ │ Compat    │ │
│  └──────────┘ └───────────┘ └──────┘ └───────────┘ │
└─────────────────────────────────────────────────────┘
```

### 核心流程

1. **Schema 转换**：将 Pydantic 模型转换为 LLM 能理解的格式（根据 Mode 不同，转换为 Function/Tool Schema、JSON Schema、或嵌入 prompt 的描述文本）

2. **请求构造**：根据 Mode 构造不同的 LLM 请求：
   - `TOOLS`：构造 `tools` 参数，包含函数定义
   - `JSON`：设置 `response_format={"type": "json_object"}`
   - `MD_JSON`：在 prompt 中添加"请返回JSON"的指令

3. **响应解析**：从 LLM 响应中提取结构化数据：
   - `TOOLS`：从 `tool_calls` 中提取参数 JSON
   - `JSON`：解析 `message.content` 为 JSON
   - `MD_JSON`：用正则从 Markdown 代码块中提取 JSON

4. **验证与重试**：将解析后的 JSON 传入 Pydantic 模型验证。验证失败时：
   - 将 `ValidationError` 格式化为自然语言
   - 将错误信息作为 assistant 消息附加到对话中
   - 重新发送请求给 LLM

### 设计模式

1. **装饰器/包装器模式**：Instructor 通过 `patch()` 或 `from_openai()` 将原生 OpenAI 客户端包装，在 `create()` 调用时注入 Schema 处理逻辑，而不修改原有 API。

2. **策略模式（Mode）**：不同的 Mode 实现了相同的接口（Schema 转换 + 响应解析），运行时可切换策略。

3. **Pydantic 优先**：Instructor 完全依赖 Pydantic 进行数据验证和 Schema 生成，避免了重复实现验证逻辑。用户定义的 Pydantic 模型同时承担三个角色：
   - **Schema 定义**：自动转换为 JSON Schema 传给 LLM
   - **数据验证**：LLM 返回后自动验证
   - **类型安全**：IDE 自动补全和类型检查

## 7. 常见注意事项和最佳实践

### 注意事项

1. **Mode 与模型的兼容性**：
   - `Mode.TOOLS` 要求模型支持 Function Calling（GPT-4o、Claude 3.5 等支持）
   - `Mode.JSON` 要求模型支持 JSON Mode（OpenAI 的 gpt-4o 等支持）
   - 开源模型（通过 Ollama）通常只能使用 `Mode.MD_JSON` 或 `Mode.JSON`
   - 选择不兼容的 Mode 会导致运行时错误或输出质量下降

2. **复杂模型的 token 开销**：
   - 嵌套模型和大量字段会生成较长的 Schema，消耗额外 token
   - `Field(description=...)` 的描述会传入 LLM，增加 prompt 长度
   - 建议保持模型简洁，避免过度嵌套（3层以上）

3. **重试成本**：
   - 每次 `max_retries` 重试都是一次完整的 LLM API 调用
   - 验证越严格，重试概率越高
   - 建议先用 `max_retries=0` 测试，观察验证通过率后再调整

4. **流式处理的限制**：
   - `create_partial()` 不支持 `max_retries`（部分对象无法验证）
   - `create_iterable()` 中的单个对象验证失败会跳过该对象

5. **Pydantic V2 要求**：Instructor 要求 Pydantic V2+，V1 不兼容。

### 最佳实践

1. **为每个字段添加 description**：
   ```python
   # 不好：LLM 不知道如何填充
   class Result(BaseModel):
       x: str
       y: float

   # 好：清晰的描述引导LLM正确输出
   class Result(BaseModel):
       x: str = Field(description="产品的唯一标识符")
       y: float = Field(description="价格，单位为人民币元", ge=0)
   ```

2. **使用枚举约束分类任务**：
   ```python
   from enum import Enum

   class Category(str, Enum):
       TECH = "technology"
       SPORTS = "sports"
       POLITICS = "politics"
       ENTERTAINMENT = "entertainment"
   ```
   枚举类型将可选值直接暴露给 LLM，大幅提高分类准确率。

3. **合理设置 max_retries**：
   - 简单提取（1-3个字段）：`max_retries=1` 即可
   - 复杂验证（自定义验证器）：`max_retries=3`
   - 关键任务：`max_retries=5`，但需监控成本

4. **选择合适的流式方法**：
   - 需要实时显示构建过程 → `create_partial()`
   - 需要逐个返回列表项 → `create_iterable()`
   - 需要完整验证的对象 → `create()`（非流式）

5. **利用 create_with_completion 监控成本**：
   ```python
   result, completion = client.create_with_completion(
       response_model=MyModel,
       messages=[...],
   )
   total_tokens = completion.usage.total_tokens
   ```

6. **中文场景建议**：
   - `Field(description=...)` 使用中文描述，LLM 更容易理解
   - 枚举值使用英文（方便代码处理），描述使用中文
   - 避免在模型字段名中使用中文（影响代码可读性），用 description 中文说明

7. **渐进式构建复杂 Schema**：
   - 先从简单模型开始验证基本提取能力
   - 再逐步添加字段、约束和验证器
   - 过于复杂的 Schema 可能导致 LLM 困惑，反而降低提取质量
