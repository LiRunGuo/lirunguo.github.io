---
title: "Pydantic 数据验证库"
excerpt: "BaseModel、数据验证、序列化、Field约束、与LLM结构化输出配合"
collection: llm-libs
permalink: /llm-libs/04-pydantic
category: core
toc: true
---


## 1. 简介与在 LLM 开发中的作用

Pydantic 是 Python 生态中最流行的数据验证库，基于 Python 类型注解（Type Hints）在运行时进行数据校验和序列化。它将 Python 的类型系统从"仅用于静态检查"提升为"运行时强保证"，让数据模型的定义、验证、序列化一体化完成。

Pydantic 的核心思想是：**用类型注解定义数据结构，框架自动完成验证、转换和序列化**。

在 LLM（大语言模型）开发中，Pydantic 扮演着至关重要的角色：

- **LLM API 请求/响应模型定义**：为 OpenAI、Anthropic 等 API 的请求参数和响应数据定义强类型模型
- **结构化输出验证**：确保 LLM 的输出符合预定义的 JSON Schema，实现可靠的函数调用（Function Calling）
- **与 instructor/outlines 配合**：instructor 库基于 Pydantic 模型驱动 LLM 生成结构化输出；outlines 使用 Pydantic 模型约束生成过程
- **配置管理**：API Key、模型参数、推理配置等的类型安全加载
- **数据管道验证**：在 ETL 管道中确保每一步数据的类型和约束正确

---

## 2. 安装方式

```bash
# 基础安装（Pydantic V2）
pip install pydantic

# 常用搭配安装
pip install pydantic email-validator    # 邮箱验证支持

# LLM 开发推荐安装
pip install pydantic instructor openai  # 结构化输出全家桶

# Conda 安装
conda install pydantic

# 安装后验证
python -c "import pydantic; print(pydantic.__version__)"  # 输出: 2.x.x
```

**版本说明**：本文档基于 Pydantic V2 编写。V2 相比 V1 有重大改变（使用 Rust 重写核心、API 变更），如果你从 V1 迁移，请参考官方迁移指南。

---

## 3. 核心功能详解

### 3.1 BaseModel — 模型定义

BaseModel 是 Pydantic 的核心基类，所有数据模型都继承自它。

```python
from pydantic import BaseModel
from typing import Optional, List

# 基本模型定义
class LLMPrompt(BaseModel):
    prompt: str
    max_tokens: int = 100         # 带默认值的字段
    temperature: float = 0.7      # 带默认值的字段
    top_p: float = 1.0

# 实例化
request = LLMPrompt(prompt="解释量子计算")
print(request.prompt)          # 解释量子计算
print(request.max_tokens)      # 100
print(request.temperature)     # 0.7

# 模型字段会自动类型转换（coercion）
request = LLMPrompt(prompt="Hello", max_tokens="200", temperature=0)
print(request.max_tokens)      # 200（str → int 自动转换）
print(type(request.max_tokens)) # <class 'int'>

# 访问模型字段元数据
print(LLMPrompt.model_fields)
# {'prompt': FieldInfo(...), 'max_tokens': FieldInfo(...), ...}

# 模型转字典
print(request.model_dump())
# {'prompt': 'Hello', 'max_tokens': 200, 'temperature': 0.0, 'top_p': 1.0}

# 模型转JSON字符串
print(request.model_dump_json())
# '{"prompt":"Hello","max_tokens":200,"temperature":0.0,"top_p":1.0}'
```

**BaseModel 核心类方法**：

| 方法 | 说明 |
|------|------|
| `model_dump()` | 序列化为字典 |
| `model_dump_json()` | 序列化为 JSON 字符串 |
| `model_validate(data)` | 从字典/对象验证并创建实例 |
| `model_validate_json(json_str)` | 从 JSON 字符串验证并创建实例 |
| `model_json_schema()` | 生成 JSON Schema |
| `model_rebuild()` | 重建模型（解决前向引用） |
| `model_copy()` | 复制模型实例 |

### 3.2 字段类型

Pydantic 支持丰富的字段类型，并能自动进行类型转换和验证。

```python
from pydantic import BaseModel
from typing import Optional, List, Dict, Literal, Union
from datetime import datetime
from uuid import UUID

class LLMConfig(BaseModel):
    # ---- 基本类型 ----
    name: str                                    # 字符串
    version: int                                 # 整数
    score: float                                 # 浮点数
    is_active: bool                              # 布尔值

    # ---- 可选类型 ----
    description: Optional[str] = None            # 可选，默认None
    tags: List[str] = []                         # 列表，默认空列表

    # ---- 复杂类型 ----
    metadata: Dict[str, str] = {}                # 字典
    model_type: Literal['gpt', 'claude', 'llama'] = 'gpt'  # 字面量枚举

    # ---- 日期时间 ----
    created_at: datetime = datetime.now()        # 日期时间

    # ---- 联合类型 ----
    value: Union[str, int] = 0                   # 字符串或整数

# 实例化与验证
config = LLMConfig(
    name="MyModel",
    version="2",      # str → int 自动转换
    score="0.95",     # str → float 自动转换
    is_active=1,      # int → bool 自动转换
    model_type="gpt"
)
print(config.version)    # 2 (int)
print(config.score)      # 0.95 (float)
print(config.is_active)  # True (bool)
```

**类型自动转换规则**：

| 源类型 | 目标类型 | 转换规则 |
|--------|---------|---------|
| `str` | `int` | `"123"` → `123` |
| `str` | `float` | `"3.14"` → `3.14` |
| `str` | `bool` | `"true"` → `True`（不区分大小写） |
| `int` | `bool` | `1` → `True`, `0` → `False` |
| `str` | `datetime` | ISO格式自动解析 |
| `str` | `UUID` | 合法UUID字符串自动解析 |

### 3.3 默认值与默认工厂

```python
from pydantic import BaseModel, Field
from typing import List, Dict
import time

class ModelConfig(BaseModel):
    # 静态默认值
    name: str = "default"
    temperature: float = 0.7

    # 使用 Field 定义默认值（推荐）
    max_tokens: int = Field(default=100, description="最大生成token数")

    # 可变默认值必须使用 default_factory
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, str] = Field(default_factory=dict)

    # 动态默认值
    created_at: float = Field(default_factory=time.time)

# ⚠️ 错误示范：可变默认值直接赋值
# class Bad(BaseModel):
#     tags: List[str] = []  # 所有实例共享同一个列表！

# ✅ 正确做法
# class Good(BaseModel):
#     tags: List[str] = Field(default_factory=list)
```

---

## 4. 数据验证

### 4.1 自动类型检查

Pydantic 在实例化时自动进行类型验证和转换，验证失败抛出 `ValidationError`。

```python
from pydantic import BaseModel, ValidationError

class APIRequest(BaseModel):
    model: str
    temperature: float
    max_tokens: int

# 正常验证
req = APIRequest(model="gpt-4", temperature=0.7, max_tokens=100)
print(req)  # model='gpt-4' temperature=0.7 max_tokens=100

# 验证失败
try:
    req = APIRequest(model="gpt-4", temperature="not_a_number", max_tokens=100)
except ValidationError as e:
    print(e.error_count())   # 错误数量
    print(e.errors())        # 错误详情列表
    # [{'type': 'float_parsing', 'loc': ('temperature',), 'msg': '...', 'input': 'not_a_number'}]

# 缺少必填字段
try:
    req = APIRequest(model="gpt-4")
except ValidationError as e:
    print(e.errors())
    # [{'type': 'missing', 'loc': ('temperature',), 'msg': 'Field required', ...}]
    # [{'type': 'missing', 'loc': ('max_tokens',), 'msg': 'Field required', ...}]
```

**ValidationError 的关键属性**：

| 属性/方法 | 说明 |
|----------|------|
| `e.errors()` | 错误详情列表，每项包含 `type`、`loc`、`msg`、`input` |
| `e.error_count()` | 错误数量 |
| `e.json()` | 错误信息序列化为 JSON |
| `str(e)` | 人类可读的错误信息 |

### 4.2 自定义验证器 @field_validator

`@field_validator` 用于对单个字段添加自定义验证逻辑。

```python
from pydantic import BaseModel, field_validator, ValidationError

class ChatRequest(BaseModel):
    model: str
    temperature: float
    max_tokens: int
    prompt: str

    # 验证单个字段 —— 在类型转换之后执行
    @field_validator('temperature')
    @classmethod
    def temperature_range(cls, v: float) -> float:
        """确保温度在 [0, 2] 范围内"""
        if not 0 <= v <= 2:
            raise ValueError(f'temperature 必须在 0-2 之间，当前值: {v}')
        return v

    @field_validator('max_tokens')
    @classmethod
    def max_tokens_positive(cls, v: int) -> int:
        """确保 max_tokens 为正整数"""
        if v <= 0:
            raise ValueError('max_tokens 必须为正整数')
        return v

    @field_validator('model')
    @classmethod
    def supported_model(cls, v: str) -> str:
        """确保模型名称合法"""
        supported = {'gpt-4', 'gpt-3.5-turbo', 'claude-3', 'llama-3'}
        if v not in supported:
            raise ValueError(f'不支持的模型: {v}，支持的模型: {supported}')
        return v

    # 多字段同时验证
    @field_validator('prompt')
    @classmethod
    def prompt_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError('prompt 不能为空')
        return v.strip()

# 测试
try:
    req = ChatRequest(model="gpt-5", temperature=3.0, max_tokens=-1, prompt="  ")
except ValidationError as e:
    for err in e.errors():
        print(f"字段 {err['loc']}: {err['msg']}")
```

**field_validator 参数**：

| 参数 | 说明 |
|------|------|
| `*fields` | 要验证的字段名 |
| `mode` | `'after'`（默认，类型转换后验证）或 `'before'`（类型转换前验证） |
| `check_fields` | 是否检查字段存在（默认 True） |

### 4.3 模型级验证 @model_validator

当验证逻辑涉及多个字段之间的约束时，使用 `@model_validator`。

```python
from pydantic import BaseModel, model_validator, ValidationError

class GenerationConfig(BaseModel):
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 50
    frequency_penalty: float = 0.0

    @model_validator(mode='after')
    def check_sampling_params(self) -> 'GenerationConfig':
        """temperature 和 top_p 不能同时设置"""
        if self.temperature < 0.01 and self.top_p < 1.0:
            raise ValueError('不能同时使用低 temperature 和低 top_p 采样')
        return self

    @model_validator(mode='after')
    def check_penalty_range(self) -> 'GenerationConfig':
        """frequency_penalty 范围检查"""
        if not -2.0 <= self.frequency_penalty <= 2.0:
            raise ValueError('frequency_penalty 必须在 -2.0 到 2.0 之间')
        return self

# 测试
try:
    config = GenerationConfig(temperature=0.0, top_p=0.5)
except ValidationError as e:
    print(e)
```

---

## 5. 序列化与反序列化

### 5.1 model_dump — 序列化为字典

```python
from pydantic import BaseModel, Field
from typing import Optional

class LLMResponse(BaseModel):
    text: str
    score: float
    model_name: str
    tokens_used: int
    internal_id: Optional[int] = None

response = LLMResponse(
    text="量子计算利用量子叠加和纠缠...",
    score=0.95,
    model_name="gpt-4",
    tokens_used=150,
    internal_id=42
)

# 完整序列化
print(response.model_dump())
# {'text': '量子计算利用量子叠加和纠缠...', 'score': 0.95, 'model_name': 'gpt-4', 'tokens_used': 150, 'internal_id': 42}

# 排除指定字段
print(response.model_dump(exclude={'internal_id'}))
# {'text': '量子计算利用量子叠加和纠缠...', 'score': 0.95, 'model_name': 'gpt-4', 'tokens_used': 150}

# 只包含指定字段
print(response.model_dump(include={'text', 'score'}))
# {'text': '量子计算利用量子叠加和纠缠...', 'score': 0.95}

# 排除未设置的字段（默认值为None且未修改的字段）
print(response.model_dump(exclude_none=True))
# {'text': '...', 'score': 0.95, 'model_name': 'gpt-4', 'tokens_used': 150, 'internal_id': 42}

# 排除默认值
print(response.model_dump(exclude_defaults=True))
# {'text': '量子计算利用量子叠加和纠缠...', 'score': 0.95, 'model_name': 'gpt-4', 'tokens_used': 150}
```

**model_dump 参数**：

| 参数 | 说明 |
|------|------|
| `mode` | `'python'`（默认，返回 Python 类型）或 `'json'`（返回 JSON 兼容类型） |
| `include` | 只包含指定字段 |
| `exclude` | 排除指定字段 |
| `exclude_none` | 排除值为 None 的字段 |
| `exclude_defaults` | 排除使用默认值的字段 |
| `exclude_unset` | 排除未显式设置的字段 |

### 5.2 model_dump_json — 序列化为 JSON 字符串

```python
# 序列化为 JSON 字符串
json_str = response.model_dump_json()
print(json_str)
# '{"text":"量子计算利用量子叠加和纠缠...","score":0.95,"model_name":"gpt-4","tokens_used":150,"internal_id":42}'

# 带缩进
json_str = response.model_dump_json(indent=2)

# 排除字段
json_str = response.model_dump_json(exclude={'internal_id'})
```

### 5.3 model_validate — 反序列化（从字典）

```python
from pydantic import BaseModel, ValidationError

class ChatMessage(BaseModel):
    role: str
    content: str

# 从字典创建（带验证）
data = {"role": "user", "content": "Hello!"}
msg = ChatMessage.model_validate(data)
print(msg)  # role='user' content='Hello!'

# 验证失败
try:
    ChatMessage.model_validate({"role": "user"})  # 缺少 content
except ValidationError as e:
    print(e)
```

### 5.4 model_validate_json — 反序列化（从 JSON 字符串）

```python
# 从 JSON 字符串创建
json_str = '{"role": "assistant", "content": "你好！"}'
msg = ChatMessage.model_validate_json(json_str)
print(msg)  # role='assistant' content='你好！'

# 验证失败
try:
    ChatMessage.model_validate_json('{"role": "user"}')
except ValidationError as e:
    print(e)
```

### 5.5 嵌套模型的序列化

```python
from pydantic import BaseModel
from typing import List

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: float = 0.7

# 从嵌套字典创建
req = ChatRequest.model_validate({
    "model": "gpt-4",
    "messages": [
        {"role": "system", "content": "你是一个助手"},
        {"role": "user", "content": "你好"}
    ]
})

# 嵌套序列化
print(req.model_dump())
# {'model': 'gpt-4', 'messages': [{'role': 'system', 'content': '你是一个助手'}, {'role': 'user', 'content': '你好'}], 'temperature': 0.7}

print(req.model_dump_json())
# '{"model":"gpt-4","messages":[{"role":"system","content":"你是一个助手"},{"role":"user","content":"你好"}],"temperature":0.7}'
```

---

## 6. Field — 字段约束与元数据

`Field` 函数用于为模型字段添加约束条件和元数据信息。

```python
from pydantic import BaseModel, Field
from typing import Optional

class ModelParams(BaseModel):
    # ---- 数值约束 ----
    temperature: float = Field(
        default=0.7,
        ge=0.0,          # 大于等于 (greater than or equal)
        le=2.0,          # 小于等于 (less than or equal)
        description="采样温度，越高越随机"
    )

    top_p: float = Field(
        default=1.0,
        gt=0.0,          # 大于 (greater than)
        le=1.0,          # 小于等于
        description="核采样概率阈值"
    )

    max_tokens: int = Field(
        default=100,
        gt=0,            # 大于 0
        le=4096,         # 最大 4096
        description="最大生成 token 数"
    )

    # ---- 字符串约束 ----
    prompt: str = Field(
        min_length=1,    # 最小长度
        max_length=10000, # 最大长度
        description="输入提示文本"
    )

    model_name: str = Field(
        default="gpt-4",
        pattern=r'^(gpt-4|gpt-3.5-turbo|claude-3|llama-3)$',  # 正则约束
        description="模型名称"
    )

    # ---- 列表约束 ----
    stop_sequences: list[str] = Field(
        default_factory=list,
        max_length=4,    # 列表最大长度
        description="停止序列"
    )

    # ---- 字段描述（生成JSON Schema时使用） ----
    frequency_penalty: float = Field(
        default=0.0,
        ge=-2.0,
        le=2.0,
        description="频率惩罚，降低重复token的概率"
    )

# 验证约束
from pydantic import ValidationError

try:
    params = ModelParams(
        temperature=3.0,    # 超出 le=2.0
        top_p=0.0,          # 不满足 gt=0.0
        max_tokens=0,       # 不满足 gt=0
        prompt="",          # 不满足 min_length=1
        model_name="unknown"  # 不匹配 pattern
    )
except ValidationError as e:
    for err in e.errors():
        print(f"字段 {err['loc']}: {err['msg']}")

# 正常创建
params = ModelParams(prompt="解释量子计算", temperature=0.5, max_tokens=500)
print(params.model_dump())
```

**Field 约束参数汇总**：

| 参数 | 适用类型 | 说明 |
|------|---------|------|
| `gt` | 数值 | 大于 (>) |
| `ge` | 数值 | 大于等于 (>=) |
| `lt` | 数值 | 小于 (<) |
| `le` | 数值 | 小于等于 (<=) |
| `multiple_of` | 数值 | 必须是某数的倍数 |
| `min_length` | 字符串/列表 | 最小长度 |
| `max_length` | 字符串/列表 | 最大长度 |
| `pattern` | 字符串 | 正则表达式约束 |
| `default` | 所有 | 默认值 |
| `default_factory` | 所有 | 默认值工厂函数 |
| `description` | 所有 | 字段描述 |
| `alias` | 所有 | 字段别名 |
| `title` | 所有 | 字段标题（用于 JSON Schema） |
| `examples` | 所有 | 示例值（用于 JSON Schema） |

### 6.1 字段别名 alias

```python
from pydantic import BaseModel, Field

class APIResponse(BaseModel):
    # Python 用下划线，JSON/API 用驼峰
    model_name: str = Field(alias="modelName")
    token_count: int = Field(alias="tokenCount")
    finish_reason: str = Field(alias="finishReason")

# 使用别名创建
resp = APIResponse(modelName="gpt-4", tokenCount=150, finishReason="stop")

# 序列化时使用别名
print(resp.model_dump(by_alias=True))
# {'modelName': 'gpt-4', 'tokenCount': 150, 'finishReason': 'stop'}

# 不使用别名（默认用 Python 字段名）
print(resp.model_dump())
# {'model_name': 'gpt-4', 'token_count': 150, 'finish_reason': 'stop'}
```

---

## 7. 模型配置 model_config

通过 `model_config` 可以全局配置模型的行为。

```python
from pydantic import BaseModel, ConfigDict, Field

class StrictModel(BaseModel):
    model_config = ConfigDict(
        # ---- 额外字段处理 ----
        extra='forbid',          # 禁止额外字段（默认 'ignore'）
        # extra='ignore'        # 忽略额外字段
        # extra='allow'         # 允许额外字段

        # ---- 严格模式 ----
        strict=True,             # 禁止自动类型转换，要求精确类型匹配

        # ---- 验证赋值 ----
        validate_assignment=True, # 属性赋值时也进行验证（默认 False）

        # ---- 别名相关 ----
        populate_by_name=True,   # 允许同时使用字段名和别名创建

        # ---- 字段排序 ----
        # 使用定义顺序而非字母排序
    )

    name: str
    value: int

# 测试 extra='forbid'
try:
    StrictModel(name="test", value=1, extra_field="oops")
except Exception as e:
    print(f"额外字段被禁止: {e}")

# 测试 strict=True
try:
    StrictModel(name="test", value="1")  # str 不能自动转 int
except Exception as e:
    print(f"严格模式: {e}")

# 测试 validate_assignment
class ValidatedModel(BaseModel):
    model_config = ConfigDict(validate_assignment=True)
    score: float

m = ValidatedModel(score=0.9)
try:
    m.score = "not_a_number"  # 赋值时也会验证
except Exception as e:
    print(f"赋值验证: {e}")
```

**ConfigDict 常用配置**：

| 配置项 | 默认值 | 说明 |
|--------|-------|------|
| `extra` | `'ignore'` | 额外字段处理：`'ignore'`/`'allow'`/`'forbid'` |
| `strict` | `False` | 严格模式，禁止自动类型转换 |
| `validate_assignment` | `False` | 属性赋值时是否验证 |
| `populate_by_name` | `False` | 是否允许用字段名而非别名创建 |
| `frozen` | `False` | 是否不可变（True 时类似 dataclass(frozen=True)） |
| `from_attributes` | `False` | 是否从对象属性读取（兼容 ORM 模式） |
| `use_enum_values` | `False` | 是否使用枚举值而非枚举对象 |
| `str_strip_whitespace` | `False` | 是否自动去除字符串首尾空白 |
| `str_min_length` | `None` | 全局字符串最小长度 |

### 7.1 frozen 模式（不可变模型）

```python
from pydantic import BaseModel, ConfigDict

class FrozenConfig(BaseModel):
    model_config = ConfigDict(frozen=True)
    api_key: str
    base_url: str

config = FrozenConfig(api_key="sk-xxx", base_url="https://api.openai.com/v1")

try:
    config.api_key = "new-key"  # 不可修改
except Exception as e:
    print(f"不可变模型: {e}")

# frozen 模型可以作为字典键或放入集合
print(hash(config))  # 可哈希
```

---

## 8. 在 LLM 开发中的典型使用场景和代码示例

### 8.1 LLM API 请求/响应模型定义

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum

class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class Message(BaseModel):
    role: Role
    content: str
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str = Field(
        pattern=r'^(gpt-4|gpt-3.5-turbo|claude-3|llama-3)$',
        description="模型ID"
    )
    messages: List[Message] = Field(
        min_length=1,
        description="对话消息列表"
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=None, gt=0)
    stop: Optional[List[str]] = Field(default=None, max_length=4)
    stream: bool = False

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: Literal["stop", "length", "content_filter"]

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage

# 使用示例
request = ChatCompletionRequest(
    model="gpt-4",
    messages=[
        Message(role=Role.SYSTEM, content="你是一个Python专家"),
        Message(role=Role.USER, content="解释装饰器")
    ],
    temperature=0.5,
    max_tokens=500
)

print(request.model_dump_json(indent=2))
```

### 8.2 结构化输出验证

```python
from pydantic import BaseModel, Field, ValidationError
from typing import List, Optional

# 定义期望的LLM输出结构
class Entity(BaseModel):
    name: str = Field(description="实体名称")
    entity_type: str = Field(description="实体类型，如人名、地名、组织")
    description: str = Field(description="实体描述")

class Relation(BaseModel):
    subject: str = Field(description="主语实体")
    predicate: str = Field(description="关系谓词")
    object: str = Field(description="宾语实体")

class KnowledgeExtraction(BaseModel):
    """从文本中提取的知识三元组"""
    entities: List[Entity] = Field(description="提取的实体列表")
    relations: List[Relation] = Field(description="提取的关系列表")
    summary: str = Field(description="文本摘要")

# 验证LLM输出
llm_output = """
{
    "entities": [
        {"name": "爱因斯坦", "entity_type": "人名", "description": "理论物理学家"},
        {"name": "相对论", "entity_type": "理论", "description": "物理学基础理论"}
    ],
    "relations": [
        {"subject": "爱因斯坦", "predicate": "提出", "object": "相对论"}
    ],
    "summary": "爱因斯坦提出了相对论"
}
"""

try:
    result = KnowledgeExtraction.model_validate_json(llm_output)
    print(f"提取到 {len(result.entities)} 个实体, {len(result.relations)} 个关系")
    print(result.model_dump_json(indent=2))
except ValidationError as e:
    print(f"LLM输出验证失败: {e}")
```

### 8.3 与 instructor 配合实现结构化生成

```python
# pip install instructor openai
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List

# 定义输出模型
class SentimentAnalysis(BaseModel):
    sentiment: str = Field(description="情感极性: positive/negative/neutral")
    confidence: float = Field(ge=0.0, le=1.0, description="置信度")
    keywords: List[str] = Field(description="关键词列表")

# 使用 instructor 包装客户端
client = instructor.from_openai(OpenAI())

# 自动让LLM生成符合模型的结构化输出
result = client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "分析以下文本的情感：这个产品真的太棒了，我非常满意！"}
    ],
    response_model=SentimentAnalysis,  # Pydantic 模型驱动生成
)

print(f"情感: {result.sentiment}")
print(f"置信度: {result.confidence}")
print(f"关键词: {result.keywords}")
```

### 8.4 配置管理与环境变量

```python
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings  # pip install pydantic-settings

class LLMAppSettings(BaseSettings):
    """LLM 应用配置，自动从环境变量读取"""

    # API 配置
    openai_api_key: str = Field(description="OpenAI API Key")
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="API 基础URL"
    )

    # 模型配置
    default_model: str = Field(default="gpt-4")
    default_temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=1000, gt=0)

    # 应用配置
    debug: bool = Field(default=False)
    log_level: str = Field(default="INFO")

    @field_validator('openai_api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v.startswith('sk-'):
            raise ValueError('API Key 必须以 sk- 开头')
        return v

    model_config = {
        "env_prefix": "LLM_",        # 环境变量前缀: LLM_OPENAI_API_KEY
        "env_file": ".env",          # 从 .env 文件读取
        "extra": "ignore"
    }

# 使用（自动从环境变量/ .env 文件读取）
# settings = LLMAppSettings()
# print(settings.openai_api_key)
# print(settings.default_model)
```

### 8.5 生成 JSON Schema（与 Function Calling 配合）

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class SearchParams(BaseModel):
    """搜索参数"""
    query: str = Field(description="搜索查询关键词")
    max_results: int = Field(default=5, gt=0, description="最大返回结果数")
    language: Optional[str] = Field(default=None, description="语言过滤")

# 生成 JSON Schema（可直接用于 OpenAI Function Calling）
schema = SearchParams.model_json_schema()
print(schema)
# {
#   'properties': {
#     'query': {'description': '搜索查询关键词', 'title': 'Query', 'type': 'string'},
#     'max_results': {'default': 5, 'description': '最大返回结果数', ...},
#     'language': {'anyOf': [{'type': 'string'}, {'type': 'null'}], 'default': None, ...}
#   },
#   'required': ['query'],
#   'title': 'SearchParams',
#   'type': 'object'
# }

# 用于 OpenAI Function Calling
import openai

tools = [
    {
        "type": "function",
        "function": {
            "name": "search",
            "description": "搜索相关信息",
            "parameters": SearchParams.model_json_schema()
        }
    }
]
```

---

## 9. 数学原理

### 9.1 类型系统的形式化

Pydantic 的验证系统可以理解为类型到验证函数的映射：

$$\text{validate}: \text{Type} \times \text{Value} \rightarrow \text{ValidValue} \cup \text{ValidationError}$$

对于复合类型，验证是递归的：

- `List[T]` → 对列表中每个元素验证类型 `T`
- `Optional[T]` → 值为 `None` 或验证类型 `T`
- `Union[T1, T2]` → 依次尝试 `T1`、`T2`，取第一个成功的

### 9.2 约束验证的数学表达

Field 约束本质上是在值空间上定义可行域：

- `ge=a`: $x \geq a$
- `gt=a`: $x > a$
- `le=b`: $x \leq b$
- `lt=b`: $x < b$
- `min_length=n`: $|s| \geq n$
- `max_length=m`: $|s| \leq m$
- `pattern=r`: $s \in L(r)$，其中 $L(r)$ 是正则表达式 $r$ 的语言

一个字段的所有约束构成可行域的交集：

$$x \in \bigcap_{i} C_i$$

其中 $C_i$ 是第 $i$ 个约束定义的可行集。

---

## 10. 代码原理 / 架构原理

### 10.1 Pydantic V2 核心架构

Pydantic V2 的核心验证逻辑使用 Rust 编写（`pydantic-core`），通过 PyO3 暴露给 Python。

```
pydantic (Python 层)
├── BaseModel              # 用户接口层
│   ├── model_validate()   # 调用 core_schema 验证
│   ├── model_dump()       # 序列化
│   └── model_fields       # 字段元数据
├── field_validator        # 注册自定义验证器
├── model_validator        # 注册模型级验证器
└── Field                  # 字段约束定义

pydantic-core (Rust 层)
├── SchemaValidator        # 核心验证引擎
│   ├── validate_python()  # Python 对象验证
│   ├── validate_json()    # JSON 字符串验证
│   └── validate_assignment()  # 赋值验证
├── SchemaSerializer       # 序列化引擎
│   ├── to_python()        # 序列化为 Python 对象
│   └── to_json()          # 序列化为 JSON
└── core_schema            # 验证模式定义
```

### 10.2 model_fields 元数据机制

当定义一个 Pydantic 模型时，框架会在类创建时收集所有字段信息，存入 `model_fields` 字典：

```python
from pydantic import BaseModel, Field

class Example(BaseModel):
    name: str
    age: int = Field(gt=0, description="年龄")

# model_fields 存储了每个字段的完整元数据
print(Example.model_fields)
# {
#   'name': FieldInfo(annotation=str, required=True, ...),
#   'age': FieldInfo(annotation=int, required=False, default=..., gt=0, description='年龄', ...)
# }
```

`model_fields` 的每个 `FieldInfo` 对象包含：

| 属性 | 说明 |
|------|------|
| `annotation` | 类型注解 |
| `default` | 默认值 |
| `default_factory` | 默认值工厂 |
| `alias` | 别名 |
| `title` | 标题 |
| `description` | 描述 |
| `gt/ge/lt/le` | 数值约束 |
| `min_length/max_length` | 长度约束 |
| `pattern` | 正则约束 |
| `metadata` | 额外元数据列表 |

### 10.3 验证流程

Pydantic 的验证流程分为以下几个阶段：

```
输入数据
    │
    ▼
┌──────────────────┐
│ 1. 前置验证器     │  mode='before' 的 field_validator / model_validator
│   (原始值处理)    │  可以修改输入数据
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 2. 类型转换       │  str→int, str→float, str→bool, str→datetime 等
│   (coercion)     │  严格模式(strict=True)下跳过
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 3. 后置验证器     │  mode='after' 的 field_validator
│   (字段级)       │  可以修改字段值
└────────┬─────────┘
         │
         ▼
┌──────────────────┐
│ 4. 模型验证器     │  mode='after' 的 model_validator
│   (模型级)       │  可以访问所有字段，进行跨字段验证
└────────┬─────────┘
         │
         ▼
    验证完成 → 返回模型实例
```

### 10.4 JSON Schema 生成原理

Pydantic 的 `model_json_schema()` 方法将模型定义转换为 JSON Schema，转换规则如下：

| Pydantic 类型 | JSON Schema 类型 |
|--------------|-----------------|
| `str` | `{"type": "string"}` |
| `int` | `{"type": "integer"}` |
| `float` | `{"type": "number"}` |
| `bool` | `{"type": "boolean"}` |
| `List[T]` | `{"type": "array", "items": {T的schema}}` |
| `Optional[T]` | `{"anyOf": [{T的schema}, {"type": "null"}]}` |
| `Field(gt=0)` | `{"exclusiveMinimum": 0}` |
| `Field(ge=0)` | `{"minimum": 0}` |
| `Field(min_length=1)` | `{"minLength": 1}` |
| `Field(pattern=r'...')` | `{"pattern": "..."}` |

这使得 Pydantic 模型可以直接驱动 LLM 的结构化输出（如 OpenAI Function Calling 的 `parameters` 字段）。

---

## 11. 常见注意事项和最佳实践

### 11.1 可变默认值

```python
from pydantic import BaseModel, Field
from typing import List, Dict

# ❌ 错误：直接使用可变默认值
# class Bad(BaseModel):
#     tags: List[str] = []          # 所有实例共享同一个列表！
#     meta: Dict[str, str] = {}     # 所有实例共享同一个字典！

# ✅ 正确：使用 default_factory
class Good(BaseModel):
    tags: List[str] = Field(default_factory=list)
    meta: Dict[str, str] = Field(default_factory=dict)
```

### 11.2 性能考虑

```python
from pydantic import BaseModel, ConfigDict

# 对于高频创建的模型，可以预编译验证器
class HighPerfModel(BaseModel):
    model_config = ConfigDict(
        # 严格模式更快（跳过类型转换）
        # 但要求输入类型精确匹配
        # strict=True,

        # 关闭不需要的验证
        validate_assignment=False,  # 不需要赋值验证时关闭
    )
    name: str
    value: int

# model_validate 比 构造函数 稍慢（因为多一层间接调用）
# 对于性能敏感场景，直接用构造函数
m = HighPerfModel(name="test", value=1)  # 比 model_validate 快
```

### 11.3 模型继承

```python
from pydantic import BaseModel

class BaseRequest(BaseModel):
    model: str
    temperature: float = 0.7

class ChatRequest(BaseRequest):
    messages: list[dict]

class CompletionRequest(BaseRequest):
    prompt: str

# 子类会继承父类的所有字段和验证器
# 注意：model_config 也会被继承
```

### 11.4 LLM 开发最佳实践

1. **始终为 LLM 输出定义 Pydantic 模型**：不要直接解析 JSON 字符串，用 `model_validate_json` 带验证地解析
2. **使用 `description` 生成清晰的 JSON Schema**：LLM 会根据 Schema 中的描述生成更准确的输出
3. **使用 `Literal` 限制枚举值**：避免 LLM 生成不在预期范围内的值
4. **使用 `Optional` 标记可选字段**：明确哪些字段是必需的
5. **`extra='forbid'` 防止幻觉字段**：LLM 可能生成多余字段，使用 forbid 模式可及时发现
6. **嵌套模型替代扁平字典**：对复杂输出使用嵌套模型，层次更清晰
7. **对 API Key 等敏感信息使用 `pydantic-settings`**：从环境变量安全加载

```python
# LLM 结构化输出验证最佳实践模板
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Literal

class LLMOutput(BaseModel):
    """LLM 输出模型的最佳实践模板"""

    model_config = ConfigDict(
        extra='forbid',           # 禁止额外字段（防止幻觉）
        str_strip_whitespace=True # 自动去除空白
    )

    # 使用 Literal 限制枚举
    category: Literal["positive", "negative", "neutral"]

    # 使用 Field 约束和描述
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="分类置信度，0-1之间"
    )

    # 可选字段明确标记
    reasoning: Optional[str] = Field(
        default=None,
        description="分类理由（可选）"
    )

    # 嵌套模型
    keywords: List[str] = Field(
        default_factory=list,
        max_length=10,
        description="关键词列表，最多10个"
    )

# 使用
import json

def parse_llm_output(raw_json: str) -> LLMOutput:
    """安全解析 LLM 输出"""
    try:
        return LLMOutput.model_validate_json(raw_json)
    except Exception as e:
        # 处理验证失败（如回退到默认值或重新请求LLM）
        raise ValueError(f"LLM 输出验证失败: {e}")
```

---

## 总结

Pydantic 通过 Python 类型注解实现运行时数据验证，是 LLM 开发中不可或缺的工具。掌握 BaseModel 定义、Field 约束、自定义验证器、序列化/反序列化、模型配置等核心功能，能够确保 LLM API 调用的类型安全、验证 LLM 输出的结构正确性、实现配置的安全管理。其基于 `model_fields` 的元数据机制和 JSON Schema 生成能力，使其成为连接 LLM 与结构化数据的标准桥梁。
