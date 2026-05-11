---
title: "SGLang 结构化生成引擎"
excerpt: "RadixAttention基数树KV cache复用、编程原语、约束解码"
collection: llm-libs
permalink: /llm-libs/12-sglang
category: inference
---


## 1. 简介

SGLang（Structured Generation Language）是一个面向大语言模型的结构化生成编程框架，由加州大学伯克利分校 Lianmin Zheng 等人开发。SGLang 的核心创新在于通过编程原语抽象 LLM 的生成过程，并基于 RadixAttention 技术实现 KV cache 的智能复用，在多轮对话、Agent 工作流和结构化输出等场景中显著提升推理效率。

### 在 LLM 开发中的作用

- **编程抽象**：通过 `sgl.gen()`、`sgl.select()` 等原语将 LLM 生成过程表达为可组合的程序，而非简单的字符串拼接。
- **KV cache 复用**：RadixAttention 利用基数树自动识别和复用公共前缀的 KV cache，在多轮对话和 Agent 场景中节省大量计算。
- **约束解码**：支持正则表达式、JSON Schema 等约束，确保输出符合指定格式。
- **高性能推理**：内置连续批处理和张量并行，吞吐量与 vLLM 相当甚至更高。
- **与 vLLM 的关系**：SGLang 的后端推理引擎可以基于 vLLM，但上层提供了更丰富的编程抽象和更高效的前缀复用机制。

---

## 2. 安装方式

### 基本安装

```bash
# 使用 pip 安装
pip install sglang

# 安装所有依赖（包括FlashInfer等）
pip install "sglang[all]"

# 从源码安装
git clone https://github.com/sgl-project/sglang.git
cd sglang
pip install -e ".[all]"
```

### 依赖说明

- Python >= 3.9
- PyTorch >= 2.1
- NVIDIA GPU（计算能力 >= 8.0，即 Ampere 架构及以上，推荐）
- FlashInfer（可选，用于加速注意力计算）

### 验证安装

```python
import sglang as sgl
print(sgl.__version__)
```

---

## 3. 核心类与函数详细说明

### 3.1 编程原语

SGLang 的核心设计理念是将 LLM 的生成过程表达为结构化的程序。每个生成步骤都是一个原语调用，多个原语可以组合成复杂的生成流程。

#### sgl.gen() — 自由生成

`sgl.gen()` 是最基本的生成原语，指示模型从当前状态自由生成文本。

```python
@sgl.function
def simple_qa(s, question):
    s += question
    s += sgl.gen("answer", max_tokens=256, temperature=0.7)
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | 必填 | 生成结果的变量名，后续可通过 `s["name"]` 访问 |
| `max_tokens` | int | 128 | 最大生成 token 数 |
| `temperature` | float | 1.0 | 采样温度 |
| `top_p` | float | 1.0 | 核采样概率阈值 |
| `top_k` | int | -1 | Top-k 采样 |
| `frequency_penalty` | float | 0.0 | 频率惩罚 |
| `presence_penalty` | float | 0.0 | 存在惩罚 |
| `stop` | str/list | None | 停止词 |
| `ignore_eos` | bool | False | 是否忽略 EOS |
| `regex` | str | None | 正则约束（约束解码） |
| `choices` | list | None | 限定输出选项（等价于 sgl.select） |
| `dtype` | str | None | 强制输出数据类型 |

#### sgl.select() — 选择生成

`sgl.select()` 限定模型从给定选项中选择一个输出，适用于分类、判断等任务。

```python
@sgl.function
def sentiment_analysis(s, text):
    s += "分析以下文本的情感倾向：\n"
    s += text + "\n"
    s += "情感倾向："
    s += sgl.select("sentiment", choices=["正面", "负面", "中性"])
```

**参数说明**：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `name` | str | 必填 | 选择结果的变量名 |
| `choices` | list | 必填 | 可选的选项列表 |
| `temperature` | float | 0.0 | 采样温度（默认0，确定性选择） |

**返回值**：从 `choices` 中选出的一个字符串，可通过 `s["name"]` 访问。

#### sgl.image() — 图像输入

`sgl.image()` 用于多模态模型，将图像作为输入传递给 LLM。

```python
@sgl.function
def image_qa(s, image_path, question):
    s += sgl.image(image_path)
    s += question
    s += sgl.gen("answer", max_tokens=256)
```

#### 完整原语组合示例

```python
import sglang as sgl

@sgl.function
def complex_workflow(s, topic):
    # 步骤1：生成主题大纲
    s += "请为以下主题生成大纲：" + topic + "\n"
    s += "大纲："
    s += sgl.gen("outline", max_tokens=256, temperature=0.7)

    # 步骤2：判断大纲质量
    s += "\n这份大纲是否足够详细？"
    s += sgl.select("quality", choices=["详细", "简略", "需要改进"])

    # 步骤3：根据判断结果生成详细内容
    s += "\n请根据大纲展开详细内容：\n"
    s += sgl.gen("detail", max_tokens=512, temperature=0.8)
```

### 3.2 Runtime（运行时）

`Runtime` 是 SGLang 的推理运行时，负责模型加载、请求调度和 KV cache 管理。

#### 创建 Runtime

```python
from sglang import Runtime

# 基本创建
runtime = Runtime(
    model_path="meta-llama/Llama-2-7b-hf",  # 模型路径
)

# 完整参数创建
runtime = Runtime(
    model_path="meta-llama/Llama-2-7b-hf",
    tp_size=1,                    # 张量并行大小
    mem_fraction_static=0.88,     # 静态分配的显存比例
    base_gpu_id=0,                # 起始GPU ID
    enable_radix_cache=True,      # 启用RadixAttention（默认True）
    enable_torch_compile=False,   # 启用TorchCompile优化
)
```

#### 批量推理

```python
import sglang as sgl

# 初始化运行时
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")

# 定义生成函数
@sgl.function
def qa(s, question):
    s += "问题：" + question + "\n答案："
    s += sgl.gen("answer", max_tokens=128, temperature=0.7)

# 批量运行
questions = ["什么是AI？", "什么是深度学习？", "什么是NLP？"]
states = qa.run_batch(
    [{"question": q} for q in questions],
    runtime=runtime,
    num_threads="auto",  # 自动选择并发线程数
)

# 获取结果
for state in states:
    print(state["answer"])

# 清理缓存
runtime.flush_cache()
```

#### flush_cache()

```python
runtime.flush_cache()  # 清空RadixAttention缓存的所有KV cache
```

**使用场景**：当切换到完全不同的任务或需要释放显存时调用。正常推理中不需要频繁调用，RadixAttention 会自动管理缓存的淘汰。

### 3.3 约束解码

SGLang 支持多种约束解码方式，确保 LLM 输出符合指定格式。

#### 正则表达式约束

```python
@sgl.function
def extract_info(s, text):
    s += "从以下文本中提取日期：\n"
    s += text + "\n"
    s += "日期："
    # 约束输出为日期格式
    s += sgl.gen("date", regex=r"\d{4}-\d{2}-\d{2}")

@sgl.function
def extract_number(s, text):
    s += "提取价格：\n"
    s += text + "\n价格："
    # 约束输出为数字格式
    s += sgl.gen("price", regex=r"\d+\.?\d*")
```

#### JSON 约束

```python
import json

@sgl.function
def json_output(s, query):
    s += "请以JSON格式回答：\n"
    s += query + "\n"
    s += "```json\n"
    # 使用regex约束JSON格式
    s += sgl.gen(
        "json_result",
        max_tokens=512,
        stop="```",
        regex=r'\{[^}]*\}',  # 约束为JSON对象
    )

# 使用示例
runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")
state = json_output.run(
    query="列出三种编程语言及其特点",
    runtime=runtime,
)
result = json.loads(state["json_result"])
```

#### 选择约束

```python
@sgl.function
def classify(s, text):
    s += "分类以下文本：\n" + text + "\n类别："
    s += sgl.select("category", choices=["科技", "体育", "娱乐", "政治"])

@sgl.function
def yes_no_qa(s, question):
    s += question + "\n答案："
    s += sgl.select("answer", choices=["是", "否"])
```

### 3.4 OpenAI 兼容服务器

SGLang 提供与 OpenAI API 兼容的 HTTP 服务器。

#### 启动服务器

```bash
# 基本启动
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --port 8000

# 完整参数启动
python -m sglang.launch_server \
    --model-path meta-llama/Llama-2-7b-hf \
    --port 8000 \
    --tp 2 \
    --mem-fraction-static 0.88 \
    --enable-radix-cache \
    --chat-template chat_template.jinja
```

#### 常用服务器参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--model-path` | 模型路径（必填） | - |
| `--port` | 监听端口 | 8000 |
| `--tp` | 张量并行数 | 1 |
| `--mem-fraction-static` | 静态显存分配比例 | 0.88 |
| `--enable-radix-cache` | 启用 RadixAttention | True |
| `--disable-radix-cache` | 禁用 RadixAttention | False |
| `--chunked-prefill-size` | 分块预填充大小 | -1 |
| `--chat-template` | 聊天模板路径 | 自动检测 |

#### API 调用示例

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
)

# Chat Completions
response = client.chat.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    messages=[
        {"role": "system", "content": "你是一个有帮助的AI助手。"},
        {"role": "user", "content": "解释什么是深度学习"},
    ],
    max_tokens=256,
    temperature=0.7,
)
print(response.choices[0].message.content)

# Completions
response = client.completions.create(
    model="meta-llama/Llama-2-7b-hf",
    prompt="深度学习是",
    max_tokens=128,
)
print(response.choices[0].text)
```

---

## 4. 典型使用场景与代码示例

### 4.1 多轮对话（利用 RadixAttention 复用 KV cache）

```python
import sglang as sgl

@sgl.function
def multi_turn_chat(s, messages):
    """多轮对话，自动复用历史KV cache"""
    for msg in messages:
        if msg["role"] == "user":
            s += "用户：" + msg["content"] + "\n"
        elif msg["role"] == "assistant":
            s += "助手：" + msg["content"] + "\n"
    s += "助手："
    s += sgl.gen("response", max_tokens=256, temperature=0.7)

runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")

# 第一轮对话
messages_1 = [{"role": "user", "content": "什么是机器学习？"}]
state_1 = multi_turn_chat.run(messages=messages_1, runtime=runtime)
print(state_1["response"])

# 第二轮对话（复用第一轮的KV cache）
messages_2 = messages_1 + [
    {"role": "assistant", "content": state_1["response"]},
    {"role": "user", "content": "它和深度学习有什么区别？"},
]
state_2 = multi_turn_chat.run(messages=messages_2, runtime=runtime)
print(state_2["response"])
# RadixAttention自动识别前两轮对话的公共前缀并复用KV cache
```

### 4.2 Agent 工作流

```python
import sglang as sgl
import json

@sgl.function
def agent_workflow(s, query):
    """ReAct风格的Agent工作流"""
    # 第1步：思考
    s += "查询：" + query + "\n"
    s += "思考：我需要先分析这个问题。\n"
    s += "思考："
    s += sgl.gen("thought_1", max_tokens=128, temperature=0.7)

    # 第2步：决定行动
    s += "\n行动类型："
    s += sgl.select("action_type", choices=["搜索", "计算", "回答"])

    # 第3步：根据行动类型分支
    if s["action_type"] == "回答":
        s += "\n最终答案："
        s += sgl.gen("answer", max_tokens=256, temperature=0.5)
    else:
        s += "\n行动内容："
        s += sgl.gen("action_content", max_tokens=128)
        s += "\n观察结果：[模拟观察]\n最终答案："
        s += sgl.gen("answer", max_tokens=256, temperature=0.5)

runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")

# 运行Agent（多个Agent调用共享相同的前缀结构）
queries = [
    "法国的首都人口是多少？",
    "Python的GIL是什么？",
    "量子纠缠的原理是什么？",
]
states = agent_workflow.run_batch(
    [{"query": q} for q in queries],
    runtime=runtime,
)
for state in states:
    print(f"答案: {state['answer']}")
```

### 4.3 结构化数据提取

```python
import sglang as sgl
import json

@sgl.function
def extract_structured_data(s, text):
    """从非结构化文本中提取结构化数据"""
    s += "从以下文本中提取信息：\n"
    s += text + "\n\n"

    # 使用约束解码确保输出格式
    s += "姓名："
    s += sgl.gen("name", max_tokens=20, stop="\n")
    s += "\n年龄："
    s += sgl.gen("age", regex=r"\d{1,3}", max_tokens=5)
    s += "\n职业："
    s += sgl.gen("occupation", max_tokens=30, stop="\n")
    s += "\n城市："
    s += sgl.gen("city", max_tokens=20, stop="\n")
    s += "\n情感："
    s += sgl.select("sentiment", choices=["积极", "消极", "中性"])

runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")

texts = [
    "张三今年28岁，是一名软件工程师，居住在北京，对AI技术充满热情。",
    "李四今年35岁，是一名医生，居住在上海，最近工作压力很大。",
]

states = extract_structured_data.run_batch(
    [{"text": t} for t in texts],
    runtime=runtime,
)

for state in states:
    print(json.dumps({
        "name": state["name"],
        "age": state["age"],
        "occupation": state["occupation"],
        "city": state["city"],
        "sentiment": state["sentiment"],
    }, ensure_ascii=False, indent=2))
```

### 4.4 自我一致性（Self-Consistency）推理

```python
import sglang as sgl
from collections import Counter

@sgl.function
def math_reasoning(s, problem):
    """数学推理：生成多条推理路径"""
    s += "问题：" + problem + "\n"
    s += "请逐步推理：\n"
    s += sgl.gen("reasoning", max_tokens=512, temperature=0.7)
    s += "\n因此，答案是："
    s += sgl.gen("answer", max_tokens=20, stop="\n")

runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")

problem = "一个农场有鸡和兔子共35个头，94只脚，问鸡和兔子各多少只？"

# 生成多条推理路径
states = math_reasoning.run_batch(
    [{"problem": problem} for _ in range(8)],  # 8条推理路径
    runtime=runtime,
    num_threads=8,
)

# 投票选择最常见答案
answers = [s["answer"].strip() for s in states]
most_common = Counter(answers).most_common(1)[0]
print(f"多数投票答案: {most_common[0]} (出现{most_common[1]}次)")
```

### 4.5 LLM 作为 Judge（评估）

```python
import sglang as sgl

@sgl.function
def llm_judge(s, question, answer_a, answer_b):
    """使用LLM作为裁判评估两个回答"""
    s += "请评估以下两个回答的质量：\n\n"
    s += "问题：" + question + "\n\n"
    s += "回答A：" + answer_a + "\n\n"
    s += "回答B：" + answer_b + "\n\n"
    s += "评估标准：准确性、完整性、清晰度\n\n"
    s += "更好的回答是："
    s += sgl.select("winner", choices=["A", "B", "两者相当"])

runtime = sgl.Runtime(model_path="meta-llama/Llama-2-7b-hf")

question = "什么是量子纠缠？"
answer_a = "量子纠缠是两个粒子之间的关联..."
answer_b = "量子纠缠指两个量子系统之间存在非经典的相关性..."

state = llm_judge.run(
    question=question,
    answer_a=answer_a,
    answer_b=answer_b,
    runtime=runtime,
)
print(f"裁判结果: {state['winner']}")
```

---

## 5. 数学原理

### 5.1 RadixAttention：基于基数树的 KV cache 管理

RadixAttention 是 SGLang 的核心创新，使用基数树（Radix Tree，又称 Patricia Trie）来管理和复用 KV cache。

#### 问题背景

在多轮对话和 Agent 场景中，多个请求通常共享大量公共前缀（如系统提示、历史对话）。传统的 PagedAttention 虽然解决了显存碎片化问题，但无法自动识别和复用跨请求的公共前缀。

#### 基数树（Radix Tree）原理

基数树是一种压缩前缀树，其核心思想是合并只有一个子节点的连续节点，从而减少树的深度。

**基本性质**：

1. 每个节点代表一个 token 序列（可以是多个 token）
2. 从根到某节点的路径拼接起来就是该节点代表的完整序列
3. 共享前缀的序列在树中共享路径

**基数树示例**：

```
输入序列：
  A: "系统提示 | 你是助手 | 问题1"
  B: "系统提示 | 你是助手 | 问题2"
  C: "系统提示 | 你是助手 | 问题1 | 回答1 | 问题3"

基数树结构：
  root
   └── "系统提示 | 你是助手"（共享前缀，KV cache 只计算一次）
        ├── "问题1"（KV cache 复用共享前缀）
        │    └── "回答1 | 问题3"（继续复用）
        └── "问题2"（KV cache 复用共享前缀）
```

#### RadixAttention 工作流程

1. **前缀匹配**：对于新请求，在基数树中查找最长公共前缀
2. **复用 KV cache**：匹配到的前缀部分的 KV cache 直接复用，无需重新计算
3. **增量计算**：只计算新 token 的 KV cache，并插入基数树
4. **缓存淘汰**：当显存不足时，使用 LRU（最近最少使用）策略淘汰叶子节点

#### 查找复杂度

基数树的查找复杂度为 $O(L)$，其中 $L$ 是序列长度。

**分析**：

- 在最坏情况下，需要遍历从根到叶的路径
- 路径长度等于序列中的边数，由于基数树合并了单子节点，边数不超过 token 数
- 因此查找复杂度为 $O(L)$

**与朴素方法的对比**：

- 朴素方法（逐 token 比较）：$O(N \cdot L)$，$N$ 是缓存序列数
- 基数树方法：$O(L)$，与缓存序列数无关

#### 显存节省

假设有 $M$ 个请求共享长度为 $P$ 的前缀，每个请求的独有部分长度为 $U_i$：

**无缓存时**的总 KV cache 显存：

$$
M_{\text{no\_cache}} = M \cdot (P + U_i) \cdot d
$$

**RadixAttention**的总 KV cache 显存：

$$
M_{\text{radix}} = (P + \sum_{i=1}^{M} U_i) \cdot d
$$

**节省比例**：

$$
\text{节省比例} = 1 - \frac{M_{\text{radix}}}{M_{\text{no\_cache}}} = 1 - \frac{P + \sum U_i}{M \cdot (P + \bar{U})} = \frac{(M-1) \cdot P}{M \cdot (P + \bar{U})}
$$

当 $M$ 较大且 $P \gg \bar{U}$（长前缀、短独有部分）时，节省比例接近 $(M-1)/M$。

### 5.2 KV cache 复用的数学分析

对于标准 Transformer 的自注意力计算：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

其中 $K$ 和 $V$ 即为 KV cache。当多个请求共享前缀时：

- **共享前缀部分**：$K_{\text{prefix}}$ 和 $V_{\text{prefix}}$ 只需计算一次
- **独有部分**：每个请求只计算自己的 $K_{\text{unique}}^{(i)}$ 和 $V_{\text{unique}}^{(i)}$
- **完整 KV**：$K^{(i)} = [K_{\text{prefix}}; K_{\text{unique}}^{(i)}]$，$V^{(i)} = [V_{\text{prefix}}; V_{\text{unique}}^{(i)}]$

**节省的计算量**（FLOPs）：

对于每个注意力层，prefill 阶段的计算量为：

$$
\text{FLOPs}_{\text{attention}} = 2 \cdot L \cdot d_k \cdot (L + d_k)
$$

共享前缀节省的计算量：

$$
\text{FLOPs}_{\text{saved}} = M \cdot 2 \cdot P \cdot d_k \cdot (P + d_k) - 2 \cdot P \cdot d_k \cdot (P + d_k) = (M-1) \cdot 2 \cdot P \cdot d_k \cdot (P + d_k)
$$

### 5.3 约束解码的数学原理

约束解码通过修改 token 采样概率分布，确保输出符合指定约束。

**标准采样**：

$$
P(y_t | y_{<t}) = \frac{\exp(\text{logit}_{y_t} / T)}{\sum_{y'} \exp(\text{logit}_{y'} / T)}
$$

**约束采样**：

$$
P(y_t | y_{<t}) = \begin{cases}
\frac{\exp(\text{logit}_{y_t} / T)}{\sum_{y' \in V_{\text{valid}}} \exp(\text{logit}_{y'} / T)} & \text{if } y_t \in V_{\text{valid}} \\
0 & \text{otherwise}
\end{cases}
$$

其中 $V_{\text{valid}}$ 是在当前约束下允许的 token 集合。SGLang 使用有限状态自动机（FSA）来高效确定每个位置的 $V_{\text{valid}}$：

1. 将正则表达式编译为 FSA
2. 在每个生成步骤，根据 FSA 的当前状态确定允许的 token
3. 生成 token 后，根据 token 值转移 FSA 状态

---

## 6. 代码原理与架构原理

### 6.1 整体架构

```
┌──────────────────────────────────────────────────────┐
│                  SGLang Frontend                     │
│  ┌──────────────┐  ┌──────────────┐                 │
│  │ sgl.function │  │ sgl.gen()    │                 │
│  │ sgl.select() │  │ sgl.image()  │                 │
│  │ (编程原语)    │  │ (生成原语)   │                 │
│  └──────┬───────┘  └──────┬───────┘                 │
│         └────────┬────────┘                         │
└──────────────────┼───────────────────────────────────┘
                   │
┌──────────────────▼───────────────────────────────────┐
│               SGLang Runtime                         │
│  ┌─────────────┐  ┌──────────────────────────────┐  │
│  │ Scheduler   │  │ RadixAttention Manager       │  │
│  │ (调度器)     │  │ ┌────────────────────────┐  │  │
│  └──────┬──────┘  │ │   Radix Tree            │  │  │
│         │         │ │  ┌─────┐                │  │  │
│  ┌──────▼──────┐  │ │  │Root │                │  │  │
│  │ ModelRunner │  │ │  └──┬──┘                │  │  │
│  │ (模型执行)   │  │ │  ┌──┴──────────┐        │  │  │
│  └──────┬──────┘  │ │  │Prefix Node  │        │  │  │
│         │         │ │  └──┬─────┬────┘        │  │  │
│  ┌──────▼──────┐  │ │  ┌──┴──┐ ┌┴────┐       │  │  │
│  │ CUDA Kernel │  │ │  │Req A│ │Req B│       │  │  │
│  │ (GPU计算)    │  │ │  └─────┘ └─────┘       │  │  │
│  └─────────────┘  │ └────────────────────────┘  │  │
│                   └──────────────────────────────┘  │
└──────────────────────────────────────────────────────┘
```

### 6.2 编程原语的执行模型

SGLang 的编程原语采用延迟执行（Lazy Execution）模型：

1. **构建阶段**：`@sgl.function` 装饰的函数被调用时，不立即执行生成，而是构建一个执行图
2. **优化阶段**：Runtime 对执行图进行分析，识别可以批处理的操作
3. **执行阶段**：按优化后的计划执行生成，利用 RadixAttention 复用 KV cache

```python
# 构建阶段：创建SGLang程序
@sgl.function
def my_program(s, x):
    s += "问题：" + x + "\n"
    s += "思考："
    s += sgl.gen("thought", max_tokens=128)
    s += "\n答案："
    s += sgl.gen("answer", max_tokens=64)

# 执行阶段：运行程序
state = my_program.run(x="什么是AI？", runtime=runtime)
```

### 6.3 RadixAttention 的实现

```python
# RadixAttention核心逻辑（简化版）
class RadixAttention:
    def __init__(self):
        self.tree = RadixTree()       # 基数树
        self.cache = {}               # token_hash -> KV cache blocks
        self.lru_queue = []           # LRU淘汰队列

    def match_prefix(self, token_ids):
        """在基数树中查找最长公共前缀"""
        node = self.tree.root
        matched_length = 0
        for i, token in enumerate(token_ids):
            if token in node.children:
                node = node.children[token]
                matched_length = i + 1
            else:
                break
        return matched_length, node

    def insert(self, token_ids, kv_cache_blocks):
        """将新序列的KV cache插入基数树"""
        matched_len, matched_node = self.match_prefix(token_ids)

        # 复用已匹配部分的KV cache
        reused_blocks = self._get_blocks(matched_node)

        # 只为新增部分分配和计算KV cache
        new_tokens = token_ids[matched_len:]
        new_blocks = self._allocate_and_compute(new_tokens)

        # 插入基数树
        self.tree.insert(matched_node, new_tokens, new_blocks)

        return reused_blocks, new_blocks

    def evict(self, num_blocks_needed):
        """LRU策略淘汰KV cache"""
        while num_blocks_needed > 0 and self.lru_queue:
            victim = self.lru_queue.pop(0)  # 淘汰最久未使用
            if victim.is_leaf():
                self._free_blocks(victim.blocks)
                self.tree.remove(victim)
                num_blocks_needed -= len(victim.blocks)
```

### 6.4 与 vLLM 的架构对比

| 特性 | vLLM | SGLang |
|------|------|--------|
| KV cache 管理 | PagedAttention（分页） | RadixAttention（基数树） |
| 跨请求复用 | 前缀缓存（可选） | 自动前缀复用（默认） |
| 编程模型 | 字符串 API | 结构化编程原语 |
| 约束解码 | 不原生支持 | 原生支持（regex、select） |
| 后端 | 自有推理引擎 | 可选 vLLM 或自有引擎 |
| 最佳场景 | 通用推理服务 | Agent/多轮对话/结构化输出 |

### 6.5 连续批处理与调度

SGLang 的调度器与 vLLM 类似，采用连续批处理策略，但增加了对 RadixAttention 的感知：

```python
# SGLang调度器核心逻辑（简化版）
class SGLangScheduler:
    def schedule(self, waiting_queue, running_queue):
        """每次迭代的调度"""
        # 1. 处理running队列（继续解码）
        for req in running_queue:
            if req.is_finished():
                running_queue.remove(req)
                # 注意：完成的请求的KV cache保留在基数树中
                # 以便后续请求复用
            else:
                req.decode_step()

        # 2. 从waiting队列调度新请求
        for req in waiting_queue:
            # 检查RadixAttention中的前缀匹配
            matched_len = self.radix_attn.match_prefix(req.token_ids)

            # 只需为新增部分分配显存
            new_tokens = len(req.token_ids) - matched_len
            if self.has_memory_for(new_tokens):
                req.set_prefix_match(matched_len)
                running_queue.append(req)
                waiting_queue.remove(req)
```

---

## 7. 常见注意事项与最佳实践

### 7.1 RadixAttention 使用建议

```python
# 推荐场景：多个请求共享前缀
# - 多轮对话：历史消息作为共享前缀
# - RAG：系统提示和检索上下文作为共享前缀
# - Agent工作流：工具描述和指令作为共享前缀
runtime = sgl.Runtime(
    model_path="model-name",
    enable_radix_cache=True,  # 默认True，建议保持
)

# 不推荐场景：每个请求完全独立，无共享前缀
# 此时RadixAttention会增加少量查找开销
# 可以禁用以获得更好的单请求性能
runtime = sgl.Runtime(
    model_path="model-name",
    enable_radix_cache=False,  # 完全独立请求时
)
```

### 7.2 批量推理优化

```python
# 使用run_batch进行批量推理，而不是循环调用run
questions = ["问题1", "问题2", "问题3", ...]

# 好的做法：批量推理
states = qa.run_batch(
    [{"question": q} for q in questions],
    runtime=runtime,
    num_threads="auto",  # 自动选择最优并发数
)

# 不好的做法：逐个推理
states = []
for q in questions:
    state = qa.run(question=q, runtime=runtime)
    states.append(state)
```

### 7.3 约束解码最佳实践

```python
@sgl.function
def constrained_output(s, query):
    # 1. 使用select代替gen+stop进行分类
    s += "情感："
    s += sgl.select("sentiment", choices=["正面", "负面", "中性"])
    # 比以下方式更高效、更准确：
    # s += sgl.gen("sentiment", max_tokens=5, stop=["正面", "负面", "中性"])

    # 2. 使用regex确保格式正确
    s += "\n日期："
    s += sgl.gen("date", regex=r"\d{4}-\d{2}-\d{2}")
    # 比依赖模型自行格式化更可靠

    # 3. 约束范围不要过于严格
    # 好的做法：合理的正则范围
    s += "\n金额："
    s += sgl.gen("amount", regex=r"\d+\.?\d{0,2}")

    # 不好的做法：过于严格的约束可能导致模型困惑
    # s += sgl.gen("amount", regex=r"12345\.67")  # 太严格
```

### 7.4 显存管理

```python
# 调整静态显存比例
runtime = sgl.Runtime(
    model_path="model-name",
    mem_fraction_static=0.88,  # 默认0.88
    # 如果OOM，可以降低此值（但会降低KV cache容量）
    # 如果显存充足，可以适当提高
)

# 定期清理缓存（通常不需要）
# RadixAttention会自动管理缓存淘汰
runtime.flush_cache()  # 仅在切换完全不同的任务时使用
```

### 7.5 与 vLLM 对比选择指南

| 场景 | 推荐 | 原因 |
|------|------|------|
| 通用推理 API 服务 | vLLM | 成熟稳定，社区支持广泛 |
| 多轮对话系统 | SGLang | RadixAttention 自动复用历史 KV cache |
| Agent 工作流 | SGLang | 编程原语适合表达复杂流程 |
| 结构化输出 | SGLang | 原生约束解码支持 |
| 批量离线推理 | vLLM 或 SGLang | 性能接近，根据是否需要约束解码选择 |
| 生产环境部署 | vLLM | 更成熟，文档更完善 |
| 实验性项目 | SGLang | 编程抽象更灵活 |

### 7.6 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| RadixAttention 命中率低 | 请求间无共享前缀 | 确保系统提示一致，或禁用 RadixAttention |
| 约束解码输出不完整 | 约束过于严格或 max_tokens 太小 | 放宽约束、增大 max_tokens |
| OOM | 显存不足 | 降低 `mem_fraction_static`、使用量化 |
| 多 GPU 推理慢 | 通信开销 | 确保 NVLink 连接、增大 batch size |
| select 输出不在选项中 | 模型能力不足 | 使用更大的模型或调整提示 |

### 7.7 性能优化清单

1. **启用 RadixAttention**：默认开启，对多轮对话和 Agent 场景至关重要
2. **使用 `run_batch`**：批量推理比逐个调用效率高得多
3. **合理设置 `num_threads`**：通常设为 `"auto"` 即可
4. **使用约束解码**：`select` 比自由生成+后处理更高效
5. **共享前缀设计**：在系统提示中尽可能使用一致的模板
6. **适时清理缓存**：切换任务时调用 `flush_cache()`
7. **调整 `mem_fraction_static`**：根据实际显存情况微调
8. **使用量化模型**：AWQ/GPTQ 量化可显著减少显存占用
