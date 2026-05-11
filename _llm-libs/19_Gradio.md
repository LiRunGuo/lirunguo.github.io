---
title: "Gradio 交互式UI框架"
excerpt: "Interface/Blocks、Chatbot组件、流式输出、HF Spaces部署"
collection: llm-libs
permalink: /llm-libs/19-gradio
category: inference
toc: true
---


## 1. 简介与在 LLM 开发中的作用

Gradio 是一个开源 Python 库，用于快速构建机器学习和深度学习模型的交互式 Web 界面。它的核心理念是：**用最少的代码创建可交互的 Demo**。

Gradio 的核心特点：

- **极简 API**：几行代码即可创建完整 Web 界面
- **内置组件**：文本框、图片、音频、视频、Chatbot 等开箱即用
- **自动分享**：`share=True` 一键生成公网链接
- **HuggingFace 集成**：一键部署到 HuggingFace Spaces
- **流式输出**：原生支持 LLM 逐 token 流式展示

### Gradio 在 LLM 开发中的角色

1. **LLM Demo 快速搭建**：将模型推理函数包装为交互式 Web 应用，快速展示效果
2. **交互式评估**：通过 Chatbot 界面进行人工评估和红队测试
3. **原型验证**：在产品开发早期快速验证 LLM 应用可行性
4. **模型对比**：并排展示不同模型的生成效果
5. **数据标注**：构建 LLM 输出质量的标注工具

---

## 2. 安装方式

### 基础安装

```bash
pip install gradio
```

### 完整安装（含额外依赖）

```bash
pip install gradio[full]  # 安装所有可选依赖
```

### 开发相关

```bash
pip install gradio                    # 核心库
pip install gradio-client             # Python 客户端，用于编程方式调用 Gradio API
pip install httpx                     # 异步 HTTP 客户端
```

### 验证安装

```python
import gradio as gr
print(gr.__version__)
```

---

## 3. 核心类/函数/工具的详细说明

### 3.1 Interface — 快速创建 UI

`gr.Interface` 是 Gradio 最核心的高层 API，用最少的代码创建输入→处理→输出的界面。

```python
import gradio as gr

def greet(name: str) -> str:
    return f"你好，{name}！"

demo = gr.Interface(
    fn=greet,                          # 处理函数，接收 inputs 的值作为参数
    inputs="text",                     # 输入组件类型（简写或组件对象）
    outputs="text",                    # 输出组件类型
    title="问候应用",                   # 界面标题
    description="输入名字获取问候",      # 界面描述
    examples=[                         # 示例输入，用户可点击快速填入
        ["张三"],
        ["李四"],
    ],
)

demo.launch()
```

**Interface 关键参数**：

| 参数 | 类型 | 说明 |
|------|------|------|
| `fn` | Callable | 处理函数，输入参数对应 inputs，返回值对应 outputs |
| `inputs` | str/Component/list | 输入组件，可以是类型字符串或组件对象 |
| `outputs` | str/Component/list | 输出组件 |
| `title` | str | 界面标题 |
| `description` | str | 界面描述（支持 Markdown） |
| `examples` | list | 示例数据，显示在界面下方 |
| `live` | bool | 设为 True 时，输入变化自动触发（无需点按钮） |
| `allow_flagging` | str | 标记模式："never"/"auto"/"manual" |
| `cache_examples` | bool | 是否缓存示例的运行结果 |

**组件类型简写**：

| 简写字符串 | 对应组件 | 适用场景 |
|-----------|---------|---------|
| `"text"` | `gr.Textbox` | 文本输入/输出 |
| `"textbox"` | `gr.Textbox` | 多行文本 |
| `"number"` | `gr.Number` | 数值 |
| `"slider"` | `gr.Slider` | 滑块 |
| `"checkbox"` | `gr.Checkbox` | 复选框 |
| `"dropdown"` | `gr.Dropdown` | 下拉选择 |
| `"image"` | `gr.Image` | 图片 |
| `"audio"` | `gr.Audio` | 音频 |
| `"file"` | `gr.File` | 文件上传 |

**多输入多输出**：

```python
import gradio as gr

def analyze(text: str, model: str, temperature: float) -> tuple:
    """多输入 → 多输出"""
    summary = f"摘要：{text[:20]}..."
    sentiment = "正面"
    word_count = len(text)
    return summary, sentiment, word_count

demo = gr.Interface(
    fn=analyze,
    inputs=[
        gr.Textbox(label="输入文本", lines=5, placeholder="请输入要分析的文本..."),
        gr.Dropdown(choices=["gpt-3.5", "gpt-4", "claude-3"], value="gpt-3.5", label="模型选择"),
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
    ],
    outputs=[
        gr.Textbox(label="摘要"),
        gr.Textbox(label="情感"),
        gr.Number(label="字数"),
    ],
    title="文本分析工具",
)

demo.launch()
```

### 3.2 Blocks — 灵活布局与事件绑定

`gr.Blocks` 提供比 Interface 更灵活的低层 API，支持自定义布局、组件组合和事件绑定。

```python
import gradio as gr

with gr.Blocks(title="LLM 对话应用") as demo:
    # 使用 Markdown 添加标题和说明
    gr.Markdown("# LLM 对话应用\n输入问题，获取 AI 回复")

    with gr.Row():                           # 水平布局
        with gr.Column(scale=3):             # 左侧列，比例3
            prompt = gr.Textbox(
                label="输入",
                placeholder="请输入你的问题...",
                lines=5,
            )
            with gr.Row():                   # 嵌套水平布局
                model_select = gr.Dropdown(
                    choices=["gpt-3.5-turbo", "gpt-4", "claude-3"],
                    value="gpt-3.5-turbo",
                    label="模型",
                )
                temperature = gr.Slider(
                    minimum=0.0, maximum=2.0,
                    value=0.7, step=0.1,
                    label="Temperature",
                )
            submit_btn = gr.Button("提交", variant="primary")  # primary 样式
            clear_btn = gr.Button("清空")

        with gr.Column(scale=2):             # 右侧列，比例2
            output = gr.Textbox(label="回复", lines=15)

    # 事件绑定
    submit_btn.click(
        fn=lambda prompt, model, temp: f"[{model}] 回复：{prompt}",
        inputs=[prompt, model_select, temperature],
        outputs=output,
    )
    clear_btn.click(
        fn=lambda: (None, None),             # 清空输入和输出
        inputs=[],
        outputs=[prompt, output],
    )

demo.launch()
```

**Blocks 布局组件**：

| 组件 | 用途 |
|------|------|
| `gr.Row()` | 水平排列子组件 |
| `gr.Column()` | 垂直排列子组件，`scale` 控制比例 |
| `gr.Tab()` / `gr.TabItem()` | 标签页切换 |
| `gr.Accordion()` | 可折叠区域 |
| `gr.Group()` | 无边框分组 |
| `gr.TabbedInterface()` | 多 Interface 标签页 |

**事件绑定方法**：

```python
# 组件支持的事件
component.click(fn, inputs, outputs)       # 点击事件
component.change(fn, inputs, outputs)      # 值变化事件
component.submit(fn, inputs, outputs)      # 回车提交（Textbox）
component.select(fn, inputs, outputs)      # 选中事件（Dropdown等）
component.upload(fn, inputs, outputs)      # 上传事件（File/Image）
component.clear(fn, inputs, outputs)       # 清除事件

# 流式输出事件（核心！用于 LLM 逐 token 输出）
component.click(fn, inputs, outputs, stream=True)   # 流式模式
```

### 3.3 Chatbot 组件 — 多轮对话

Chatbot 是 Gradio 专门为 LLM 对话场景设计的核心组件。

#### ChatInterface — 最简对话界面

```python
import gradio as gr

def respond(message: str, history: list) -> str:
    """
    参数：
    - message: 用户当前输入的消息
    - history: 历史对话列表，格式为 [[user_msg, assistant_msg], ...]

    返回：助手回复的文本
    """
    # history 自动维护对话上下文
    context = "\n".join([f"用户：{h[0]}\n助手：{h[1]}" for h in history])
    response = f"基于上下文回复：{message}"
    return response

demo = gr.ChatInterface(
    fn=respond,
    title="LLM 对话助手",
    description="基于大语言模型的对话系统",
    chatbot=gr.Chatbot(height=500),            # 自定义 Chatbot 组件
    textbox=gr.Textbox(placeholder="输入消息...", container=False, scale=7),
    additional_inputs=[
        gr.Slider(minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"),
        gr.Dropdown(choices=["gpt-3.5", "gpt-4"], value="gpt-3.5", label="模型"),
    ],
    # additional_inputs 会出现在聊天框下方
)

demo.launch()
```

**ChatInterface 关键参数**：

| 参数 | 说明 |
|------|------|
| `fn` | 对话处理函数，签名 `(message, history) -> str` |
| `chatbot` | 自定义 Chatbot 组件 |
| `textbox` | 自定义输入框组件 |
| `additional_inputs` | 额外输入控件（如 Temperature） |
| `additional_inputs_accordion` | 额外输入的折叠面板配置 |
| `retry_btn` / `undo_btn` / `clear_btn` | 按钮文本配置 |

#### 手动构建 Chatbot（更灵活）

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# LLM 多轮对话")

    chatbot = gr.Chatbot(
        height=500,
        bubble_full_width=False,          # 气泡不占满宽度
        avatar_images=(None, "🤖"),       # 用户和助手的头像
        show_copy_button=True,            # 显示复制按钮
    )

    with gr.Row():
        msg_input = gr.Textbox(
            placeholder="输入消息...",
            show_label=False,
            scale=8,
        )
        submit_btn = gr.Button("发送", variant="primary", scale=1)
        clear_btn = gr.Button("清空", scale=1)

    # 状态管理
    state = gr.State([])  # 存储对话历史

    def respond(message: str, history: list) -> tuple:
        """手动管理对话状态"""
        history.append([message, ""])  # 先添加用户消息

        # 模拟生成回复
        response = f"回复：{message}"
        history[-1][1] = response  # 更新助手回复

        return "", history  # 清空输入框，更新聊天记录

    submit_btn.click(
        fn=respond,
        inputs=[msg_input, state],
        outputs=[msg_input, chatbot],
    )
    msg_input.submit(
        fn=respond,
        inputs=[msg_input, state],
        outputs=[msg_input, chatbot],
    )
    clear_btn.click(
        fn=lambda: ([], []),
        inputs=[],
        outputs=[chatbot, state],
    )

demo.launch()
```

### 3.4 流式输出 — yield 生成器

流式输出是 Gradio 在 LLM 场景下最重要的功能，通过 Python 生成器实现逐 token 输出。

#### 基础流式输出

```python
import gradio as gr
import time

def stream_respond(message: str, history: list):
    """
    流式输出函数：
    - 使用 yield 逐步返回部分结果
    - Gradio 自动将每次 yield 的内容追加到输出
    - 函数结束时输出完成
    """
    response = f"这是对 '{message}' 的详细回复，包含多个token的流式输出。"
    partial = ""

    for char in response:
        partial += char
        time.sleep(0.05)        # 模拟逐 token 生成延迟
        yield partial           # 每次返回当前累积的完整文本

demo = gr.ChatInterface(fn=stream_respond, title="流式对话")
demo.launch()
```

#### 带历史上下文的流式对话

```python
import gradio as gr
import time

def stream_chat(message: str, history: list, model: str, temperature: float):
    """
    流式多轮对话函数

    参数：
    - message: 当前用户消息
    - history: 对话历史 [[user, assistant], ...]
    - model: 模型名称（来自 additional_inputs）
    - temperature: 采样温度（来自 additional_inputs）
    """
    # 构建上下文
    context = ""
    for user_msg, assistant_msg in history:
        context += f"用户：{user_msg}\n助手：{assistant_msg}\n"
    context += f"用户：{message}\n"

    # 模拟流式生成
    full_response = f"[{model}] 基于上下文({len(history)}轮对话)回复：{message}"
    partial = ""

    for char in full_response:
        partial += char
        time.sleep(0.03)
        yield partial

demo = gr.ChatInterface(
    fn=stream_chat,
    title="流式多轮对话",
    additional_inputs=[
        gr.Dropdown(choices=["gpt-3.5", "gpt-4", "claude-3"], value="gpt-3.5", label="模型"),
        gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature"),
    ],
)

demo.launch()
```

#### Blocks 模式下的流式 Chatbot

```python
import gradio as gr
import time

with gr.Blocks() as demo:
    gr.Markdown("# LLM 流式对话")

    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="输入消息...", show_label=False)

    def respond(message: str, chat_history: list):
        """Blocks 下的流式对话需要 yield chat_history"""
        chat_history.append([message, ""])  # 添加空回复

        response = f"回复：{message}，这是流式生成的内容。"
        for i, char in enumerate(response):
            chat_history[-1][1] += char     # 逐步更新最后一条回复
            time.sleep(0.03)
            yield chat_history              # yield 更新后的历史

    msg.submit(respond, [msg, chatbot], [chatbot])

demo.launch()
```

### 3.5 自定义组件与状态管理

#### State 组件

```python
import gradio as gr

with gr.Blocks() as demo:
    gr.Markdown("# 带状态的 LLM 对话")

    chatbot = gr.Chatbot(height=400)
    msg = gr.Textbox(placeholder="输入消息...")

    # 使用 State 存储复杂状态
    conversation_state = gr.State({
        "history": [],
        "total_tokens": 0,
        "model": "gpt-4",
    })

    def respond(message: str, state: dict):
        # 更新状态
        state["history"].append({"role": "user", "content": message})
        state["total_tokens"] += len(message)

        # 生成回复
        response = f"回复：{message}"
        state["history"].append({"role": "assistant", "content": response})
        state["total_tokens"] += len(response)

        # 转换为 Chatbot 格式
        chat_history = []
        for i in range(0, len(state["history"]), 2):
            user_msg = state["history"][i]["content"]
            assistant_msg = state["history"][i+1]["content"] if i+1 < len(state["history"]) else ""
            chat_history.append([user_msg, assistant_msg])

        return chat_history, state

    msg.submit(respond, [msg, conversation_state], [chatbot, conversation_state])

demo.launch()
```

#### 条件显示/隐藏组件

```python
import gradio as gr

with gr.Blocks() as demo:
    mode = gr.Radio(["对话模式", "补全模式"], value="对话模式", label="选择模式")

    # 对话模式组件
    with gr.Row(visible=True) as chat_row:
        chatbot = gr.Chatbot(height=400)
        chat_input = gr.Textbox(placeholder="输入对话消息...")

    # 补全模式组件
    with gr.Row(visible=False) as complete_row:
        prompt_input = gr.Textbox(label="提示词", lines=5)
        complete_output = gr.Textbox(label="补全结果", lines=10)

    def switch_mode(mode: str):
        if mode == "对话模式":
            return gr.Row(visible=True), gr.Row(visible=False)
        else:
            return gr.Row(visible=False), gr.Row(visible=True)

    mode.change(
        fn=switch_mode,
        inputs=mode,
        outputs=[chat_row, complete_row],
    )

demo.launch()
```

### 3.6 常用输入/输出组件

```python
import gradio as gr

with gr.Blocks() as demo:
    # 文本组件
    text = gr.Textbox(label="文本", placeholder="输入文本...", lines=3, max_lines=10)
    code = gr.Code(label="代码", language="python", interactive=True)

    # 数值组件
    number = gr.Number(label="数值", value=0, precision=2)
    slider = gr.Slider(minimum=0, maximum=100, value=50, step=1, label="滑块", info="拖动选择")

    # 选择组件
    dropdown = gr.Dropdown(choices=["选项1", "选项2", "选项3"], value="选项1", label="下拉选择", multiselect=False)
    radio = gr.Radio(choices=["A", "B", "C"], value="A", label="单选")
    checkbox = gr.Checkbox(label="勾选框", value=False)
    checkbox_group = gr.CheckboxGroup(choices=["特性1", "特性2", "特性3"], label="多选")

    # 文件组件
    file = gr.File(label="文件上传", file_count="single", file_types=[".txt", ".pdf"])
    files = gr.File(label="多文件上传", file_count="multiple")

    # 图片组件
    image = gr.Image(label="图片上传", type="pil", height=300)

    # 音频组件
    audio = gr.Audio(label="音频上传", type="filepath")
    mic = gr.Audio(label="录音", sources=["microphone"], type="filepath")

    # 视频
    video = gr.Video(label="视频上传")

    # DataFrame
    dataframe = gr.Dataframe(
        headers=["列1", "列2", "列3"],
        datatype=["str", "number", "str"],
        row_count=5,
        col_count=3,
        label="数据表格",
    )

    # JSON
    json_data = gr.JSON(label="JSON 数据")

    # HTML
    html = gr.HTML(value="<h3>自定义 HTML 内容</h3>")

    # Markdown
    markdown = gr.Markdown(value="**粗体** 和 *斜体*")

demo.launch()
```

### 3.7 分享与部署

#### 本地运行与分享

```python
demo.launch(
    server_name="0.0.0.0",   # 监听地址，0.0.0.0 允许外部访问
    server_port=7860,         # 端口号，默认 7860
    share=True,               # 生成公网分享链接（72小时有效）
    auth=("admin", "password"),  # 用户名密码认证
    # auth=lambda u, p: u == "admin",  # 自定义认证函数
    inbrowser=True,           # 自动打开浏览器
)
```

#### 挂载到 FastAPI

```python
from fastapi import FastAPI
import gradio as gr

# 创建 Gradio 应用
def predict(text):
    return f"回复：{text}"

demo = gr.Interface(fn=predict, inputs="text", outputs="text")

# 创建 FastAPI 应用
app = FastAPI(title="LLM Service")

# 将 Gradio 挂载到 FastAPI
app = gr.mount_gradio_app(
    app,                           # FastAPI 实例
    demo,                          # Gradio 应用
    path="/ui",                    # 挂载路径，访问 /ui 打开界面
)

# 现在：
# - /docs → FastAPI Swagger 文档
# - /ui   → Gradio 交互界面
# - /api/... → FastAPI API 端点
```

#### 部署到 HuggingFace Spaces

```bash
# 1. 创建 Space 仓库
# 在 https://huggingface.co/new-space 创建，选择 Gradio SDK

# 2. 项目结构
# app.py         ← Gradio 应用入口（必须是 app.py）
# requirements.txt
# README.md      ← 包含 metadata 配置

# 3. requirements.txt
gradio>=4.0.0
torch
transformers

# 4. 推送代码
git remote add space https://huggingface.co/spaces/your-name/your-space
git push space main
```

---

## 4. 在 LLM 开发中的典型使用场景和代码示例

### 场景一：LLM 对话 Demo

```python
import gradio as gr
import time

# 模拟 LLM 推理
def mock_llm_stream(message: str, history: list, model: str, temperature: float):
    """模拟 LLM 流式推理"""
    # 构建完整 prompt
    messages = []
    for user_msg, assistant_msg in history:
        messages.append(f"User: {user_msg}")
        messages.append(f"Assistant: {assistant_msg}")
    messages.append(f"User: {message}")

    # 模拟流式生成
    full_response = f"基于模型 {model} (temp={temperature})，对你问题 '{message}' 的回复：这是一个模拟的 LLM 生成过程，实际使用时替换为真实模型调用。"
    partial = ""

    for char in full_response:
        partial += char
        time.sleep(0.03)
        yield partial

demo = gr.ChatInterface(
    fn=mock_llm_stream,
    title="LLM 对话 Demo",
    description="基于大语言模型的对话演示，支持流式输出",
    additional_inputs=[
        gr.Dropdown(
            choices=["gpt-3.5-turbo", "gpt-4", "claude-3-sonnet", "claude-3-opus"],
            value="gpt-3.5-turbo",
            label="模型选择",
        ),
        gr.Slider(
            minimum=0.0, maximum=2.0, value=0.7, step=0.1,
            label="Temperature",
            info="值越高，输出越随机",
        ),
        gr.Slider(
            minimum=100, maximum=4096, value=2048, step=100,
            label="Max Tokens",
        ),
    ],
    chatbot=gr.Chatbot(
        height=600,
        show_copy_button=True,
        bubble_full_width=False,
    ),
)

demo.launch()
```

### 场景二：模型对比评测工具

```python
import gradio as gr
import time

def generate_model_a(prompt: str, temperature: float):
    """模型A的推理"""
    response = f"[模型A] 对 '{prompt}' 的回复：这是模型A生成的详细内容..."
    for char in response:
        time.sleep(0.02)
        yield char

def generate_model_b(prompt: str, temperature: float):
    """模型B的推理"""
    response = f"[模型B] 对 '{prompt}' 的回复：这是模型B生成的不同风格的内容..."
    for char in response:
        time.sleep(0.02)
        yield char

with gr.Blocks(title="模型对比评测") as demo:
    gr.Markdown("# LLM 模型对比评测工具\n输入提示词，同时对比两个模型的输出")

    with gr.Row():
        prompt = gr.Textbox(label="提示词", lines=5, placeholder="输入测试提示词...", scale=4)
        temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature", scale=1)
        submit_btn = gr.Button("生成", variant="primary", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 模型 A (GPT-3.5)")
            output_a = gr.Textbox(label="输出A", lines=15, show_copy_button=True)
        with gr.Column():
            gr.Markdown("### 模型 B (GPT-4)")
            output_b = gr.Textbox(label="输出B", lines=15, show_copy_button=True)

    with gr.Row():
        vote_a = gr.Button("A 更好", variant="secondary")
        vote_b = gr.Button("B 更好", variant="secondary")
        vote_tie = gr.Button("差不多")
        vote_result = gr.Textbox(label="投票结果", interactive=False)

    votes = gr.State({"A": 0, "B": 0, "tie": 0})

    submit_btn.click(fn=generate_model_a, inputs=[prompt, temperature], outputs=output_a)
    submit_btn.click(fn=generate_model_b, inputs=[prompt, temperature], outputs=output_b)

    def record_vote(choice: str, vote_counts: dict):
        vote_counts[choice] += 1
        return vote_counts, f"A: {vote_counts['A']} | B: {vote_counts['B']} | 平局: {vote_counts['tie']}"

    vote_a.click(fn=lambda v: record_vote("A", v), inputs=[votes], outputs=[votes, vote_result])
    vote_b.click(fn=lambda v: record_vote("B", v), inputs=[votes], outputs=[votes, vote_result])
    vote_tie.click(fn=lambda v: record_vote("tie", v), inputs=[votes], outputs=[votes, vote_result])

demo.launch()
```

### 场景三：RAG 交互式问答

```python
import gradio as gr
import time

def rag_query(question: str, top_k: int, temperature: float):
    """RAG 流式问答"""
    # 模拟检索过程
    retrieved = [
        f"文档片段{i}: 与 '{question}' 相关的内容..." for i in range(top_k)
    ]

    # 模拟生成过程
    response = f"基于 {top_k} 个检索文档的回答：\n\n"
    for doc in retrieved:
        response += f"- {doc}\n"
    response += f"\n综合回答：关于 '{question}'，以上文档提供了以下信息..."

    partial = ""
    for char in response:
        partial += char
        time.sleep(0.02)
        yield partial, "\n".join(retrieved)  # 同时输出回答和检索来源

with gr.Blocks(title="RAG 问答系统") as demo:
    gr.Markdown("# RAG 交互式问答系统\n输入问题，系统检索相关文档并生成回答")

    with gr.Row():
        with gr.Column(scale=2):
            question = gr.Textbox(label="问题", lines=3, placeholder="输入你的问题...")
            with gr.Row():
                top_k = gr.Slider(1, 10, value=3, step=1, label="检索文档数")
                temperature = gr.Slider(0.0, 2.0, value=0.5, step=0.1, label="Temperature")
            submit_btn = gr.Button("查询", variant="primary")

        with gr.Column(scale=1):
            sources = gr.Textbox(label="检索来源", lines=10, interactive=False)

    answer = gr.Textbox(label="回答", lines=10, show_copy_button=True)

    submit_btn.click(
        fn=rag_query,
        inputs=[question, top_k, temperature],
        outputs=[answer, sources],
    )

demo.launch()
```

### 场景四：LLM 参数调优工具

```python
import gradio as gr
import time
import random

def generate_with_params(prompt: str, temperature: float, top_p: float, top_k: int,
                          max_tokens: int, repetition_penalty: float, num_samples: int):
    """带参数的 LLM 生成，展示参数对输出的影响"""
    results = []
    for i in range(num_samples):
        random.seed(42 + i)  # 可复现
        # 模拟不同参数下的生成结果
        if temperature < 0.3:
            text = f"[低温度({temperature})] 确定性回复：{prompt}的答案是明确的。"
        elif temperature > 1.5:
            text = f"[高温度({temperature})] 创意性回复：{prompt}引发了无限遐想，如同星辰大海般浩瀚。"
        else:
            text = f"[中温度({temperature})] 平衡回复：关于{prompt}，可以从多个角度来分析。"

        results.append(f"--- Sample {i+1} ---\n{text}")

    return "\n\n".join(results)

with gr.Blocks(title="LLM 参数调优") as demo:
    gr.Markdown("# LLM 参数调优工具\n调整参数观察对生成结果的影响")

    with gr.Row():
        with gr.Column(scale=2):
            prompt = gr.Textbox(label="提示词", lines=3, value="解释人工智能")
            output = gr.Textbox(label="生成结果", lines=20, show_copy_button=True)

        with gr.Column(scale=1):
            temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.1, label="Temperature",
                                     info="控制随机性")
            top_p = gr.Slider(0.0, 1.0, value=0.9, step=0.05, label="Top-p",
                               info="核采样概率阈值")
            top_k = gr.Slider(1, 100, value=50, step=1, label="Top-k",
                               info="候选token数量")
            max_tokens = gr.Slider(64, 4096, value=512, step=64, label="Max Tokens")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.0, step=0.1,
                                            label="Repetition Penalty")
            num_samples = gr.Slider(1, 5, value=3, step=1, label="生成样本数")
            generate_btn = gr.Button("生成", variant="primary")

    generate_btn.click(
        fn=generate_with_params,
        inputs=[prompt, temperature, top_p, top_k, max_tokens, repetition_penalty, num_samples],
        outputs=output,
    )

demo.launch()
```

---

## 5. 数学原理

### 5.1 Temperature 采样

LLM 的 Temperature 参数控制输出概率分布的"锐度"：

原始 logits $z_i$ 经过 Temperature 缩放后计算 softmax：

$$P(x_i) = \frac{e^{z_i / T}}{\sum_{j} e^{z_j / T}}$$

- $T \to 0$：分布趋向 one-hot（确定性输出，选概率最高的 token）
- $T = 1$：原始分布
- $T \to \infty$：分布趋向均匀（完全随机）

### 5.2 Top-p（核采样）

Top-p 采样选择累积概率达到 $p$ 的最小 token 集合：

$$\text{选择最小集合 } S \text{ 使得 } \sum_{x_i \in S} P(x_i) \geq p$$

然后在 $S$ 内按概率重新归一化采样。$p=0.9$ 意味着只考虑概率前 90% 的 token。

### 5.3 Top-k 采样

Top-k 简单地限制候选 token 数量为前 $k$ 个：

$$S = \text{argsort}(P(x_i))[:k]$$

Top-k 与 Top-p 的区别：Top-k 固定候选数量，Top-p 固定概率覆盖，Top-p 更自适应。

---

## 6. 代码原理 / 架构原理

### 6.1 整体架构

```
┌──────────────────────────────────────┐
│            浏览器 (React 前端)          │
│  ┌──────────┐  ┌──────────────────┐  │
│  │ Gradio UI │  │  WebSocket Client │  │
│  └─────┬────┘  └────────┬─────────┘  │
└────────┼─────────────────┼───────────┘
         │ HTTP/WS         │
└────────┼─────────────────┼───────────┘
│        ↓                 ↓           │
│  ┌──────────┐  ┌──────────────────┐  │
│  │  FastAPI  │  │  WebSocket Server│  │
│  │  (REST)   │  │  (实时通信)        │  │
│  └─────┬────┘  └────────┬─────────┘  │
│        │                │            │
│  ┌─────┴────────────────┴─────────┐  │
│  │       Gradio Python 后端         │  │
│  │  ┌─────────┐ ┌───────────────┐  │  │
│  │  │ 事件系统  │ │ 组件管理器      │  │  │
│  │  └─────────┘ └───────────────┘  │  │
│  │  ┌─────────┐ ┌───────────────┐  │  │
│  │  │ 流式队列  │ │ 状态管理       │  │  │
│  │  └─────────┘ └───────────────┘  │  │
│  └────────────────────────────────┘  │
│             Python 进程              │
└──────────────────────────────────────┘
```

### 6.2 前后端通信机制

Gradio 使用两种通信方式：

**1. HTTP REST API**（普通请求）：
```
前端 → POST /api/predict → 后端处理函数 → 返回结果
```

**2. WebSocket**（流式输出和实时更新）：
```
前端 ←→ WebSocket /queue/join ←→ 后端事件队列
         ↑
   流式输出时，后端持续推送
   yield 的部分结果到前端
```

### 6.3 流式输出的实现原理

```python
# Gradio 流式输出的简化实现流程：

# 1. 前端发起请求，建立 WebSocket 连接
# 2. 后端将请求放入事件队列
# 3. 后端调用处理函数（generator）
# 4. 每次 yield 时：
#    → 将部分结果放入 WebSocket 消息队列
#    → 前端接收消息，追加显示内容
# 5. 生成器结束时，标记完成

# 关键代码路径：
# gr.ChatInterface / Blocks 事件绑定 (stream=True)
#     ↓
# 处理函数（generator，yield 部分结果）
#     ↓
# EventStream → WebSocket → 前端 React 组件更新
```

### 6.4 组件渲染原理

```
Python 组件定义                React 前端组件
─────────────                ──────────────
gr.Textbox          →        <TextArea>
gr.Chatbot          →        <ChatBot>
gr.Slider           →        <Slider>
gr.Image            →        <ImageUpload>
gr.Dropdown         →        <Dropdown>
gr.Button           →        <Button>

映射关系：
1. Python 定义组件 → 生成 JSON Schema
2. 前端根据 Schema 渲染 React 组件
3. 用户交互 → 发送事件到后端
4. 后端处理 → 返回更新值 → 前端重新渲染
```

### 6.5 mount_gradio_app 原理

```python
# mount_gradio_app 的简化实现
def mount_gradio_app(app, blocks, path):
    """将 Gradio Blocks 挂载到 FastAPI/Starlette 应用"""
    # 1. Gradio 底层也是基于 Starlette 的 ASGI 应用
    # 2. 使用 Starlette 的 Mount 中间件将 Gradio 路由
    #    挂载到指定 path 下
    # 3. 路由结构：
    #    /path/          → Gradio UI 页面
    #    /path/api/      → Gradio REST API
    #    /path/queue/    → Gradio WebSocket 队列
    #    /path/assets/   → 静态资源
    from starlette.routing import Mount
    app.routes.append(Mount(path, app=blocks.app))
    return app
```

---

## 7. 常见注意事项和最佳实践

### 7.1 流式输出最佳实践

```python
# 1. 流式函数必须使用 yield，不能 return
# 正确：
def stream_fn(message, history):
    for token in generate_tokens(message):
        yield token  # 逐步输出

# 错误：return 会直接结束，不会流式输出
def wrong_fn(message, history):
    return full_response

# 2. ChatInterface 中，yield 返回的是当前累积的完整文本
# Gradio 会自动替换输出内容（而非追加）
def chat_stream(message, history):
    partial = ""
    for token in generate_tokens(message):
        partial += token
        yield partial  # 每次返回完整累积文本

# 3. Blocks 中的 Chatbot 流式需要更新历史
def blocks_stream(message, chat_history):
    chat_history.append([message, ""])
    for token in generate_tokens(message):
        chat_history[-1][1] += token
        yield chat_history  # yield 更新后的完整历史
```

### 7.2 性能优化

```python
# 1. 使用 queue 限制并发
demo = gr.Interface(fn=slow_inference, inputs="text", outputs="text")
demo.queue(
    max_size=20,          # 最大排队请求数
    default_concurrency_limit=2,  # 并发处理数
)

# 2. 避免在处理函数中加载模型（应全局加载）
# 错误：每次请求都加载模型
def predict(text):
    model = load_model()  # 非常慢！
    return model.predict(text)

# 正确：全局加载一次
model = load_model()  # 应用启动时加载

def predict(text):
    return model.predict(text)

# 3. 大输出使用流式，减少首字节时间
# 4. 使用 cache_examples 缓存示例结果
demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    examples=[["示例1"], ["示例2"]],
    cache_examples=True,  # 启动时预计算示例结果并缓存
)
```

### 7.3 安全注意事项

```python
# 1. 生产环境禁用 share
demo.launch(share=False)  # share=True 仅用于临时分享

# 2. 添加认证
demo.launch(auth=("username", "password"))

# 3. 自定义认证
def auth_fn(username, password):
    # 可对接数据库验证
    return username in ALLOWED_USERS and verify_password(username, password)

demo.launch(auth=auth_fn)

# 4. 防止 SSRF — 不要直接将用户输入作为 URL 请求
# 5. 限制文件上传大小和类型
gr.File(file_types=[".txt", ".pdf"], label="文档上传")  # 限制文件类型
```

### 7.4 部署最佳实践

```python
# 1. 生产环境配置
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,                 # 禁用公网分享
    max_threads=4,               # 最大线程数
)

# 2. 使用 queue 管理并发
demo.queue(
    max_size=50,                 # 排队上限
    default_concurrency_limit=4, # 并发限制
)

# 3. Docker 部署
```

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7860

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: "3"
services:
  gradio-app:
    build: .
    ports:
      - "7860:7860"
    environment:
      - GRADIO_SERVER_NAME=0.0.0.0
      - GRADIO_SERVER_PORT=7860
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]  # GPU 支持
    restart: unless-stopped
```

### 7.5 调试技巧

```python
# 1. 启用调试模式
demo.launch(debug=True)  # 显示详细错误信息

# 2. 查看事件日志
import logging
logging.basicConfig(level=logging.INFO)

# 3. 测试组件事件绑定
# 使用 Gradio 的 Python 客户端测试
from gradio_client import Client

client = Client("http://localhost:7860/")
result = client.predict("测试输入", api_name="/predict")
print(result)

# 4. 处理超时
demo.queue().launch(
    max_threads=4,
    # 长时间推理需要增大超时
)
```

### 7.6 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 流式输出不生效 | 函数未使用 yield | 改为生成器函数，使用 yield |
| ChatInterface 不更新历史 | 返回格式不对 | 返回字符串，Gradio 自动管理历史 |
| 界面加载慢 | 模型在请求时加载 | 全局预加载模型 |
| share 链接无法访问 | 72小时后过期 | 使用正式部署方式 |
| GPU 内存不足 | 多用户并发推理 | 使用 queue 限制并发数 |
| 上传文件处理失败 | 文件路径或类型问题 | 检查 type 参数（"filepath"/"bytes"/"pil"） |
| 事件不触发 | 组件未正确绑定 | 检查 inputs/outputs 参数和组件变量 |

### 7.7 与其他框架的集成

```python
# 1. Gradio + FastAPI
from fastapi import FastAPI
import gradio as gr

app = FastAPI()

@app.get("/api/models")
async def list_models():
    return {"models": ["gpt-3.5", "gpt-4"]}

demo = gr.Interface(fn=predict, inputs="text", outputs="text")
app = gr.mount_gradio_app(app, demo, path="/ui")

# 2. Gradio + LangChain
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-3.5-turbo", streaming=True)

def langchain_chat(message, history):
    # 将 Gradio 历史转为 LangChain 消息格式
    messages = [HumanMessage(content=msg) for msg, _ in history]
    messages.append(HumanMessage(content=message))

    partial = ""
    for chunk in llm.stream(messages):
        partial += chunk.content
        yield partial

demo = gr.ChatInterface(fn=langchain_chat)
demo.launch()
```

---

## 参考链接

- Gradio 官方文档：https://www.gradio.app/docs/
- Gradio GitHub：https://github.com/gradio-app/gradio
- Gradio Playground：https://www.gradio.app/playground
- HuggingFace Spaces：https://huggingface.co/spaces
- Gradio Client 文档：https://www.gradio.app/docs/python-client/client
