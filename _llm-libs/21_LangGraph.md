---
title: "LangGraph Agent编排框架"
excerpt: "StateGraph、节点/边/条件路由、持久化、Human-in-the-loop、多Agent"
collection: llm-libs
permalink: /llm-libs/21-langgraph
category: agent
---


## 1. 库的简介和在LLM开发中的作用

LangGraph 是由 LangChain 团队开发的一个用于构建**有状态、多参与者** LLM 应用的框架。它以**有限状态机（FSM）**为核心思想，将 Agent 的工作流建模为**有向图**，其中节点代表计算步骤（如调用 LLM、执行工具），边代表状态转移和控制流。

在 LLM 开发中，LangGraph 解决了以下核心问题：

- **复杂工作流编排**：将多步骤的 LLM 调用、工具使用、条件分支组织为清晰的图结构
- **状态管理**：在多轮对话和复杂流程中自动维护和传递状态
- **人机交互**：支持在关键节点暂停执行，等待人类审批或输入
- **持久化**：支持将对话和执行状态保存到内存、SQLite、Postgres 等存储后端
- **多 Agent 协作**：支持 Supervisor、Swarm 等多种多 Agent 编排模式

与 LangChain 的链式调用不同，LangGraph 提供了更灵活的图结构，允许循环、条件分支和状态回溯，非常适合构建复杂的 Agent 系统。

## 2. 安装方式

```bash
# 基础安装
pip install langgraph

# 安装带 LangChain 集成的版本
pip install langgraph-langchain

# 持久化支持
pip install langgraph-checkpoint-sqlite    # SQLite 持久化
pip install langgraph-checkpoint-postgres  # PostgreSQL 持久化

# 安装全部依赖
pip install langgraph[all]
```

## 3. 核心类/函数/工具的详细说明

### 3.1 StateGraph — 图的创建与构建

`StateGraph` 是 LangGraph 的核心类，用于定义状态图的结构。

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

# 定义状态结构
class AgentState(TypedDict):
    messages: Annotated[list, add]  # 使用 add 操作符实现消息追加
    next_agent: str                  # 下一个要执行的 Agent
    current_step: int                # 当前步骤计数

# 创建状态图
# 参数: 状态类型
graph = StateGraph(AgentState)
```

**参数说明**：
- `state_type`：状态的类型注解，通常为 `TypedDict` 子类
- 状态字段可使用 `Annotated[类型, reducer]` 指定更新策略，`reducer` 是一个函数，定义如何将新值合并到现有状态中

### 3.2 状态管理 — TypedDict 定义状态与更新机制

```python
from typing import TypedDict, Annotated
from operator import add

# 基本状态定义
class SimpleState(TypedDict):
    query: str
    response: str

# 带 reducer 的状态定义
class RichState(TypedDict):
    # Annotated[list, add] 表示新值会追加到列表，而非替换
    messages: Annotated[list, add]
    # 无 reducer，新值直接覆盖旧值
    current_tool: str
    # 使用自定义 reducer
    scores: Annotated[list, add]

# 自定义 reducer 函数
def merge_dicts(existing: dict, new: dict) -> dict:
    """合并字典，新值覆盖已有键"""
    if existing is None:
        return new
    return {**existing, **new}

class MergedState(TypedDict):
    config: Annotated[dict, merge_dicts]
```

**状态更新机制**：
- 节点函数返回一个字典，字典的键对应状态字段
- 无 `reducer` 的字段：返回值直接覆盖当前状态
- 有 `reducer` 的字段：调用 `reducer(当前值, 返回值)` 进行合并
- `operator.add` 是最常用的 reducer，实现列表追加

### 3.3 节点(Node) — 函数定义与状态读写

节点是图中的计算单元，每个节点是一个接收状态、返回状态更新的函数。

```python
def analyze_query(state: AgentState) -> dict:
    """分析用户查询的节点"""
    # 读取状态
    messages = state["messages"]
    last_message = messages[-1] if messages else ""

    # 执行计算（如调用 LLM）
    analysis = f"分析结果: {last_message}"

    # 返回状态更新（只返回需要更新的字段）
    return {
        "messages": [f"分析完成: {analysis}"],
        "current_step": state.get("current_step", 0) + 1
    }

def generate_response(state: AgentState) -> dict:
    """生成回复的节点"""
    return {
        "messages": ["这是生成的回复"],
        "next_agent": "end"
    }

# 添加节点到图
# 参数: 节点名称, 节点函数
graph.add_node("analyze", analyze_query)
graph.add_node("generate", generate_response)
```

**关键要点**：
- 节点函数接收完整的 `state` 字典作为参数
- 节点函数返回一个**部分状态字典**，只包含需要更新的字段
- 节点名称在图中必须唯一
- 节点函数可以是同步或异步函数

### 3.4 边(Edge) — 普通边与条件路由

#### 普通边（无条件转移）

```python
# 从 START 到指定节点
graph.add_edge(START, "analyze")

# 从一个节点到另一个节点
graph.add_edge("analyze", "generate")

# 从一个节点到 END（结束）
graph.add_edge("generate", END)
```

#### 条件路由（Conditional Edges）

条件边根据当前状态动态决定下一个节点。

```python
def route_by_intent(state: AgentState) -> str:
    """根据意图路由到不同节点"""
    next_agent = state.get("next_agent", "")
    if next_agent == "search":
        return "search_node"
    elif next_agent == "calculate":
        return "calculate_node"
    else:
        return "generate"

# 添加条件边
# 参数: 源节点, 路由函数, 目标节点映射
graph.add_conditional_edges(
    "analyze",              # 源节点
    route_by_intent,        # 路由函数，返回目标节点名称
    {                       # 可选：路由值到节点名的映射
        "search_node": "search",
        "calculate_node": "calculate",
        "generate": "generate"
    }
)

# 更简洁的条件边（路由函数直接返回节点名）
graph.add_conditional_edges(
    "analyze",
    route_by_intent
    # 不提供映射时，路由函数返回值即为节点名
)
```

**条件路由的执行流程**：
1. 源节点执行完毕
2. 调用路由函数，传入当前状态
3. 路由函数返回目标节点名称（或映射键）
4. 根据返回值确定下一个执行的节点

### 3.5 持久化 — MemorySaver、SqliteSaver、PostgresSaver

持久化允许保存和恢复图的执行状态，是实现多轮对话和故障恢复的关键。

#### MemorySaver（内存持久化）

```python
from langgraph.checkpoint.memory import MemorySaver

# 创建内存检查点
memory = MemorySaver()

# 编译图时传入检查点
app = graph.compile(checkpointer=memory)

# 使用 thread_id 区分不同会话
config = {"configurable": {"thread_id": "session-1"}}

# 第一次调用
result1 = app.invoke({"messages": ["你好"], "next_agent": "", "current_step": 0}, config)

# 第二次调用（会延续之前的对话状态）
result2 = app.invoke({"messages": ["请继续"]}, config)

# 获取状态快照
snapshot = app.get_state(config)
print(snapshot.values)  # 当前状态值
print(snapshot.next)    # 下一个要执行的节点
```

#### SqliteSaver（SQLite 持久化）

```python
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3

# 创建 SQLite 连接
conn = sqlite3.connect("chat_history.db", check_same_thread=False)

# 创建检查点
checkpointer = SqliteSaver(conn)

# 编译
app = graph.compile(checkpointer=checkpointer)
```

#### PostgresSaver（PostgreSQL 持久化）

```python
from langgraph.checkpoint.postgres import PostgresSaver

# 使用连接字符串
checkpointer = PostgresSaver.from_conn_string(
    "postgresql://user:password@localhost:5432/mydb"
)

# 首次使用需要创建表
# checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)
```

### 3.6 人机交互(Human-in-the-loop) — interrupt 与 Command

人机交互允许在图执行过程中暂停，等待人类输入后继续。

#### 使用 interrupt 暂停执行

```python
from langgraph.types import interrupt, Command

def human_review_node(state: AgentState) -> dict:
    """需要人工审核的节点"""
    # interrupt() 会暂停图执行，返回值传递给调用者
    human_feedback = interrupt("请审核以下内容是否正确：\n" + str(state["messages"]))

    # 人类反馈通过 Command 传入后，此处获得反馈内容
    return {
        "messages": [f"人工审核结果: {human_feedback}"],
    }

# 添加节点和边
graph.add_node("review", human_review_node)
graph.add_edge("analyze", "review")
graph.add_conditional_edges("review", lambda s: s.get("next_agent", "generate"))

# 编译时启用 interrupt
app = graph.compile(checkpointer=memory, interrupt_before=["review"])

# 执行到 review 节点前会暂停
config = {"configurable": {"thread_id": "review-session"}}
result = app.invoke({"messages": ["需要审核的内容"], "next_agent": "", "current_step": 0}, config)

# 查看当前状态
state = app.get_state(config)
print(state.next)  # 显示下一个要执行的节点

# 提供人工反馈并继续执行
app.invoke(
    Command(resume="审核通过，内容正确"),
    config
)
```

#### interrupt_before 和 interrupt_after

```python
# 在指定节点执行前暂停
app = graph.compile(checkpointer=memory, interrupt_before=["review"])

# 在指定节点执行后暂停
app = graph.compile(checkpointer=memory, interrupt_after=["review"])

# 也可以同时指定
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["review"],
    interrupt_after=["generate"]
)
```

### 3.7 子图 — 嵌套图结构

子图允许将复杂的工作流拆分为可复用的模块。

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict

# 定义子图状态
class SubGraphState(TypedDict):
    query: str
    result: str
    sub_step_count: int

# 创建子图
def process_step_1(state: SubGraphState) -> dict:
    return {"sub_step_count": state.get("sub_step_count", 0) + 1}

def process_step_2(state: SubGraphState) -> dict:
    return {"result": f"处理完成: {state['query']}"}

sub_graph = StateGraph(SubGraphState)
sub_graph.add_node("step1", process_step_1)
sub_graph.add_node("step2", process_step_2)
sub_graph.add_edge(START, "step1")
sub_graph.add_edge("step1", "step2")
sub_graph.add_edge("step2", END)

compiled_sub = sub_graph.compile()

# 定义主图状态
class MainState(TypedDict):
    messages: Annotated[list, add]
    task: str
    final_result: str

def call_subgraph(state: MainState) -> dict:
    """在主图节点中调用子图"""
    result = compiled_sub.invoke({
        "query": state["task"],
        "sub_step_count": 0
    })
    return {"final_result": result["result"]}

# 主图
main_graph = StateGraph(MainState)
main_graph.add_node("subprocess", call_subgraph)
main_graph.add_edge(START, "subprocess")
main_graph.add_edge("subprocess", END)

app = main_graph.compile()
```

**状态转换**：子图与主图之间的状态需要手动映射。在调用子图的节点函数中，将主图状态转换为子图状态传入，再将子图输出映射回主图状态。

### 3.8 编译与运行 — compile()、invoke()、stream()

#### compile()

```python
# 基本编译
app = graph.compile()

# 带持久化编译
app = graph.compile(checkpointer=memory)

# 带中断编译
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["review"],
    interrupt_after=[]
)
```

#### invoke()

```python
# 同步调用，返回最终状态
result = app.invoke(
    {"messages": ["你好"], "next_agent": "", "current_step": 0},
    config={"configurable": {"thread_id": "test"}}
)
print(result)  # 最终状态字典
```

#### stream()

```python
# 流式输出，逐步返回每个节点的执行结果
for event in app.stream(
    {"messages": ["你好"], "next_agent": "", "current_step": 0},
    config={"configurable": {"thread_id": "stream-test"}}
):
    # event 是一个字典 {节点名: 节点输出}
    for node_name, node_output in event.items():
        print(f"节点 {node_name} 输出: {node_output}")

# 流式输出的不同模式
for event in app.stream(
    input_data,
    config=config,
    stream_mode="values"   # 输出完整状态值
):
    print(event)

# stream_mode 选项:
# "values" — 每步输出完整状态
# "updates" — 每步输出状态增量（默认）
# "messages" — 输出 LLM 消息流
```

### 3.9 多Agent — Supervisor模式与Swarm模式

#### Supervisor 模式

Supervisor 模式中，一个中心 Agent 负责协调其他 Agent 的调用。

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class SupervisorState(TypedDict):
    messages: Annotated[list, add]
    next_agent: str

def supervisor(state: SupervisorState) -> dict:
    """监督者节点：决定下一步调用哪个 Agent"""
    messages = state["messages"]
    last_msg = messages[-1] if messages else ""

    # 简单路由逻辑（实际中通常由 LLM 决定）
    if "搜索" in last_msg:
        next_agent = "researcher"
    elif "写代码" in last_msg:
        next_agent = "coder"
    elif "计算" in last_msg:
        next_agent = "calculator"
    else:
        next_agent = "FINISH"

    return {"next_agent": next_agent}

def researcher(state: SupervisorState) -> dict:
    """研究 Agent"""
    return {"messages": ["研究结果: 找到了相关信息..."]}

def coder(state: SupervisorState) -> dict:
    """编码 Agent"""
    return {"messages": ["代码已生成: print('hello')"]}

def calculator(state: SupervisorState) -> dict:
    """计算 Agent"""
    return {"messages": ["计算结果: 42"]}

# 路由函数
def route_agent(state: SupervisorState) -> str:
    next_agent = state.get("next_agent", "FINISH")
    if next_agent == "FINISH":
        return END
    return next_agent

# 构建图
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)
graph.add_node("calculator", calculator)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_agent)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")
graph.add_edge("calculator", "supervisor")

app = graph.compile()

# 执行
result = app.invoke({"messages": ["我需要写代码"], "next_agent": ""})
```

#### Swarm 模式

Swarm 模式中，Agent 之间可以直接相互移交控制权，无需中心协调者。

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class SwarmState(TypedDict):
    messages: Annotated[list, add]
    current_agent: str

def agent_a(state: SwarmState) -> dict:
    """Agent A 可以将控制权交给 B"""
    messages = state["messages"]
    last_msg = messages[-1] if messages else ""

    if "需要 B 处理" in last_msg:
        return {
            "messages": ["Agent A: 移交给 Agent B"],
            "current_agent": "agent_b"
        }
    return {
        "messages": ["Agent A: 我已处理完毕"],
        "current_agent": "FINISH"
    }

def agent_b(state: SwarmState) -> dict:
    """Agent B 可以将控制权交回 A"""
    return {
        "messages": ["Agent B: 处理完成，交回 Agent A"],
        "current_agent": "agent_a"
    }

def swarm_router(state: SwarmState) -> str:
    current = state.get("current_agent", "")
    if current == "FINISH":
        return END
    return current

graph = StateGraph(SwarmState)
graph.add_node("agent_a", agent_a)
graph.add_node("agent_b", agent_b)

graph.add_edge(START, "agent_a")
graph.add_conditional_edges("agent_a", swarm_router)
graph.add_conditional_edges("agent_b", swarm_router)

app = graph.compile()
result = app.invoke({"messages": ["需要 B 处理这个问题"], "current_agent": ""})
```

## 4. 在LLM开发中的典型使用场景和代码示例

### 场景一：构建带工具调用的 ReAct Agent

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from operator import add

# 定义状态
class ReActState(TypedDict):
    messages: Annotated[list, add]
    tool_calls: Annotated[list, add]
    final_answer: str

# 模拟 LLM 调用
def call_llm(state: ReActState) -> dict:
    messages = state["messages"]
    last_msg = messages[-1] if messages else ""

    # 简化：根据关键词决定是否调用工具
    if "天气" in last_msg:
        return {"messages": ["需要调用天气工具"], "tool_calls": ["get_weather"]}
    elif "计算" in last_msg:
        return {"messages": ["需要调用计算工具"], "tool_calls": ["calculator"]}
    else:
        return {"messages": ["我可以直接回答这个问题"], "final_answer": "这是最终答案"}

# 模拟工具执行
def execute_tools(state: ReActState) -> dict:
    tool_calls = state.get("tool_calls", [])
    results = []
    for tool in tool_calls:
        if tool == "get_weather":
            results.append("北京今天晴天，25度")
        elif tool == "calculator":
            results.append("计算结果: 42")
    return {
        "messages": [f"工具结果: {r}" for r in results],
        "tool_calls": []  # 清空工具调用列表
    }

# 路由：决定是否需要执行工具
def should_use_tools(state: ReActState) -> str:
    if state.get("tool_calls"):
        return "tools"
    if state.get("final_answer"):
        return END
    return "llm"

# 构建图
graph = StateGraph(ReActState)
graph.add_node("llm", call_llm)
graph.add_node("tools", execute_tools)

graph.add_edge(START, "llm")
graph.add_conditional_edges("llm", should_use_tools, {"tools": "tools", END: END})
graph.add_edge("tools", "llm")  # 工具执行后回到 LLM

app = graph.compile(checkpointer=MemorySaver())

# 执行
result = app.invoke(
    {"messages": ["北京今天天气怎么样？"], "tool_calls": [], "final_answer": ""},
    config={"configurable": {"thread_id": "react-1"}}
)
print(result)
```

### 场景二：多轮对话系统

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated
from operator import add

class ChatState(TypedDict):
    messages: Annotated[list, add]
    user_intent: str

def classify_intent(state: ChatState) -> dict:
    last_msg = state["messages"][-1] if state["messages"] else ""
    if "帮助" in last_msg or "求助" in last_msg:
        return {"user_intent": "help"}
    elif "订单" in last_msg:
        return {"user_intent": "order"}
    else:
        return {"user_intent": "chat"}

def handle_help(state: ChatState) -> dict:
    return {"messages": ["我是客服助手，请问有什么可以帮助您的？"]}

def handle_order(state: ChatState) -> dict:
    return {"messages": ["请提供您的订单号，我来帮您查询。"]}

def handle_chat(state: ChatState) -> dict:
    return {"messages": ["好的，我明白了。还有其他问题吗？"]}

def route_intent(state: ChatState) -> str:
    intent = state.get("user_intent", "chat")
    return {
        "help": "help_node",
        "order": "order_node",
        "chat": "chat_node"
    }.get(intent, "chat_node")

graph = StateGraph(ChatState)
graph.add_node("classify", classify_intent)
graph.add_node("help_node", handle_help)
graph.add_node("order_node", handle_order)
graph.add_node("chat_node", handle_chat)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route_intent)
graph.add_edge("help_node", END)
graph.add_edge("order_node", END)
graph.add_edge("chat_node", END)

app = graph.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "chat-session"}}

# 多轮对话
result1 = app.invoke({"messages": ["我需要帮助"], "user_intent": ""}, config)
result2 = app.invoke({"messages": ["查看我的订单"], "user_intent": ""}, config)
```

## 5. 数学原理

### 有限状态机（FSM）与图论基础

LangGraph 的核心数学模型是**有限状态机**，形式化定义为一个五元组：

$$M = (Q, \Sigma, \delta, q_0, F)$$

其中：
- $Q$ = 有限状态集合（对应图中的节点）
- $\Sigma$ = 输入字母表（对应状态转移的触发条件）
- $\delta: Q \times \Sigma \rightarrow Q$ = 状态转移函数（对应图中的边和条件路由）
- $q_0 \in Q$ = 初始状态（对应 START 节点）
- $F \subseteq Q$ = 终止状态集合（对应 END 节点）

**与标准 FSM 的区别**：LangGraph 扩展了经典 FSM，引入了：
1. **状态携带数据**：每个状态不仅是一个标记，还关联一个 TypedDict 数据结构
2. **条件转移**：$\delta$ 可以基于当前状态数据动态计算
3. **循环**：允许状态之间的循环转移（经典 FSM 本身支持，但很多 FSM 实现不支持）
4. **层级结构**：子图支持状态的嵌套，形成层级状态机

### 状态更新的数学表达

设状态为 $S = \{s_1, s_2, ..., s_n\}$，节点函数 $f$ 返回更新 $\Delta S$，则状态更新规则为：

- 无 reducer 字段：$s_i' = \Delta s_i$
- 有 reducer 字段：$s_i' = r_i(s_i, \Delta s_i)$，其中 $r_i$ 是 reducer 函数

例如，`Annotated[list, add]` 的更新为：
$$s_i' = s_i \oplus \Delta s_i$$

其中 $\oplus$ 表示列表拼接操作。

## 6. 代码原理/架构原理

### 架构概览

```
┌─────────────────────────────────────────────┐
│              应用层 (Application)             │
│  invoke() / stream() / get_state()          │
├─────────────────────────────────────────────┤
│           编译层 (CompiledGraph)              │
│  图验证 → 节点调度 → 状态管理 → 持久化        │
├─────────────────────────────────────────────┤
│              定义层 (StateGraph)              │
│  add_node / add_edge / add_conditional_edges │
├─────────────────────────────────────────────┤
│            持久化层 (Checkpointer)            │
│  MemorySaver / SqliteSaver / PostgresSaver   │
└─────────────────────────────────────────────┘
```

### 执行流程

1. **定义阶段**：通过 `StateGraph` API 定义图的节点和边
2. **编译阶段**：`compile()` 验证图的完整性（所有节点可达、无悬空边），生成可执行的 `CompiledGraph`
3. **执行阶段**：
   - 从 START 节点开始
   - 调度器按拓扑顺序执行节点
   - 每个节点执行后更新状态
   - 条件边根据状态选择下一步
   - 遇到 END 节点结束
   - 遇到 interrupt 暂停并保存状态
4. **持久化**：每个节点执行后，checkpointer 保存状态快照

### 关键设计模式

**Pregel 模型**：LangGraph 的执行引擎基于 Pregel 模型（与 Apache Spark 的执行模型类似），每个"超步"（superstep）中：
- 所有可执行的节点并行运行
- 节点之间通过状态通信
- 超步结束后统一更新状态

## 7. 常见注意事项和最佳实践

### 注意事项

1. **状态不可变性**：节点函数不应直接修改输入状态，应返回新的状态更新
2. **图的连通性**：编译时会检查所有节点是否可达，不可达节点会报错
3. **循环检测**：LangGraph 允许循环，但需注意避免无限循环，建议设置最大迭代次数
4. **thread_id 必须唯一**：使用持久化时，不同的对话/会话必须使用不同的 `thread_id`
5. **异步支持**：节点函数可以是 async 的，但 stream/invoke 需要使用对应的异步版本（`ainvoke`、`astream`）

### 最佳实践

```python
# 1. 为复杂状态使用 TypedDict 并明确 reducer
class RobustState(TypedDict):
    messages: Annotated[list, add]    # 消息追加
    context: dict                     # 覆盖更新
    retry_count: int                  # 覆盖更新

# 2. 条件路由中使用枚举避免拼写错误
from enum import Enum

class AgentName(str, Enum):
    RESEARCHER = "researcher"
    CODER = "coder"
    FINISH = "FINISH"

def router(state) -> str:
    # 使用枚举值
    return AgentName.RESEARCHER

# 3. 设置递归限制防止无限循环
app = graph.compile(
    checkpointer=memory,
    interrupt_before=["review"],
)
result = app.invoke(
    input_data,
    config={
        "configurable": {"thread_id": "safe-session"},
        "recursion_limit": 25  # 最大递归/迭代次数
    }
)

# 4. 使用 stream_mode 调试
for event in app.stream(input_data, config=config, stream_mode="updates"):
    print(event)  # 观察每步的状态变化

# 5. 子图状态转换要显式映射
def call_subgraph(state: MainState) -> dict:
    # 显式映射主图状态到子图状态
    sub_input = {
        "query": state["task"],
        "sub_step_count": 0
    }
    result = compiled_sub.invoke(sub_input)
    # 显式映射子图结果到主图状态
    return {"final_result": result["result"]}

# 6. 人机交互时务必使用 checkpointer
# interrupt 需要 checkpointer 来保存和恢复状态
app = graph.compile(checkpointer=memory, interrupt_before=["approval_node"])
```

### 调试技巧

```python
# 获取执行历史
config = {"configurable": {"thread_id": "debug-session"}}
state_history = list(app.get_state_history(config))
for state in state_history:
    print(f"步骤: {state.metadata}, 下一节点: {state.next}")

# 查看图结构（生成 Mermaid 图）
print(app.get_graph().draw_mermaid())
```
