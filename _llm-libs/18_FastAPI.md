---
title: "FastAPI 异步Web框架"
excerpt: "路由/Pydantic集成、SSE流式响应、中间件、LLM推理API服务"
collection: llm-libs
permalink: /llm-libs/18-fastapi
category: inference
toc: true
---


## 1. 简介与在 LLM 开发中的作用

FastAPI 是一个现代、高性能的 Python Web 框架，基于 Python 3.8+ 的类型提示构建。它的核心特点包括：

- **极速性能**：与 NodeJS 和 Go 相当，是 Python 最快的框架之一
- **快速开发**：类型提示驱动的开发体验，减少约 200% 的人为错误
- **自动文档**：自动生成 OpenAPI（Swagger）和 JSON Schema 文档
- **异步原生**：基于 ASGI 标准，原生支持 async/await

### FastAPI 在 LLM 开发中的角色

在 LLM 应用开发中，FastAPI 扮演着**模型推理服务框架**和**LLM API 服务层**的关键角色：

1. **模型推理服务**：将 LLM 模型封装为 RESTful API，供前端或其他服务调用
2. **流式输出服务**：通过 SSE（Server-Sent Events）实现 LLM 生成内容的实时流式推送
3. **RAG 服务**：构建检索增强生成的 API 管道
4. **AI Agent 服务**：为多 Agent 系统提供 API 通信层
5. **模型管理**：提供模型加载、卸载、版本切换的 API 接口

---

## 2. 安装方式

### 基础安装

```bash
pip install fastapi
```

### 安装 ASGI 服务器（必需）

```bash
pip install uvicorn[standard]
```

### 一次性安装（推荐）

```bash
pip install "fastapi[all]"
```

这会安装 FastAPI 及其所有可选依赖，包括 uvicorn、pydantic 等。

### 开发依赖

```bash
pip install fastapi uvicorn python-multipart  # 文件上传支持
pip install httpx                             # 异步测试客户端
pip install gunicorn                          # 生产部署
```

---

## 3. 核心类/函数/工具的详细说明

### 3.1 应用创建：FastAPI 类

```python
from fastapi import FastAPI

app = FastAPI(
    title="LLM 推理服务",           # API 标题，显示在 Swagger 文档中
    description="基于大语言模型的推理服务",  # API 描述
    version="1.0.0",                # API 版本
    docs_url="/docs",               # Swagger UI 路径，默认 /docs
    redoc_url="/redoc",             # ReDoc 文档路径，默认 /redoc
    openapi_url="/openapi.json",    # OpenAPI schema 路径
)
```

**关键参数**：
- `title`：API 文档标题
- `description`：支持 Markdown 格式的详细描述
- `version`：语义化版本号
- `docs_url`/`redoc_url`：设为 `None` 可禁用文档
- `openapi_url`：设为 `None` 可完全禁用 OpenAPI schema

### 3.2 路由定义

#### 基本路由

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")                          # GET 请求
async def root():
    return {"message": "LLM 服务已启动"}

@app.post("/predict")                  # POST 请求
async def predict():
    return {"result": "预测结果"}

# 支持所有 HTTP 方法：@app.get, @app.post, @app.put, @app.delete,
# @app.patch, @app.options, @app.head, @app.trace
```

#### 路径参数（Path Parameters）

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/models/{model_id}")
async def get_model(model_id: str):
    """路径参数直接在 URL 路径中定义，用花括号包裹"""
    return {"model_id": model_id}

# 带类型约束的路径参数
@app.get("/items/{item_id}")
async def get_item(item_id: int):   # 自动类型转换和验证，非整数返回422
    return {"item_id": item_id}

# 使用 Path 添加约束和元数据
from fastapi import Path

@app.get("/models/{model_id}")
async def get_model(
    model_id: int = Path(
        ...,                        # ... 表示必填
        title="模型ID",
        description="唯一标识模型的整数ID",
        ge=1,                       # 最小值 (greater than or equal)
        le=1000,                    # 最大值 (less than or equal)
    )
):
    return {"model_id": model_id}
```

#### 查询参数（Query Parameters）

```python
from fastapi import FastAPI, Query

app = FastAPI()

@app.get("/generate")
async def generate(
    prompt: str = Query(..., description="生成提示词", min_length=1, max_length=2000),
    max_tokens: int = Query(512, description="最大生成token数", ge=1, le=4096),
    temperature: float = Query(0.7, description="采样温度", ge=0.0, le=2.0),
    top_p: float = Query(0.9, description="核采样阈值", ge=0.0, le=1.0),
    model: str = Query("gpt-3.5-turbo", description="模型名称"),
):
    return {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "model": model,
    }

# 可选查询参数（使用 Optional 或默认值 None）
from typing import Optional

@app.get("/search")
async def search(
    query: str,
    limit: Optional[int] = Query(None, description="返回结果数量限制"),
):
    return {"query": query, "limit": limit}
```

### 3.3 请求模型：Pydantic BaseModel 集成

Pydantic 是 FastAPI 的数据验证核心，通过定义 BaseModel 来声明请求体的结构和类型。

```python
from pydantic import BaseModel, Field
from typing import Optional, List
from fastapi import FastAPI

app = FastAPI()

# 定义请求模型
class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="输入提示词", min_length=1)
    max_tokens: int = Field(512, description="最大生成token数", ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="采样温度")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="核采样概率")
    stop: Optional[List[str]] = Field(None, description="停止生成的字符串列表")
    stream: bool = Field(False, description="是否启用流式输出")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prompt": "请解释量子计算的基本原理",
                    "max_tokens": 1024,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stream": False,
                }
            ]
        }
    }

# 定义响应模型
class GenerateResponse(BaseModel):
    id: str = Field(..., description="生成请求唯一ID")
    text: str = Field(..., description="生成的文本内容")
    model: str = Field(..., description="使用的模型名称")
    usage: dict = Field(..., description="token使用统计")

# 使用请求和响应模型
@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    # request 对象已通过 Pydantic 验证，所有字段类型正确
    generated_text = f"对 '{request.prompt}' 的回复"  # 实际中调用模型推理
    return GenerateResponse(
        id="gen-001",
        text=generated_text,
        model="gpt-3.5-turbo",
        usage={"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60},
    )
```

**Pydantic 验证特性**：
- 自动类型转换（如字符串 "123" → 整数 123）
- 约束验证（`ge`, `le`, `min_length`, `max_length`, `regex`）
- 嵌套模型支持
- 自定义验证器（`@field_validator`）
- 请求体验证失败时自动返回 422 错误和详细错误信息

### 3.4 异步处理

#### async def 与 await

```python
import asyncio
from fastapi import FastAPI

app = FastAPI()

# 异步路由处理函数
@app.post("/generate")
async def generate(prompt: str):
    # 使用 await 调用异步函数，不阻塞事件循环
    result = await call_llm_api(prompt)
    return {"result": result}

async def call_llm_api(prompt: str) -> str:
    """模拟异步调用 LLM API"""
    await asyncio.sleep(1)  # 模拟网络IO等待
    return f"Generated text for: {prompt}"
```

#### 异步数据库访问

```python
from fastapi import FastAPI, Depends
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

app = FastAPI()

# 异步数据库引擎
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
async_session = async_sessionmaker(engine, expire_on_commit=False)

class Base(DeclarativeBase):
    pass

# 依赖注入：获取异步数据库会话
async def get_db():
    async with async_session() as session:
        yield session

@app.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: int,
    db: AsyncSession = Depends(get_db),  # 依赖注入
):
    result = await db.execute(
        "SELECT * FROM conversations WHERE id = :id",
        {"id": conversation_id}
    )
    return result.fetchone()
```

#### 同步与异步的选择

```python
# 纯 IO 操作 → 用 async def
@app.get("/data")
async def read_data():
    data = await fetch_from_remote()  # 异步IO
    return data

# CPU 密集型操作 → 用普通 def（FastAPI 自动放入线程池）
@app.post("/process")
def process_data(data: str):  # 注意：普通 def，非 async def
    result = heavy_computation(data)  # CPU密集型计算
    return {"result": result}

# 混合场景：在 async def 中运行 CPU 密集型任务
import asyncio
from concurrent.futures import ProcessPoolExecutor

process_pool = ProcessPoolExecutor()

@app.post("/inference")
async def inference(prompt: str):
    loop = asyncio.get_event_loop()
    # 将CPU密集型任务放入进程池，避免阻塞事件循环
    result = await loop.run_in_executor(
        process_pool,
        run_model_inference,  # 同步函数
        prompt
    )
    return {"result": result}
```

### 3.5 中间件

#### CORS 中间件

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# 配置跨域资源共享
app.add_middleware(
    CORSMiddleware,
    allow_origins=[                      # 允许的源
        "http://localhost:3000",          # 前端开发服务器
        "https://your-frontend.com",      # 生产前端
    ],
    allow_credentials=True,              # 允许携带Cookie
    allow_methods=["*"],                 # 允许的HTTP方法
    allow_headers=["*"],                 # 允许的请求头
)
```

#### 自定义中间件

```python
import time
from fastapi import FastAPI, Request

app = FastAPI()

# 请求计时中间件
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    # 调用下一个中间件或路由处理函数
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# API Key 验证中间件
@app.middleware("http")
async def verify_api_key(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key != "your-secret-key":
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"detail": "Invalid API Key"}
        )
    response = await call_next(request)
    return response
```

### 3.6 SSE 流式响应

这是 FastAPI 在 LLM 开发中最核心的功能之一，用于实现 LLM 生成内容的实时推送。

#### 基础 StreamingResponse

```python
import asyncio
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

app = FastAPI()

@app.post("/stream/generate")
async def stream_generate(prompt: str):
    """SSE 流式生成接口"""

    async def event_generator():
        # 模拟 LLM 逐 token 生成
        tokens = ["你", "好", "，", "我", "是", "AI", "助", "手"]
        for i, token in enumerate(tokens):
            # SSE 数据格式：data: {json}\n\n
            data = json.dumps({
                "token": token,
                "index": i,
                "finished": i == len(tokens) - 1
            }, ensure_ascii=False)
            yield f"data: {data}\n\n"
            await asyncio.sleep(0.1)  # 模拟生成延迟

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",   # SSE MIME 类型
        headers={
            "Cache-Control": "no-cache",   # 禁止缓存
            "Connection": "keep-alive",     # 保持连接
            "X-Accel-Buffering": "no",      # Nginx 禁用缓冲
        }
    )
```

#### 完整的 LLM 流式推理服务

```python
import asyncio
import json
import uuid
from typing import AsyncGenerator
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI(title="LLM 流式推理服务")

class StreamRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="输入提示词")
    max_tokens: int = Field(512, ge=1, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)

async def stream_llm_response(prompt: str, max_tokens: int, temperature: float) -> AsyncGenerator[str, None]:
    """模拟 LLM 流式推理（实际中替换为真实模型调用）"""
    request_id = str(uuid.uuid4())

    # 模拟逐 token 生成
    full_text = f"这是对 '{prompt}' 的详细回复内容，包含了多个token的流式输出演示。"
    tokens = list(full_text)  # 中文按字符拆分模拟

    for i, token in enumerate(tokens):
        chunk = {
            "id": f"chatcmpl-{request_id[:8]}",
            "object": "chat.completion.chunk",
            "choices": [{
                "index": 0,
                "delta": {"content": token},
                "finish_reason": None if i < len(tokens) - 1 else "stop"
            }]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0.05)

    yield "data: [DONE]\n\n"  # SSE 结束标记

@app.post("/v1/chat/completions")
async def chat_completions(request: StreamRequest):
    """OpenAI 兼容的流式聊天补全接口"""
    return StreamingResponse(
        stream_llm_response(request.prompt, request.max_tokens, request.temperature),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )
```

#### 客户端接收 SSE 流式响应

```python
# Python 客户端示例
import httpx
import asyncio
import json

async def stream_chat():
    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            "http://localhost:8000/v1/chat/completions",
            json={"prompt": "解释深度学习", "max_tokens": 512, "temperature": 0.7},
            timeout=60.0,
        ) as response:
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]  # 去掉 "data: " 前缀
                    if data == "[DONE]":
                        print("\n生成完成")
                        break
                    chunk = json.loads(data)
                    content = chunk["choices"][0]["delta"].get("content", "")
                    print(content, end="", flush=True)

asyncio.run(stream_chat())
```

### 3.7 文件上传

```python
from fastapi import FastAPI, UploadFile, File, HTTPException
from typing import List

app = FastAPI()

# 单文件上传
@app.post("/upload/document")
async def upload_document(file: UploadFile = File(..., description="上传的文档文件")):
    """
    参数说明：
    - file: UploadFile 对象，包含 filename, content_type, file 等属性
    - File(...): ... 表示必填
    """
    # 读取文件内容
    content = await file.read()

    # 获取文件信息
    return {
        "filename": file.filename,          # 原始文件名
        "content_type": file.content_type,  # MIME类型
        "size": len(content),               # 文件大小（字节）
    }

# 多文件上传
@app.post("/upload/documents")
async def upload_documents(files: List[UploadFile] = File(..., description="多个文档文件")):
    results = []
    for file in files:
        content = await file.read()
        results.append({
            "filename": file.filename,
            "size": len(content),
        })
    return {"uploaded": len(results), "files": results}

# 带大小限制的文件上传
@app.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="仅支持 JPEG/PNG 图片")

    content = await file.read()
    max_size = 10 * 1024 * 1024  # 10MB
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="文件大小不能超过10MB")

    return {"filename": file.filename, "size": len(content)}
```

### 3.8 依赖注入

```python
from fastapi import FastAPI, Depends, HTTPException, Header
from typing import Optional

app = FastAPI()

# 函数依赖
async def get_query_params(q: Optional[str] = None):
    return {"q": q}

@app.get("/search")
async def search(params: dict = Depends(get_query_params)):
    return params

# 类依赖 — API Key 验证
class APIKeyValidator:
    def __init__(self, valid_keys: list):
        self.valid_keys = valid_keys

    def __call__(self, x_api_key: str = Header(...)):
        if x_api_key not in self.valid_keys:
            raise HTTPException(status_code=403, detail="Invalid API Key")
        return x_api_key

validate_api_key = APIKeyValidator(valid_keys=["key-001", "key-002"])

@app.post("/generate")
async def generate(prompt: str, api_key: str = Depends(validate_api_key)):
    return {"prompt": prompt, "api_key": api_key}

# 子依赖链
async def get_token(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid token format")
    return authorization[7:]

async def get_current_user(token: str = Depends(get_token)):
    # 验证 token 并返回用户信息
    if token == "valid-token":
        return {"user_id": "user-001", "role": "admin"}
    raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/me")
async def read_users_me(user: dict = Depends(get_current_user)):
    return user
```

### 3.9 异常处理

```python
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

app = FastAPI()

# 内置 HTTP 异常
@app.get("/models/{model_id}")
async def get_model(model_id: str):
    if model_id not in ["gpt-4", "gpt-3.5-turbo"]:
        raise HTTPException(
            status_code=404,
            detail=f"模型 {model_id} 不存在",
            headers={"X-Error": "Model Not Found"},  # 可选自定义响应头
        )
    return {"model_id": model_id}

# 自定义异常类
class ModelNotReadyError(Exception):
    def __init__(self, model_id: str):
        self.model_id = model_id

# 注册异常处理器
@app.exception_handler(ModelNotReadyError)
async def model_not_ready_handler(request: Request, exc: ModelNotReadyError):
    return JSONResponse(
        status_code=503,
        content={
            "error": "model_not_ready",
            "detail": f"模型 {exc.model_id} 正在加载中，请稍后重试",
            "model_id": exc.model_id,
        }
    )

@app.get("/inference/{model_id}")
async def inference(model_id: str):
    if model_id == "gpt-4-loading":
        raise ModelNotReadyError(model_id)
    return {"result": "ok"}
```

---

## 4. 在 LLM 开发中的典型使用场景和代码示例

### 场景一：LLM 推理 API 服务

```python
import asyncio
import json
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title="LLM Inference Service", version="1.0.0")

# 模拟模型加载
loaded_models = {"gpt-3.5-turbo": True, "gpt-4": True}

class Message(BaseModel):
    role: str = Field(..., description="角色：system/user/assistant")
    content: str = Field(..., description="消息内容")

class ChatRequest(BaseModel):
    model: str = Field(..., description="模型名称")
    messages: List[Message] = Field(..., min_length=1, description="对话消息列表")
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(2048, ge=1, le=8192)
    stream: bool = Field(False, description="是否流式输出")

class ChatResponse(BaseModel):
    id: str
    model: str
    choices: List[dict]
    usage: dict

@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    """非流式聊天补全"""
    if request.model not in loaded_models:
        raise HTTPException(status_code=404, detail=f"模型 {request.model} 不存在")

    # 模拟推理
    last_message = request.messages[-1].content
    response_text = f"收到你的问题：{last_message}。这是模拟的回复。"

    return ChatResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:8]}",
        model=request.model,
        choices=[{
            "index": 0,
            "message": {"role": "assistant", "content": response_text},
            "finish_reason": "stop"
        }],
        usage={"prompt_tokens": 20, "completion_tokens": 30, "total_tokens": 50}
    )

@app.post("/v1/chat/completions/stream")
async def chat_completions_stream(request: ChatRequest):
    """流式聊天补全"""
    if request.model not in loaded_models:
        raise HTTPException(status_code=404, detail=f"模型 {request.model} 不存在")

    async def generate():
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        last_message = request.messages[-1].content
        tokens = list(f"收到你的问题：{last_message}。这是流式回复。")

        for i, token in enumerate(tokens):
            chunk = {
                "id": request_id,
                "object": "chat.completion.chunk",
                "choices": [{
                    "index": 0,
                    "delta": {"content": token} if i > 0 else {"role": "assistant", "content": token},
                    "finish_reason": None if i < len(tokens) - 1 else "stop"
                }]
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
            await asyncio.sleep(0.03)

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )
```

### 场景二：RAG（检索增强生成）服务

```python
from fastapi import FastAPI, Depends
from pydantic import BaseModel, Field
from typing import List

app = FastAPI(title="RAG Service")

class Document(BaseModel):
    content: str
    metadata: dict = {}

class QueryRequest(BaseModel):
    query: str = Field(..., description="用户查询")
    top_k: int = Field(5, ge=1, le=20, description="检索文档数量")
    model: str = Field("gpt-3.5-turbo", description="生成模型")

class QueryResponse(BaseModel):
    query: str
    retrieved_docs: List[Document]
    answer: str
    sources: List[str]

@app.post("/rag/query", response_model=QueryResponse)
async def rag_query(request: QueryRequest):
    # 1. 检索相关文档（实际中调用向量数据库）
    retrieved_docs = await retrieve_documents(request.query, request.top_k)

    # 2. 构建提示词
    context = "\n".join([doc.content for doc in retrieved_docs])
    prompt = f"基于以下参考资料回答问题：\n{context}\n\n问题：{request.query}"

    # 3. 调用 LLM 生成回答
    answer = await call_llm(prompt, model=request.model)

    return QueryResponse(
        query=request.query,
        retrieved_docs=retrieved_docs,
        answer=answer,
        sources=[doc.metadata.get("source", "") for doc in retrieved_docs],
    )

async def retrieve_documents(query: str, top_k: int) -> List[Document]:
    """模拟向量检索"""
    return [Document(content="模拟检索到的文档内容", metadata={"source": "doc1.txt"})]

async def call_llm(prompt: str, model: str) -> str:
    """模拟 LLM 调用"""
    return f"基于检索到的文档，关于 '{prompt}' 的回答"
```

### 场景三：模型管理 API

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

app = FastAPI(title="Model Management Service")

# 模型状态存储
model_registry: Dict[str, dict] = {
    "gpt-3.5-turbo": {"status": "loaded", "device": "cuda:0", "memory_mb": 2048},
    "gpt-4": {"status": "unloaded", "device": None, "memory_mb": 8192},
}

class ModelLoadRequest(BaseModel):
    model_name: str
    device: Optional[str] = "cuda:0"
    quantization: Optional[str] = None  # "4bit", "8bit", None

@app.get("/models")
async def list_models():
    """列出所有可用模型"""
    return {"models": model_registry}

@app.post("/models/{model_name}/load")
async def load_model(model_name: str, request: ModelLoadRequest):
    """加载模型到显存"""
    if model_name not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    if model_registry[model_name]["status"] == "loaded":
        raise HTTPException(status_code=409, detail="模型已加载")

    # 模拟加载过程
    model_registry[model_name] = {
        "status": "loaded",
        "device": request.device,
        "memory_mb": model_registry[model_name]["memory_mb"],
    }
    return {"message": f"模型 {model_name} 已加载到 {request.device}"}

@app.post("/models/{model_name}/unload")
async def unload_model(model_name: str):
    """卸载模型释放显存"""
    if model_name not in model_registry:
        raise HTTPException(status_code=404, detail="模型不存在")
    if model_registry[model_name]["status"] == "unloaded":
        raise HTTPException(status_code=409, detail="模型未加载")

    model_registry[model_name]["status"] = "unloaded"
    model_registry[model_name]["device"] = None
    return {"message": f"模型 {model_name} 已卸载"}
```

---

## 5. 数学原理

### 5.1 SSE（Server-Sent Events）协议

SSE 是一种基于 HTTP 的单向实时通信协议，其数据格式遵循以下规范：

```
data: {payload}\n\n
```

每个消息由 `data:` 字段和 JSON 载荷组成，以两个换行符 `\n\n` 结尾。服务端可保持 HTTP 连接不断开，持续推送数据。

**与 WebSocket 的比较**：

| 特性 | SSE | WebSocket |
|------|-----|-----------|
| 通信方向 | 单向（服务器→客户端） | 双向 |
| 协议 | HTTP | WS |
| 重连机制 | 内置自动重连 | 需手动实现 |
| 数据格式 | 纯文本 | 文本/二进制 |
| 适用场景 | LLM 流式输出、通知推送 | 聊天、实时协作 |

LLM 场景下，生成是单向的（模型→用户），SSE 更轻量且天然兼容 HTTP 生态。

### 5.2 异步 IO 的事件循环模型

FastAPI 基于 ASGI 异步模型，其核心是事件循环（Event Loop）：

```
事件循环 (Event Loop)
├── 注册 IO 操作（网络请求、数据库查询）
├── IO 等待时挂起当前协程，切换执行其他协程
├── IO 完成时恢复挂起的协程
└── 所有协程并发执行，无需多线程
```

**数学表达**：设 N 个请求各需 IO 等待时间 $t_i$，同步模型总时间为 $\sum_{i=1}^{N} t_i$，异步模型约为 $\max(t_1, t_2, ..., t_N)$，在 IO 密集型场景下吞吐量提升可达 N 倍。

---

## 6. 代码原理 / 架构原理

### 6.1 ASGI 架构

```
客户端请求
    ↓
┌─────────────────────────┐
│     Uvicorn (ASGI Server) │  ← 监听端口，接收 HTTP 请求
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     Starlette (ASGI Toolkit) │  ← 路由匹配、中间件管道、请求/响应对象
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     FastAPI               │  ← 路由装饰器、依赖注入、OpenAPI 生成
└────────────┬────────────┘
             ↓
┌─────────────────────────┐
│     Pydantic              │  ← 数据验证、序列化、类型转换
└─────────────────────────┘
```

**三层架构**：
1. **Uvicorn**：ASGI 服务器，负责 HTTP 协议解析和底层网络 IO
2. **Starlette**：ASGI 工具包，提供路由、中间件、请求/响应抽象
3. **FastAPI**：在 Starlette 之上添加类型驱动的 API 开发体验

### 6.2 请求处理流程

```
1. Uvicorn 接收 HTTP 请求 → 转换为 ASGI scope
2. Starlette 中间件管道（CORS → 自定义中间件 → ...）
3. 路由匹配（URL → 处理函数）
4. FastAPI 依赖注入解析（Depends 链）
5. Pydantic 请求体验证（JSON → BaseModel）
6. 执行路由处理函数（async def / def）
7. Pydantic 响应序列化（BaseModel → JSON）
8. 中间件管道（逆序）
9. Uvicorn 返回 HTTP 响应
```

### 6.3 依赖注入机制

FastAPI 的依赖注入基于 Python 的类型提示和 `Depends` 类：

```python
# Depends 的简化实现原理
class Depends:
    def __init__(self, dependency):
        self.dependency = dependency

# FastAPI 在路由调用前递归解析依赖链
# 1. 检查函数参数类型提示
# 2. 如果参数有 Depends()，调用对应的依赖函数
# 3. 将返回值注入到路由函数参数中
# 4. 支持子依赖（依赖函数本身也有 Depends 参数）
```

### 6.4 OpenAPI 文档自动生成

FastAPI 通过检查路由函数的类型提示、Pydantic 模型和默认值，自动生成 OpenAPI 3.0 规范：

```python
# 路由函数定义
@app.post("/generate")
async def generate(request: GenerateRequest) -> GenerateResponse:
    ...

# FastAPI 自动提取：
# - 请求体 schema → GenerateRequest 的 JSON Schema
# - 响应 schema → GenerateResponse 的 JSON Schema
# - 参数约束 → Field/ge/le/min_length 等
# - 描述信息 → Field(description=...) 和 docstring
# → 生成 /openapi.json → Swagger UI (/docs) 和 ReDoc (/redoc) 渲染
```

---

## 7. 常见注意事项和最佳实践

### 7.1 性能优化

```python
# 1. CPU 密集型任务不要用 async def，用普通 def 或放入线程/进程池
# 错误：会阻塞事件循环
@app.post("/inference")
async def inference(prompt: str):
    result = model.generate(prompt)  # CPU密集，阻塞所有请求
    return result

# 正确：让 FastAPI 自动放入线程池
@app.post("/inference")
def inference(prompt: str):  # 普通 def
    result = model.generate(prompt)
    return result

# 2. 大文件用流式响应，避免内存溢出
from fastapi.responses import StreamingResponse

@app.get("/download/{file_id}")
async def download(file_id: str):
    def iterfile():
        with open(f"large_file_{file_id}.bin", "rb") as f:
            while chunk := f.read(64 * 1024):  # 64KB 分块
                yield chunk
    return StreamingResponse(iterfile(), media_type="application/octet-stream")
```

### 7.2 生产部署

```bash
# 使用 Gunicorn + Uvicorn Workers（推荐生产部署方式）
gunicorn main:app \
    --workers 4 \              # Worker 数量，通常为 CPU 核心数
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8000 \
    --timeout 120 \            # LLM 推理可能较慢，增大超时
    --max-requests 1000 \      # 定期重启 worker 防止内存泄漏
    --max-requests-jitter 50   # 随机偏移，避免所有 worker 同时重启
```

#### Docker 部署

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["gunicorn", "main:app", \
     "--workers", "4", \
     "--worker-class", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", \
     "--timeout", "120"]
```

### 7.3 安全注意事项

```python
# 1. 限制请求体大小
from fastapi import FastAPI, Request
from fastapi.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

class MaxBodySizeMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, max_body_size: int = 10 * 1024 * 1024):  # 10MB
        super().__init__(app)
        self.max_body_size = max_body_size

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_size:
            from fastapi.responses import JSONResponse
            return JSONResponse(status_code=413, content={"detail": "请求体过大"})
        return await call_next(request)

# 2. API 速率限制（使用 slowapi）
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter

@app.post("/generate")
@limiter.limit("10/minute")  # 每分钟最多10次请求
async def generate(request: Request, prompt: str):
    return {"result": "ok"}

# 3. 输入验证 — 始终使用 Pydantic 模型，不要直接接受原始 dict
# 4. 禁用生产环境的文档端点
app = FastAPI(docs_url=None, redoc_url=None)  # 生产环境
```

### 7.4 流式响应注意事项

```python
# 1. 确保正确设置 SSE 响应头
return StreamingResponse(
    generator(),
    media_type="text/event-stream",  # 必须是 text/event-stream
    headers={
        "Cache-Control": "no-cache",     # 防止代理缓存
        "X-Accel-Buffering": "no",       # 防止 Nginx 缓冲
    }
)

# 2. SSE 数据格式必须严格遵循规范
# 正确格式
yield f"data: {json.dumps(payload)}\n\n"  # 两个换行符

# 3. 客户端断开连接时及时清理资源
from fastapi import Request

@app.post("/stream")
async def stream(request: Request):
    async def generate():
        try:
            for token in model.stream_generate():
                if await request.is_disconnected():  # 检测客户端是否断开
                    break
                yield f"data: {token}\n\n"
        finally:
            cleanup_resources()  # 释放 GPU 显存等

    return StreamingResponse(generate(), media_type="text/event-stream")

# 4. Nginx 反向代理配置
# proxy_buffering off;  # 必须关闭 Nginx 缓冲
# proxy_read_timeout 300s;  # LLM 生成可能较慢，增大超时
```

### 7.5 项目结构建议

```
llm_service/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI 应用入口
│   ├── api/                 # API 路由
│   │   ├── __init__.py
│   │   ├── v1/
│   │   │   ├── chat.py      # 聊天补全路由
│   │   │   ├── models.py    # 模型管理路由
│   │   │   └── embeddings.py # 向量嵌入路由
│   ├── core/                # 核心配置
│   │   ├── config.py        # 配置管理
│   │   ├── security.py      # 安全相关
│   │   └── middleware.py     # 中间件
│   ├── models/              # Pydantic 模型
│   │   ├── request.py
│   │   └── response.py
│   ├── services/            # 业务逻辑
│   │   ├── llm_service.py   # LLM 推理封装
│   │   └── rag_service.py   # RAG 服务
│   └── dependencies.py      # 依赖注入
├── tests/                   # 测试
├── Dockerfile
├── requirements.txt
└── README.md
```

### 7.6 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| 流式响应被缓冲 | Nginx/代理缓冲 | 设置 `X-Accel-Buffering: no`，Nginx 加 `proxy_buffering off` |
| 大模型推理阻塞其他请求 | async 中执行 CPU 密集操作 | 使用普通 `def` 或 `run_in_executor` |
| 422 Validation Error | 请求数据类型不匹配 | 检查 Pydantic 模型定义和请求体格式 |
| 连接超时 | LLM 推理时间长 | 增大 `timeout` 设置，前端设置较长超时 |
| 内存泄漏 | 全局变量累积 | 使用 `max-requests` 定期重启 worker |

---

## 参考链接

- FastAPI 官方文档：https://fastapi.tiangolo.com/
- Starlette 文档：https://www.starlette.io/
- Pydantic 文档：https://docs.pydantic.dev/
- Uvicorn 文档：https://www.uvicorn.org/
- SSE 规范：https://html.spec.whatwg.org/multipage/server-sent-events.html
