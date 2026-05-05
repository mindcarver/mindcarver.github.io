# AI 应用后端结构设计

## 文章导读

前文已经分别覆盖 API 调用、结构化输出、Function Calling 和 Prompt 模板管理，这一篇要把这些散件组装成一个完整的后端服务。

很多学 AI 的人会停留在"写脚本"阶段——一个 Python 文件里塞了 API 调用、Prompt、数据处理，跑起来能出结果就觉得够了。这在演示阶段没问题，但一旦需要给别人用、需要稳定运行、需要持续迭代，脚本就撑不住了。你需要一个结构清晰的后端服务。

这篇文章讨论的不是某个框架的高级用法，而是"怎么把 AI 能力组织成一个可维护的后端系统"。路由怎么设计、请求和响应怎么定义、服务层怎么拆分、配置怎么管理、日志怎么打、错误怎么返回、项目目录怎么组织——这些都是工程问题，和 AI 本身无关，但决定了你的 AI 能力能不能被可靠地交付。

读完这篇文章后，你应该能搭建一个结构清晰的 FastAPI 项目，设计规范的 API 接口，把 LLM Client、结构化输出、Prompt 模板整合到一个统一的架构中。

## 核心概念

### 一、FastAPI 基础

FastAPI 是 Python 生态中构建 API 服务的首选框架。它快（基于异步 ASGI）、简洁（用类型注解自动生成文档）、好用（自动数据校验和序列化）。对于 AI 应用后端来说，FastAPI 几乎是最合适的选择。

为什么选 FastAPI 而不是 Flask 或 Django？Flask 是同步框架，AI 应用经常需要等待模型响应（可能几十秒），同步框架在等待期间会阻塞线程，影响并发能力。FastAPI 是异步框架，等待模型响应期间可以处理其他请求，并发能力更强。Django 太重了，它内置了 ORM、模板引擎、管理后台等很多 AI 应用用不到的东西，学习成本和复杂度都不划算。

FastAPI 的核心概念不多，但对于 AI 应用后端来说，需要重点理解以下几个：

路由（Router）。路由决定了什么 URL 路径对应什么处理函数。比如 POST /analyze/industry 对应行业分析处理函数。

依赖注入（Dependency Injection）。FastAPI 支持依赖注入，可以在处理函数中自动获取数据库连接、配置对象等依赖。这避免了在函数内部手动创建这些对象。

请求体（Request Body）。用 Pydantic 模型定义请求体的结构，FastAPI 自动解析和校验。非法的请求会在进入处理函数之前就被拦截。

响应模型（Response Model）。用 Pydantic 模型定义响应的结构，FastAPI 自动序列化和校验。保证你的 API 返回的数据格式是可预期的。

中间件（Middleware）。中间件是在请求处理之前或之后执行的逻辑。常见的用途包括日志记录、错误处理、认证鉴权。

FastAPI 还有一个被低估的优势：它自动生成交互式 API 文档。你定义好路由和请求/响应模型后，访问 /docs 路径就能看到一个可测试的 API 文档页面。这在开发阶段非常方便，也方便后续给其他人看你的接口设计。

### 二、路由设计

路由设计是 API 后端的骨架。好的路由设计让使用者直觉地知道怎么调用你的接口。

路由设计的第一原则是 RESTful 风格。虽然 AI 应用不一定要严格遵守 REST 规范，但遵循一些基本约定能让 API 更易理解：

用名词表示资源，用 HTTP 方法表示操作。/industries 表示行业资源，POST /industries/analyze 表示分析行业的操作。

用路径参数表示具体的资源实例。/reports/{report_id} 表示获取某个具体的报告。

用查询参数表示过滤条件。/reports?type=industry&status=completed 表示获取已完成类型的行业分析报告。

对于 AI 应用来说，路由通常分为以下几组：

分析类接口。/analyze/industry（行业分析）、/analyze/role（岗位分析）、/analyze/process（流程分析）。这类接口接收分析请求，调用模型，返回结构化的分析结果。

任务类接口。/tasks（创建任务）、/tasks/{task_id}（查询任务状态）、/tasks/{task_id}/cancel（取消任务）。这类接口用于处理长时间运行的分析任务。

管理类接口。/admin/prompts（Prompt 模板管理）、/admin/cache（缓存管理）、/admin/logs（日志查询）。这类接口用于系统管理。

路由的层级不宜过深。超过三层的路径（如 /api/v1/analyze/industry/advanced）会让 API 难以记忆和使用。如果逻辑复杂，考虑用查询参数或请求体参数来区分，而不是增加路径层级。

### 三、请求与响应设计

请求和响应的设计决定了 API 的易用性和可维护性。

请求体的设计原则：

必要参数和可选参数分离。必要参数放在请求体中，可选参数可以用查询参数或请求体中的可选字段。对于行业分析，行业名称是必要参数，分析深度、输出格式是可选参数。

用 Pydantic 模型定义请求结构。Pydantic 会自动做类型校验和类型转换。如果用户传入的 temperature 是字符串 "0.7"，Pydantic 会自动转换为浮点数 0.7。如果类型完全不兼容（如传入字符串 "high"），Pydantic 会返回清晰的错误信息。

提供默认值。AI 模型的很多参数有合理的默认值。temperature 默认 0.7，max_tokens 默认 2000。这些默认值应该在后端定义，而不是要求前端每次都传。

响应体的设计原则：

统一响应格式。不管接口成功还是失败，响应格式应该是一致的。比如统一包含 success（布尔）、data（数据）、error（错误信息）、request_id（请求追踪 ID）四个字段。这样前端处理响应时不用针对每个接口写不同的逻辑。

包含请求追踪 ID。每次请求生成一个唯一的 request_id，在响应中返回。这个 ID 应该贯穿整个请求链路——包括日志、模型调用、数据库查询。当出问题时，用户提供 request_id，你就能在日志中快速定位。

返回有意义的错误信息。不要只返回"调用失败"这种模糊的信息。要告诉用户为什么失败——是参数错误、模型不可用、还是超时。同时要在后端日志中记录详细的错误堆栈，但不要把堆栈信息直接返回给用户（可能有安全风险）。

### 四、服务层设计

服务层（Service Layer）是后端架构的核心。它把业务逻辑和路由层分离，让代码更易维护。

服务层的价值：

路由层只负责 HTTP 相关的事务——接收请求、解析参数、调用服务、返回响应。业务逻辑全部在服务层实现。

服务层的函数可以被多个路由复用。比如行业分析服务既可以被同步分析接口调用，也可以被异步任务接口调用。

服务层易于测试。路由层依赖 HTTP 请求对象，测试时需要构造 mock 请求。服务层的函数是纯业务逻辑，测试时直接传入参数即可。

服务层的拆分原则：

按业务领域拆分。LLMService 负责模型调用、PromptService 负责 Prompt 管理、RAGService 负责检索增强生成。每个服务负责一个明确的领域。

按职责拆分。每个服务类内部，初始化负责配置和依赖注入、公共方法负责对外暴露能力、私有方法负责内部实现细节。

服务层的依赖管理：

服务类之间会有依赖。RAGService 依赖 LLMService（生成 Embedding）、AnalysisService 依赖 RAGService（检索相关文档）。这些依赖应该在服务类初始化时通过依赖注入传入，而不是在服务类内部直接创建。

这种设计让测试更容易。你可以给 AnalysisService 注入一个 mock 的 RAGService，而不需要真正调用向量数据库。

### 五、配置管理

配置管理是后端工程的基础设施，但经常被忽视。

配置管理的核心原则：

配置和代码分离。不要把 API Key、数据库连接字符串写死在代码里。这些信息应该放在配置文件或环境变量中。

环境隔离。开发环境、测试环境、生产环境的配置应该分开。开发环境用测试数据库和较低的模型调用频率限制，生产环境用真实数据库和完整的限流配置。

敏感信息保护。API Key、数据库密码等敏感信息应该通过环境变量传入，而不是写在配置文件中。配置文件可以提交到代码仓库，但包含敏感信息的 .env 文件应该被 .gitignore 排除。

FastAPI 项目中常用的配置管理方式是用 Pydantic Settings：

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # 应用配置
    app_name: str = "AI Application"
    debug: bool = False

    # API 配置
    api_v1_prefix: str = "/api/v1"

    # LLM 配置
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"
    default_model: str = "gpt-4o-mini"

    # 其他配置
    max_tokens: int = 2000
    timeout: int = 30

    class Config:
        env_file = ".env"

settings = Settings()
```

这种方式的优点是配置有类型约束、有默认值、可以从环境变量自动读取。

### 六、日志系统

日志是系统可观测性的基础。没有日志的系统就像在黑盒中运行，出问题时很难定位。

日志的设计原则：

结构化日志。不要用 print 语句输出日志，不要用不规则的字符串格式。应该用 JSON 格式输出结构化日志——包含时间戳、日志级别、模块、消息、请求 ID 等字段。结构化日志可以被日志系统（如 ELK、Loki）索引和查询。

日志级别要合理。DEBUG 级别记录详细的调试信息（只在开发环境开启）、INFO 级别记录正常的业务流程（如请求到达、任务完成）、WARNING 级别记录异常但不影响运行的情况（如重试成功）、ERROR 级别记录需要关注的错误（如模型调用失败）。

日志要包含上下文。只记录"调用失败"是不够的。要记录是哪个操作失败、失败时的参数是什么、错误信息是什么、请求 ID 是多少。这些上下文信息对于排查问题至关重要。

FastAPI 项目中可以这样配置日志：

```python
import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.name,
            "request_id": getattr(record, "request_id", None)
        }
        return json.dumps(log_data, ensure_ascii=False)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler()
    ]
)
```

### 七、错误处理

统一的错误处理让 API 更可靠、更易用。

错误处理的设计原则：

不要让未捕获的异常暴露给用户。如果代码中抛出异常但没有被捕获，FastAPI 会返回 500 错误和默认的错误页面。这不友好。应该用 try-except 捕获异常，返回结构化的错误信息。

区分错误类型。参数错误应该返回 400，认证错误返回 401，权限不足返回 403，资源不存在返回 404，模型服务不可用返回 503。不同的错误码让前端可以做不同的处理。

错误信息要清晰。不要返回"系统错误"这种模糊的信息。要告诉用户问题是什么——"行业名称不能为空"、"模型调用超时，请稍后重试"。

FastAPI 中可以用异常处理器统一处理错误：

```python
from fastapi import HTTPException

class AppException(Exception):
    def __init__(self, message: str, code: int = 500):
        self.message = message
        self.code = code

@app.exception_handler(AppException)
async def app_exception_handler(request, exc: AppException):
    return JSONResponse(
        status_code=exc.code,
        content={"success": False, "error": exc.message}
    )
```

### 八、项目目录组织

清晰的目录结构让项目易于维护和扩展。

推荐的 FastAPI 项目目录结构：

```
ai_app/
├── app/
│   ├── api/
│   │   ├── v1/
│   │   │   ├── endpoints/
│   │   │   │   ├── chat.py
│   │   │   │   ├── analysis.py
│   │   │   │   └── admin.py
│   │   │   └── router.py
│   ├── core/
│   │   ├── config.py
│   │   ├── security.py
│   │   └── logging.py
│   ├── models/
│   │   ├── schemas.py
│   │   └── database.py
│   ├── services/
│   │   ├── llm_service.py
│   │   ├── rag_service.py
│   │   └── prompt_service.py
│   └── main.py
├── tests/
├── prompts/
├── outputs/
└── requirements.txt
```

这个结构把不同职责的代码放在不同的目录中：api 目录处理 HTTP 接口，core 目录处理配置和基础设施，models 目录定义数据模型，services 目录实现业务逻辑。

## 实战示例

### 完整的 FastAPI 主入口

```python
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.router import api_router
from app.core.config import settings
from app.core.logging import logger

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
    openapi_url=f"{settings.api_v1_prefix}/openapi.json"
)

# CORS 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 请求日志中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path}",
        extra={"duration": duration, "status": response.status_code}
    )
    return response

# 路由注册
app.include_router(api_router, prefix=settings.api_v1_prefix)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
```

### 服务层示例

```python
from app.core.config import settings

class LLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url
        )

    async def chat(self, messages: list, temperature: float = 0.7):
        try:
            response = self.client.chat.completions.create(
                model=settings.default_model,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            raise LLMError(f"模型调用失败: {str(e)}")
```

## 核心结论

后端架构设计决定了 AI 应用能否稳定、可维护地运行。今天建立的是工程实践，不是 AI 算法，但这些工程实践是 AI 能力被可靠交付的基础。

核心要点：

FastAPI 是构建 AI 应用后端的最佳选择——异步、自动文档、类型安全。

路由设计要 RESTful、层级清晰、命名直观。

请求和响应要统一格式、包含追踪 ID、返回有意义的错误信息。

服务层要把业务逻辑和路由层分离，按业务领域拆分。

配置管理要和代码分离、环境隔离、敏感信息保护。

日志要结构化、合理分级、包含上下文。

错误处理要统一、分类清晰、信息明确。

项目目录结构要按职责划分、易于扩展。

## 相关延伸

Week 2 的最终阶段，你要把今天学的架构能力用到实战中——实现一个完整的行业分析 API Demo。这个 Demo 会把本周所有知识串起来，从用户输入到最终报告，走通整个链路。
