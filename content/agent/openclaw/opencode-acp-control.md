# OpenClaw 通过 opencode-acp-control 操控 OpenCode 写代码

## 简介

`opencode-acp-control` 是一个强大的 OpenClaw 技能，它允许 AI 助手通过编程方式控制 OpenCode（一个 AI 编码助手）来自动化代码开发任务。通过这个技能，OpenClaw 可以像人类开发者一样与 OpenCode 交互，但速度更快、更可靠。

### 什么是 opencode-acp-control？

opencode-acp-control 是基于 OpenCode 的 **ACP (Agent Client Protocol)** 协议实现的控制技能。它使用 JSON-RPC 2.0 标准与 OpenCode 进行通信，实现了完整的会话管理、任务分发和进度监控功能。

## 为什么需要它？

### 解决的核心问题

1. **TUI 交互限制**
   - OpenCode 的默认界面是 TUI（文本用户界面），难以程序化控制
   - 无法批量处理多个开发任务
   - 难以集成到自动化工作流中

2. **手动编码效率低**
   - 重复性代码编写耗时
   - 多文件项目管理复杂
   - 缺乏统一的项目脚手架

3. **AI 协作需求**
   - OpenClaw 需要一个可靠的方式来委托代码开发任务
   - 需要实时监控开发进度
   - 需要处理复杂的多步骤开发流程

## 安装方法

### 从 ClawHub 安装

```bash
# 搜索技能
clawhub search "opencode-acp-control"

# 安装技能
clawhub install opencode-acp-control
```

安装后，技能会被放置在：
```
~/.openclaw/workspace/skills/opencode-acp-control/
```

### 验证安装

检查技能是否正确安装：

```bash
ls -la ~/.openclaw/workspace/skills/opencode-acp-control/
```

应该看到 `SKILL.md` 和相关文件。

## 工作原理

### ACP 协议

ACP (Agent Client Protocol) 是 OpenCode 提供的标准化通信协议，基于 JSON-RPC 2.0 规范。它定义了客户端（OpenClaw）与服务端（OpenCode）之间的通信格式。

### JSON-RPC 通信流程

```
OpenClaw (客户端)          OpenCode (服务端)
      |                           |
      |  1. initialize            |
      |-------------------------->|
      |  <-- capabilities         |
      |                           |
      |  2. session/new           |
      |-------------------------->|
      |  <-- sessionId            |
      |                           |
      |  3. session/prompt        |
      |-------------------------->|
      |  <-- session/update (流式)|
      |  <-- session/update       |
      |  <-- stopReason           |
      |                           |
```

### 核心概念

1. **会话 (Session)**
   - 每个开发任务在独立的会话中执行
   - 会话包含工作目录、上下文和历史记录
   - 可以列出、恢复或分叉会话

2. **提示 (Prompt)**
   - 发送给 OpenCode 的开发任务描述
   - 支持文本和图片
   - 可以包含详细的需求和约束

3. **更新 (Update)**
   - OpenCode 实时返回的进度信息
   - 包括思考过程、工具调用、文件操作等
   - 流式传输，实时反馈

## 使用步骤

### 1. 启动 OpenCode ACP

```javascript
// 使用 OpenClaw 的 exec 工具启动
mcp_exec({
  background: true,
  command: "opencode acp",
  pty: true,
  workdir: "/path/to/project"
})
```

### 2. 初始化连接

发送初始化请求：

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "method": "initialize",
  "params": {
    "protocolVersion": 1,
    "clientCapabilities": {
      "fs": {
        "readTextFile": true,
        "writeTextFile": true
      },
      "terminal": true
    },
    "clientInfo": {
      "name": "openclaw",
      "title": "OpenClaw",
      "version": "1.0.0"
    }
  }
}
```

响应示例：

```json
{
  "jsonrpc": "2.0",
  "id": 0,
  "result": {
    "protocolVersion": 1,
    "agentCapabilities": {
      "loadSession": true,
      "mcpCapabilities": {
        "http": true,
        "sse": true
      }
    },
    "agentInfo": {
      "name": "OpenCode",
      "version": "1.2.10"
    }
  }
}
```

### 3. 创建会话

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "session/new",
  "params": {
    "cwd": "/path/to/project",
    "mcpServers": []
  }
}
```

使用 `session/list` 获取 sessionId：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "session/list",
  "params": {}
}
```

### 4. 发送开发任务

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "method": "session/prompt",
  "params": {
    "sessionId": "ses_xxx",
    "prompt": [
      {
        "type": "text",
        "text": "创建一个 React TODO 应用，包含添加、删除、标记完成功能"
      }
    ]
  }
}
```

### 5. 监控进度

持续轮询获取更新：

```javascript
// 每 2-5 秒轮询一次
mcp_process({
  action: "poll",
  sessionId: "xxx",
  timeout: 5000
})
```

更新类型：

- `agent_thought_chunk` - AI 的思考过程
- `agent_message_chunk` - AI 的回复消息
- `tool_call` - 工具调用（如创建文件）
- `tool_call_update` - 工具执行进度
- `usage_update` - Token 使用情况

完成标志：

```json
{
  "jsonrpc": "2.0",
  "id": 3,
  "result": {
    "stopReason": "end_turn"
  }
}
```

## 实战案例

### 案例 1：Skills Board 项目

**任务**：创建一个技能面板 Web 应用

**代码**：

```javascript
// 1. 启动 OpenCode ACP
await mcp_exec({
  background: true,
  command: "opencode acp",
  pty: true,
  workdir: "/Users/mac08/workspace/test"
})

// 2. 初始化
await mcp_process({
  action: "write",
  data: JSON.stringify({
    jsonrpc: "2.0",
    id: 0,
    method: "initialize",
    params: { /* ... */ }
  }),
  sessionId: "xxx"
})

// 3. 创建会话并发送任务
await mcp_process({
  action: "write",
  data: JSON.stringify({
    jsonrpc: "2.0",
    id: 3,
    method: "session/prompt",
    params: {
      sessionId: "ses_xxx",
      prompt: [{
        type: "text",
        text: "开发一个 skills board 项目。创建以下文件：\n1. index.html - 技能面板页面\n2. skills.sh - 管理脚本\n3. style.css - 样式文件\n4. app.js - JavaScript 交互逻辑"
      }]
    }
  }),
  sessionId: "xxx"
})

// 4. 监控进度
while (true) {
  const update = await mcp_process({
    action: "poll",
    sessionId: "xxx",
    timeout: 10000
  })
  
  if (update.includes("stopReason")) {
    break
  }
}
```

**结果**：
- ✅ 4 个文件自动创建
- ✅ 完整的前端应用
- ✅ 现代化的 UI 设计
- ✅ 本地存储持久化

### 案例 2：TypeScript TODO 应用

**任务**：使用 TypeScript 开发 TODO 应用

**特点**：
- 完整的 TypeScript 项目结构
- 类型定义、模块化设计
- 自动编译和依赖安装

**生成的文件**：
```
todo-app/
├── package.json
├── tsconfig.json
├── index.html
├── styles.css
├── src/
│   ├── types.ts
│   ├── storage.ts
│   ├── ui.ts
│   └── app.ts
└── dist/          # 编译后的 JS 文件
```

**代码量**：1027 行

### 案例 3：zVec RAG 系统

**任务**：基于阿里 zVec 开发 RAG 系统

**复杂度**：
- Python FastAPI 后端
- 文档处理（PDF/TXT/MD/DOCX）
- 向量存储集成
- RAG 问答引擎
- RESTful API

**生成的文件**：
```
zvec-rag-system/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── .env.example
└── app/
    ├── api/
    │   └── routes.py
    └── services/
        ├── document_processor.py
        ├── vector_store.py
        └── rag_engine.py
```

**代码量**：1027 行

**开发时间**：约 5 分钟

## 优势对比

### vs TUI 交互

| 特性 | TUI 交互 | opencode-acp-control |
|------|----------|----------------------|
| 自动化 | ❌ 需要人工操作 | ✅ 完全自动化 |
| 批量任务 | ❌ 一次一个 | ✅ 支持批量 |
| 进度监控 | ❌ 难以追踪 | ✅ 实时反馈 |
| 集成性 | ❌ 独立工具 | ✅ 无缝集成 |
| 可靠性 | ⚠️ 依赖界面 | ✅ 协议保证 |

### vs 手动编码

| 特性 | 手动编码 | opencode-acp-control |
|------|----------|----------------------|
| 速度 | 慢 | 快（5-10 分钟完成项目） |
| 一致性 | 因人而异 | 高度一致 |
| 错误率 | 较高 | 较低（AI 辅助） |
| 文档 | 需要额外编写 | 自动生成 |
| 学习曲线 | 陡峭 | 平缓 |

## 最佳实践

### 1. 任务描述要清晰

**好的示例**：
```
创建一个 TypeScript TODO 应用。要求：
1. 使用 TypeScript + HTML + CSS
2. 功能：添加、删除、标记完成
3. 本地存储持久化
4. 响应式设计
```

**不好的示例**：
```
做一个 TODO 应用
```

### 2. 合理设置超时

```javascript
// 简单任务：10-30 秒
timeout: 30000

// 中等任务：1-2 分钟
timeout: 120000

// 复杂任务：3-5 分钟
timeout: 300000
```

### 3. 处理错误和重试

```javascript
let retries = 3
while (retries > 0) {
  try {
    const result = await sendPrompt(task)
    if (result.success) break
  } catch (error) {
    retries--
    if (retries === 0) throw error
    await sleep(5000) // 等待 5 秒后重试
  }
}
```

### 4. 监控 Token 使用

```json
{
  "sessionUpdate": "usage_update",
  "used": 99250,
  "size": 200000,
  "cost": {
    "input": 0.15,
    "output": 0.60
  }
}
```

### 5. 保存会话 ID

```javascript
// 保存 sessionId 以便后续恢复
const sessionId = "ses_xxx"
localStorage.setItem('lastSessionId', sessionId)

// 恢复会话
const savedSessionId = localStorage.getItem('lastSessionId')
```

## 注意事项

### 1. 环境要求

- ✅ OpenCode 已安装并配置
- ✅ 有效的 AI 提供商凭证（OpenAI/Anthropic/GLM 等）
- ✅ Git 仓库（OpenCode 需要在 git 仓库中运行）

### 2. 性能考虑

- 复杂任务可能需要 5-10 分钟
- Token 消耗较大（每个项目约 50K-100K tokens）
- 建议使用成本较低的模型（如 GLM-5）

### 3. 错误处理

常见错误：

1. **会话创建失败**
   - 检查工作目录是否存在
   - 确保是 git 仓库

2. **长时间无响应**
   - 任务可能太复杂，简化描述
   - 检查 OpenCode 进程是否卡住

3. **文件未生成**
   - 检查权限
   - 查看 OpenCode 日志

### 4. 安全性

- ⚠️ 不要在提示中包含敏感信息（API keys、密码等）
- ⚠️ 审查生成的代码，特别是涉及网络请求的部分
- ⚠️ 使用 `.env` 文件管理敏感配置

## 总结

`opencode-acp-control` 是 OpenClaw 生态中的一个强大工具，它通过标准化的 ACP 协议实现了对 OpenCode 的程序化控制。主要优势包括：

✅ **自动化**：完全自动化的代码开发流程
✅ **可靠性**：基于标准协议，稳定可靠
✅ **高效性**：5-10 分钟完成完整项目
✅ **可扩展**：支持复杂的多文件项目
✅ **可监控**：实时进度反馈和错误处理

通过本文介绍的方法，你可以：
1. 快速搭建项目脚手架
2. 自动化重复性开发任务
3. 提高代码质量和一致性
4. 专注于架构设计而非具体实现

**下一步**：
- 尝试简单的项目（如 TODO 应用）
- 探索更复杂的场景（如 RAG 系统）
- 集成到你的开发工作流中

Happy Coding! 
