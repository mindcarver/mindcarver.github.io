# MemOS 本地部署 + OpenClaw 集成指南

本文档记录如何将本地部署的 MemOS 2.0 与 OpenClaw 对接，实现 AI 智能体的长期记忆能力。

---

## 架构概览

```
OpenClaw (智能体) ──▶ MemOS 插件 (生命周期钩子) ──▶ MemOS API (FastAPI)
│
┌─────┴─────┐
│ │
Neo4j Qdrant
(图数据库) (向量库)
```

插件通过两个钩子完成记忆的读写：

- `before_agent_start`：调用 `/product/search`，把相关记忆注入上下文
- `agent_end`：调用 `/product/add`，把本轮对话写入记忆

MemOS 支持四种记忆类型：text_mem（文本）、act_mem（行为）、para_mem（参数）、pref_mem（偏好），底层用 Neo4j 图图存储、Qdrant 做向量检索。

---

## 前置要求

| 软件                    | 版本 | 用途                  |
| ----------------------- | ---- | --------------------- |
| Docker & Docker Compose | 最新 | 运行 MemOS 及依赖服务 |
| Node.js                 | 18+  | OpenClaw 运行环境     |

确认 MemOS 相关容器都跑起来了：

```bash
docker compose -f /path/to/memOS/docker/docker-compose.yml ps
# 应该能看到 neo4j、qdrant、memos 三个容器
```

---

## 安装插件

插件仓库：https://github.com/mindcarver/MemOS-Cloud-OpenClaw-Plugin

```bash
git clone https://github.com/mindcarver/MemOS-Cloud-OpenClaw-Plugin.git
```

### 方式一：load.paths 引用（推荐）

直接在配置里指向本地仓库，改代码不用重新拷贝，重启 gateway 就生效：

```json
{
  "plugins": {
    "entries": {
      "memos-local-openclaw-plugin": { "enabled": true }
    }
  },
  "load": {
    "paths": ["/your/path/to/MemOS-Cloud-OpenClaw-Plugin"]
  }
}
```

### 方式二：手动拷贝

```bash
mkdir -p ~/.openclaw/extensions/memos-local-openclaw-plugin
cp -r /path/to/MemOS-Cloud-OpenClaw-Plugin/* ~/.openclaw/extensions/memos-local-openclaw-plugin/
```

然后在 `~/.openclaw/openclaw.json` 里启用：

```json
{
  "plugins": {
    "entries": {
      "memos-local-openclaw-plugin": { "enabled": true }
    }
  }
}
```

---

## 配置

### 环境变量 (`~/.openclaw/.env`)

```bash
MEMOS_BASE_URL=http://127.0.0.1:8000/product
MEMOS_API_KEY=dummy-key-for-local-memos
OPENAI_API_KEY=dummy-key-for-local-memos
```

几点说明：

- 用 `127.0.0.1` 而不是 `localhost`，Node.js 的 fetch 在某些环境下会优先解析 IPv6 导致连不上
- 两个 API Key 填 dummy 值就行，本地 MemOS 不校验认证，但插件逻辑里会检查是否为空，不填会输出警告
- 如果你不在意警告日志，不配也能正常跑

### 插件参数（可选）

在 `openclaw.json` 里可以细调插件行为：

```json
{
  "plugins": {
    "entries": {
      "memos-local-openclaw-plugin": {
        "enabled": true,
        "config": {
          "userId": "openclaw-user",
          "memCubeId": "openclaw-user",
          "memoryLimitNumber": 5,
          "preferenceLimitNumber": 6,
          "maxMessageChars": 20000,
          "includeAssistant": true,
          "captureStrategy": "last_turn",
          "timeoutMs": 5000,
          "retries": 1
        }
      }
    }
  }
}
```

| 参数                      | 默认值               | 说明                                      |
| ----------------------- | ----------------- | --------------------------------------- |
| `userId`                | `"openclaw-user"` | 用户标识                                    |
| `memCubeId`             | 同 userId          | 记忆空间 ID，用于隔离不同用户的记忆                     |
| `memoryLimitNumber`     | 5                 | 回回时返回的记忆条数                              |
| `preferenceLimitNumber` | 6                 | 回回时返回的偏好条数                              |
| `maxMessageChars`       | 20000             | 单条消息截断长度                                |
| `includeAssistant`      | true              | 保存记忆时是否包含助手回复                           |
| `captureStrategy`       | `"last_turn"`     | `last_turn` 只存最后一轮，`full_session` 存整个会话 |
| `timeoutMs`             | 5000              | API 请求超时（毫秒）                            |
| `retries`               | 1                 | 请求失败重试次数                                |

---

## API 接口

### 搜索记忆 `POST /product/search`

```bash
curl -X POST http://127.0.0.1:8000/product/search \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "openclaw-user",
    "query": "用户的名字是什么",
    "top_k": 5,
    "mem_cube_id": "openclaw-user"
  }'
```

返回结构：

```json
{
  "code": 200,
  "data": {
    "memory_detail_list": [
      {
        "memory_key": "用户姓名",
        "memory_value": "张三",
        "create_time": "2026-02-11 10:30",
        "similarity": 0.95
      }
    ],
    "preference_detail_list": [
      {
        "preference": "用户喜欢使用 Linux 系统",
        "preference_type": "Implicit Preference"
      }
    ]
  }
}
```

### 添加记忆 `POST /product/add`

```bash
curl -X POST http://127.0.0.1:8000/product/add \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "openclaw-user",
    "messages": [
      {"role": "user", "content": "我叫张三，住在北京"},
      {"role": "assistant", "content": "你好张三，很高兴认识你"}
    ],
    "mem_cube_id": "openclaw-user",
    "source": "openclaw"
  }'
```

注意 `messages` 必须是数组，不能传 `JSON.stringify()` 后的字符串。

---

## 从云版迁移到本地版的改动

如果你是从云版插件改过来的，主要改了三处：

**1. API 密钥改为可选** (`lib/memos-cloud-api.js`)

云版强制要求 Bearer token，本地版改成有 key 就带、没有也能请求：

```javascript
const headers = { "Content-Type": "application/json" }
if (apiKey) {
  headers.Authorization = `Bearer ${apiKey}`
}
```

**2. messages 传原始数组** (`index.js`)

云版把 messages 做了 `JSON.stringify()`，本地 MemOS 2.0 要求直接传数组：

```javascript
// 云版（错误）
messages: JSON.stringify(messages)

// 本地版（正确）
messages: messages
```

**3. baseUrl 用 127.0.0.1**

避免 Node.js fetch 的 IPv6 解析问题。

---

## 故障排除

### `TypeError: fetch failed`

```
[memory] sync failed (session-start): TypeError: fetch failed
```

按顺序排查：

1. MemOS 跑起来了吗？`curl http://127.0.0.1:8000/product/health`
2. `.env` 里是不是写了 `localhost`？改成 `127.0.0.1`
3. `.env` 里有没有配 API Key？没配的话插件会跳过请求

### 搜索不到记忆

- 确认 `user_id` 一致——添加和搜索用的是同一个
- 确认之前确实添加过记忆（空库搜不到东西）
- 检查 Qdrant 状态：`curl http://localhost:6333/collections`

### 调试命令

```bash
# MemOS 日志
docker logs memos-container

# Qdrant 状态
curl http://localhost:6333/collections

# Neo4j 状态
curl http://localhost:7474

# 直接测试搜索
curl -X POST http://127.0.0.1:8000/product/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"openclaw-user","query":"测试","top_k":5}'
```

---

## 验证集成

跑通这三步就说明对接成功了：

```bash
# 1. 写入一条记忆
curl -X POST http://127.0.0.1:8000/product/add \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","messages":[{"role":"user","content":"我叫测试用户，我的爱好是编程"}]}'

# 2. 搜索刚才写入的记忆
curl -X POST http://127.0.0.1:8000/product/search \
  -H "Content-Type: application/json" \
  -d '{"user_id":"test_user","query":"测试用户的爱好是什么","top_k":5}'

# 3. 在 OpenClaw 里测试端到端
# 发送："记住，我喜欢用 Linux"
# 再问："我喜欢什么操作系统？"
# 看回复里有没有用到记忆
```

---

## 文件清单

| 文件                        | 说明                               |
| --------------------------- | ---------------------------------- |
| `~/.openclaw/.env`          | 环境变量（baseUrl、API Key）       |
| `~/.openclaw/openclaw.json` | OpenClaw 主配置，启用插件          |
| `lib/memos-cloud-api.js`    | API 通信层（搜索、添加、配置构建） |
| `index.js`                  | 插件入口，注册生命周期钩子         |

---

## 相关链接

- [MemOS-Cloud-OpenClaw-Plugin](https://github.com/mindcarver/MemOS-Cloud-OpenClaw-Plugin) — 插件仓库
- [MemOS](https://github.com/MemTensor/MemOS) — MemOS 源码
- [OpenClaw](https://github.com/openclaw/openclaw) — OpenClaw 文档
- [插件开发指南](./openclaw-plugin-guide.md)
