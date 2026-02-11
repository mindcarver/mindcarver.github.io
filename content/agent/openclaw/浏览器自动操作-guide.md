# OpenClaw 网页自动化控制完整指南
> 通过 OpenClaw + mcporter + chrome-devtools MCP 实现对浏览器的自动化控制

>

> 最后更新：2026-02-11


## 目录
1. [整体架构](#1-整体架构)

2. [前置条件](#2-前置条件)

3. [环境搭建](#3-环境搭建)

4. [配置详解](#4-配置详解)

5. [使用方式](#5-使用方式)

6. [可用操作一览](#6-可用操作一览)

7. [踩坑记录与解决方案](#7-踩坑记录与解决方案)

8. [实战示例](#8-实战示例)

9. [进阶：OpenClaw 内置 Browser 工具](#9-进阶openclaw-内置-browser-工具)

10. [两种方案对比](#10-两种方案对比)


## 1. 整体架构
```

用户 (Telegram/Web)

│

▼

OpenClaw Gateway (AI Agent)

│

├── 方案 A: mcporter CLI ──► chrome-devtools MCP Server ──► Chrome (你的浏览器)

│ │

│ mcporter daemon (保持连接)

│

└── 方案 B: OpenClaw 内置 browser 工具 ──► 独立 Chrome 实例 (profile=openclaw)

```
**方案 A（推荐）**：通过 mcporter 调用 chrome-devtools MCP，连接你正在使用的 Chrome 浏览器，直接操作你看到的页面。
**方案 B**：OpenClaw 自带的 browser 工具，会启动一个独立的 Chrome 实例（与你日常使用的 Chrome 分开）。


## 2. 前置条件
### 2.1 软件版本要求
| 组件 | 最低版本 | 当前版本 | 说明 |

|------|---------|---------|------|

| Node.js | v18+ | v22.22.0 | mcporter 和 chrome-devtools-mcp 的运行环境 |

| npm | v8+ | v10.9.4 | 包管理器 |

| Google Chrome | v144+ | v144.0.7559.133 | 需要 144+ 才支持 `--autoConnect` 模式 |

| OpenClaw | v2026.2+ | v2026.2.9 | AI Agent 框架 |

| mcporter | v0.7+ | v0.7.3 | MCP 客户端 CLI |

| chrome-devtools-mcp | v0.16+ | v0.17.0 | Chrome DevTools MCP Server |
### 2.2 安装
```bash

# mcporter（全局安装）

npm install -g mcporter
# chrome-devtools-mcp 不需要手动安装，mcporter 会通过 npx 自动拉取

```


## 3. 环境搭建
### 3.1 Chrome 开启 Remote Debugging
**这是最关键的一步。** 有两种方式：
#### 方式一：chrome://inspect（推荐，无需重启 Chrome）
1. 在 Chrome 地址栏输入 `chrome://inspect/#remote-debugging`

2. 点击页面上的启用按钮

3. 完成。Chrome 会在 `127.0.0.1:9222` 开启调试端口
> ⚠️ 注意：这种方式开启的 debugging server 的 HTTP 端点（如 `/json/version`）可能返回 404，但 autoConnect 模式不依赖 HTTP 端点，它通过 Chrome 的 user-data-dir 来发现和连接。
#### 方式二：命令行启动（需要重启 Chrome）
```bash

# macOS

/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --remote-debugging-port=9222
# 如果 Chrome 已经在运行，需要先完全退出再用这个命令启动

```
### 3.2 配置 mcporter
创建或编辑 `~/.mcporter/mcporter.json`：
```json

{

"mcpServers": {

"chrome-devtools": {

"command": "npx",

"args": [

"-y",

"chrome-devtools-mcp@latest",

"--autoConnect"

]

}

},

"imports": []

}

```
也可以用 CLI 命令创建：
```bash

mcporter config add chrome-devtools \

--command npx \

--arg -y \

--arg chrome-devtools-mcp@latest \

--arg --autoConnect \

--scope home

```
**关键参数说明：**
| 参数 | 作用 |

|------|------|

| `--autoConnect` | 自动连接已运行的 Chrome（Chrome 144+ 支持），不启动新实例 |

| `--browserUrl http://127.0.0.1:9222` | 通过 HTTP 端点连接（替代 autoConnect，适用于旧版 Chrome） |

| `--isolated` | 每次使用临时 profile 目录（避免锁冲突，但不保留浏览器状态） |

| `--headless` | 无头模式运行（无 UI） |

| `--userDataDir <path>` | 自定义 Chrome profile 目录 |
### 3.3 启动 mcporter daemon
```bash

# 启动 daemon（保持 chrome-devtools MCP server 进程常驻）

mcporter daemon start
# 检查状态

mcporter daemon status

# 输出示例：

# Daemon pid 97977 — socket: /Users/mac08/.mcporter/daemon/daemon-e13f27eeb8f9.sock

# - chrome-devtools: connected (last used 2026-02-11T10:45:38.188Z)
# 停止

mcporter daemon stop
# 重启

mcporter daemon restart

```
**为什么需要 daemon？**
- chrome-devtools-mcp 启动时会连接 Chrome，这个过程需要几秒

- 没有 daemon 时，每次 `mcporter call` 都会启动一个新的 MCP server 进程

- 如果上一个进程还没完全退出，新进程就会因为 Chrome profile 锁冲突而失败

- daemon 模式下，MCP server 进程保持运行，所有调用复用同一个连接
### 3.4 验证连接
```bash

# 列出 Chrome 中所有打开的页面

mcporter call chrome-devtools.list_pages
# 如果能看到你的 tab 列表，说明连接成功

```


## 4. 配置详解
### 4.1 mcporter 配置文件位置
| 路径 | 作用 | 优先级 |

|------|------|--------|

| `~/.mcporter/mcporter.json` | 全局配置（home scope） | 中 |

| `<project>/config/mcporter.json` | 项目级配置 | 高 |

| `~/.claude.json` | Claude Code 导入的配置 | 低（作为 import 源） |
mcporter 会自动从 `~/.claude.json`、`~/.config/opencode/opencode.json` 等编辑器配置中导入 MCP server 定义。如果在 `~/.mcporter/mcporter.json` 中定义了同名 server，会覆盖导入的配置。
### 4.2 lifecycle（生命周期）配置
mcporter 内置了对 chrome-devtools 的 keep-alive 支持：
```javascript

// mcporter 源码中的默认 keep-alive 列表

const DEFAULT_KEEP_ALIVE = new Set(['chrome-devtools', 'mobile-mcp', 'playwright']);

```
这意味着名为 `chrome-devtools` 的 server 会自动被 daemon 管理，无需额外配置 lifecycle。
如果需要手动控制：
```json

{

"mcpServers": {

"chrome-devtools": {

"command": "npx",

"args": ["-y", "chrome-devtools-mcp@latest", "--autoConnect"],

"lifecycle": "keep-alive"

}

}

}

```
也可以通过环境变量控制：
```bash

# 强制启用 keep-alive

export MCPORTER_KEEPALIVE=chrome-devtools
# 强制禁用 keep-alive

export MCPORTER_DISABLE_KEEPALIVE=chrome-devtools
# 所有 server 都启用

export MCPORTER_KEEPALIVE=*

```
### 4.3 OpenClaw 侧的配置
OpenClaw 通过 `exec` 工具调用 `mcporter call` 命令来操作浏览器。不需要在 OpenClaw 的配置中做额外设置，只要确保：
1. `mcporter` 在 PATH 中可用

2. mcporter daemon 正在运行

3. Chrome 已开启 remote debugging


## 5. 使用方式
### 5.1 通过 OpenClaw 对话
直接用自然语言告诉 OpenClaw：
```

"打开 http://localhost:8017/ 然后从左侧拖两个节点到画布上"

"帮我截个图看看当前页面"

"点击页面上的登录按钮"

"在搜索框里输入 xxx"

```
OpenClaw 会自动调用 mcporter 来执行操作。
### 5.2 通过命令行直接调用
```bash

# 列出页面

mcporter call chrome-devtools.list_pages
# 导航到 URL

mcporter call chrome-devtools.navigate_page --args '{"type":"url","url":"http://localhost:8017/"}'
# 获取页面快照（文本形式的 DOM 树）

mcporter call chrome-devtools.take_snapshot
# 截图

mcporter call chrome-devtools.take_screenshot --args '{"filePath":"./screenshot.png"}'
# 点击元素（uid 从 snapshot 获取）

mcporter call chrome-devtools.click --args '{"uid":"1_17"}'
# 拖拽

mcporter call chrome-devtools.drag --args '{"from_uid":"1_17","to_uid":"1_43"}'
# 输入文本

mcporter call chrome-devtools.fill --args '{"uid":"1_158","value":"hello world"}'
# 按键

mcporter call chrome-devtools.press_key --args '{"key":"Enter"}'
# 执行 JavaScript

mcporter call chrome-devtools.evaluate_script --args '{"function":"() => document.title"}'

```


## 6. 可用操作一览
chrome-devtools MCP 提供 26 个工具：
### 页面管理
| 工具 | 说明 |

|------|------|

| `list_pages` | 列出所有打开的页面 |

| `select_page` | 选择一个页面作为操作目标 |

| `new_page` | 打开新页面 |

| `close_page` | 关闭页面 |

| `navigate_page` | 导航（URL/前进/后退/刷新） |

| `resize_page` | 调整页面尺寸 |
### 页面交互
| 工具 | 说明 |

|------|------|

| `take_snapshot` | 获取页面的无障碍树快照（文本形式，包含 uid） |

| `take_screenshot` | 截图（支持全页/元素/指定格式） |

| `click` | 点击元素（支持双击） |

| `hover` | 悬停在元素上 |

| `drag` | 拖拽元素到另一个元素 |

| `fill` | 在输入框中填入文本 |

| `fill_form` | 批量填写表单 |

| `press_key` | 按键/组合键 |

| `upload_file` | 上传文件 |

| `wait_for` | 等待指定文本出现 |

| `handle_dialog` | 处理浏览器弹窗 |
### 开发者工具
| 工具 | 说明 |

|------|------|

| `evaluate_script` | 在页面中执行 JavaScript |

| `list_console_messages` | 列出控制台消息 |

| `get_console_message` | 获取特定控制台消息 |

| `list_network_requests` | 列出网络请求 |

| `get_network_request` | 获取特定网络请求详情 |
### 性能分析
| 工具 | 说明 |

|------|------|

| `performance_start_trace` | 开始性能追踪 |

| `performance_stop_trace` | 停止性能追踪 |

| `performance_analyze_insight` | 分析性能洞察 |
### 模拟
| 工具 | 说明 |

|------|------|

| `emulate` | 模拟网络/CPU/地理位置/UA/暗色模式/视口 |


## 7. 踩坑记录与解决方案
### 7.1 "browser already running" 错误
**症状：**

```

The browser is already running for /Users/mac08/.cache/chrome-devtools-mcp/chrome-profile.

Use --isolated to run multiple browser instances.

```
**原因：** chrome-devtools-mcp 默认会启动一个新的 Chrome 实例，使用固定的 profile 目录。如果上一个实例没有完全退出，新实例就会因为 `SingletonLock` 文件冲突而失败。
**解决方案：**
```bash

# 1. 杀掉所有残留进程

pkill -9 -f "chrome-devtools-mcp"

ps aux | grep "user-data-dir=.*chrome-devtools-mcp" | grep -v grep | awk '{print $2}' | xargs kill -9
# 2. 删除锁文件

rm -f ~/.cache/chrome-devtools-mcp/chrome-profile/SingletonLock
# 3. 使用 --autoConnect 模式（根本解决）

# 修改配置为 autoConnect，不再启动新 Chrome 实例

```
### 7.2 mcporter daemon 崩溃后无法重连
**症状：** daemon 停止后，后续调用全部失败。
**解决方案：**
```bash

# 完整重置流程

mcporter daemon stop

pkill -9 -f "chrome-devtools-mcp"

rm -f ~/.cache/chrome-devtools-mcp/chrome-profile/SingletonLock

mcporter daemon start

```
### 7.3 --stdio 模式下连续调用冲突
**症状：** 不用 daemon 时，第一次 `mcporter call` 成功，第二次就报 "browser already running"。
**原因：** 每次 `mcporter call --stdio` 都会启动一个新的 chrome-devtools-mcp 进程。第一次启动了 Chrome，第二次又想启动就冲突了。
**解决方案：** 必须使用 daemon 模式，让同一个 MCP server 进程处理所有调用。
### 7.4 autoConnect 连接不上
**可能原因：**

1. Chrome 版本低于 144

2. 没有在 `chrome://inspect/#remote-debugging` 中启用

3. Chrome 没有运行
**排查步骤：**
```bash

# 检查 Chrome 版本

/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome --version
# 检查是否有 debugging 端口在监听

lsof -iTCP -sTCP:LISTEN -P -n | grep -i chrome
# 尝试直接连接

curl -s http://127.0.0.1:9222/json/version

```
### 7.5 拖拽操作成功但节点重叠
**症状：** `drag` 返回成功，但多个节点堆叠在同一位置。
**原因：** 拖拽的目标位置（to_uid）是整个画布元素，所有节点都被放到了画布的同一个默认位置。
**解决方案：** 拖拽后通过 `evaluate_script` 调用画布的 API 来调整节点位置：
```bash

mcporter call chrome-devtools.evaluate_script --args '{

"function": "() => { const graph = ...; graph.update(node, { x: 400, y: 300 }); }"

}'

```


## 8. 实战示例
### 8.1 打开页面并拖拽节点
```bash

# 1. 确保 daemon 运行

mcporter daemon start
# 2. 导航到目标页面

mcporter call chrome-devtools.navigate_page \

--args '{"type":"url","url":"http://localhost:8017/"}'
# 3. 获取页面快照，找到元素 uid

mcporter call chrome-devtools.take_snapshot
# 4. 拖拽节点到画布

mcporter call chrome-devtools.drag \

--args '{"from_uid":"1_17","to_uid":"1_43"}'
# 5. 截图确认

mcporter call chrome-devtools.take_screenshot \

--args '{"filePath":"./result.png"}'

```
### 8.2 自动填写表单并提交
```bash

# 1. 快照获取表单元素 uid

mcporter call chrome-devtools.take_snapshot
# 2. 批量填写

mcporter call chrome-devtools.fill_form --args '{

"elements": [

{"uid": "1_10", "value": "username"},

{"uid": "1_12", "value": "password123"}

]

}'
# 3. 点击提交

mcporter call chrome-devtools.click --args '{"uid":"1_15"}'
# 4. 等待页面跳转

mcporter call chrome-devtools.wait_for --args '{"text":"Welcome"}'

```
### 8.3 监控页面数据
```bash

# 定时执行 JavaScript 获取页面数据

mcporter call chrome-devtools.evaluate_script --args '{

"function": "() => { return JSON.parse(localStorage.getItem(\"strategyData\")); }"

}'

```


## 9. 进阶：OpenClaw 内置 Browser 工具
OpenClaw 还有一个内置的 browser 工具，不依赖 mcporter。
### 使用场景
- 需要一个干净的、独立的浏览器环境

- 不想影响你正在使用的 Chrome

- 需要 OpenClaw 完全控制浏览器生命周期
### 基本操作
OpenClaw 在对话中会自动使用 browser 工具，支持：
- `browser start` — 启动独立 Chrome 实例

- `browser navigate` — 导航到 URL

- `browser snapshot` — 获取页面快照

- `browser screenshot` — 截图

- `browser act` — 执行操作（click/type/drag/evaluate 等）

- `browser stop` — 关闭浏览器
### 与 mcporter 方案的区别
browser 工具启动的是一个完全独立的 Chrome 实例（使用 `~/.openclaw/browser/openclaw/user-data` 作为 profile），与你日常使用的 Chrome 完全隔离。


## 10. 两种方案对比
| 特性 | 方案 A: mcporter + chrome-devtools | 方案 B: OpenClaw browser 工具 |

|------|-----------------------------------|-------------------------------|

| 连接方式 | 连接你已有的 Chrome | 启动独立 Chrome 实例 |

| 能看到你的 tab | ✅ 是 | ❌ 否 |

| 需要额外配置 | 需要配置 mcporter + daemon | 开箱即用 |

| 稳定性 | daemon 模式下稳定 | 稳定 |

| 操作丰富度 | 26 个工具（含性能分析、网络监控） | 基础操作（click/type/drag/evaluate） |

| 适合场景 | 操作你正在使用的页面、调试、监控 | 自动化测试、爬取、独立任务 |

| Chrome 版本要求 | 144+（autoConnect）或任意版本（browserUrl） | 任意版本 |


## 快速启动清单
```bash

# ✅ 一次性设置（只需做一次）

npm install -g mcporter

mcporter config add chrome-devtools --command npx --arg -y --arg chrome-devtools-mcp@latest --arg --autoConnect --scope home
# ✅ 每次使用前

# 1. 确保 Chrome 已打开

# 2. 在 Chrome 中访问 chrome://inspect/#remote-debugging 并启用

# 3. 启动 daemon

mcporter daemon start
# ✅ 验证

mcporter call chrome-devtools.list_pages

# 应该能看到你所有打开的 tab
# 🎉 开始使用

# 直接在 OpenClaw 对话中用自然语言描述你想做的操作

```
