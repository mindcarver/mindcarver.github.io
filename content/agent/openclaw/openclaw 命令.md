# OpenClaw CLI 命令参考文档

OpenClaw 是一个强大的 AI 助手框架,提供了丰富的命令行工具来管理各种功能。

## 目录

- [Gateway 命令](#gateway-命令)

- [Channels 命令](#channels-命令)

- [Models 命令](#models-命令)

- [Config 命令](#config-命令)

- [Node 命令](#node-命令)

- [Browser 命令](#browser-命令)

- [Memory 命令](#memory-命令)

- [Logs 命令](#logs-命令)

- [Skills 命令](#skills-命令)

- [Security 命令](#security-命令)

- [Cron 命令](#cron-命令)

- [Approvals 命令](#approvals-命令)

- [ACP 命令](#acp-命令)

---

## Gateway 命令

Gateway 是 OpenClaw 的核心 WebSocket 服务,负责协调所有组件。

### gateway run

运行 WebSocket Gateway (前台)

```bash

openclaw gateway run

```

**用途:** 启动 Gateway 服务,通常在开发或调试时使用前台模式。

---

### gateway status

显示 Gateway 服务状态并探测 Gateway

```bash

openclaw gateway status

openclaw gateway status --url ws://localhost:18789

openclaw gateway status --probe

openclaw gateway status --deep --json

```

**选项:**

- `--url <url>` - Gateway WebSocket URL (默认使用 config/remote/local)

- `--token <token>` - Gateway token (如需要)

- `--password <password>` - Gateway password (密码认证)

- `--timeout <ms>` - 超时时间(默认: 10000)

- `--no-probe` - 跳过 RPC 探测

- `--deep` - 扫描系统级服务

- `--json` - 输出 JSON

**用途:** 查看 Gateway 是否正常运行,以及连接状态、健康检查等信息。

---

### gateway install

安装 Gateway 服务 (launchd/systemd/schtasks)

```bash

openclaw gateway install

openclaw gateway install --port 18789 --runtime bun

openclaw gateway install --token my-token --force

```

**选项:**

- `--port <port>` - Gateway 端口

- `--runtime <runtime>` - 守护进程运行时 (node|bun),默认: node

- `--token <token>` - Gateway token (token 认证)

- `--force` - 已安装时重新安装/覆盖

- `--json` - 输出 JSON

**用途:** 将 Gateway 安装为系统服务,开机自启动。

---

### gateway uninstall

卸载 Gateway 服务

```bash

openclaw gateway uninstall

```

**用途:** 移除已安装的 Gateway 系统服务。

---

### gateway start

启动 Gateway 服务

```bash

openclaw gateway start

```

**用途:** 启动已安装的 Gateway 服务(后台运行)。

---

### gateway stop

停止 Gateway 服务

```bash

openclaw gateway stop

```

**用途:** 停止运行的 Gateway 服务。

---

### gateway restart

重启 Gateway 服务

```bash

openclaw gateway restart

```

**用途:** 重启 Gateway 服务以应用配置更改。

---

### gateway call

调用 Gateway 方法

```bash

openclaw gateway call health

openclaw gateway call status --params '{"verbose":true}'

openclaw gateway call system.presence --params '{"agent":"main"}'

```

**选项:**

- `--params <json>` - JSON 对象字符串作为参数(默认: {})

- `--json` - 输出 JSON

**用途:** 直接调用 Gateway RPC 方法,用于调试和高级操作。

---

### gateway usage-cost

从会话日志中获取使用成本摘要

```bash

openclaw gateway usage-cost

openclaw gateway usage-cost --days 7

```

**选项:**

- `--days <days>` - 包含的天数(默认: 30)

- `--json` - 输出 JSON

**用途:** 查看最近的 API 使用成本和 token 统计。

---

### gateway health

获取 Gateway 健康状态

```bash

openclaw gateway health

```

**用途:** 检查 Gateway 各个通道和服务的健康状态。

---

### gateway probe

显示 Gateway 可达性 + 发现 + 健康 + 状态摘要 (本地 + 远程)

```bash

openclaw gateway probe

openclaw gateway probe --ssh user@remote-host

openclaw gateway probe --url ws://gateway.example.com:18789

openclaw gateway probe --ssh-auto --timeout 5000

```

**选项:**

- `--url <url>` - 显式指定 Gateway WebSocket URL(仍然探测 localhost)

- `--ssh <target>` - 远程 Gateway 隧道的 SSH 目标 (user@host 或 user@host:port)

- `--ssh-identity <path>` - SSH 身份文件路径

- `--ssh-auto` - 尝试从 Bonjour 发现推导 SSH 目标

- `--token <token>` - Gateway token(应用于所有探测)

- `--password <password>` - Gateway password(应用于所有探测)

- `--timeout <ms>` - 总体探测预算(默认: 3000)

- `--json` - 输出 JSON

**用途:** 全面检查本地和远程 Gateway 的可用性、连接状态和健康情况。

---

### gateway discover

通过 Bonjour 发现 Gateway (本地 + 宽域如果配置)

```bash

openclaw gateway discover

openclaw gateway discover --timeout 3000

openclaw gateway discover --json

```

**选项:**

- `--timeout <ms>` - 每个命令超时时间(默认: 2000)

- `--json` - 输出 JSON

**用途:** 发现网络中可用的 Gateway 实例。

---

## Channels 命令

管理聊天渠道账号,支持多种平台如 WhatsApp、Telegram、Discord、Slack 等。

### channels list

列出已配置的渠道 + 认证配置

```bash

openclaw channels list

openclaw channels list --json

openclaw channels list --no-usage

```

**选项:**

- `--no-usage` - 跳过模型提供商使用/配额快照

- `--json` - 输出 JSON

**用途:** 查看所有已添加的渠道及其配置状态。

---

### channels status

显示 Gateway 渠道状态(本地深度检测使用 `status --deep`)

```bash

openclaw channels status

openclaw channels status --probe --timeout 15000

openclaw channels status --json

```

**选项:**

- `--probe` - 探测渠道凭据

- `--timeout <ms>` - 超时时间(默认: 10000)

- `--json` - 输出 JSON

**用途:** 检查各渠道的连接状态和可用性。

---

### channels capabilities

显示提供商能力 (intents/scopes + 支持的功能)

```bash

openclaw channels capabilities

openclaw channels capabilities --channel discord --account my-bot

openclaw channels capabilities --target discord:channel-id

```

**选项:**

- `--channel <name>` - 渠道名称

- `--account <id>` - 账户 ID(需要 --channel)

- `--target <dest>` - 权限审核的目标渠道 (Discord channel:<id>)

- `--timeout <ms>` - 超时时间(默认: 10000)

- `--json` - 输出 JSON

**用途:** 查看渠道支持的权限、意图和功能。

---

### channels resolve

将渠道/用户名解析为 ID

```bash

openclaw channels resolve @username

openclaw channels resolve @username --channel telegram

openclaw channels resolve group-name --kind group --account my-account

```

**选项:**

- `--channel <name>` - 渠道名称

- `--account <id>` - 账户 ID

- `--kind <kind>` - 目标类型 (auto|user|group,默认: auto)

- `--json` - 输出 JSON

**用途:** 将用户名或群组名转换为系统 ID。

---

### channels logs

从 Gateway 日志文件显示最近的渠道日志

```bash

openclaw channels logs

openclaw channels logs --channel whatsapp --lines 500

openclaw channels logs --channel all --json

```

**选项:**

- `--channel <name>` - 渠道名称(默认: all)

- `--lines <n>` - 行数(默认: 200)

- `--json` - 输出 JSON

**用途:** 查看特定渠道的日志,用于调试连接问题。

---

### channels add

添加或更新渠道账号

```bash

openclaw channels add --channel telegram --token BOT_TOKEN

openclaw channels add --channel discord --bot-token xoxb-...

openclaw channels add --channel slack --bot-token xoxb-... --app-token xapp-...

openclaw channels add --channel signal --signal-number +15551234567

openclaw channels add --channel whatsapp --auth-dir /path/to/auth

```

**选项:**

- `--channel <name>` - 渠道名称

- `--account <id>` - 账户 ID(默认账户)

- `--name <name>` - 账户的显示名称

- `--token <token>` - Bot token (Telegram/Discord)

- `--token-file <path>` - Bot token 文件 (Telegram)

- `--bot-token <token>` - Slack bot token (xoxb-...)

- `--app-token <token>` - Slack app token (xapp-...)

- `--signal-number <e164>` - Signal 账号 (E.164 格式)

- `--cli-path <path>` - CLI 路径 (signal-cli 或 imsg)

- `--db-path <path>` - iMessage 数据库路径

- `--service <service>` - iMessage 服务 (imessage|sms|auto)

- `--region <region>` - iMessage 区域 (SMS)

- `--auth-dir <path>` - WhatsApp 认证目录覆盖

- `--http-url <url>` - Signal HTTP daemon 基础 URL

- `--http-host <host>` - Signal HTTP 主机

- `--http-port <port>` - Signal HTTP 端口

- `--webhook-path <path>` - Webhook 路径 (Google Chat/BlueBubbles)

- `--webhook-url <url>` - Google Chat webhook URL

- `--audience-type <type>` - Google Chat 受众类型 (app-url|project-number)

- `--audience <value>` - Google Chat 受众值 (app URL 或项目编号)

- `--homeserver <url>` - Matrix homeserver URL

- `--user-id <id>` - Matrix 用户 ID

- `--access-token <token>` - Matrix 访问令牌

- `--password <password>` - Matrix 密码

- `--device-name <name>` - Matrix 设备名称

- `--initial-sync-limit <n>` - Matrix 初始同步限制

- `--ship <ship>` - Tlon ship 名称 (~sampel-palnet)

- `--url <url>` - Tlon ship URL

- `--code <code>` - Tlon 登录码

- `--group-channels <list>` - Tlon 群组渠道(逗号分隔)

- `--dm-allowlist <list>` - Tlon DM 白名单(逗号分隔的 ships)

- `--auto-discover-channels` - Tlon 自动发现群组渠道

- `--no-auto-discover-channels` - 禁用 Tlon 自动发现

- `--use-env` - 使用环境令牌(仅默认账户)

**用途:** 添加新的聊天渠道配置,使 OpenClaw 能够与该平台通信。

---

### channels remove

禁用或删除渠道账号

```bash

openclaw channels remove --channel telegram

openclaw channels remove --channel discord --account my-bot

openclaw channels remove --channel whatsapp --delete

```

**选项:**

- `--channel <name>` - 渠道名称

- `--account <id>` - 账户 ID(默认账户)

- `--delete` - 删除配置条目(无提示)

**用途:** 移除或禁用已配置的渠道账号。

---

### channels login

链接渠道账号(如果支持)

```bash

openclaw channels login --channel whatsapp

openclaw channels login --channel telegram --verbose

```

**选项:**

- `--channel <channel>` - 渠道别名(默认: whatsapp)

- `--account <id>` - 账户 ID (accountId)

- `--verbose` - 详细连接日志

**用途:** 启动 OAuth 或其他链接流程以连接到渠道。

---

### channels logout

登出渠道会话(如果支持)

```bash

openclaw channels logout --channel whatsapp

openclaw channels logout --account my-account

```

**选项:**

- `--channel <channel>` - 渠道别名(默认: whatsapp)

- `--account <id>` - 账户 ID (accountId)

**用途:** 断开渠道连接并清除会话。

---

## Models 命令

模型发现、扫描和配置管理。

### models list

列出模型(默认显示已配置)

```bash

openclaw models list

openclaw models list --all

openclaw models list --local --provider anthropic

openclaw models list --json --plain

```

**选项:**

- `--all` - 显示完整模型目录

- `--local` - 过滤本地模型

- `--provider <name>` - 按提供商过滤

- `--json` - 输出 JSON

- `--plain` - 纯文本输出

**用途:** 浏览可用的 AI 模型,查看配置状态。

---

### models status

显示已配置模型状态

```bash

openclaw models status

openclaw models status --json --plain

openclaw models status --probe --probe-provider anthropic

openclaw models status --check

```

**选项:**

- `--json` - 输出 JSON

- `--plain` - 纯文本输出

- `--check` - 如果认证过期/已过期则非零退出 (1=过期/缺失, 2=即将过期)

- `--probe` - 探测已配置的提供商认证(实时)

- `--probe-provider <name>` - 仅探测单个提供商

- `--probe-profile <id>` - 仅探测特定认证配置 ID (可重复或逗号分隔)

- `--probe-timeout <ms>` - 每次探测超时时间

- `--probe-concurrency <n>` - 并发探测数

- `--probe-max-tokens <n>` - 探测最大 token 数(尽力而为)

- `--agent <id>` - 要检查的代理 ID

**用途:** 查看当前模型配置,认证状态和可用性。

---

### models set

设置默认模型

```bash

openclaw models set claude-3-5-sonnet-20241022

openclaw models set gpt-4o

```

**参数:**

- `<model>` - 模型 ID 或别名

**用途:** 设置 AI 助手使用的默认文本模型。

---

### models set-image

设置图片模型

```bash

openclaw models set-image dall-e-3

openclaw models set-image stable-diffusion-xl

```

**参数:**

- `<model>` - 模型 ID 或别名

**用途:** 设置 AI 助手使用的默认图片生成模型。

---

### models aliases list

列出模型别名

```bash

openclaw models aliases list

openclaw models aliases list --json

```

**选项:**

- `--json` - 输出 JSON

- `--plain` - 纯文本输出

**用途:** 查看所有自定义的模型别名。

---

### models aliases add

添加或更新模型别名

```bash

openclaw models aliases add fast gpt-3.5-turbo

openclaw models aliases add claude claude-3-opus-20240229

```

**参数:**

- `<alias>` - 别名名称

- `<model>` - 模型 ID 或别名

**用途:** 创建模型的简短别名,方便快速引用。

---

### models aliases remove

移除模型别名

```bash

openclaw models aliases remove fast

```

**参数:**

- `<alias>` - 别名名称

**用途:** 删除不再使用的模型别名。

---

### models fallbacks list

列出备用模型

```bash

openclaw models fallbacks list

openclaw models fallbacks list --json

```

**选项:**

- `--json` - 输出 JSON

- `--plain` - 纯文本输出

**用途:** 查看配置的备用模型列表,主模型不可用时自动切换。

---

### models fallbacks add

添加备用模型

```bash

openclaw models fallbacks add gpt-4o-mini

openclaw models fallbacks add claude-3-haiku

```

**参数:**

- `<model>` - 模型 ID 或别名

**用途:** 添加模型到备用列表,当主模型失败时自动回退。

---

### models fallbacks remove

移除备用模型

```bash

openclaw models fallbacks remove gpt-4o-mini

```

**参数:**

- `<model>` - 模型 ID 或别名

**用途:** 从备用列表中移除指定模型。

---

### models fallbacks clear

清除所有备用模型

```bash

openclaw models fallbacks clear

```

**用途:** 清空备用模型列表,禁用自动回退功能。

---

### models image-fallbacks list

列出图片备用模型

```bash

openclaw models image-fallbacks list

```

**选项:**

- `--json` - 输出 JSON

- `--plain` - 纯文本输出

**用途:** 查看图片生成任务的备用模型列表。

---

### models image-fallbacks add

添加图片备用模型

```bash

openclaw models image-fallbacks add dall-e-2

```

**参数:**

- `<model>` - 模型 ID 或别名

**用途:** 添加图片生成备用模型。

---

### models image-fallbacks remove

移除图片备用模型

```bash

openclaw models image-fallbacks remove dall-e-2

```

**参数:**

- `<model>` - 模型 ID 或别名

**用途:** 从图片备用列表中移除模型。

---

### models image-fallbacks clear

清除所有图片备用模型

```bash

openclaw models image-fallbacks clear

```

**用途:** 清空图片备用模型列表。

---

### models scan

扫描 OpenRouter 免费模型以查找工具 + 图片功能

```bash

openclaw models scan

openclaw models scan --min-params 7 --provider anthropic

openclaw models scan --yes --set-default --set-image

openclaw models scan --no-probe --json

```

**选项:**

- `--min-params <b>` - 最小参数大小(十亿)

- `--max-age-days <days>` - 跳过超过 N 天的模型

- `--provider <name>` - 按提供商前缀过滤

- `--max-candidates <n>` - 最大备用候选数(默认: 6)

- `--timeout <ms>` - 每次探测超时时间

- `--concurrency <n>` - 探测并发数

- `--no-probe` - 跳过实时探测;仅列出免费候选

- `--yes` - 无提示接受默认值

- `--no-input` - 禁用提示(使用默认值)

- `--set-default` - 设置 agents.defaults.model 为第一个选择

- `--set-image` - 设置 agents.defaults.imageModel 为第一个图片选择

- `--json` - 输出 JSON

**用途:** 自动发现适合的免费或低成本 AI 模型并配置。

---

### models auth add

交互式认证助手 (setup-token 或 paste token)

```bash

openclaw models auth add

```

**用途:** 引导式添加模型提供商认证信息。

---

### models auth login

运行提供商插件认证流程 (OAuth/API key)

```bash

openclaw models auth login --provider anthropic

openclaw models auth login --provider openai --method api-key --set-default

```

**选项:**

- `--provider <id>` - 插件注册的提供商 ID

- `--method <id>` - 提供商认证方法 ID

- `--set-default` - 应用提供商的默认模型推荐

**用途:** 通过 OAuth 或 API 密钥登录到模型提供商。

---

### models auth setup-token

运行提供商 CLI 以创建/同步 token (需要 TTY)

```bash

openclaw models auth setup-token --provider anthropic

openclaw models auth setup-token --provider openai --yes

```

**选项:**

- `--provider <name>` - 提供商 ID (默认: anthropic)

- `--yes` - 跳过确认

**用途:** 使用提供商的 CLI 工具管理 API 令牌。

---

### models auth paste-token

将 token 粘贴到 auth-profiles.json 并更新配置

```bash

openclaw models auth paste-token --provider anthropic

openclaw models auth paste-token --provider openai --profile-id openai:manual --expires-in 365d

```

**选项:**

- `--provider <name>` - 提供商 ID (例如 anthropic)

- `--profile-id <id>` - 认证配置 ID (默认: <provider>:manual)

- `--expires-in <duration>` - 可选过期时间(例如 365d, 12h),存储为绝对 expiresAt

**用途:** 直接粘贴 API token 到认证配置文件。

---

### models auth login-github-copilot

通过 GitHub 设备流程登录 GitHub Copilot (需要 TTY)

```bash

openclaw models auth login-github-copilot

openclaw models auth login-github-copilot --profile-id github-copilot:custom --yes

```

**选项:**

- `--profile-id <id>` - 认证配置 ID (默认: github-copilot:github)

- `--yes` - 无提示覆盖现有配置

**用途:** 为 GitHub Copilot 设置认证。

---

### models auth order get

显示每代理认证配置顺序覆盖 (来自 auth-profiles.json)

```bash

openclaw models auth order get --provider anthropic

openclaw models auth order get --provider anthropic --agent main --json

```

**选项:**

- `--provider <name>` - 提供商 ID (例如 anthropic)

- `--agent <id>` - 代理 ID (默认: 已配置的默认代理)

- `--json` - 输出 JSON

**用途:** 查看特定代理的模型认证使用顺序。

---

### models auth order set

设置每代理认证配置顺序覆盖 (锁定轮换到此列表)

```bash

openclaw models auth order set --provider anthropic anthropic:default anthropic:premium

openclaw models auth order set --provider openai openai:gpt4 openai:gpt35 --agent my-agent

```

**选项:**

- `--provider <name>` - 提供商 ID (例如 anthropic)

- `--agent <id>` - 代理 ID (默认: 已配置的默认代理)

- `<profileIds...>` - 认证配置 ID (例如 anthropic:default)

**用途:** 为特定代理指定模型认证配置的固定使用顺序。

---

### models auth order clear

清除每代理认证配置顺序覆盖 (回退到 config/round-robin)

```bash

openclaw models auth order clear --provider anthropic

openclaw models auth order clear --provider openai --agent my-agent

```

**选项:**

- `--provider <name>` - 提供商 ID (例如 anthropic)

- `--agent <id>` - 代理 ID (默认: 已配置的默认代理)

**用途:** 移除认证顺序覆盖,恢复默认轮换行为。

---

## Config 命令

配置管理工具。不带子命令运行时启动配置向导。

### config (无子命令)

启动配置向导

```bash

openclaw config

openclaw config --section gateway --section channels

```

**选项:**

- `--section <section>` - 配置向导节(可重复),无子命令时使用

**用途:** 交互式配置 OpenClaw 的各项设置。

---

### config get

通过点路径获取配置值

```bash

openclaw config get agents.defaults.model

openclaw config get gateway.port --json

openclaw config get channels.whatsapp.useEnv

```

**选项:**

- `--json` - 输出 JSON

**用途:** 读取特定配置项的值,支持嵌套路径访问。

---

### config set

通过点路径设置配置值

```bash

openclaw config set agents.defaults.model claude-3-5-sonnet

openclaw config set gateway.mode local

openclaw config set "channels.whatsapp.accounts[0].name" MyWhatsApp

openclaw config set agents.defaults.maxTokens 4096 --json

```

**选项:**

- `--json` - 解析值为 JSON5(必需)

**用途:** 修改配置文件中的特定设置值。

---

### config unset

通过点路径移除配置值

```bash

openclaw config unset agents.defaults.model

openclaw config unset channels.whatsapp.useEnv

```

**用途:** 从配置中删除指定的配置项。

---

## Node 命令

运行无头节点主机 (system.run/system.which)

### node run

运行无头节点主机 (前台)

```bash

openclaw node run

openclaw node run --host 192.168.1.100 --port 18789

openclaw node run --node-id my-node --display-name "My Node"

```

**选项:**

- `--host <host>` - Gateway 主机

- `--port <port>` - Gateway 端口

- `--tls` - 对 gateway 连接使用 TLS

- `--tls-fingerprint <sha256>` - 预期 TLS 证书指纹 (sha256)

- `--node-id <id>` - 覆盖节点 ID (清除配对令牌)

- `--display-name <name>` - 覆盖节点显示名称

**用途:** 启动节点主机,连接到 Gateway 并注册为可用节点。

---

### node status

显示节点主机状态

```bash

openclaw node status

openclaw node status --json

```

**选项:**

- `--json` - 输出 JSON

**用途:** 检查节点主机的运行状态和连接信息。

---

### node install

安装节点主机服务 (launchd/systemd/schtasks)

```bash

openclaw node install

openclaw node install --host gateway.example.com --tls

openclaw node install --node-id remote-node --runtime bun

```

**选项:**

- `--host <host>` - Gateway 主机

- `--port <port>` - Gateway 端口

- `--tls` - 对 gateway 连接使用 TLS

- `--tls-fingerprint <sha256>` - 预期 TLS 证书指纹

- `--node-id <id>` - 覆盖节点 ID

- `--display-name <name>` - 覆盖节点显示名称

- `--runtime <runtime>` - 服务运行时 (node|bun),默认: node

- `--force` - 已安装时重新安装/覆盖

- `--json` - 输出 JSON

**用途:** 将节点主机安装为系统服务。

---

### node uninstall

卸载节点主机服务

```bash

openclaw node uninstall

```

**用途:** 移除已安装的节点主机系统服务。

---

### node stop

停止节点主机服务

```bash

openclaw node stop

```

**用途:** 停止运行的节点主机服务。

---

### node restart

重启节点主机服务

```bash

openclaw node restart

```

**用途:** 重启节点主机服务。

---

## Browser 命令

管理 OpenClaw 的专用浏览器 (Chrome/Chromium)。

### browser status

显示浏览器状态

```bash

openclaw browser status

openclaw browser status --browser-profile my-profile --json

```

**选项:**

- `--browser-profile <name>` - 浏览器配置名称(默认来自配置)

- `--json` - 输出机器可读 JSON

**用途:** 查看浏览器实例的运行状态、CDP 端口、配置等。

---

### browser start

启动浏览器(已运行时无操作)

```bash

openclaw browser start

openclaw browser start --browser-profile custom

```

**选项:**

- `--browser-profile <name>` - 浏览器配置名称

**用途:** 启动或确保浏览器进程正在运行。

---

### browser stop

停止浏览器(尽力而为)

```bash

openclaw browser stop

openclaw browser stop --browser-profile my-profile

```

**选项:**

- `--browser-profile <name>` - 浏览器配置名称

**用途:** 关闭浏览器进程。

---

### browser reset-profile

重置浏览器配置(移动到回收站)

```bash

openclaw browser reset-profile

openclaw browser reset-profile --browser-profile custom

```

**选项:**

- `--browser-profile <name>` - 浏览器配置名称

**用途:** 清除浏览器用户数据、缓存等。

---

### browser tabs

列出打开的标签页

```bash

openclaw browser tabs

openclaw browser tabs --json

```

**用途:** 查看浏览器中当前打开的所有标签页。

---

### browser tab new

打开新标签页 (about:blank)

```bash

openclaw browser tab new

```

**用途:** 在浏览器中创建新的空白标签页。

---

### browser tab select

按索引聚焦标签页 (从1开始)

```bash

openclaw browser tab select 2

```

**参数:**

- `<index>` - 标签页索引(从1开始)

**用途:** 切换到指定索引的标签页。

---

### browser tab close

按索引关闭标签页 (从1开始); 默认: 第一个标签页

```bash

openclaw browser tab close

openclaw browser tab close 3

```

**参数:**

- `[index]` - 标签页索引(从1开始,可选)

**用途:** 关闭指定或第一个标签页。

---

### browser open

在新标签页中打开 URL

```bash

openclaw browser open https://example.com

```

**参数:**

- `<url>` - 要打开的 URL

**用途:** 在浏览器中导航到指定 URL。

---

### browser focus

按 target id (或唯一前缀)聚焦标签页

```bash

openclaw browser focus tab-123

```

**参数:**

- `<targetId>` - 目标 ID 或唯一前缀

**用途:** 切换到具有特定 ID 的标签页。

---

### browser close

关闭标签页 (target id 可选)

```bash

openclaw browser close

openclaw browser close tab-456

```

**参数:**

- `[targetId]` - 目标 ID 或唯一前缀(可选)

**用途:** 关闭当前或指定 ID 的标签页。

---

### browser profiles

列出所有浏览器配置

```bash

openclaw browser profiles

```

**用途:** 查看所有已创建的浏览器配置及其状态。

---

### browser create-profile

创建新浏览器配置

```bash

openclaw browser create-profile --name my-profile

openclaw browser create-profile --name work --color #0066CC --cdp-url http://localhost:9222

```

**选项:**

- `--name <name>` - 配置名称(小写、数字、连字符) - 必需

- `--color <hex>` - 配置颜色(hex 格式,例如 #0066CC)

- `--cdp-url <url>` - 远程 Chrome 的 CDP URL (http/https)

- `--driver <driver>` - 配置驱动器 (openclaw|extension),默认: openclaw

**用途:** 创建新的浏览器配置,用于隔离不同的浏览环境。

---

### browser delete-profile

删除浏览器配置

```bash

openclaw browser delete-profile --name old-profile

```

**选项:**

- `--name <name>` - 要删除的配置名称 - 必需

**用途:** 删除不再需要的浏览器配置。

---

## Memory 命令

内存搜索工具,管理知识库和会话历史。

### memory status

显示内存搜索索引状态

```bash

openclaw memory status

openclaw memory status --agent main --deep

openclaw memory status --agent my-agent --index --force

openclaw memory status --json

```

**选项:**

- `--agent <id>` - 代理 ID (默认: 默认代理)

- `--json` - 输出 JSON

- `--deep` - 探测嵌入提供商可用性

- `--index` - 如果脏则重新索引(隐含 --deep)

- `--force` - 强制完全重新索引

- `--verbose` - 详细日志

**用途:** 查看内存索引的状态、索引文件数、嵌入模型等。

---

### memory index

重新索引内存文件

```bash

openclaw memory index

openclaw memory index --agent main --force

```

**选项:**

- `--agent <id>` - 代理 ID (默认: 默认代理)

- `--force` - 强制完全重新索引

- `--verbose` - 详细日志

**用途:** 重新构建内存搜索索引,添加新文件或更新已有内容。

---

### memory search

搜索内存文件

```bash

openclaw memory search "how to setup OpenClaw"

openclaw memory search "authentication" --max-results 5 --min-score 0.7

openclaw memory search "database" --agent my-agent --json

```

**选项:**

- `--agent <id>` - 代理 ID (默认: 默认代理)

- `--max-results <n>` - 最大结果数

- `--min-score <n>` - 最小分数

- `--json` - 输出 JSON

**用途:** 在知识库中搜索相关内容,使用向量检索找到最佳匹配。

---

## Logs 命令

通过 RPC 追踪 Gateway 文件日志。

### logs

通过 RPC 追踪 Gateway 文件日志

```bash

openclaw logs

openclaw logs --limit 1000 --follow

openclaw logs --interval 2000 --json

openclaw logs --plain --no-color

```

**选项:**

- `--limit <n>` - 最大返回行数(默认: 200)

- `--max-bytes <n>` - 最大读取字节数(默认: 250000)

- `--follow` - 跟踪日志输出

- `--interval <ms>` - 轮询间隔毫秒数(默认: 1000)

- `--json` - 发射 JSON 日志行

- `--plain` - 纯文本输出(无 ANSI 样式)

- `--no-color` - 禁用 ANSI 颜色

**用途:** 实时查看 Gateway 日志,用于调试和监控。

---

## Skills 命令

列出和检查可用的技能。

### skills (无子命令)

默认操作 - 列出技能

```bash

openclaw skills

```

**用途:** 显示所有可用的技能及其状态。

---

### skills list

列出所有可用技能

```bash

openclaw skills list

openclaw skills list --eligible --verbose

openclaw skills list --json

```

**选项:**

- `--json` - 输出为 JSON

- `--eligible` - 仅显示就绪(可使用)的技能

- `-v, --verbose` - 显示更多详细信息包括缺失要求

**用途:** 浏览可用的技能插件,查看哪些已启用、禁用或缺少依赖。

---

### skills info

显示有关技能的详细信息

```bash

openclaw skills info memory-search

openclaw skills info browser --json

```

**选项:**

- `--json` - 输出为 JSON

**用途:** 查看技能的详细描述、要求、安装选项等。

---

### skills check

检查哪些技能就绪 vs 缺少要求

```bash

openclaw skills check

openclaw skills check --json

```

**选项:**

- `--json` - 输出为 JSON

**用途:** 综合报告所有技能的状态,识别需要安装或配置的依赖项。

---

## Security 命令

安全工具(审核)。

### security audit

审核配置 + 本地状态以查找常见安全隐患

```bash

openclaw security audit

openclaw security audit --deep

openclaw security audit --fix

openclaw security audit --deep --fix --json

```

**选项:**

- `--deep` - 尝试实时 Gateway 探测(尽力而为)

- `--fix` - 应用安全修复(收紧默认值 + chmod 状态/配置)

- `--json` - 打印 JSON

**用途:** 扫描系统配置和文件权限,发现并修复安全漏洞。

---

## Cron 命令

管理 cron 任务 (通过 Gateway)。

### cron (无子命令)

默认操作 - 显示帮助

```bash

openclaw cron

```

**用途:** 显示 cron 命令的帮助信息。

---

### cron status

查看 cron 任务状态

```bash

openclaw cron status

```

**用途:** 查看计划任务的执行状态和下次运行时间。

---

### cron list

列出所有 cron 任务

```bash

openclaw cron list

```

**用途:** 显示所有配置的定时任务。

---

### cron add

添加新的 cron 任务

```bash

openclaw cron add "0 */6 * * *" my-task"

```

**参数:**

- `<cron-expression>` - cron 表达式

- `<command>` - 要执行的命令

**用途:** 创建新的定时任务。

---

### cron remove

移除 cron 任务

```bash

openclaw cron remove <task-id>

```

**用途:** 删除指定的定时任务。

---

### cron edit

编辑 cron 任务

```bash

openclaw cron edit <task-id>

```

**用途:** 修改现有的定时任务配置。

---

## Approvals 命令

管理执行批准(gateway 或节点主机)。

### approvals get

获取执行批准快照

```bash

openclaw approvals get

openclaw approvals get --node remote-node

openclaw approvals get --gateway --json

```

**选项:**

- `--node <node>` - 目标节点 ID/名称/IP

- `--gateway` - 强制 gateway 批准

- `--json` - 输出 JSON

**用途:** 查看当前的执行批准配置。

---

### approvals set

用 JSON 文件替换执行批准

```bash

openclaw approvals set --file approvals.json

openclaw approvals set --stdin

openclaw approvals set --node my-node --file ./approvals.json

```

**选项:**

- `--node <node>` - 目标节点 ID/名称/IP

- `--gateway` - 强制 gateway 批准

- `--file <path>` - 要上传的 JSON 文件路径

- `--stdin` - 从 stdin 读取 JSON

**用途:** 批量更新执行批准规则。

---

### approvals allowlist

编辑每代理允许列表

```bash

openclaw approvals allowlist

```

**用途:** 管理每个代理的命令/脚本执行白名单。

---

### approvals allowlist add

将 glob 模式添加到允许列表

```bash

openclaw approvals allowlist add "~/Projects/**/bin/rg"

openclaw approvals allowlist add --agent main --node <id|name|ip> "/usr/bin/uptime"

openclaw approvals allowlist add --agent "*" "/usr/bin/uname"

```

**选项:**

- `--node <node>` - 目标节点 ID/名称/IP

- `--gateway` - 强制 gateway 批准

- `--agent <id>` - 代理 ID (默认: "*")

**用途:** 允许特定命令或模式,无需手动批准。

---

### approvals allowlist remove

从允许列表中移除 glob 模式

```bash

openclaw approvals allowlist remove "~/Projects/**/bin/rg"

openclaw approvals allowlist remove --node my-node --agent my-agent "/usr/bin/uname"

```

**选项:**

- `--node <node>` - 目标节点 ID/名称/IP

- `--gateway` - 强制 gateway 批准

- `--agent <id>` - 代理 ID (默认: "*")

**用途:** 从白名单中移除命令模式。

---

## ACP 命令

运行由 Gateway 支持的 ACP bridge。

### acp (无子命令)

运行 ACP bridge

```bash

openclaw acp

```

**选项:**

- `--url <url>` - Gateway WebSocket URL (默认使用 gateway.remote.url 当配置时)

- `--token <token>` - Gateway token (如需要)

- `--password <password>` - Gateway password (如需要)

- `--session <key>` - 默认会话密钥 (例如 agent:main:main)

- `--session-label <label>` - 默认会话标签以解析

- `--require-existing` - 如果会话密钥/标签不存在则失败

- `--reset-session` - 首次使用前重置会话密钥

- `--no-prefix-cwd` - 不用工作目录作为前缀提示

- `--verbose, -v` - 详细日志到 stderr

**用途:** 启动 ACP bridge,连接本地 ACP 服务器到 Gateway。

---

### acp client

针对本地 ACP bridge 运行交互式 ACP 客户端

```bash

openclaw acp client

openclaw acp client --cwd ~/my-project --server-verbose --verbose

```

**选项:**

- `--cwd <dir>` - ACP 会话的工作目录

- `--server <command>` - ACP 服务器命令(默认: openclaw)

- `--server-args <args...>` - ACP 服务器的额外参数

- `--server-verbose` - 在 ACP 服务器上启用详细日志

- `--verbose, -v` - 详细客户端日志

**用途:** 启动交互式 ACP 客户端,用于直接与 OpenClaw 的 AI 功能交互。

---

## 全局选项

大多数命令支持以下全局选项:

- `--json` - 输出 JSON 格式,便于脚本解析

- `--verbose, -v` - 显示详细日志信息

- `--help, -h` - 显示帮助信息

- `--version, -V` - 显示版本号

## 文档链接

完整文档请访问: https://docs.openclaw.ai/

---

*最后更新: 2026-02-10*
