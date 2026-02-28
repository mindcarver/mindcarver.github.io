# Oh My OpenCode 使用指南

> 把单一的 AI 编程助手变成一个自律的 AI 开发军团

---

## 📚 目录

1. [核心概念](#核心概念)
2. [两种工作模式](#两种工作模式)
3. [核心特性](#核心特性)
4. [命令详解](#命令详解)
5. [Agent 系统](#agent-系统)
6. [技能系统](#技能系统)
7. [实用技巧](#实用技巧)
8. [配置文件](#配置文件)
9. [常见场景](#常见场景)
10. [安装与卸载](#安装与卸载)

---

## 核心概念

### 什么是 Oh My OpenCode？

Oh My OpenCode 是 OpenCode 的超级插件，通过多 Agent 协作系统，将单一的 AI 编程助手升级为具备高度自主性的 AI 开发军团。

### 核心魔法词：`ultrawork` / `ulw`

在提示词中加入 `ultrawork` 或 `ulw`，即可触发全自动工作模式：

| 能力 | 说明 |
|------|------|
| 🔍 自动探索 | 深度分析代码库结构和依赖关系 |
| 📋 自动计划 | 智能拆解任务，制定执行步骤 |
| 🤖 自动分配 | 调用专家 Agent 处理各自专长 |
| ✅ 自动验证 | 运行测试和诊断确保质量 |
| 🔄 持续迭代 | 任务未完成则自动优化直到 100% |

---

## 两种工作模式

### 模式一：Ultrawork 模式 (快速工作)

**适用场景**：快速任务、明确目标、信任 AI 自主决策

#### 基本语法

在提示词中加入 `"ulw"` 或 `"ultrawork"`：

```bash
# 功能开发
ulw 添加用户认证功能到我的 Next.js 应用

# Bug 修复
ulw 修复所有测试失败

# 代码重构
ulw 重构登录页面，添加更好的 UI

# 性能优化
ulw 优化数据库查询性能
```

#### 工作流程

```
用户输入 ulw 任务
       ↓
Sisyphus 接收，分析意图
       ↓
自动探索代码库模式
       ↓
并行调用专家 Agent
  (Hephaestus, Oracle, Librarian...)
       ↓
验证结果（测试、诊断）
       ↓
   未完成？继续迭代
       ↓
  100% 完成 → 结束
```

---

### 模式二：Prometheus 模式 (精密工作)

**适用场景**：复杂项目、关键生产变更、需要文档化决策

#### 执行步骤

**步骤 1：进入 Prometheus 模式**

```bash
# 在 OpenCode TUI 中按 Tab 键
# 进入 Prometheus 规划师模式
```

**步骤 2：描述工作**

```bash
我要为电商系统添加完整的支付流程，支持多种支付方式
```

**步骤 3：接受采访**

Prometheus 会通过一系列问题来明确需求：

```
Prometheus: 需要支持哪些支付方式？
你: 支付宝、微信支付、信用卡

Prometheus: 是否需要支付失败重试机制？
你: 是的，最多重试 3 次

Prometheus: 订单状态如何流转？
你: 待支付 → 已支付 → 发货中 → 已完成
```

**步骤 4：确认计划**

Prometheus 生成详细计划并保存到：

```
.sisyphus/plans/支付流程计划.md
```

计划包含：
- ✓ 任务列表
- ✓ 验收标准
- ✓ 风险评估
- ✓ 技术选型建议

**步骤 5：执行计划**

```bash
# 执行 Prometheus 生成的计划
/start-work

# 或指定特定计划
/start-work 支付流程计划
```

---

## 核心特性

### 🚪 IntentGate — 意图门

真正行动前先分析用户真实意图，避免字面误解。

```
用户输入 → IntentGate 分析 → 真实意图 → 合适的 Agent
         "优化登录"           "重写登录组件"    Frontend Architect
```

**核心能力：**
- 🔍 深入分析用户请求背后的真实需求
- 🎯 区分表面需求 vs 深层需求
- 🚫 避免字面理解导致的错误行动
- 📋 智能任务分类和路由

---

### 🔗 Hash-Anchored Edit Tool — 哈希锚定编辑

解决 "Harness Problem" 的核心创新。每行代码带有内容哈希标签：

```javascript
11#VK| function hello() {
22#XJ|   return "world";
33#MB| }
```

**核心优势：**
- ✅ 内容哈希验证每次修改
- 🚫 零 stale-line 错误
- 🔒 文件变更后自动拒绝编辑
- 📈 成功率提升：6.7% → 68.3%

**工作原理：**
```
1. Agent 读取文件 → 每行获得哈希标签 (LINE#ID)
2. 编辑时引用标签 → 验证内容是否匹配
3. 哈希不匹配 → 拒绝编辑，重新读取
4. 哈希匹配 → 安全应用修改
```

---

### 🛠️ LSP + AST-Grep — IDE 级精度

| 功能 | 命令 | 用途 |
|------|------|------|
| **LSP 重命名** | `lsp_rename` | 跨文件符号重命名 |
| **LSP 跳转** | `lsp_goto_definition` | 跳转到定义 |
| **LSP 查找引用** | `lsp_find_references` | 查找所有引用 |
| **LSP 诊断** | `lsp_diagnostics` | 预构建错误检查 |
| **AST-Grep** | `ast_grep` | 模式感知的代码搜索 |

**支持语言：** 25+ 种主流编程语言

---

### 🧠 Background Agents — 后台并行专家

同时发射 5+ 个专家 Agent 并行工作：

```
┌─────────────────────────────────────┐
│  Sisyphus (主控)                     │
└──────┬───────────────────────────────┘
       │
       ├─→ Explore (代码搜索) ──────┐
       ├─→ Librarian (文档查找) ────┤
       ├─→ Oracle (架构审查) ───────┤ → 并行执行
       ├─→ Hephaestus (代码实现) ───┤   上下文精简
       └─→ Metis (计划咨询) ─────────┘   结果汇总
```

**核心优势：**
- ⚡ 上下文保持精简
- 🔄 结果准备好后汇总
- 🚀 多倍效率提升

---

### 📚 Built-in MCPs — 内置 MCP 服务器

| MCP | 功能 | 用途 |
|-----|------|------|
| **Exa** | Web 搜索 | 实时信息检索 |
| **Context7** | 官方文档 | 技术文档查询 |
| **Grep.app** | GitHub 搜索 | 开源代码搜索 |

**始终在线**，无需额外配置。

---

### 🔌 Skill-Embedded MCPs — 技能嵌入式 MCP

MCP 服务器不再占用上下文窗口：

```
传统方式:               技能嵌入式方式:
┌─────────────┐         ┌─────────────┐
│ Agent Context │         │ Agent Context │
├─────────────┤         ├─────────────┤
│ Work Task    │         │ Work Task    │
│ + MCP Server │  ❌     │              │  ✅
│ (huge)       │         │ Skill → MCP  │
└─────────────┘         └─────────────┘
```

**核心优势：**
- 🎯 按需启动 MCP
- 📦 任务范围限定
- 🗑️ 完成后自动清理

---

### ✅ Todo Enforcer — 任务强制执行器

Agent 空闲？系统自动把它拉回任务：

```
Agent 进入空闲状态
       ↓
Todo Enforcer 检测到
       ↓
自动重新激活 Agent
       ↓
回到任务直到完成
```

**保证：** 任务必须完成，绝不半途而废。

---

### 💬 Comment Checker — 注释检查器

确保代码注释干净专业：

```
❌ AI 生成的冗余注释:
// This function returns the sum of two numbers
function add(a, b) { return a + b; }

✅ 经过 Comment Checker:
function add(a, b) { return a + b; }
```

**核心能力：**
- 🚫 移除 AI 生成的废话注释
- ✅ 保留有价值的说明
- 📖 让代码读起来像资深开发者写的

---

### 🖥️ Tmux Integration — 完整交互终端

Agent 停留在会话中，支持所有交互式工具：

| 功能 | 支持 |
|------|------|
| **REPLs** | Python, Node, IRB... |
| **Debuggers** | gdb, lldb... |
| **TUI Apps** | htop, vim, tmux... |
| **Interactive Shell** | 全部支持 |

---

### 🤖 Agent 分类映射

Agent 不选模型，选任务类型：

| Category | 用途 | 自动映射模型 |
|----------|------|-------------|
| `visual-engineering` | 前端、UI/UX、设计 | 创意模型 |
| `deep` | 自主研究 + 执行 | 推理模型 |
| `quick` | 单文件修改、拼写 | 快速模型 |
| `ultrabrain` | 复杂逻辑、架构决策 | 智慧模型 |

---

## 命令详解

### 核心命令列表

| 命令 | 功能 | 示例 |
|------|------|------|
| `ulw` / `ultrawork` | 全自动工作模式 | `ulw 添加认证` |
| `/start-work` | 执行 Prometheus 计划 | `/start-work` |
| `/init-deep` | 生成项目上下文文件 | `/init-deep` |
| `/ulw-loop` | Ultrawork 持续循环 | `/ulw-loop "构建 API"` |
| `/ralph-loop` | 自引用开发循环 | `/ralph-loop "重构模块"` |
| `/cancel-ralph` | 取消 Ralph Loop | `/cancel-ralph` |

---

### `/init-deep` - 深度初始化

在项目中生成层级化的 `AGENTS.md` 文件系统，实现智能上下文自动注入。

#### 核心原理

`/init-deep` 通过深度遍历项目目录结构，在每个关键层级创建 `AGENTS.md` 文件，形成分层的知识体系：

```
层级化上下文架构
┌─────────────────────────────────────────────┐
│ 项目根目录 AGENTS.md                         │
│ ├─ 项目整体架构                              │
│ ├─ 技术栈说明                                │
│ ├─ 全局编码规范                              │
│ └─ 目录结构说明                              │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ src/AGENTS.md                                │
│ ├─ 模块职责说明                              │
│ ├─ 模块间依赖关系                            │
│ └─ src 级别的编码约定                        │
└─────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────┐
│ src/components/AGENTS.md                     │
│ ├─ 组件设计模式                              │
│ ├─ 组件间通信约定                            │
│ └─ 组件级技术细节                            │
└─────────────────────────────────────────────┘
```

#### 工作流程

```
1. 目录遍历
   ├─ 从项目根目录开始
   ├─ 递归扫描所有子目录
   └─ 识别需要创建 AGENTS.md 的层级

2. 结构分析
   ├─ 分析目录结构和模块组织
   ├─ 识别代码模式和框架使用
   └─ 提取项目元数据（依赖、配置等）

3. 内容生成
   ├─ 为每个层级生成适当的 AGENTS.md
   ├─ 父级文件包含宏观信息
   ├─ 子级文件包含具体细节
   └─ 保持层级间的一致性

4. 上下文关联
   ├─ 子级文件继承父级上下文
   ├─ 建立层级的引用关系
   └─ 确保信息不重复不遗漏
```

#### 用法

```bash
# 基础用法 - 在当前项目生成层级化 AGENTS.md
/init-deep

# 强制创建新文件（覆盖已有文件）
/init-deep --create-new

# 限制遍历深度（默认无限）
/init-deep --max-depth=3

# 排除特定目录
/init-deep --exclude=node_modules,dist,.git

# 仅更新已有文件，不创建新文件
/init-deep --update-only
```

#### 生成的文件结构示例

```
my-project/
├── AGENTS.md                          # 🌐 项目全局上下文
│   ├── 项目概述与目标
│   ├── 技术栈（Next.js + TypeScript + Tailwind）
│   ├── 架构模式（Monorepo / Microservices）
│   ├── 全局编码规范
│   └── 环境配置说明
│
├── src/
│   ├── AGENTS.md                      # 📦 模块级上下文
│   │   ├── src 目录职责
│   │   ├── 模块组织方式
│   │   ├── 模块间依赖关系
│   │   └── 数据流说明
│   │
│   ├── components/
│   │   ├── AGENTS.md                  # 🎨 组件级上下文
│   │   │   ├── 组件设计模式
│   │   │   ├── 状态管理方案
│   │   │   ├── 组件通信约定
│   │   │   └── UI 框架使用规范
│   │   │
│   │   ├── Button/
│   │   │   └── AGENTS.md              # 🔧 具体组件上下文
│   │   │       ├── Button 组件职责
│   │   │       ├── Props 接口定义
│   │   │       ├── 使用示例
│   │   │       └── 相关组件
│   │   │
│   │   └── Form/
│   │       └── AGENTS.md              # 🔧 具体组件上下文
│   │           ├── Form 组件职责
│   │           ├── 表单验证规则
│   │           └── 数据提交流程
│   │
│   ├── lib/
│   │   └── AGENTS.md                  # ⚙️ 工具库上下文
│   │       ├── 通用工具函数
│   │       ├── API 封装
│   │       └── 数据处理逻辑
│   │
│   └── hooks/
│       └── AGENTS.md                  # 🪝 自定义 Hooks 上下文
│           ├── Hooks 设计模式
│           ├── 副作用处理规范
│           └── 性能优化约定
│
├── tests/
│   └── AGENTS.md                      # 🧪 测试上下文
│       ├── 测试策略
│       ├── 测试框架选择
│       ├── 覆盖率要求
│       └── Mock 数据管理
│
└── docs/
    └── AGENTS.md                      # 📚 文档上下文
        ├── 文档结构
        ├── 文档工具
        └── 文档生成流程
```

#### AGENTS.md 内容层级设计

| 层级 | 内容范围 | 粒度 | Agent 使用场景 |
|------|----------|------|----------------|
| **根目录** | 项目架构、技术栈、全局规范 | 高层级宏观 | 理解项目整体、跨模块任务 |
| **一级目录** | 模块职责、模块关系、数据流 | 模块级中观 | 单模块开发、模块间协作 |
| **二级目录** | 组件设计、接口定义、实现细节 | 组件级微观 | 具体功能实现、Bug 修复 |
| **叶子目录** | 具体实现、使用示例、注意事项 | 细节级精确 | 精确修改、性能优化 |

#### Agent 自动上下文读取机制

当 Agent 执行任务时，会自动读取相关层级的 `AGENTS.md`：

```
用户任务: "修改 Button 组件的点击行为"
       ↓
Sisyphus 分析任务范围 → 涉及 src/components/Button
       ↓
自动读取上下文:
  1. 根目录 AGENTS.md → 了解项目全局规范
  2. src/AGENTS.md → 理解组件模块组织方式
  3. src/components/AGENTS.md → 掌握组件设计模式
  4. src/components/Button/AGENTS.md → 精确理解 Button 组件
       ↓
基于完整上下文执行任务 → 修改符合项目规范和设计模式
```

#### 最佳实践

**✅ 何时使用 `/init-deep`**

| 场景 | 原因 |
|------|------|
| 新项目初始化 | 建立完整的知识体系 |
| 接手遗留项目 | 快速理解代码结构 |
| 大型重构前 | 记录当前架构作为参考 |
| 团队协作项目 | 统一项目认知和规范 |
| 复杂项目结构 | 多模块/微服务架构 |

**❌ 何时不需要 `/init-deep`**

| 场景 | 原因 |
|------|------|
| 简单单文件脚本 | 层级化意义不大 |
| 临时测试项目 | 不值得投入 |
| 明确的单模块项目 | 单个 AGENTS.md 足够 |
| 个人熟悉的小项目 | 已经了解项目结构 |

#### 不同项目类型的 init-deep 策略

**前端项目**
```bash
# 推荐深度：3-4 层
/init-deep --max-depth=4

# 重点目录：
# - src/components (组件级上下文)
# - src/lib (工具函数)
# - src/hooks (自定义 Hooks)
# - public/assets (静态资源)
```

**后端项目**
```bash
# 推荐深度：3-5 层
/init-deep --max-depth=5

# 重点目录：
# - src/controllers (控制器逻辑)
# - src/services (业务逻辑)
# - src/models (数据模型)
# - src/routes (路由定义)
```

**Monorepo 项目**
```bash
# 推荐深度：不限（默认）
/init-deep

# 为每个 package 单独生成 AGENTS.md
# packages/
# ├── package-a/AGENTS.md
# ├── package-b/AGENTS.md
# └── shared/AGENTS.md
```

#### 维护建议

```bash
# 项目结构重大变更后重新生成
/init-deep --create-new

# 定期更新以保持同步
/init-deep --update-only

# 排除不需要的目录
/init-deep --exclude=node_modules,dist,build,.next
```

#### 手动增强 AGENTS.md

自动生成的 `AGENTS.md` 是基础框架，可以手动补充：

```markdown
<!-- 自动生成的内容 -->

## 手动补充内容

### 团队约定
- 代码审查要求
- Git 提交规范
- 发布流程说明

### 业务上下文
- 产品功能说明
- 业务术语表
- 用户场景描述

### 技术债务
- 已知问题列表
- 计划重构项
- 性能瓶颈点
```

#### 与 Agent 协作的优势

| 特性 | 单一 AGENTS.md | 层级化 AGENTS.md |
|------|----------------|------------------|
| 上下文精确度 | 可能过载 | 按需加载 |
| 更新维护成本 | 集中维护但易冲突 | 分散维护各自独立 |
| Agent 理解效率 | 需过滤无关信息 | 直接定位相关层级 |
| 大型项目适应性 | 文件过大难以维护 | 天然适合复杂结构 |

---

### `/ulw-loop` - 持续工作循环

让 AI 一直工作直到任务 100% 完成。

#### 用法

```bash
# 基础用法
/ulw-loop "实现完整的用户管理系统"

# 带最大迭代次数
/ulw-loop "重构所有旧代码" --max-iterations=50
```

---

### `/ralph-loop` - 自引用循环

更智能的循环，AI 会自我评估和改进。

#### 用法

```bash
# 开始循环
/ralph-loop "构建一个博客系统"

# 设置最大迭代次数
/ralph-loop "优化性能" --max-iterations=100

# 停止循环
/cancel-ralph
```

---

## Agent 系统

### 主 Agent

#### Sisyphus (西西弗斯) — 主指挥官

| 属性 | 描述 |
|------|------|
| 默认模型 | claude-opus-4-6 / kimi-k2.5 / glm-5 |
| 思考预算 | 32,000 tokens |
| 角色定位 | 主编排者 |

**核心特点：**
- 🎯 **意图门** — 真正行动前先分析用户真实意图
- 🔄 **编排大师** — 调度专家团队
- ⚡ **激进并行** — 同时发射多个 Agent 并行工作
- 📋 **Todo 强制执行** — 确保任务必须完成
- 🚫 **绝不半途而废** — 任务完成前绝不停止

**适合：**
- 作为所有任务的入口点
- 复杂多步骤任务的编排
- 代码实现、Bug 修复、功能开发

---

#### Hephaestus (赫菲斯托斯) — 自主深度工作者

| 属性 | 描述 |
|------|------|
| 默认模型 | gpt-5.3-codex |
| 角色定位 | 正牌工匠 |
| 运行模式 | 完全自主 |

**核心特点：**
- 🛠️ **自主执行** — 只需给目标，不给具体做法
- 🔍 **自动探索** — 自动理解代码库模式
- 🚫 **不需保姆** — 从头到尾独立执行
- 💪 **深度工作** — 专注于单一复杂任务

**适合：**
- 需要完全自主执行的复杂编码任务
- 大型重构或迁移任务

---

#### Prometheus (普罗米修斯) — 战略规划师

| 属性 | 描述 |
|------|------|
| 默认模型 | claude-opus-4-6 / kimi-k2.5 / glm-5 |
| 思考预算 | 32,000+ tokens |
| 角色定位 | 规划者，不是实现者 |

**核心特点：**
- 🗣️ **面试模式** — 像主管一样采访用户，深挖需求
- 📝 **生成工作计划** — 产出计划文件
- ⚠️ **只规划不实现** — 系统层面禁止修改代码
- 🔗 **集成研究 Agent** — 自动调用专家收集信息
- ✅ **Momus 审核** — 高精度模式下计划会被审核

**适合：**
- 动手前需要理清思路的复杂项目
- 大型功能设计前的需求分析
- 生成详细的实施计划

**触发方式：** `/start-work` 命令或按 `[Tab]` 键

---

#### Metis — 计划顾问

| 属性 | 描述 |
|------|------|
| 默认模型 | claude-opus-4-6 / kimi-k2.5 / glm-5 |
| 角色定位 | 计划顾问 |

**核心特点：**
- 📋 **计划咨询** — 审查和优化执行计划
- 🔄 **迭代改进** — 持续完善计划细节
- 🎯 **可行性评估** — 评估计划执行难度
- 📊 **资源评估** — 分析所需资源

**适合：**
- 审查 Prometheus 生成的计划
- 优化执行步骤
- 识别潜在风险

**调用：** Prometheus 自动调用，或 `@metis`

---

### 子 Agent (学科专家)

#### Oracle — 只读咨询顾问

| 属性 | 描述 |
|------|------|
| 成本 | 昂贵 (EXPENSIVE) |
| 权限 | 只读，不修改文件 |
| 模型选择 | 高质量推理模型 |

**核心特点：**
- 🧠 深度架构分析
- 🔍 只读咨询 — 不动手，只给建议
- 🛡️ 安全/性能审查
- 🐛 困难调试 — 2 次以上修复失败后咨询

**适合：**
- 复杂架构决策咨询
- 完成重要工作后的自我审查
- 困难 Bug 的诊断（多次尝试失败后）
- 多系统权衡分析

**调用：** `@oracle`

---

#### Librarian — 参考文献搜索员

| 属性 | 描述 |
|------|------|
| 成本 | 便宜 (CHEAP) |
| 搜索范围 | 外部资源 |
| 工具 | GitHub CLI, Context7, Web Search |

**核心特点：**
- 📚 搜索外部资源 — 官方文档、OSS 实现、API 示例
- 🔗 多仓库分析
- 🌐 最佳实践查找
- ⚡ 后台并行

**适合：**
- "How do I use library?"
- "What's the best practice for framework feature?"
- 查找开源项目的实现示例
- 获取官方 API 文档

**调用：** `@librarian`

---

#### Explore — 上下文 Grep

| 属性 | 描述 |
|------|------|
| 成本 | 免费 (FREE) |
| 搜索范围 | 当前代码库 |
| 定位 | 代码模式发现 |

**核心特点：**
- 🔍 上下文感知搜索
- 🎯 模式发现 — 跨层级的代码模式识别
- ⚡ 后台并行
- 🧩 多角度搜索

**适合：**
- 多个搜索角度需要的场景
- 不熟悉的模块结构探索
- 跨层级的模式发现

**调用：** `@explore`

---

#### Multimodal Looker — 多模态观察者

| 属性 | 描述 |
|------|------|
| 能力 | 图像/视觉内容理解 |
| 用途 | 截图分析、UI 审查、图表理解 |

**核心特点：**
- 👁️ **图像理解** — 分析截图和设计稿
- 🎨 **UI 审查** — 视觉设计评估
- 📊 **图表分析** — 数据可视化内容理解

**适合：**
- 分析网页截图
- 审查 UI 设计
- 理解图表内容

---

### Agent 调用语法

使用 `@agent-name` 直接调用专家 Agent：

```bash
# 让 Oracle 审查架构
@oracle 审查这个设计并提议架构方案

# 让 Librarian 查找实现模式
@librarian 这个功能是如何实现的？找找类似模式

# 让 Explore 搜索文件
@explore 找所有支付相关的文件
```

---

## 技能系统

### 内置技能

| 技能 | 功能 | 使用场景 |
|------|------|----------|
| `playwright` | 浏览器自动化 | 网页测试、截图、交互 |
| `agent-browser` | Agent 专用浏览器 | 爬虫、表单填写 |
| `git-master` | Git 专家 | 原子提交、rebase 操作 |
| `frontend-ui-ux` | UI/UX 设计 | 前端界面设计 |

---

### `agent-browser` 使用示例

```bash
# 打开网页
agent-browser open https://example.com

# 获取页面快照（带元素引用）
agent-browser snapshot -i
# 输出: textbox "Email" [ref=e1], button "Submit" [ref=e3]

# 填写表单
agent-browser fill @e1 "user@example.com"
agent-browser fill @e2 "password123"

# 点击按钮
agent-browser click @e3

# 等待加载完成
agent-browser wait --load networkidle

# 截图
agent-browser screenshot result.png

# 关闭浏览器
agent-browser close
```

---

### 禁用特定技能

编辑配置文件 `~/.config/opencode/oh-my-opencode.json`：

```json
{
  "disabled_skills": ["playwright", "agent-browser"]
}
```

---

## 实用技巧

### 1. 组合使用技巧

```bash
# 精密 + 快速：先用 Prometheus 规划，再用 ulw 执行
[按 Tab] → Prometheus 计划 → /start-work

# 快速迭代：直接 ulw
ulw 实现登录功能

# 长期项目：ralph-loop
/ralph-loop "这个周末构建一个完整的 SaaS"
```

---

### 2. 项目初始化工作流

```bash
# 1. 进入新项目
cd my-project

# 2. 生成上下文文件
/init-deep

# 3. 开始工作
ulw 添加基础功能
```

---

### 3. 处理复杂任务

```bash
# 使用 Prometheus 模式
[按 Tab] 进入 Prometheus
→ 描述复杂需求
→ 回答采访问题
→ 查看生成的计划 .sisyphus/plans/*.md
→ /start-work 执行
```

---

### 4. 快速修复 Bug

```bash
# 直接用 ultrawork
ulw 修复登录页面的验证问题
```

---

### 5. 重构代码

```bash
# 使用 ralph-loop 持续优化
/ralph-loop "重构整个用户模块，提升性能"
```

---

## 配置文件

### 文件位置

| 文件 | 位置 | 作用 |
|------|------|------|
| 主配置 | `~/.config/opencode/opencode.json` | MCP、插件配置 |
| Agent配置 | `~/.config/opencode/oh-my-opencode.json` | 模型映射 |
| 项目配置 | `.opencode/oh-my-opencode.json` | 项目级覆盖 |
| 计划文件 | `.sisyphus/plans/*.md` | Prometheus 生成的计划 |

---

## 常见场景

### 场景 1：新功能开发

```bash
ulw 添加用户评论功能，支持回复和点赞
```

---

### 场景 2：Bug 修复

```bash
ulw 修复支付页面在 Safari 上的显示问题
```

---

### 场景 3：性能优化

```bash
ulw 优化首页加载速度，目标 LCP < 2s
```

---

### 场景 4：代码重构

```bash
/ralph-loop "把所有 callback 改成 async/await"
```

---

### 场景 5：复杂项目

```bash
[Tab] → Prometheus 模式
→ "构建一个完整的电商后台"
→ 回答问题
→ /start-work
```

---

## 快速参考

### 命令速查表

| 模式 | 触发方式 | 适用场景 |
|------|----------|----------|
| Ultrawork | `ulw <任务>` | 快速开发、明确目标 |
| Prometheus | `[Tab]` 键 | 复杂项目、需要规划 |
| 持续循环 | `/ulw-loop "<任务>"` | 长期任务、需要迭代 |
| 自引用循环 | `/ralph-loop "<任务>"` | 自我优化、智能改进 |

---

### Agent 速查表

| Agent | 擅长 | 调用方式 |
|-------|------|----------|
| Sisyphus | 总指挥 | 自动触发 |
| Hephaestus | 代码实现 | 自动调用 |
| Prometheus | 需求规划 | `[Tab]` 键 |
| Oracle | 架构审查 | `@oracle` |
| Librarian | 模式查找 | `@librarian` |
| Explore | 文件搜索 | `@explore` |

---

### 委托决策表

| 需求类型 | 委托给 | 触发条件 |
|----------|--------|----------|
| 架构决策 | Oracle | 多系统权衡、不熟悉模式 |
| 自我审查 | Oracle | 完成重要实现后 |
| 困难调试 | Oracle | 2 次以上修复失败后 |
| 外部文档/库研究 | Librarian | 不熟悉的包/库 |
| 代码库探索 | Explore | 找现有结构、模式 |

---

### 协作流程示例

```
用户请求复杂功能
    │
    ▼
┌─────────────────────────────────────────────┐
│  Sisyphus (主编排者)                         │
│  ├─ 意图门：分析真实意图                     │
│  └─ 决定：需要规划，启动 Prometheus          │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Prometheus (规划师)                         │
│  ├─ 并行启动 Explore (搜代码模式)            │
│  ├─ 并行启动 Librarian (搜外部文档)          │
│  └─ 咨询 Oracle (架构建议)                   │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Momus (审核员)                              │
│  └─ 审核计划：通过/拒绝                      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Sisyphus/Hephaestus (执行)                  │
│  └─ 按计划实现代码                           │
└─────────────────────────────────────────────┘
```

---

### 最佳实践

1. **日常开发** → 直接用 `ulw` 命令
2. **复杂新功能** → 先 `/start-work` 让 Prometheus 规划
3. **研究性问题** → 并行启动 `Explore` + `Librarian`
4. **困难 Bug** → 先自己试，2 次失败后咨询 `Oracle`
5. **架构决策** → 直接咨询 `Oracle`

---

## 安装与卸载

### 安装

**推荐方式（让 Agent 帮你安装）：**

将以下提示词粘贴给任何 LLM Agent（Claude Code, Cursor, AmpCode 等）：

```
Install and configure oh-my-opencode by following the instructions here:
https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/refs/heads/dev/docs/guide/installation.md
```

**手动安装（不推荐，容易出错）：**

```bash
# 获取安装指南
curl -s https://raw.githubusercontent.com/code-yeongyu/oh-my-opencode/refs/heads/dev/docs/guide/installation.md

# 然后按照指南手动配置（不推荐）
```

> ⚠️ **安全警告**: `ohmyopencode.com` 不是官方站点！
> - Oh My OpenCode 是**免费开源**的
> - 官方下载: https://github.com/code-yeongyu/oh-my-opencode/releases
> - 请勿在第三方站点输入支付信息

### 卸载

**步骤 1：从 OpenCode 配置中移除插件**

编辑 `~/.config/opencode/opencode.json`（或 `opencode.jsonc`），从 `plugin` 数组中删除 `"oh-my-opencode"`：

```bash
# 使用 jq 自动处理
jq '.plugin = [.plugin[] | select(. != "oh-my-opencode")]' \
    ~/.config/opencode/opencode.json > /tmp/oc.json && \
    mv /tmp/oc.json ~/.config/opencode/opencode.json
```

**步骤 2：删除配置文件（可选）**

```bash
# 删除用户配置
rm -f ~/.config/opencode/oh-my-opencode.json ~/.config/opencode/oh-my-opencode.jsonc

# 删除项目配置（如果存在）
rm -f .opencode/oh-my-opencode.json .opencode/oh-my-opencode.jsonc
```

**步骤 3：验证卸载**

```bash
opencode --version
# 插件应该不再加载
```

---

## 模型推荐

### 经济实惠组合

即使只用以下订阅，ultrawork 也能很好地工作：

| 服务 | 订阅 | 价格 |
|------|------|------|
| ChatGPT | Subscription | $20/月 |
| Kimi Code | Subscription | $0.99/月（首月） |
| GLM Coding | Plan | $10/月 |

> 💡 如果你符合按 Token 付费的条件，使用 Kimi 和 Gemini 模型不会花费太多。

### Agent-模型映射

| Agent | 推荐模型 | 特点 |
|-------|----------|------|
| Sisyphus | claude-opus-4-6 / kimi-k2.5 / glm-5 | 强大编排能力 |
| Hephaestus | gpt-5.3-codex | 代码实现专家 |
| Prometheus | claude-opus-4-6 / kimi-k2.5 / glm-5 | 深度规划思考 |
| Oracle | 高质量推理模型 | 架构分析 |
| Librarian | 便宜模型 | 文档搜索 |
| Explore | 免费模型 | 代码库搜索 |

### 多模型支持

Oh My OpenCode 不锁定任何单一提供商：

- 🤖 **Claude / Kimi / GLM** — 用于编排
- 🧠 **GPT** — 用于推理
- ⚡ **Minimax** — 用于速度
- 🎨 **Gemini** — 用于创意

**未来不是选择单一赢家——而是协调所有模型。**
