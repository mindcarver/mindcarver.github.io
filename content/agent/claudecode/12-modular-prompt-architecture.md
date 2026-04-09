# Claude Code 的 Prompt 不是一整段话，而是 110+ 个乐高积木

当大多数人想象一个 AI 编程助手的 system prompt 时，脑海中浮现的往往是一段精心打磨的长文本。但如果你从 Claude Code 的编译产物中提取出全部 prompt，你会发现一个截然不同的图景：截至 v2.1.91，Claude Code 的 system prompt 由 **254 个独立的 markdown 文件**组成，分布在六个功能层级中，总计约 163,000 tokens。这些文件就像乐高积木一样，根据运行环境、用户配置和当前状态被动态组装成最终的上下文。

这个发现来自 Piebald AI 维护的 [claude-code-system-prompts](https://github.com/Piebald-AI/claude-code-system-prompts) 项目。该项目从 Claude Code 的 npm 包编译产物中逆向提取所有 prompt 片段，并在 CHANGELOG 中跟踪了自 v2.0.14 以来 140 个版本的变更历史。本文基于该项目的实际文件内容，拆解这套模块化 prompt 架构的设计。

## 一、为什么不用一段 monolith prompt

传统上，AI 应用的 system prompt 往往是一个巨大的字符串，包含角色定义、行为规范、工具使用说明、输出格式要求等所有内容，一次性灌入上下文。这种做法有几个问题：

**维护成本高。** 一旦需要修改某个工具的使用说明，必须在整段文本中定位相关内容，容易改错地方或遗漏。

**无法按需裁剪。** 用户在 minimal mode 下不需要 hooks、LSP、插件系统等功能的 prompt，但 monolith prompt 要么全部加载，要么不加载。

**版本追踪困难。** 一段巨大的文本在版本间如何变化？哪些部分被修改了？很难精确追踪。

**token 浪费。** 当用户只是在一个 Python 项目中做简单的 bug 修复时，加载关于 Java SDK、Go SDK、PHP SDK 的 API 参考文档纯粹是浪费上下文窗口。

Claude Code 的做法是：将 prompt 拆分为 254 个独立文件，每个文件关注一个明确的职责，然后根据场景动态组装。这不是一次性加载的巨石，而是一个按需拼装的乐高系统。

## 二、六层分类体系：254 个积木的角色

[source: README.md]

根据 README.md 中的分类，254 个 prompt 文件被组织为六个层级：

| 层级 | 文件数 | Token 总量 | 范围 | 职责 |
|------|--------|-----------|------|------|
| Agent Prompts | 32 | ~29,600 | 78-3,325 | 子代理和工具函数的独立 prompt |
| Data | 27 | ~68,500 | 292-5,106 | 嵌入的模板数据和 API 参考文档 |
| System Prompts | 68 | ~17,200 | 16-1,882 | 主 system prompt 的组成部分 |
| System Reminders | 37 | ~5,300 | 12-1,297 | 运行时状态通知 |
| Tool Descriptions | 75 | ~17,600 | 11-2,037 | 内置工具的描述文本 |
| Skills | 15 | ~25,500 | 412-5,541 | 专业化任务的技能 prompt |

这些层级不是随机分类，而是构成了一个清晰的架构：System Prompts 定义核心行为准则，Tool Descriptions 告诉模型如何使用工具，Agent Prompts 赋能子代理，Data 提供知识库，System Reminders 传递实时状态，Skills 处理特定领域任务。

## 三、每层的角色与典型示例

### 3.1 Agent Prompts：子代理的独立人格

[source: agent-prompt-explore.md, agent-prompt-general-purpose.md]

Claude Code 内部有多种子代理（subagent），每种代理有自己的完整 system prompt。当主代理将任务委派给子代理时，子代理并不共享主代理的 prompt，而是获得一个独立的、专门化的指令集。

例如 Explore 代理是一个只读的代码搜索专家（494 tokens），其 prompt 明确声明：

> "This is a READ-ONLY exploration task. You are STRICTLY PROHIBITED from: Creating new files, Modifying existing files, Deleting files..."

[source: agent-prompt-explore.md]

Explore 代理的 agentMetadata 定义了它的工具权限：禁止使用 Agent、Edit、Write、NotebookEdit 等工具，只能搜索和阅读。它甚至指定了使用的模型为 `haiku`，一个更快速、更经济的模型选择。

与之对比，General Purpose 代理（285 tokens）拥有全部工具访问权限，用于处理需要搜索、分析和编辑代码的复杂多步任务。

[source: agent-prompt-general-purpose.md]

最庞大的 agent prompt 是安全监控器的两部分，合计超过 6,400 tokens，定义了自主代理的完整安全策略，包括哪些操作必须被阻止、哪些可以放行。

### 3.2 Data：嵌入的知识库

[source: README.md]

Data 层是 token 消耗最大的层，27 个文件占了约 68,500 tokens。这些文件是 Claude Code 内嵌的参考文档，覆盖了 Claude API 在 11 种语言/平台上的使用指南。

最大的几个 Data 文件包括：

- `data-tool-use-reference-python.md`（5,106 tokens）- Python 工具使用参考
- `data-tool-use-reference-typescript.md`（5,033 tokens）- TypeScript 工具使用参考
- `data-claude-api-reference-java.md`（4,506 tokens）- Java SDK 参考
- `data-tool-use-concepts.md`（4,139 tokens）- 工具使用概念

这些文件显然不会在每次对话中全部加载。当用户使用 `/skill build-with-claude-api` 技能时，系统会根据用户选择的语言，仅加载对应语言的参考文档。模块化带来的直接好处是按需加载，避免了大量 token 浪费。

### 3.3 System Prompts：核心行为准则

[source: README.md, system-prompt-doing-tasks-software-engineering-focus.md]

System Prompts 层包含 68 个文件，定义了 Claude Code 的核心行为准则。很多文件极其简短，却每个都承担一个明确的单一职责：

最短的只有 16 tokens：
> "Your responses should be short and concise."

[source: system-prompt-tone-and-style-concise-output-short.md]

另一个 104 tokens 的文件定义了核心身份定位：
> "The user will primarily request you to perform software engineering tasks... When given an unclear or generic instruction, consider it in the context of these software engineering tasks and the current working directory."

[source: system-prompt-doing-tasks-software-engineering-focus.md]

"一个文件一条规则"的设计贯穿整个 System Prompts 层。有一组以 `doing-tasks-` 为前缀的文件，每个只有 24-78 tokens，分别规定了：不要做不必要的新增、不要过早抽象、不要给时间估算、不要添加不必要的错误处理、先读后改、注意安全。这些文件可以独立增删而不影响其他规则。

### 3.4 System Reminders：运行时的状态信使

[source: system-reminder-token-usage.md, system-reminder-plan-mode-is-active-5-phase.md]

System Reminders 是在对话过程中根据运行时事件注入的提示片段。它们不是一开始就加载的，而是在特定事件发生时才出现。

最简单的例子是 token 使用提醒（39 tokens）：
> "Token usage: ${ATTACHMENT_OBJECT.used}/${ATTACHMENT_OBJECT.total}; ${ATTACHMENT_OBJECT.remaining} remaining"

[source: system-reminder-token-usage.md]

这会在对话接近上下文窗口限制时被注入，提醒模型注意 token 预算。

最复杂的 System Reminder 是 5 阶段 Plan Mode（1,297 tokens），它定义了一个完整的计划工作流：初始理解、设计、审查、第四阶段（动态生成）、退出计划模式。这个提醒只在用户进入 plan mode 时才被注入。

[source: system-reminder-plan-mode-is-active-5-phase.md]

其他 System Reminders 覆盖了丰富的事件类型：文件被外部修改、文件在 IDE 中被打开、Hook 执行成功或失败、新的诊断信息被检测到、会话从另一台机器继续等等。每个事件对应一个独立的 reminder 文件，互不干扰。

### 3.5 Tool Descriptions：工具的使用手册

[source: README.md, tool-description-bash-overview.md]

75 个 Tool Description 文件是数量最多的层。其中 Bash 工具的描述被拆分成了约 40 个独立文件，每个文件描述 Bash 工具的一个方面：

- `tool-description-bash-overview.md`（19 tokens）- 只有开篇一句话
- `tool-description-bash-sandbox-default-to-sandbox.md`（38 tokens）- 沙箱默认策略
- `tool-description-bash-git-never-skip-hooks.md`（59 tokens）- 不要跳过 git hooks
- `tool-description-bash-sleep-keep-short.md`（29 tokens）- 保持 sleep 时间在 1-5 秒
- `tool-description-bash-sandbox-evidence-unix-socket-errors.md`（11 tokens）- 沙箱失败证据之一

这种极端的粒度拆分意味着：如果 Anthropic 想修改沙箱策略，只需改动沙箱相关的几个文件，其他 30 多个 Bash 工具的描述文件完全不受影响。

### 3.6 Skills：按需激活的专业技能

[source: skill-verify-skill.md, README.md]

Skills 是专业化任务的完整 prompt 包。与 System Prompts 不同，Skills 不会默认加载，只在用户通过 `/skill` 命令或特定触发条件调用时才注入。

最大的 Skill 是 `build-with-claude-api`（5,541 tokens），它是一个完整的路由指南，包括语言检测、平台选择和架构概览。其次是 `init-claudemd-and-skill-setup`（4,618 tokens），提供了从代码库探索到用户访谈到迭代提案的完整流程。

[source: README.md]

Verify Skill（2,158 tokens）体现了 Skills 层的设计哲学：它明确告诉模型不要做什么（不要跑测试、不要写单元测试来验证功能），而是要求模型启动应用、实际操作、观察行为、收集证据。这种专业化的行为规范不可能通过通用 system prompt 来实现，必须是独立的技能模块。

[source: skill-verify-skill.md]

## 四、条件注入：哪些 prompt 是动态加载的

[source: system-prompt-minimal-mode.md, system-prompt-auto-mode.md, system-prompt-learning-mode.md]

254 个 prompt 文件并不都在每次会话中加载。Claude Code 有一套精细的条件注入机制，根据环境和配置决定加载哪些模块。

**运行模式决定加载什么。** 三种运行模式各有专属 prompt：

- **Minimal mode**（164 tokens）：明确列出要跳过的功能（hooks、LSP、插件、auto-memory、后台预取等），要求通过 CLI 参数显式提供上下文 [source: system-prompt-minimal-mode.md]
- **Auto mode**（255 tokens）：激活持续自主执行，要求模型立即行动、减少中断、偏好行动而非规划 [source: system-prompt-auto-mode.md]
- **Learning mode**（1,042 tokens）：在完成任务的同时鼓励用户学习，要求模型在 20 行以上的代码中邀请用户编写 2-10 行关键逻辑 [source: system-prompt-learning-mode.md]

这三种模式的 prompt 互斥，不会同时加载。

**运行时事件触发 System Reminders。** Plan mode 的 5 阶段提醒（1,297 tokens）只在用户进入 plan mode 时注入。Token 使用提醒只在 token 消耗达到特定阈值时出现。文件修改提醒只在外部编辑器或 linter 修改了文件时触发。

**平台决定工具描述。** PowerShell 相关的描述（1,455 tokens + 285 tokens）只在 Windows 环境下加载。Bash 沙箱相关的大量子规则（合计数百 tokens）根据沙箱配置状态决定是否注入。

**Data 文件按需引用。** 27 个语言特定的 API 参考文档不会默认加载到上下文中。它们在特定 Skill 被激活时，按用户选择的语言加载对应文档。

## 五、模板变量系统：168 个运行时插值点

[source: system-reminder-token-usage.md, tool-description-bash-git-commit-and-pr-creation-instructions.md, system-reminder-plan-mode-is-active-5-phase.md]

254 个 prompt 文件中，有 99 个文件包含模板变量，总计 168 个不同的变量名。这些变量在运行时由 Claude Code 的 JavaScript 代码进行插值，替换为实际值。

模板变量大致分为几类：

**工具名称引用。** `${BASH_TOOL_NAME}`、`${EDIT_TOOL_NAME}`、`${GREP_TOOL_NAME}`、`${WRITE_TOOL_NAME}` 等。这种设计允许工具名称在代码中统一修改后，所有引用该工具的 prompt 自动更新，无需逐个文件修改。

**运行时对象。** `${ATTACHMENT_OBJECT.used}/${ATTACHMENT_OBJECT.total}` 用于 token 使用提醒，`${AGGREGATED_USAGE_DATA}` 用于使用分析，`${DIAGNOSTICS_SUMMARY}` 用于诊断信息。

**条件控制。** 在 5 阶段 Plan Mode 的 prompt 中，可以看到这样的条件逻辑：

```
${PLAN_V2_PLAN_AGENT_COUNT>1?`Use up to ${PLAN_V2_PLAN_AGENT_COUNT} agents for complex tasks...`:""}
```

当配置只允许一个 plan 代理时，关于多代理协作的整段说明文本会被完全省略，不消耗任何 token。

**函数调用。** `${GET_PHASE_FOUR_FN()}` 是一个运行时函数调用，其返回值被插入到 prompt 中。这种模式允许将复杂的条件逻辑封装在代码中，保持 prompt 文件本身的简洁。

[source: system-reminder-plan-mode-is-active-5-phase.md]

Git commit 和 PR 创建的指令中引用了 `${BASH_TOOL_NAME}`、`${COMMIT_CO_AUTHORED_BY_CLAUDE_CODE}`、`${PR_GENERATED_WITH_CLAUDE_CODE}` 等变量，使得签名格式、工具引用等可以集中管理。

[source: tool-description-bash-git-commit-and-pr-creation-instructions.md]

这些变量在提取出的文件中以字面量形式出现，因为它们是从编译后的 JavaScript 中静态提取的，实际插值发生在运行时。正如 CLAUDE.md 中所说："Template variables like `${BASH_TOOL_NAME}` are interpolated at runtime by Claude Code -- they appear as literal strings in these files."

[source: CLAUDE.md]

## 六、从 CHANGELOG 看模块化的优势

[source: CHANGELOG.md]

CHANGELOG.md 记录了 140 个版本的 prompt 变更，其中新增了 187 个 prompt 文件，移除了 87 个。这个变更记录本身就是模块化架构优势的直接证据。

**独立修改不影响其他。** 在 v2.1.90 中，安全监控器的 block/allow 规则被更新，memory consolidation 的索引格式被调整，verification specialist 的文件发现逻辑被改进。这些变更分别发生在不同的文件中，互不影响。如果是 monolith prompt，任何一处修改都需要在巨大的文本中精确定位，冒着意外修改其他规则的风险。

**安全地添加和移除。** v2.1.84 一次性新增了 5 个 prompt（General Purpose agent、PowerShell 支持、request_teach_access），同时移除了 8 个不再需要的 prompt（旧的 Explore strengths、旧的 compact 分析指令、over-engineering 指导等）。这种大规模的重构在模块化架构中是安全的：移除一个文件不会影响其他文件的内容。

**精确的版本追踪。** 每个文件的 YAML frontmatter 中都有 `ccVersion` 字段，标记该文件最后一次被修改的版本。例如 `system-prompt-tone-and-style-concise-output-short.md` 的 ccVersion 是 2.1.53，意味着自从那个版本以来，这个"保持简洁"的指令再没被修改过。

**变更的 token 成本可见。** CHANGELOG 的每个版本条目都标注了 token 变化量。例如 v2.1.89 新增了 +3,986 tokens（Buddy Mode、Ultraplan、Computer Use MCP 等新功能），v2.1.88 减少了 -1,627 tokens（移除了 System Section、精简了 Verify Skill）。每个版本的 prompt 成本变化一目了然。

## 七、token 经济学：254 个片段如何组合

[source: README.md]

254 个文件总计约 163,700 tokens，但没有任何一次会话会加载所有文件。实际加载的 prompt 子集取决于：

1. **核心行为层**（始终加载）：System Prompts 中关于软件工程行为、工具使用偏好、安全规范等约 68 个文件，约 17,200 tokens。

2. **工具描述层**（根据平台选择加载）：75 个文件中，Bash 相关的约 40 个文件大部分会加载，PowerShell 相关文件仅在 Windows 加载，特定工具的附加说明根据配置决定。实际加载约 12,000-15,000 tokens。

3. **Agent Prompts**（按需激活）：32 个文件中，只有 Explore 和 Plan 等少数代理会在常规会话中触发。一个典型的会话可能加载 2-3 个 agent prompt，约 1,500-3,000 tokens。

4. **System Reminders**（事件触发）：37 个文件中，token 使用提醒和会话继续等少数会始终加载，大部分只在特定事件时注入。一次会话可能累积 1,000-3,000 tokens 的 reminders。

5. **Data**（技能触发）：27 个文件约 68,500 tokens 不会默认加载，只在相关 Skill 被激活时按语言加载对应文档。单次加载约 2,000-5,000 tokens。

6. **Skills**（用户调用）：15 个文件完全按需加载，默认不消耗任何 token。

一个典型的编码会话，system prompt 的实际 token 消耗大约在 30,000-45,000 tokens，远低于 163,700 的总量上限。模块化架构让 Claude Code 能够将 token 预算精确地分配给当前需要的指令，而不是浪费在无关的内容上。

## 八、与 monolith prompt 的对比

将 Claude Code 的模块化方案与传统的单段 prompt 做对比，差异是结构性的：

| 维度 | Monolith Prompt | Claude Code 模块化方案 |
|------|----------------|----------------------|
| 文件组织 | 单一字符串或少量段 | 254 个独立 markdown 文件 |
| 加载策略 | 全量加载 | 按运行模式、环境、事件动态组装 |
| 修改影响 | 改动一处可能影响全文 | 修改一个文件不影响其他 |
| 版本追踪 | diff 整段文本 | 按文件精确追踪，每个文件独立 ccVersion |
| 运行时适配 | 硬编码或极少条件分支 | 168 个模板变量，支持条件逻辑和函数调用 |
| token 效率 | 固定成本，无法裁剪 | 163K 总量，典型会话仅加载 30K-45K |
| 子代理支持 | 共享主 prompt 或复制 | 每个代理有独立 prompt，定义专属工具权限和模型选择 |

最本质的差异在于：monolith prompt 把"说什么"和"什么时候说"混在一起，而模块化方案将这两者分离。每个 prompt 文件只负责"说什么"，"什么时候说"由代码逻辑根据运行时条件决定。这种关注点分离是软件工程的基本原则，Claude Code 将它用在了 prompt 工程上。

## 九、总结：对 AI 应用开发的启示

Claude Code 的模块化 prompt 架构提供了几条可操作的启示：

**第一，prompt 是代码，应该像代码一样组织。** 单一职责、关注点分离、模块化设计这些软件工程原则完全适用于 prompt 工程化。每个文件只做一件事，修改一个行为不需要碰其他行为。

**第二，为动态组装设计，而不是为全量加载设计。** 当 prompt 超过几千 tokens 时，就应该考虑拆分。按功能域、按使用场景、按触发条件将 prompt 拆分为独立模块，然后在运行时按需组装。这是 token 经济学的要求，也是可维护性的要求。

**第三，用模板变量建立 prompt 与代码的接口。** 168 个模板变量不是偶然的数字，它反映了 prompt 和应用程序代码之间存在一个清晰的接口层。工具名称、运行时状态、条件配置都通过变量传递，而不是硬编码在 prompt 文本中。

**第四，版本追踪是 prompt 工程化的基础设施。** 没有版本追踪，你无法知道哪个 prompt 在哪个版本被修改、新增或移除。Claude Code 的 CHANGELOG 展示了 140 个版本的变更历史，这是 prompt 作为产品组件而非一次性文本的标志。

**第五，token 成本是 prompt 架构的核心约束。** 163K tokens 的总量在任何单次调用中都无法全部加载。模块化架构通过按需组装，将实际消耗控制在 30K-45K，这是 monolith 方案不可能实现的效率。

Claude Code 的 prompt 架构本质上是一个 **prompt 编译器**：254 个源文件作为输入，运行时配置和状态作为参数，组装出针对当前会话的最优上下文。这种架构让一个拥有超过 16 万 tokens 指令库的系统，能够在每次调用中精确地加载它需要的部分，不多也不少。
