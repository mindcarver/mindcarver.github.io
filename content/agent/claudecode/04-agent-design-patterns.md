# Anthropic 怎么造 Agent？从 Claude Code 逆向出的设计模式

## 一、为什么 Claude Code 内嵌了一份 Agent 设计指南

在 Claude Code 的系统提示词中，有一份名为 "Agent Design Patterns" 的技能文档。它不是给终端用户看的操作手册，而是 Anthropic 工程师在构建 Claude Code 本身时沉淀下来的设计决策记录。这份文档覆盖了工具面设计、上下文管理、缓存策略、多 Agent 协作等问题 [source: skill-agent-design-patterns.md]。

也就是说，面对的是经过大规模工程验证的实战经验，不是理论推演。Claude Code 每天在真实代码库中被成千上万的开发者使用，它内部的每一个设计决策都经过了成本、延迟、可靠性的三重考验。本文基于这份指南以及 Claude Code 中 Agent 相关的多个系统提示词，梳理 Anthropic 的 Agent 设计方法论。

文章会依次探讨：工具面设计的核心权衡、六种工具矩阵、Programmatic Tool Calling 的组合调用机制、按需加载的工具发现系统、长运行 Agent 的上下文生存策略、Agent 特定的缓存变通方案、三种多 Agent 协作模式的对比，以及如何用 Agent Creation Architect 来设计自定义 Agent。

---

## 二、Bash vs 专用工具的提升时机

设计一个 Agent 的第一步，是决定它通过什么形状的接口与外部世界交互。Anthropic 给出了一个非常清晰的决策框架。

### 起步用 Bash，按需提升

一个 Bash 工具给 Agent 提供了广泛的编程能力——几乎可以执行任何操作。但对 Agent 背后的执行框架（harness）而言，Bash 只能看到一个不透明的命令字符串，每一个操作都是相同的形状，无法区分 `grep` 和 `git push` [source: skill-agent-design-patterns.md]。

把某个操作提升为专用工具时，框架获得了一个带有类型化参数的动作钩子，可以拦截、门控、渲染或审计。Anthropic 明确了四个提升时机：

**安全边界（Security Boundary）**：难以逆转的操作是天然的候选者。外部 API 调用、发送消息、删除数据等操作应当被用户确认所门控。一个 `send_email` 工具很容易加设门控；而 `bash -c "curl -X POST ..."` 则不行。可逆性是一个有用的判断标准 [source: skill-agent-design-patterns.md]。

**时效检查（Staleness Checks）**：一个专用的 `edit` 工具可以拒绝在文件自上次读取后被修改过的写入操作。Bash 无法强制执行这种不变量。这在多 Agent 并发编辑场景下尤为关键 [source: skill-agent-design-patterns.md]。

**渲染（Rendering）**：某些操作需要自定义的 UI 展示。Claude Code 将提问操作提升为专用工具，使其可以渲染为模态框、呈现选项列表，并阻塞 Agent 循环直到用户回答 [source: skill-agent-design-patterns.md]。

**调度（Scheduling）**：只读工具如 `glob` 和 `grep` 可以被标记为并行安全（parallel-safe）。同样的操作通过 Bash 执行时，框架无法区分并行安全的 `grep` 和不安全的 `git push`，只能全部串行化 [source: skill-agent-design-patterns.md]。

### 经验法则

规则很简单：先用 Bash 获得广度，当需要门控、渲染、审计或并行化某个操作时，再将其提升为专用工具 [source: skill-agent-design-patterns.md]。这种渐进式设计避免了过早抽象的陷阱。

---

## 三、六种工具矩阵

Anthropic 为 Claude API 提供了六种开箱即用的工具，分为 Client-side 和 Server-side 两类 [source: skill-agent-design-patterns.md]。

| 工具 | 侧别 | 适用场景 | 行为特征 |
|------|------|---------|---------|
| **Bash** | Client | 执行 Shell 命令 | Claude 发出命令，harness 执行 |
| **Text Editor** | Client | 读写和编辑文件 | 查看、创建、编辑文件 |
| **Computer Use** | Client 或 Server | 与 GUI、Web 应用交互 | 截图并发出鼠标/键盘指令 |
| **Code Execution** | Server | 在沙箱中运行代码 | Anthropic 托管容器，内置文件和 Bash 子工具 |
| **Web Search / Fetch** | Server | 获取训练截止日期之后的信息 | Claude 发出查询，Anthropic 执行并返回带引用的结果 |
| **Memory** | Client | 跨会话保存上下文 | 读写 `/memories` 目录，自行实现存储后端 |

Client-side 工具由 Anthropic 定义（名称、Schema、Claude 的使用模式），但由你的 harness 执行。Anthropic 提供参考实现。Server-side 工具完全在 Anthropic 基础设施上运行——只需在 `tools` 中声明，Claude 处理其余部分 [source: skill-agent-design-patterns.md]。

Bash 和 Text Editor 被放在 Client 端，说明 Anthropic 认为文件操作和命令执行必须由开发者完全控制。Code Execution 和 Web Search 放在 Server 端，Anthropic 愿意为计算密集型和网络访问型任务提供托管能力。Computer Use 比较特殊，可以是任意一侧——你自己运行环境就是 Client-side，用 Anthropic 托管的环境就是 Server-side。

---

## 四、Programmatic Tool Calling：用代码组合工具调用

标准的工具使用模式下，每次工具调用都是一次往返：Claude 调用工具，结果进入 Claude 的上下文，Claude 推理后再调用下一个工具。三个顺序操作（读取用户档案、查询订单、检查库存）意味着三次往返。每次往返增加延迟和 Token 消耗，而大部分中间数据之后再也不会被用到 [source: skill-agent-design-patterns.md]。

### 工作原理

Programmatic Tool Calling（PTC）允许 Claude 将多次工具调用组合成一个脚本。这个脚本在 Code Execution 容器中运行。当脚本调用一个工具时，容器暂停，调用被执行（Client-side 或 Server-side），结果返回给正在运行的代码——而不是 Claude 的上下文。脚本用正常的控制流（循环、过滤、分支）处理结果。只有脚本的最终输出返回给 Claude [source: skill-agent-design-patterns.md]。

传统模式是"工具结果进入上下文，Agent 在上下文中推理"；PTC 模式是"Agent 编写代码，代码在沙箱中自主执行多步操作，只有最终结论回传上下文"。Token 成本按最终输出计费，而非中间结果。

### 适用场景

PTC 适用于有大量顺序工具调用的场景，或者中间结果很大、希望在进入上下文窗口之前先过滤的情况 [source: skill-agent-design-patterns.md]。

一个实际例子：任务需要读取十个文件、在每个文件中搜索特定模式、汇总结果。传统模式会把十个文件的内容都塞进上下文窗口；PTC 模式下，Claude 写一个脚本来完成所有读取和过滤，只把汇总表返回给上下文。

---

## 五、Tool Search + Skills：按需加载的工具发现机制

当 Agent 可用的工具数量增多时，一个关键问题浮现：不可能把所有工具的 Schema 都塞进每次请求的上下文中。Anthropic 提供了两种按需加载机制 [source: skill-agent-design-patterns.md]。

### Tool Search

当有许多工具可用但每次请求只有少数几个相关时，Tool Search 允许 Claude 搜索工具集并只加载相关的 Schema。关键细节：工具定义是追加（append）而非替换的——这保护了已有的缓存前缀 [source: skill-agent-design-patterns.md]。

### Skills

Skills 是任务特定的指令，Claude 只在相关时才加载。每个 Skill 是一个文件夹，包含一个 `SKILL.md` 文件。Skill 的描述（description）默认在上下文中，但完整内容只在任务需要时 Claude 才会读取 [source: skill-agent-design-patterns.md]。

这两种模式共享同一个设计原则：保持固定上下文尽可能小，按需加载细节。上下文中塞满无关的工具定义会稀释 Agent 对真正相关信息的注意力。

---

## 六、长运行 Agent 的上下文管理

长运行的 Agent 面临一个根本性挑战：上下文窗口是有限的，但任务可能是无限的。Anthropic 提供了三种互补的上下文管理策略 [source: skill-agent-design-patterns.md]。

### Context Editing（上下文编辑）

随着对话轮次增加，上下文会变得陈旧——旧的工具结果、已完成的思考块不再有价值。Context Editing 根据可配置的阈值清除工具结果和思考块。它保持对话记录精简，而不使用摘要。"修剪"策略：移除明确无用的内容，保留仍有价值的部分 [source: skill-agent-design-patterns.md]。

### Compaction（压缩）

当对话可能达到或超过上下文窗口限制时，Compaction 在服务端将较早的上下文总结为一个压缩块。"浓缩"策略：将大量历史信息压缩为更紧凑的表示 [source: skill-agent-design-patterns.md]。

### Memory（记忆）

当状态必须跨会话持久化时，Claude 在一个记忆目录中读写文件。进程重启后存活 [source: skill-agent-design-patterns.md]。

### 如何选择

Context Editing 和 Compaction 在会话内操作——Editing 修剪陈旧的轮次，Compaction 在接近限制时进行摘要。Memory 用于跨会话持久化。许多长运行 Agent 会同时使用这三种策略。Claude Code 本身就是一个例子：它使用 Context Editing 清理旧的 Bash 输出，使用 Compaction 处理超长对话，使用 Memory 保存跨会话的项目知识 [source: skill-agent-design-patterns.md]。

从单轮对话到多轮对话，Context Editing 解决了"对话太长"的问题。当对话远超窗口容量时，Compaction 提供了安全阀。当需要跨会话延续时，Memory 提供了持久化能力。三者构成了一个完整的上下文生命周期管理方案。

---

## 七、Agent 特定的缓存变通方案

Prompt Caching 直接影响 Agent 的运行成本，但 Agent 的工作模式对缓存提出了独特挑战。Anthropic 在 Agent Design Patterns 文档中给出了三种约束及其对应的变通方案 [source: skill-agent-design-patterns.md]。

### 约束一：会话中修改 System Prompt 会破坏缓存

**变通方案**：在 `messages` 数组中追加一个 `<system-reminder>` 块，而不是修改 System Prompt。缓存的 System Prompt 前缀保持不变。Claude Code 正是使用这种方式来处理时间更新和模式切换的 [source: skill-agent-design-patterns.md]。

### 约束二：会话中切换模型会破坏缓存

**变通方案**：为子任务生成一个使用更便宜模型的 Subagent，主循环保持在同一个模型上。Claude Code 的 Explore subagent 就是用 Haiku 以这种方式工作的 [source: skill-agent-design-patterns.md]。

这个变通方案揭示了一个深层设计思想：缓存一致性比模型灵活性更重要。主循环中的模型切换代价太高（整个缓存前缀失效），不如把需要不同模型的子任务隔离到独立的 Subagent 中。

### 约束三：会话中增删工具会破坏缓存

**变通方案**：使用 Tool Search 进行动态发现——它追加工具 Schema 而非替换它们，因此已有的缓存前缀被保留 [source: skill-agent-design-patterns.md]。

三种变通方案指向同一个设计原则：在 Agent 系统中，保持缓存前缀不变是一等优先级。System Prompt、模型、工具列表三者构成了缓存的基础，对它们的修改都应该通过"追加而非替换"的方式来规避缓存失效。

---

## 八、多 Agent 协作：Subagent vs Fork vs Swarm 三种模式对比

当单个 Agent 无法胜任复杂任务时，多 Agent 协作成为必然选择。Claude Code 内部实现了三种不同的多 Agent 模式，每种针对不同的问题空间。

### Subagent：独立上下文的专职代理

Subagent 是通过指定 `subagent_type` 创建的全新 Agent。它从零上下文开始，不继承父 Agent 的对话历史 [source: system-prompt-writing-subagent-prompts.md]。

正因为没有上下文，Subagent 的 Prompt 必须像"给一个刚走进房间的聪明同事做简报"一样写——解释你想要完成什么以及为什么、描述你已经学到或排除了什么、提供足够的背景让 Agent 能做判断而非仅仅跟随狭隘的指令 [source: system-prompt-writing-subagent-prompts.md]。

Anthropic 特别警告了一种反模式：永远不要委托理解（"Never delegate understanding"）。不要写"based on your findings, fix the bug"或"based on the research, implement it"这类话。这些短语把综合判断推给了 Agent 而不是你自己做。有效的 Prompt 应该证明你理解了问题：包含文件路径、行号、具体要改什么 [source: system-prompt-writing-subagent-prompts.md]。

一个典型的 Subagent 使用场景是"第二意见"——让 code-reviewer Agent 独立审查一个数据库迁移的安全性，它不会看到主 Agent 的分析，因此能提供独立的判断 [source: system-prompt-subagent-delegation-examples.md]。

### Fork：共享缓存的高效分支

Fork 是省略 `subagent_type` 时隐式触发的机制。它继承完整的对话上下文，并共享父 Agent 的 Prompt 缓存 [source: system-prompt-fork-usage-guidelines.md]。

核心判断标准是定性的："我还需要这个输出吗？"——而非任务大小。具体来说：

**研究任务**：开放性问题适合 Fork。如果研究可以被分解为独立的问题，可以在一条消息中并行启动多个 Fork。Fork 在这方面优于 Subagent——它继承上下文并共享缓存 [source: system-prompt-fork-usage-guidelines.md]。

**实现任务**：需要超过几个编辑的实现工作适合 Fork。前提是先完成研究再跳到实现 [source: system-prompt-fork-usage-guidelines.md]。

共享缓存是 Fork 的经济优势来源。因此 Anthropic 明确规定：不要在 Fork 上设置不同的 `model`——不同模型无法复用父 Agent 的缓存 [source: system-prompt-fork-usage-guidelines.md]。

Fork 有两条铁律：**不要偷看**——不要在运行中读取 Fork 的输出文件，因为那会把 Fork 的工具噪声拉入你的上下文，违背了 Fork 的初衷；**不要编造**——在通知到达之前，你对 Fork 的发现一无所知，永远不要以任何格式编造或预测 Fork 的结果 [source: system-prompt-fork-usage-guidelines.md]。

因为 Fork 继承了上下文，Fork 的 Prompt 是一个指令（directive）——说明要做什么，而不是说明情况是什么。要明确范围：什么在范围内、什么在范围外、另一个 Agent 在处理什么。不要重复解释背景 [source: system-prompt-fork-usage-guidelines.md]。

Fork 内部使用的 Worker Prompt 很值得研究。它以"STOP. READ THIS FIRST."开头，强调"你不是主 Agent"。规则包括：不要再生成 Subagent、不要对话或问问题、不要添加元评论、直接使用工具、如果修改文件则在报告前提交、工具调用之间不要输出文本、报告限制在 500 字以内、必须以"Scope:"开头 [source: agent-prompt-worker-fork-execution.md]。极简风格，因为 Fork 是一个执行单元，不是对话伙伴。

### Swarm：多 Agent 团队协作

Swarm 是通过 TeammateTool 实现的团队协作模式。它创建一个命名的团队，团队成员共享任务列表，通过消息系统协作 [source: tool-description-teammatetool.md]。

工作流程：创建团队、创建任务、生成团队成员、分配任务、成员完成后标记、最后优雅关闭团队 [source: tool-description-teammatetool.md]。

成员之间的通信通过 SendMessage 工具完成。消息自动投递，不需要手动检查收件箱。每个成员在完成一个回合后进入空闲状态（idle），这完全正常——空闲只是意味着在等待输入 [source: tool-description-teammatetool.md]。

选择 Agent 类型时需要根据任务需求匹配工具能力：只读 Agent（如 Explore、Plan）不能编辑文件，只适合研究、搜索或规划任务；全能力 Agent（如 general-purpose）可以访问所有工具包括文件编辑、写入和 Bash [source: tool-description-teammatetool.md]。

### 三种模式的对比

| 维度 | Subagent | Fork | Swarm |
|------|----------|------|-------|
| 上下文 | 全新，零历史 | 继承父 Agent 完整上下文 | 各成员独立上下文 |
| 缓存 | 独立缓存 | 共享父 Agent 缓存 | 各成员独立 |
| 模型 | 可选不同模型 | 必须同模型 | 可按成员配置 |
| 通信 | 异步通知 | 异步通知 | 消息系统 + 任务列表 |
| 适用场景 | 独立审查、第二意见 | 并行研究、上下文相关实现 | 复杂多步骤项目协作 |
| Prompt 风格 | 完整简报（背景+目标） | 指令式（做什么） | 任务驱动 |

---

## 九、Agent Creation Architect：如何设计一个自定义 Agent

Claude Code 内置了一个名为 "Agent Creation Architect" 的元 Agent——它帮助用户设计自定义 Agent。通过研究它的系统提示词，可以反推出 Anthropic 认为一个好的 Agent 设计应该包含哪些要素 [source: agent-prompt-agent-creation-architect.md]。

### 设计流程的六个步骤

Agent Creation Architect 遵循一个结构化的六步流程：

**第一步：提取核心意图**。识别 Agent 的基本目的、关键职责和成功标准。寻找显式需求和隐式需求。考虑 CLAUDE.md 文件中的项目特定上下文。对于代码审查 Agent，应该假设用户要求审查最近编写的代码而非整个代码库 [source: agent-prompt-agent-creation-architect.md]。

**第二步：设计专家人格**。创建一个能体现与任务相关深层领域知识的专家身份。人格应该激发信心并指导 Agent 的决策方法 [source: agent-prompt-agent-creation-architect.md]。

**第三步：编写指令**。System Prompt 应该包含：明确的行为边界和操作参数、任务执行的具体方法论、边缘情况的预判和处理指导、用户提到的特定需求或偏好、输出格式期望，以及与项目编码标准的一致性 [source: agent-prompt-agent-creation-architect.md]。

**第四步：性能优化**。包括适合该领域的决策框架、质量控制机制和自我验证步骤、高效的工作流模式，以及明确的升级或回退策略 [source: agent-prompt-agent-creation-architect.md]。

**第五步：创建标识符**。设计一个简洁、描述性的标识符，只使用小写字母、数字和连字符，通常是 2-4 个用连字符连接的单词，清楚表明 Agent 的主要功能，避免使用"helper"或"assistant"等通用术语 [source: agent-prompt-agent-creation-architect.md]。

**第六步：编写触发示例**。在 `whenToUse` 字段中包含示例，展示何时应该使用这个 Agent。示例应包含上下文、用户消息、助手消息，以及说明为何触发该 Agent 的评论 [source: agent-prompt-agent-creation-architect.md]。

### 输出格式

Agent Creation Architect 的输出是一个 JSON 对象，包含三个字段：`identifier`（唯一描述性标识符）、`whenToUse`（精确的触发条件描述，以"Use this agent when..."开头，含示例）、`systemPrompt`（完整的 System Prompt，使用第二人称编写）[source: agent-prompt-agent-creation-architect.md]。

### 关键设计原则

从 Agent Creation Architect 的设计原则中可以提炼出 Anthropic 对 Agent 设计的核心观点：

- **具体优于泛化**——避免模糊的指令
- **包含具体示例**——当它们能澄清行为时
- **全面性与清晰性的平衡**——每条指令都应增加价值
- **确保 Agent 有足够的上下文**——能处理核心任务的变体
- **主动寻求澄清**——当需要时
- **内置质量保证和自我纠正机制**

最后一条值得多说两句。Anthropic 不期望 Agent 完美无缺，而是期望 Agent 有能力发现和纠正自己的错误。"可恢复性"比"无错性"更现实。

---

## 十、Anthropic 的 Agent 设计哲学

从这八个源文件中，可以提炼出 Anthropic 的 Agent 设计哲学。它围绕五个核心思想展开。

**第一，渐进式设计。** 不是一开始就设计完美的工具面，而是从 Bash 起步获得广度，在遇到安全、渲染、审计、并行化的具体需求时再提升为专用工具。这不是懒惰，是对过度工程的警惕 [source: skill-agent-design-patterns.md]。

**第二，上下文是最稀缺的资源。** 从 Tool Search 的按需加载、到 Context Editing 的修剪、到 Compaction 的压缩、到 PTC 的"中间结果不进上下文"，所有设计都在围绕同一个目标：让每一 Token 都承载最高的信息密度。Fork 的存在本身就是为了防止工具噪声污染主 Agent 的上下文 [source: skill-agent-design-patterns.md, system-prompt-fork-usage-guidelines.md]。

**第三，缓存一致性是一等优先级。** System Prompt 追加而非修改、子任务隔离到 Subagent 以避免模型切换、工具 Schema 追加而非替换。这些变通方案说明：在 Agent 系统中，缓存失效的代价远大于代码的复杂性 [source: skill-agent-design-patterns.md]。

**第四，委托执行而非委托理解。** "Never delegate understanding"这条规则贯穿了所有的 Prompt 设计指南。无论是 Fork 的指令式 Prompt、Subagent 的完整简报、还是 Agent Creation Architect 的"包含文件路径和行号"建议，核心思想都是：综合判断由主 Agent 完成，子 Agent 只负责执行明确的任务 [source: system-prompt-writing-subagent-prompts.md, agent-prompt-agent-creation-architect.md]。

**第五，多模式协作而非一刀切。** Subagent、Fork、Swarm 三种模式各有适用场景，差异不是功能上的高低之分，而是针对不同问题空间的专门化设计。Subagent 擅长独立判断，Fork 擅长并行研究，Swarm 擅长持续协作。成熟的 Agent 系统应该能根据任务特征选择合适的协作模式 [source: system-prompt-fork-usage-guidelines.md, tool-description-teammatetool.md]。

五条原则形成了一个完整的设计框架：渐进式构建工具面、上下文效率为核心约束、缓存一致性为系统保障、"委托执行不委托理解"为协作准则、多模式选择应对不同复杂度。Anthropic 没有试图提供一个通用的 Agent 框架，而是提供了一套经过实战验证的设计决策——这些决策来自每天服务大量开发者的真实产品，不是实验室中的原型。
