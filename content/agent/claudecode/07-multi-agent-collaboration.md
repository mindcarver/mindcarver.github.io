# 一个 AI 怎么分身成三十个？Claude Code 的多 Agent 协作设计

## 一、为什么需要多 Agent 协作

单个 LLM agent 处理复杂任务时有三道硬墙：上下文窗口有限，串行执行效率低，无法同时关注多个独立维度。Claude Code 内置了三套层次分明的多 agent 协作机制来应对不同场景。

Agent tool 的定位在系统提示中写得直接：[source: tool-description-agent-when-to-launch-subagents.md] "Launch a new agent to handle complex, multi-step tasks autonomously." 它启动专门的 agent 子进程，自主完成复杂任务。根据隔离级别、通信需求和 token 成本的不同，Claude Code 提供了 Subagent（子代理）、Fork（分叉）和 Team/Swarm（团队协作）三种委派策略。

本文基于系统提示的原始文本，逐一拆解这三种模式的内部设计。

---

## 二、Subagent 机制：从零启动的专用 Agent

### 2.1 两种启动方式

Agent tool 的核心参数是 `subagent_type`。指定类型时，启动一个全新的专用 agent，没有父 agent 的对话上下文，需要通过 prompt 获得完整的任务描述。省略该参数时，行为取决于系统配置：在 fork 实验开启时，会创建一个继承完整对话上下文的 fork worker [source: tool-description-agent-when-to-launch-subagents.md]。

系统为不同场景预定义了多种 agent 类型：

| 类型 | 定位 | 工具权限 | 适用场景 |
|------|------|---------|---------|
| general-purpose | 全能型 | 全部工具 | 多步研究、代码分析、跨文件搜索 |
| Explore | 只读搜索 | Glob、Grep、Read、Bash（只读） | 快速定位文件、搜索代码模式 |
| code-reviewer | 独立审查 | 读取类工具 | 第二意见、独立代码评审 |

Explore agent 是一个典型的专用设计。它被严格限制为只读模式，系统提示中逐条列出了禁止操作：[source: agent-prompt-explore.md] "STRICTLY PROHIBITED: Creating new files, Modifying existing files, Deleting files, Running ANY commands that change system state." 目标是快速返回结果，底层使用 haiku 模型以降低成本和延迟。调用者可以通过 thoroughness 参数控制搜索深度："quick" 做基础搜索，"medium" 做中等探索，"very thorough" 做跨目录、多命名约定的全面扫描。

### 2.2 Prompt 写作的艺术

给全新 subagent 写 prompt 时，系统提示给出了一个精准的比喻：[source: system-prompt-writing-subagent-prompts.md] "Brief the agent like a smart colleague who just walked into the room — it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters."

具体来说，一个好的 subagent prompt 需要包含：
- 要完成什么、为什么要做
- 已经知道什么、排除了什么
- 周围的问题背景，让 agent 能做判断而非机械执行
- 如果需要简短回复，明确说明（如 "report in under 200 words"）

系统提示特别警告了一个常见错误：[source: system-prompt-writing-subagent-prompts.md] "Never delegate understanding." 不要写 "based on your findings, fix the bug" 或 "based on the research, implement it." 这类措辞把综合判断推给了 subagent。正确的做法是在 prompt 中包含文件路径、行号、具体要修改什么——证明你理解了问题再委派。

### 2.3 使用场景举例

系统提示给出了几个典型的 subagent 使用模式 [source: system-prompt-subagent-delegation-examples.md]：

**调查类任务**：用户问 "这个分支还有什么没做完的？"，coordinator fork 出一个 "ship-audit" agent，让它检查未提交变更、领先于 main 的 commit、测试覆盖情况、CI 相关文件变更，然后返回 punch list。Fork 的好处是这些调查的中间输出不会污染主 agent 的上下文。

**独立第二意见**：用户问 "这个迁移安全吗？"，coordinator 启动一个 code-reviewer 类型的 subagent。关键点在于使用了 `subagent_type` 而非 fork，这意味着这个 reviewer 从零开始，看不到 coordinator 之前的分析，保证了独立性。Prompt 需要包含完整的上下文：数据库规模、迁移方案、具体的安全关注点 [source: system-prompt-subagent-delegation-examples.md]。

---

## 三、Fork 机制：继承上下文的轻量分叉

### 3.1 设计思路

Fork 的触发条件很简单：省略 `subagent_type` 参数。此时 agent 创建一个自身的分叉，继承完整的对话上下文 [source: system-prompt-fork-usage-guidelines.md]。

Fork 和 subagent 的根本区别在于上下文继承。Fork 继承了父 agent 的所有对话历史和 prompt cache，启动成本很低。正因为共享 cache，系统明确建议不要给 fork 指定不同的 model：[source: system-prompt-fork-usage-guidelines.md] "a different model can't reuse the parent's cache."

判断是否该用 fork 的标准是一个定性的问题：[source: system-prompt-fork-usage-guidelines.md] "will I need this output again?" 如果中间的工具输出不值得保留在上下文中，就 fork 出去。

### 3.2 Fork Worker 的内部规则

Fork worker 的系统提示定义了严格的执行纪律 [source: agent-prompt-worker-fork-execution.md]：

1. 不再创建子 agent——自己就是执行者
2. 不与用户对话、不提问、不提建议
3. 不加元评论或旁白
4. 直接使用工具：Bash、Read、Write 等
5. 如果修改了文件，在报告前先 commit，报告包含 commit hash
6. 工具调用之间不输出文本，最后统一报告
7. 严格在委派范围内工作，超出范围的发现最多提一句
8. 报告控制在 500 词以内
9. 输出以 "Scope:" 开头，不加任何前言

输出格式也是固定的：

```
Scope: <一句话回显委派范围>
Result: <关键发现或答案>
Key files: <相关文件路径>
Files changed: <修改文件列表和 commit hash>
Issues: <需要标记的问题>
```

### 3.3 三条禁令

Fork 有三条强制规则 [source: system-prompt-fork-usage-guidelines.md]：

**不要偷看**：工具返回结果中包含 `output_file` 路径，但 coordinator 不能去 Read 或 tail 它。读取 fork 的中间输出会把它的工具噪声拉入父 agent 上下文，违背 fork 的初衷。等通知就好。

**不要猜测**：启动 fork 后，coordinator 对 fork 发现了什么一无所知。通知以 user-role message 的形式在一个后续 turn 到达，不是 coordinator 自己写的。如果用户在通知到达前追问，告诉他们 fork 还在跑，不要编造结果。

**写指令而非背景**：因为 fork 继承了上下文，prompt 是一个指令——做什么，不做什么，另一个 agent 在处理什么。不需要复述背景。

### 3.4 Worktree 隔离

Agent tool 还支持 `isolation: "worktree"` 参数，让 agent 在一个临时 git worktree 中运行 [source: tool-description-agent-usage-notes.md]。Agent 获得仓库的隔离副本。如果 agent 没做任何修改，worktree 自动清理；如果有修改，返回 worktree 路径和分支名。

这与 EnterWorktree 工具有区别。EnterWorktree 是用户主动要求在 worktree 中工作时才使用的工具，[source: tool-description-enterworktree.md] 它在 `.claude/worktrees/` 下创建隔离环境并切换整个 session 的工作目录。而 agent tool 的 `isolation` 参数是在委派时自动创建的临时隔离，不需要用户显式请求。

---

## 四、Team/Swarm 协作：多 Agent 并行工程

### 4.1 Team 的生命周期

Team 是 Claude Code 中最高级别的协作模式。创建一个 team 时（通过 TeamCreate），系统同时创建两样东西：
- team 配置文件：`~/.claude/teams/{team-name}/config.json`
- 任务列表目录：`~/.claude/tasks/{team-name}/`

Team 的工作流是一个七步循环：

1. TeamCreate 创建 team 和 task list
2. 用 TaskCreate 创建任务，设定依赖关系
3. 用 Agent tool 的 `team_name` 和 `name` 参数生成 teammate
4. 用 TaskUpdate 的 `owner` 字段分配任务给空闲的 teammate
5. Teammate 执行任务，完成后通过 TaskUpdate 标记
6. Teammate 每轮结束后自动进入 idle 状态并发送通知
7. 任务全部完成后，通过 SendMessage 发送 shutdown_request，然后调用 TeamDelete 清理资源

### 4.2 通信模型

Team 中的通信通过 SendMessage 工具实现。系统提示反复强调一个要点：[source: system-prompt-teammate-communication.md] "Just writing a response in text is not visible to others on your team — you MUST use the SendMessage tool."

SendMessage 支持两种目标：
- 精确发送：`to: "researcher"`，发给指定 teammate
- 广播：`to: "*"`，发给全部 teammate（系统标注为 "expensive, linear in team size"，应谨慎使用）

消息的传递是自动的。系统提示明确说：[source: tool-description-teammatetool.md] "Messages from teammates are automatically delivered to you. You do NOT need to manually check your inbox." 当 coordinator 正忙时（在处理一个 turn），消息排队等待，turn 结束后投递。UI 会显示一个简短的通知提示有消息等待。

### 4.3 Task 协调机制

Team 内的任务管理建立在共享 task list 之上。每个 teammate 的行为模式是 [source: tool-description-tasklist-teammate-workflow.md]：

1. 完成当前任务后，调用 TaskList 查找可用工作
2. 寻找 status 为 pending、没有 owner、blockedBy 为空的任务
3. 优先按 ID 顺序认领（低 ID 的任务往往为后续任务搭建上下文）
4. 用 TaskUpdate 设置 `owner` 为自己的名字来认领
5. 如果所有可用任务都被阻塞，通知 team lead 或帮忙解除阻塞

TaskCreate 本身定义了清晰的触发条件：[source: tool-description-taskcreate.md] 三步以上的复杂任务、需要仔细规划的多操作任务、plan mode 中、用户明确要求 todo list、用户给出多个任务项。单步简单任务不需要创建 task。

### 4.4 Teammate Idle 状态

Idle 状态是 team 协作中最容易被误解的部分。系统提示花了大量篇幅解释：[source: tool-description-teammatetool.md] "Teammates go idle after every turn — this is completely normal and expected."

几个关键事实：
- Idle 不等于完成或不可用。它只是表示 teammate 在等待输入
- Idle 的 teammate 可以接收消息，收到消息会自动唤醒
- 发消息后立刻 idle 是正常流程，不代表出错了
- Coordinator 不需要回应 idle 通知，除非要分配新工作

还有一个有用的设计细节：当 teammate A 给 teammate B 发私信时，idle 通知中会包含一条简短摘要。[source: tool-description-teammatetool.md] "This gives you visibility into peer collaboration without the full message content." Coordinator 能看到协作在发生，但不需要读完整消息内容。

### 4.5 Agent 类型选择

生成 teammate 时，选择哪个 `subagent_type` 取决于任务需要什么工具 [source: tool-description-teammatetool.md]：

- 只读 agent（如 Explore、Plan）不能编辑或写入文件。只适合研究、搜索、规划。绝不能分配实现任务。
- 全能 agent（如 general-purpose）拥有所有工具，包括文件编辑和 bash。适合需要做变更的任务。
- `.claude/agents/` 中定义的自定义 agent 可能有自己的工具限制，需要查看描述。

### 4.6 关闭流程

Team 的关闭有一套严格协议。系统提示要求：[source: system-reminder-team-shutdown.md] "You MUST shut down your team before preparing your final response." 用户在 team 完全关闭前无法收到最终回复。

关闭流程：先通过 `requestShutdown` 请求每个 teammate 优雅退出，等待 shutdown approval，然后调用 cleanup 操作清理 team，最后才能准备最终回复。

TeamDelete 工具负责资源清理：[source: tool-description-teamdelete.md] 删除 team 目录和 task 目录，清除 session 中的 team 上下文。如果 team 中还有活跃成员，TeamDelete 会失败——必须先关闭所有 teammate。

---

## 五、三种模式对比

| 维度 | Subagent | Fork | Team/Swarm |
|------|----------|------|------------|
| 上下文 | 全新，需要完整 prompt | 继承父 agent 全部上下文 | 每个 member 独立上下文 |
| 启动成本 | 中等（新建对话） | 低（共享 prompt cache） | 高（创建 team + task + member） |
| 隔离级别 | 完全隔离 | 上下文共享，执行隔离 | 完全隔离，有协调层 |
| 通信方式 | 单次返回结果 | 单次返回 + 后续通知 | 持续双向消息 |
| 并行能力 | 支持多个同时启动 | 支持多个同时启动 | 原生多 agent 并行 |
| 适用场景 | 独立调查、第二意见 | 研究、实现（省上下文） | 大型并行工程任务 |
| 生命周期 | 任务完成即销毁 | 任务完成即销毁 | 显式创建和销毁 |

选哪种模式的判断逻辑：

1. 需要多个 agent 持续协作、有任务依赖关系？用 Team/Swarm
2. 需要保持上下文、省 token？用 Fork（省略 subagent_type）
3. 需要全新视角、独立判断？用 Subagent（指定 subagent_type）
4. 需要文件系统隔离？用 Agent tool 的 `isolation: "worktree"` 参数

前台和后台的选择也有讲究：[source: tool-description-agent-usage-notes.md] 需要等 agent 结果才能继续工作时用前台模式；有独立的并行工作要做时用后台模式。后台 agent 完成后系统自动通知，不需要轮询或 sleep。

---

## 六、设计意图

Claude Code 的多 agent 体系不是功能堆叠，而是围绕上下文成本和隔离需求构建的分层方案。

Fork 解决的是上下文膨胀问题。调查类任务的中间输出（文件列表、搜索结果、命令输出）往往比结论本身大一个数量级，把这些塞进主 agent 上下文是浪费。Fork 让中间过程在子进程中完成，只把结论带回来。同时，共享 prompt cache 使得 fork 的 token 成本远低于全新 subagent。

Subagent 解决的是独立性需求。当你需要一个不受之前分析影响的第二意见，或者需要一个专用工具面（如只读的 Explore agent）时，从零启动的 subagent 提供了干净的执行环境。

Team/Swarm 解决的是工程规模问题。当任务大到需要多个 agent 分工协作，有依赖关系需要管理，有中间结果需要共享时，team 提供了任务列表、消息系统、生命周期管理。

三种模式共享一个核心思想：让 agent 自己判断何时委派、委派给谁。系统提示不是通过硬编码的规则强制委派，而是提供清晰的判断标准和充足的使用示例，让 coordinator agent 基于对任务的理解做出决策。这大概也是 Anthropic 在 agent 设计中一以贯之的思路：给 agent 足够的上下文和原则，让它自己判断，而不是穷举所有可能的分支。
