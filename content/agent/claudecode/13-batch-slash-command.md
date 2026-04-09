# 一次改 30 个文件：Claude Code 的并行修改引擎

## 1. 引言：为什么需要大规模并行代码修改

日常开发中有一类变更，规模远超"修一个 bug"的范畴。把整个代码库的日志框架从 Log4j 迁移到 SLF4J；对 50 个模块打同一个安全补丁；在数百个文件中统一命名风格。特征是每个文件的改动逻辑相似，但文件数量庞大，串行处理效率极低。

Claude Code 的 `/batch` slash command 就是干这个的。它的系统提示 [source: agent-prompt-batch-slash-command.md] 一上来就说："You are orchestrating a large, parallelizable change across this codebase." 思路直接：把大规模变更拆成 5 到 30 个独立单元，用 git worktree 做物理隔离，以 `run_in_background` 模式并行启动 worker agent，最后汇总结果。

底层逻辑不复杂。大规模代码变更中，大部分工作是可并行的。把任务切成互不依赖的独立单元，并行执行就能把耗时从 O(n) 压到接近 O(1)。Batch 的全部设计围绕拆分、隔离、验证、汇总这几个问题展开。

下面基于 `agent-prompt-batch-slash-command.md` 和 `system-prompt-worker-instructions.md` 两个源文件，拆解 Batch 引擎的三阶段编排架构和 worker 执行协议。

## 2. Phase 1: Research and Plan -- 从探索到拆分的规划阶段

Batch 的第一个阶段是标准的 Plan Mode 流程。Coordinator（编排者）先调用 `EnterPlanMode` 进入规划模式，然后做三件事：探索范围、分解单元、确定测试方案。

### 2.1 子代理探索：摸清变更范围

Coordinator 不是一个人闷头干。[source: agent-prompt-batch-slash-command.md] 要求："Launch one or more subagents (in the foreground -- you need their results) to deeply research what this instruction touches. Find all the files, patterns, and call sites that need to change. Understand the existing conventions so the migration is consistent."

两个细节。探索用的是前台子代理（foreground），因为 Coordinator 必须拿到探索结果才能继续。探索的目标也不只是找文件列表，还要搞清楚现有的代码规范（conventions）。这些规范后面会被注入每个 worker 的 prompt 中，确保 30 个 worker 各自为战时还能做出风格一致的决定。

### 2.2 分解为独立单元

探索完成后，Coordinator 将整体工作分解为 5 到 30 个独立单元。这个范围通过 `${MIN_5_UNITS}` 和 `${MAX_30_UNITS}` 配置，系统根据任务规模动态调整并行度。[source: agent-prompt-batch-slash-command.md] 给出的缩放规则是："Scale the count to the actual work: few files -> closer to ${MIN_5_UNITS}; hundreds of files -> closer to ${MAX_30_UNITS}."

单元数量跟实际工作量成正比。少量文件的变更用接近 5 个单元；数百个文件的大规模迁移可以扩展到 30 个。这种弹性避免了两种极端：拆得太细，协调开销超过收益；拆得太粗，并行度不够，等于白折腾。

### 2.3 确定 e2e 测试方案

这是 Phase 1 里最要紧的一步。Worker 运行在后台，没有交互能力，用户说什么它听不到。所以 Coordinator 必须在规划阶段就把验证方案定死。

[source: agent-prompt-batch-slash-command.md] 列出了四种 e2e 测试路径，覆盖不同类型的变更场景。如果所有路径都不适用，Coordinator 必须用 `AskUserQuestion` 工具主动询问用户，给出 2-3 个具体选项而非开放式提问。

源文件对此有一条硬性禁令："Do not skip this -- the workers cannot ask the user themselves." [source: agent-prompt-batch-slash-command.md] 道理很简单：worker 是自治的、不可交互的执行单元，所有需要人判断的事必须在规划阶段搞定。

## 3. 独立单元设计原则

Batch 的并行能力建立在"独立单元"这个核心抽象上。源文件 [source: agent-prompt-batch-slash-command.md] 对独立单元设了三个条件：

独立可实施。原文要求每个单元必须 "Be independently implementable in an isolated git worktree (no shared state with sibling units)"。任何两个单元之间不能有文件级别的依赖。如果单元 A 和单元 B 都要改同一个文件，它们就不能作为独立单元，要么合并，要么把这个文件的修改明确分配给其中一个。

可独立合并。原文要求每个单元必须 "Be mergeable on its own without depending on another unit's PR landing first"。这比"独立可实施"约束更强。即使两个单元改的是不同文件，如果 A 的逻辑正确性依赖 B 的修改先合入，它们也不该被拆开。

均匀大小。原文要求 "Be roughly uniform in size (split large units, merge trivial ones)"。为了保证负载均衡。某个单元特别大，它就是整个批次的耗时瓶颈；过小的单元会浪费 worker 的启动开销。

划分策略上，源文件给出明确的优先级："Prefer per-directory or per-module slicing over arbitrary file lists." [source: agent-prompt-batch-slash-command.md] 按目录或模块切片比按任意文件列表分组更好，因为目录和模块的边界通常对应代码的逻辑边界，更容易满足独立性要求。也让每个 worker 的变更范围更内聚，方便审查。

## 4. e2e 测试食谱：四条验证路径

Batch 的 e2e 测试覆盖了 UI、CLI、API 和既有测试套件几类场景。[source: agent-prompt-batch-slash-command.md] 列出了四条路径：

Browser Automation。适用于 UI 变更。原文描述："A `claude-in-chrome` skill or browser-automation tool (for UI changes: click through the affected flow, screenshot the result)"。Worker 通过浏览器自动化工具在修改后的界面上走一遍流程并截图。

tmux CLI 验证。适用于 CLI 变更。原文描述："A `tmux` or CLI-verifier skill (for CLI changes: launch the app interactively, exercise the changed behavior)"。Worker 在 tmux 会话中启动应用，交互式触发被修改的行为。

dev-server + curl。适用于 API 变更。原文描述："A dev-server + curl pattern (for API changes: start the server, hit the affected endpoints)"。Worker 启动开发服务器，用 curl 或类似工具请求受影响的端点，检查响应。

既有测试套件。原文描述："An existing e2e/integration test suite the worker can run"。项目已有端到端或集成测试的话，worker 直接跑。

四条路径都不适用时，Coordinator 通过 `AskUserQuestion` 向用户求助，必须提供 2-3 个具体选项。原文给的示例："Screenshot via chrome extension"、"Run `bun run dev` and curl the endpoint"、"No e2e -- unit tests are sufficient"。[source: agent-prompt-batch-slash-command.md]

测试食谱最终写成简短、可执行的步骤集："Write the recipe as a short, concrete set of steps that a worker can execute autonomously. Include any setup (start a dev server, build first) and the exact command/interaction to verify." [source: agent-prompt-batch-slash-command.md] 这个食谱会被逐字复制到每个 worker 的 prompt 中。

## 5. Phase 2: Spawn Workers -- worktree 隔离与并行启动

规划完成并获得用户批准后，进入第二阶段。

### 5.1 worktree 隔离

[source: agent-prompt-batch-slash-command.md] 对 worker 的启动方式有两条硬性要求："All agents must use `isolation: "worktree"` and `run_in_background: true`."

`isolation: "worktree"` 意味着每个 worker 在独立的 git worktree 中执行。Git worktree 是 Git 原生功能，允许同一个仓库下创建多个工作目录，每个可以独立 checkout 不同分支。30 个 worker 就有 30 个互不干扰的文件系统视图，同时改不同文件（甚至相同文件）也不会冲突。

`run_in_background: true` 让 worker 在后台异步执行。Coordinator 启动 worker 后不需要等待其完成，而是继续启动下一个 worker，最终通过系统通知（completion notification）收集结果。

### 5.2 单消息并行启动

[source: agent-prompt-batch-slash-command.md] 要求："Launch them all in a single message block so they run in parallel." 这是关键的性能指令。Coordinator 如果逐个启动 worker，每个启动请求都要等一轮 LLM 推理，30 个 worker 就要 30 轮。把所有启动请求放在一个消息块中，一轮推理就能触发全部并行启动。

"单消息块并行启动"把启动开销从 O(n) 降到 O(1)。LLM 推理延迟通常以秒计，30 路并行场景下，启动阶段从几分钟缩到几秒。

关于 agent 类型，源文件给出默认建议："Use `subagent_type: "general-purpose"` unless a more specific agent type fits." [source: agent-prompt-batch-slash-command.md] 原因是 worker 要干的事很多：读文件、写文件、跑测试、提交代码、创建 PR，都需要 general-purpose agent 的完整工具权限。

## 6. Worker 指令设计：自包含 prompt 与共享模板

Worker prompt 的设计是 Batch 架构里最精细的部分。[source: agent-prompt-batch-slash-command.md] 要求每个 worker 的 prompt 必须"fully self-contained"，包含五个部分：

1. 总体目标（The overall goal）：用户原始指令的完整内容，通过 `${USER_INSTRUCTIONS}` 变量注入。
2. 本单元具体任务（This unit's specific task）：从计划中逐字复制的标题、文件列表和变更描述。逐字复制（copied verbatim）确保 worker 接收到的任务描述跟计划中经过用户审批的内容完全一致。
3. 代码规范（Codebase conventions）：Phase 1 探索阶段发现的项目编码规范。30 个 worker 之间没有通信机制，必须靠同一套规范来做出风格一致的决定。
4. e2e 测试食谱（The e2e test recipe）：从计划中逐字复制的测试步骤，或跳过 e2e 的原因说明。
5. Worker 指令模板（The worker instructions）：通过 `${WORKER_PROMPT}` 变量注入的共享执行协议，逐字复制到每个 worker 的 prompt 中。

"共享模板 + 单元专属信息"的 prompt 架构有两个好处。共享模板确保所有 worker 遵循统一的执行流程，降低编排复杂度。单元专属信息（文件列表、变更描述）是计划阶段经过用户审批的内容，逐字复制避免信息传递失真。

Worker 指令模板还有一个设计特点：它作为独立的系统提示文件存在。`system-prompt-worker-instructions.md` [source: system-prompt-worker-instructions.md] 单独维护，通过 `${WORKER_PROMPT}` 注入每个 worker 的 prompt。模块化设计让 worker 执行协议可以独立于 Batch 编排逻辑迭代。

## 7. Phase 3: Track Progress -- 状态追踪与结果汇总

所有 worker 启动后，Coordinator 进入第三阶段：追踪进度。

### 7.1 初始状态表格

Coordinator 在启动 worker 后立即渲染一个初始状态表。[source: agent-prompt-batch-slash-command.md] 的模板格式：

```
| # | Unit | Status | PR |
|---|------|--------|----|
| 1 | <title> | running | -- |
| 2 | <title> | running | -- |
```

这个表格在启动时就建立了全局视图，用户一目了然。

### 7.2 异步更新

Worker 在后台异步执行，每完成一个就发送完成通知。Coordinator 解析通知中的 `PR: <url>` 行，重新渲染状态表格，将对应单元的状态更新为 `done` 或 `failed`，填入 PR 链接。

[source: agent-prompt-batch-slash-command.md] 要求："As background-agent completion notifications arrive, parse the `PR: <url>` line from each agent's result and re-render the table with updated status (`done` / `failed`) and PR links."

状态更新是增量的。每收到一个通知就更新一次，不等所有 worker 完成。用户能实时看到进度。

### 7.3 失败记录与最终汇总

对于未能产出 PR 的 worker，Coordinator 保留简要的失败原因："Keep a brief failure note for any agent that did not produce a PR." [source: agent-prompt-batch-slash-command.md]

所有 worker 报告完毕后，Coordinator 渲染最终表格和一行总结。原文给的示例："22/24 units landed as PRs"。[source: agent-prompt-batch-slash-command.md] 简洁，用户一眼看到成功率，再通过表格里的 PR 链接逐个审查或合并。

## 8. Worker Instructions 分析：五步执行协议

`system-prompt-worker-instructions.md` [source: system-prompt-worker-instructions.md] 定义了每个 worker 完成代码修改后必须遵循的五步协议。每一步有明确的职责边界：

Simplify（简化）。原文要求："Invoke the `${SKILL_TOOL_NAME}` tool with `skill: "simplify"` to review and clean up your changes." Worker 调用 simplify 技能检查自己的修改有没有冗余或低效代码。简化放在第一步，确保后续测试和提交的代码已经过初步清理。

Run unit tests（运行单元测试）。原文要求："Run the project's test suite (check for package.json scripts, Makefile targets, or common commands like `npm test`, `bun test`, `pytest`, `go test`). If tests fail, fix them." Worker 主动探测项目的测试框架和运行命令，执行测试，失败时自行修复。这体现了 worker 的自治设计：不是报告测试结果，而是有义务修好自己搞坏的东西。

Test end-to-end（端到端测试）。原文要求："Follow the e2e test recipe from the coordinator's prompt (below). If the recipe says to skip e2e for this unit, skip it." 执行 Coordinator 规划阶段确定的测试食谱。如果食谱明确说跳过，worker 可以安全跳过。

Commit and push（提交并推送）。原文要求："Commit all changes with a clear message, push the branch, and create a PR with `gh pr create`. Use a descriptive title. If `gh` is not available or the push fails, note it in your final message." Worker 用 `gh` CLI 创建 PR，工具不可用或推送失败时要在最终消息中记录原因，不能静默失败。

Report（报告）。原文要求："End with a single line: `PR: <url>` so the coordinator can track it. If no PR was created, end with `PR: none -- <reason>`." 这条严格的格式规则让 Coordinator 通过简单的文本解析提取结果，不需要复杂的结构化输出解析。

五步协议的顺序遵循"先清理、再验证、再提交、最后报告"的原则。简化在测试之前，测试的是干净代码；单元测试在 e2e 之前，快速失败降低返工成本；提交在报告之前，确保报告的 PR 链接有效。

整个 Worker Instructions 提示文件极为简洁，总共只有五条指令。这不是偶然。Worker 的 prompt 中已经包含了完整的任务描述、代码规范和测试食谱，Worker Instructions 只需要定义通用的执行流程。具体的"做什么"来自 prompt 的其他部分，"怎么做"由这个模板统一规定。

## 9. 与 Git Worktree 的协同设计

Batch 的并行能力很大程度上依赖 git worktree 提供的文件系统级隔离。

传统的并行修改通常面临两个问题：文件锁冲突和分支管理混乱。两个进程同时改同一个工作目录里的文件，产生竞态条件；每个进程各自创建分支，在同一个工作目录中来回切换，既低效又容易出错。

Git worktree 从根子上解决了这两个问题。每个 worker 跑在独立的 worktree 里，有自己的文件系统视图和分支。Worker A 在 worktree-1 的 branch-A 上改 `src/auth/login.ts`，Worker B 在 worktree-2 的 branch-B 上改 `src/api/handler.ts`，互不干扰。

这种隔离模型直接影响了独立单元的设计原则。前面提到的"独立可实施"条件——"no shared state with sibling units"——本质上是在要求：单元之间的 worktree 隔离不能被打破。如果两个单元都需要修改同一个文件，worktree 隔离就无法自然解决合并冲突，这正是独立性约束存在的根本原因。

Batch 的 `isolation: "worktree"` 参数让每次 Agent tool 调用都自动创建独立 worktree。Coordinator 不用手动管理 worktree 的创建和销毁，这个生命周期由 Claude Code 的 Agent 基础设施自动处理。Worker 完成后，worktree 可以保留（供 PR 审查）或清理。

## 10. 总结

Batch Slash Command 的核心结构三句话概括：一个人规划，一群人执行，一个人收账。

一个人规划。Coordinator 是唯一跟用户交互的角色。探索代码库、理解规范、拆分任务、确定测试方案，产出一份经过用户审批的计划。所有需要人判断的决策在这个阶段集中完成，后续执行阶段用户可以撒手不管。

一群人执行。Worker 是无状态的、自治的执行单元。接收自包含的 prompt，在隔离的 worktree 中独立工作，遵循五步执行协议，产出 PR。Worker 之间不通信、不协调，独立性是并行安全的前提。"无通信并行"牺牲了 worker 之间的协作能力，但换来了高并行度和低协调开销。

一个人收账。Coordinator 通过结构化状态表格追踪每个 worker 的完成情况，解析标准的 `PR: <url>` 报告格式，向用户呈现结果汇总。

从系统设计角度看，有几条值得借鉴的原则。第一，规划的深度决定执行的可靠性。Phase 1 中收集代码规范、确定测试方案、验证单元独立性，都是在为 Phase 2 的自治执行铺路。规划越充分，执行中出现意外需要人工干预的概率越低。第二，接口契约化降低协调成本。Worker 与 Coordinator 之间的唯一接口就是 `PR: <url>` 这一行文本，Coordinator 不需要理解 worker 的内部状态，只要解析一个标准化输出。第三，隔离粒度决定并行上限。Git worktree 提供的文件系统级隔离让 30 路并行成为可能。更粗粒度的隔离（如进程级）受限于操作系统资源；更细粒度的隔离（如行级）合并冲突的成本超过并行收益。文件级隔离恰好是个平衡点。

Batch 也有明确的边界。它不适用于需要跨文件全局一致性的变更（比如重命名一个被到处引用的公共 API），不适用于有复杂依赖链的多阶段任务（比如先改数据库 schema 再改 ORM 再改 API），也不适用于需要持续人类反馈的探索性修改。它就是为 large, parallelizable change 设计的。

[source: agent-prompt-batch-slash-command.md, system-prompt-worker-instructions.md]
