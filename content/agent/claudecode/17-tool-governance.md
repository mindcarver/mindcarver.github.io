# 能用 grep 就别用 bash：Claude Code 的工具选择哲学

## 一、引言：为什么需要 13 条独立的工具使用指导

在 Claude Code 的 system prompt 中，工具使用指导不是一段笼统的"请优先使用内置工具"，而是被拆解为 13 条独立的规则文件，每条针对一个具体的工具选择场景。它们分别对读文件、编辑文件、搜索文件名、搜索文件内容、创建文件、保留 Bash、直接搜索、委派探索、子代理指导、任务管理、技能调用，以及 Bash 工具描述中的两条补充说明做出了精确的规定。

这种拆解说明一件事：工具选择不是运行时可以随意决定的偏好，而是预先编码到 system prompt 中的硬性约束。当 LLM 面对"我该怎么读一个文件"这种基础操作时，系统根本不想让它去权衡 `cat` 和 Read tool 的利弊，而是直接规定唯一正确的答案。

原因有两层。第一层，LLM 的训练数据里全是 Bash 命令的使用模式，模型天然倾向生成 `cat`、`grep`、`sed` 这类 shell 命令，不施加强约束，模型会在大量场景下退回 Bash 路径。第二层，专用工具与 Bash 等价命令之间的差异涉及 token 效率、行为可控性、安全审计等多个维度，足以影响整个 agent 系统的质量。

本文基于这 13 条源文件的原始文本，逐一拆解 Claude Code 的工具分层体系。

---

## 二、六条"不要用 Bash 做 X"的替代方案

Claude Code 的工具替代策略覆盖了文件操作的六个核心场景。每条规则结构一致：明确指出不要用哪个 Bash 命令，要用哪个专用工具替代。

### 2.1 读文件：Read 替代 cat/head/tail/sed

[source: system-prompt-tool-usage-read-files.md] 的指令干脆明确：

> "To read files use ${READ_TOOL_NAME} instead of cat, head, tail, or sed"

四个 Bash 命令被一次列作替代对象。Read tool 通过 `offset` 和 `limit` 参数统一了全文件输出、首尾截取和单行提取四种操作，并通过行号格式化输出提供引用能力。

### 2.2 编辑文件：Edit 替代 sed/awk

[source: system-prompt-tool-usage-edit-files.md] 规定：

> "To edit files use ${EDIT_TOOL_NAME} instead of sed or awk"

`sed` 和 `awk` 在命令行中使用正则表达式，多层转义极易引入隐蔽错误。Edit tool 采用结构化的 `old_string` / `new_string` 替换模式，且 `old_string` 必须在文件中唯一匹配，防止误替换。

### 2.3 搜索文件名：Glob 替代 find/ls

[source: system-prompt-tool-usage-search-files.md] 指出：

> "To search for files use ${GLOB_TOOL_NAME} instead of find or ls"

`find` 的命令行语法复杂到令人头疼，参数组合产生的行为往往难以预测。Glob tool 将搜索简化为 pattern 匹配，返回按修改时间排序的文件路径列表。在 AI agent 的场景中，"找到匹配某种模式的文件"就是全部需求，`find` 的高级功能反而是负担。

### 2.4 搜索文件内容：Grep 替代 grep/rg

[source: system-prompt-tool-usage-search-content.md] 的规则是：

> "To search the content of files, use ${GREP_TOOL_NAME} instead of grep or rg"

被替代的对象包括 `rg`（ripgrep），说明替代原因不是性能不足，而是专用 Grep tool 提供了结构化输出能力，`output_mode`、上下文行数控制、`glob` 文件过滤等参数让搜索结果可以直接被后续工具调用消费，无需 LLM 自行解析命令行输出。

### 2.5 创建文件：Write 替代 cat heredoc / echo 重定向

[source: system-prompt-tool-usage-create-files.md] 规定：

> "To create files use ${WRITE_TOOL_NAME} instead of cat with heredoc or echo redirection"

`cat << 'EOF' > file.py` 和 `echo "content" > file.py` 的共同问题是内容与命令混合，heredoc 定界符可能与文件内容冲突，echo 重定向面临引号嵌套的转义地狱。Write tool 将路径和内容分离为两个独立参数，消除了这些歧义。

### 2.6 直接输出：用文本回复替代 echo/printf

Bash 工具描述中的专用工具对照表列出了最后一条：通信用直接输出文本（NOT echo/printf）。[source: tool-description-bash-prefer-dedicated-tools.md] 背后的设计意图很直接：当 agent 需要向用户传递信息时，不应该启动一个 Bash 进程来执行 `echo`，而应该直接生成文本回复。信息传递根本不需要工具调用。

---

## 三、"Reserve Bash"原则：token 效率、行为可控性与安全审计

[source: system-prompt-tool-usage-reserve-bash.md] 提供了 Bash 工具的定位声明：

> "Reserve using the ${BASH_TOOL_NAME} exclusively for system commands and terminal operations that require shell execution. If you are unsure and there is a relevant dedicated tool, default to using the dedicated tool and only fallback on using the ${BASH_TOOL_NAME} tool for these if it is absolutely necessary."

这条规则包含两个指令。第一，Bash 被保留给"需要 shell 执行的系统命令和终端操作"，运行测试、启动服务、执行构建、操作 git 才是 Bash 的正当领地。第二，当不确定时，默认使用专用工具，只有"绝对必要时"才回退到 Bash。这是一个"默认拒绝"原则：Bash 是例外，不是常态。

背后的动机可以从源文件中提取出几个维度。

**Token 效率**。专用工具返回结构化数据，而 Bash 命令返回原始终端输出，包含列对齐空格、颜色控制字符、进度条输出等噪音，全部计入 context window 的 token 消耗。

**行为可控性**。[source: tool-description-bash-built-in-tools-note.md] 直接陈述了这一点：

> "While the ${BASH_TOOL_NAME} tool can do similar things, it's better to use the built-in tools as they provide a better user experience and make it easier to review tool calls and give permission."

专用工具的调用签名是可预测的。Read tool 的参数是文件路径和行范围，Edit tool 的参数是旧文本和新文本。而 Bash 命令的行为取决于 shell 解析、环境变量、当前工作目录等多种运行时状态。

**安全审计**。[source: tool-description-bash-prefer-dedicated-tools.md] 用"IMPORTANT"标记强化了强制性：

> "IMPORTANT: Avoid using this tool to run ${READ_ONLY_SEARCHING_BASH_COMMANDS} commands, unless explicitly instructed or after you have verified that a dedicated tool cannot accomplish your task."

只有在两种条件下才允许使用 Bash 等价命令：被明确指示，或者已经验证专用工具无法完成任务。Bash 的使用被置于"需要正当化理由"的位置，任何非必要的 Bash 调用都应该被视为异常。

---

## 四、工具分层：直接工具、子代理与专用模式

Claude Code 的工具体系按照"简单性阈值"分层。两条规则直接描述了这种结构。

### 4.1 直接搜索：简单任务的快速路径

[source: system-prompt-tool-usage-direct-search.md] 规定：

> "For simple, directed codebase searches (e.g. for a specific file/class/function) use ${SEARCH_TOOLS} directly."

最低层是直接调用 Glob 和 Grep，适用于"简单的、有明确方向的搜索"。搜索目标已知，范围明确，结果数量有限。这是最快的路径，无需经过中间层。

### 4.2 委派探索：复杂任务的升级策略

[source: system-prompt-tool-usage-delegate-exploration.md] 定义了更高层：

> "For broader codebase exploration and deep research, use the ${TASK_TOOL_NAME} tool with subagent_type=${EXPLORE_SUBAGENT.agentType}. This is slower than using ${SEARCH_TOOLS} directly, so use this only when a simple, directed search proves to be insufficient or when your task will clearly require more than ${QUERY_LIMIT} queries."

Explore agent 被明确定位为"比直接搜索更慢"的选择，只在两种条件下激活：简单搜索已证明不够用，或者任务明确需要超过一定数量的查询。子代理拥有独立的 context window，不会被主对话的历史消息污染，但消耗额外的 token 和计算资源。

### 4.3 分层总结

综合以上规则，工具分层可以概括为三个层级：

1. 直接工具层（Glob/Grep）：目标明确的简单搜索，速度最快，资源消耗最低。
2. 子代理层（Explore agent）：广度探索，独立 context window，速度较慢但能力更强。
3. 通用执行层（General-purpose agent）：需要实际修改代码的复杂任务，完整的工具访问权限。

每一层都有明确的启用条件，不存在灰色地带。

---

## 五、Subagent Guidance：何时用子代理而非直接工具

[source: system-prompt-tool-usage-subagent-guidance.md] 提供了子代理使用的四条核心指导。

第一条确立了"任务匹配"原则：

> "Use the ${TASK_TOOL_NAME} tool with specialized agents when the task at hand matches the agent's description."

只有任务与代理的专长描述匹配时才使用子代理，防止"用大炮打蚊子"。

第二条定义了子代理的两个核心价值：

> "Subagents are valuable for parallelizing independent queries or for protecting the main context window from excessive results."

并行化独立查询，以及保护主 context window 不被大量搜索结果填满。子代理本质上是一种"信息漏斗"，宽口端接收原始数据，窄口端输出精炼结论。

第三条是刹车机制：

> "but they should not be used excessively when not needed."

直接回应过度委派的反模式。每次子代理调用意味着一个全新的 LLM 会话，消耗独立的 token 配额，委派前必须判断是否真有必要。

第四条是最容易被违反但也最关键的：

> "Importantly, avoid duplicating work that subagents are already doing - if you delegate research to a subagent, do not also perform the same searches yourself."

不要重复子代理已进行的工作。当主 agent 对子代理不够信任时，会本能地"并行"执行相同的搜索作为验证手段。这不仅是 token 浪费，更违背了设计初衷：一旦委派，就应该信任结果。

---

## 六、Task Management：TodoWrite 的使用时机

[source: system-prompt-tool-usage-task-management.md] 定义了 TodoWrite 的使用规范：

> "Break down and manage your work with the ${TODOWRITE_TOOL_NAME} tool. These tools are helpful for planning your work and helping the user track your progress."

TodoWrite 有双重功能：对内规划工作，对外展示进度。当任务包含多个步骤时，先分解为 todo 列表，再逐一执行。

规则的后半段规定了一个行为约束：

> "Mark each task as completed as soon as you are done with the task. Do not batch up multiple tasks before marking them as completed."

"Do not batch up"针对的是一种自然的优化倾向：连续完成多个任务后一次性更新状态。代价是用户失去实时进度可见性。在交互式 CLI 环境中，实时反馈是建立信任的基础，即时更新被规定为强制要求。

---

## 七、Skill Invocation：slash command 的路由机制

[source: system-prompt-tool-usage-skill-invocation.md] 定义了 slash command 与 Skill tool 的关系：

> "/<skill-name> (e.g., /commit) is shorthand for users to invoke a user-invocable skill. When executed, the skill gets expanded to a full prompt. Use the ${SKILL_TOOL_NAME} tool to execute them."

slash command 不是独立的功能调用机制，而是用户友好的语法糖。`/commit` 在执行时被展开为完整的 prompt，由 Skill tool 驱动实际执行逻辑。

规则的后半段包含严格约束：

> "IMPORTANT: Only use ${SKILL_TOOL_NAME} for skills listed in its user-invocable skills section - do not guess or use built-in CLI commands."

"不要猜测"直接针对 LLM 的幻觉式工具调用。当模型面对不熟悉的 slash command 时，可能"猜测"该命令对应某个 skill 并尝试调用。Skill tool 被严格限制在已列出的可调用技能范围内。工具调用的合法性由外部的、确定性的技能注册表决定，LLM 不能自行扩展调用范围，这是"最小权限"原则在 agent 系统中的具体实现。

---

## 八、工具治理的核心逻辑

Claude Code 的 13 条工具使用指导形成了四个原则。

**专用优于通用**。六个文件操作场景各有其专用工具，Bash 被保留为最后手段。通过在 system prompt 中反复强化"专用工具优先"的信息，Claude Code 在 LLM 的推理过程中建立了一道默认屏障，Bash 退路需要额外理由才能被激活。

**简单优于复杂**。从直接搜索到子代理委派，每一层的启用都有明确的复杂度阈值。简单搜索不需要子代理，单个文件读取不需要 Glob，直接回复不需要 echo。

**透明优于高效**。TodoWrite 的即时更新要求、专用工具的结构化输出、子代理结果必须由主 agent 转述，这些规定的共同目标是让 agent 的行为对用户可见、可理解、可审计。某些"优化"可以减少工具调用次数，但如果牺牲了透明度，就会被禁止。

**信任但不冗余**。"不要重复子代理已进行的工作"体现了一种平衡：一旦委派就信任结果，但委派本身必须经过必要性判断。这种"谨慎委派、信任执行"的模式避免了过度委派和重复验证两种反模式。

让 LLM 操作外部系统的 agent 框架都需要回答同一个问题：当模型有多种方式完成同一件事时，如何确保它选择最合适的那一种？Claude Code 的回答是：不要依赖模型的判断力，而是在 system prompt 中为每一个具体场景编码唯一的正确答案。这不是对 LLM 能力的不信任，是对系统可靠性的合理要求。当工具选择的差异涉及安全、效率和用户体验时，确定性比灵活性更重要。
