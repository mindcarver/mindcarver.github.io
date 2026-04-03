# AI 执行每一步之前，你都能插手：Claude Code 的 11 个 Hooks

## 1. 引言：为什么 Claude Code 需要生命周期钩子

Claude Code 的核心工作流是接收用户指令、调用工具、返回结果。但真实开发场景远比这个线性流程复杂：你希望在每次写入文件后自动格式化代码，在执行 Bash 命令前记录审计日志，在 session 结束时发送通知，在上下文压缩前确认保留哪些关键信息。

这些需求有一个共同特征，它们是对特定生命周期事件的自动响应。Claude Code 的 Hooks 系统就是为此而生：它允许用户在 Claude Code 的关键生命周期节点注入自定义逻辑，把一个被动的 AI 助手变成可编程的决策引擎。

正如源文件所述："Hooks run commands at specific points in Claude Code's lifecycle." [source: system-prompt-hooks-configuration.md] 每一个 Hook 都是一个插入点，它既可以是简单的 shell 命令，也可以是一个拥有完整工具访问能力的 AI Agent。

## 2. 十种 Hook 事件详解

Claude Code 定义了十种生命周期事件，每种事件对应不同的执行时机和用途：

| Event | Matcher | Purpose |
|-------|---------|---------|
| PermissionRequest | Tool name | Run before permission prompt |
| PreToolUse | Tool name | Run before tool, can block |
| PostToolUse | Tool name | Run after successful tool |
| PostToolUseFailure | Tool name | Run after tool fails |
| Notification | Notification type | Run on notifications |
| Stop | - | Run when Claude stops (including clear, resume, compact) |
| PreCompact | "manual"/"auto" | Before compaction |
| PostCompact | "manual"/"auto" | After compaction (receives summary) |
| UserPromptSubmit | - | When user submits |
| SessionStart | - | When session starts |

[source: system-prompt-hooks-configuration.md]

以下逐一分析：

**PermissionRequest** 在 Claude Code 向用户展示权限确认弹窗之前触发。它的 matcher 是 Tool name，意味着你可以针对特定工具设置权限策略。这是实现自动化权限管理的核心事件：通过返回 `allow` 或 `deny` 决策，可以完全绕过用户手动确认环节。

**PreToolUse** 在工具执行之前触发，具备阻断能力。源文件明确指出它可以 "Run before tool, can block" [source: system-prompt-hooks-configuration.md]。你可以用它做前置校验：比如阻止对 `production.config` 的写入操作，或者在执行 `rm` 命令前进行安全检查。

**PostToolUse** 在工具成功执行后触发。这是最常见的自动化场景入口：文件写入后自动格式化、代码编辑后自动运行测试。它接收完整的 `tool_response` 数据，让你可以基于执行结果做进一步处理。

**PostToolUseFailure** 在工具执行失败后触发。它填补了 PostToolUse 的盲区，专门处理异常路径。你可以用它记录失败日志、发送告警通知，或者自动重试。

**Notification** 在通知事件发生时触发。它的 matcher 是 Notification type 而非 Tool name，说明通知系统拥有独立的分类体系。

**Stop** 当 Claude 停止响应时触发。源文件特别注明它 "including clear, resume, compact" [source: system-prompt-hooks-configuration.md]，即覆盖了清除对话、恢复会话、上下文压缩等停止场景。它没有 matcher，因为停止行为是全局性的。

**PreCompact** 在上下文压缩（compaction）之前触发。matcher 可以是 `"manual"` 或 `"auto"`，区分用户手动触发和系统自动触发的压缩。你可以在压缩前指定哪些信息必须保留。

**PostCompact** 在上下文压缩完成后触发。它接收压缩后的摘要内容（"receives summary" [source: system-prompt-hooks-configuration.md]），你可以据此做质量检查或额外处理。

**UserPromptSubmit** 当用户提交新 prompt 时触发。没有 matcher，因为它作用于所有用户输入。可用于输入预处理、敏感词过滤，或记录用户行为日志。

**SessionStart** 当 session 启动时触发。这是初始化场景的理想入口：加载项目配置、设置环境变量、显示欢迎信息。

## 3. 三种 Hook 类型

Claude Code 支持三种 Hook 类型，复杂度递增：

### Command Hook

最基础的类型，直接执行 shell 命令：

```json
{ "type": "command", "command": "prettier --write $FILE", "timeout": 30 }
```

[source: system-prompt-hooks-configuration.md]

Command Hook 接收 JSON 格式的 stdin 输入，通过管道传递上下文数据。它适用于确定性的自动化操作：格式化代码、记录日志、运行测试。执行效率高，延迟可控（通过 `timeout` 字段），是日常使用最广泛的 Hook 类型。

### Prompt Hook

利用 LLM 评估条件的 Hook：

```json
{ "type": "prompt", "prompt": "Is this safe? $ARGUMENTS" }
```

[source: system-prompt-hooks-configuration.md]

Prompt Hook 的独特之处在于它引入了 AI 判断能力。当你需要的不是固定规则而是语义理解时，比如"这段代码修改是否安全"，Prompt Hook 可以让 LLM 基于上下文做出判断。

但 Prompt Hook 有一个关键限制：源文件明确指出它 "Only available for tool events: PreToolUse, PostToolUse, PermissionRequest" [source: system-prompt-hooks-configuration.md]。它不能用于 Stop、SessionStart 等非工具事件。

### Agent Hook

最强大的类型，启动一个拥有工具访问能力的 Agent：

```json
{ "type": "agent", "prompt": "Verify tests pass: $ARGUMENTS" }
```

[source: system-prompt-hooks-configuration.md]

Agent Hook 与 Prompt Hook 的可用范围相同，也仅限于 "tool events: PreToolUse, PostToolUse, PermissionRequest" [source: system-prompt-hooks-configuration.md]。但它的能力远超 Prompt Hook，Agent 可以读取文件、执行命令、搜索代码库，独立完成复杂的验证任务。

从源文件的 Agent Hook prompt 可以看出其设计理念："You are verifying a stop condition in Claude Code. Your task is to verify that the agent completed the given plan." 并且 Agent 拥有对话 transcript 的访问权限："The conversation transcript is available at: ${TRANSCRIPT_PATH}" [source: agent-prompt-agent-hook.md]。Agent Hook 不是在真空中工作，它能获取完整的会话上下文来做决策。

## 4. Hook 输入协议

所有 Hook 通过 stdin 接收 JSON 格式的输入数据。标准结构如下：

```json
{
  "session_id": "abc123",
  "tool_name": "Write",
  "tool_input": { "file_path": "/path/to/file.txt", "content": "..." },
  "tool_response": { "success": true }
}
```

[source: system-prompt-hooks-configuration.md]

各字段含义：

- `session_id`：当前会话标识符，用于关联同一 session 中的多次 Hook 调用
- `tool_name`：触发事件的工具名称，与 matcher 匹配
- `tool_input`：工具接收的完整输入参数，包括文件路径、命令内容等
- `tool_response`：仅在 PostToolUse 事件中存在，包含工具执行的返回结果

这个设计使得 Command Hook 可以通过 `jq` 精确提取所需数据。例如日志记录场景中的 `jq -r '.tool_input.command'` 就是从 stdin JSON 中提取 Bash 命令内容 [source: system-prompt-hooks-configuration.md]。

不同事件提供的输入字段有差异。PostToolUse 事件会额外提供 `tool_response`，而 PreToolUse 和 PermissionRequest 事件只有 `tool_input`。Stop、UserPromptSubmit、SessionStart 等非工具事件的输入结构更简单，7-step verification flow 中指出对这些事件 "most commands don't read stdin, so `echo '{}' | <cmd>` suffices" [source: skill-update-config-7-step-verification-flow.md]。

## 5. Hook 输出协议

Hook 通过 JSON 输出控制 Claude Code 的行为。完整的输出结构：

```json
{
  "systemMessage": "Warning shown to user in UI",
  "continue": false,
  "stopReason": "Message shown when blocking",
  "suppressOutput": false,
  "decision": "block",
  "reason": "Explanation for decision",
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "Context injected back to model"
  }
}
```

[source: system-prompt-hooks-configuration.md]

各字段详解：

**systemMessage** "Display a message to the user (all hooks)" [source: system-prompt-hooks-configuration.md]。最通用的输出字段，所有 Hook 类型都可以使用。向用户展示一条消息，适用于状态通知、警告提示等场景。

**continue** "Set to `false` to block/stop (default: true)" [source: system-prompt-hooks-configuration.md]。Hook 的核心控制机制。设为 `false` 会阻断当前操作的执行链。配合 `stopReason` 可以向用户解释阻断原因。

**stopReason** "Message shown when `continue` is false" [source: system-prompt-hooks-configuration.md]。仅在阻断时显示，让用户理解为什么操作被拦截。

**suppressOutput** "Hide stdout from transcript (default: false)" [source: system-prompt-hooks-configuration.md]。控制 Hook 自身的标准输出是否写入会话 transcript。设为 `true` 可以保持对话记录的整洁。

**decision** "block" for PostToolUse/Stop/UserPromptSubmit hooks [source: system-prompt-hooks-configuration.md]。用于在特定事件中阻断操作。但文档注明它对 PreToolUse 已废弃（deprecated），应改用 `hookSpecificOutput.permissionDecision`。

**hookSpecificOutput** 事件特定的输出结构，必须包含 `hookEventName` 字段。这是输出协议中最灵活的部分，包含几个子字段：
- `additionalContext`：文本内容，注入到模型的上下文中
- `permissionDecision`：仅 PreToolUse 可用，值为 "allow"、"deny" 或 "ask"
- `permissionDecisionReason`：仅 PreToolUse 可用，决策原因说明
- `updatedInput`：仅 PreToolUse 可用，修改后的工具输入参数

[source: system-prompt-hooks-configuration.md]

`additionalContext` 是一个值得关注的字段，它允许 Hook 向模型注入额外的上下文信息。Hook 不仅是一个守门人，还可以是一个信息增强器，在工具执行前为模型补充必要的背景知识。

## 6. 权限决策三态

在 PreToolUse 事件中，Hook 的 `hookSpecificOutput.permissionDecision` 字段可以返回三种决策：

**allow** 直接放行，跳过权限确认弹窗。适用于你信任的操作模式，比如对特定目录的写入、对特定命令前缀的执行。

**deny** 直接拒绝，阻止工具执行。适用于硬性安全规则，比如禁止修改生产环境配置文件。

**ask** 转交给用户确认，即默认的权限弹窗行为。当你不确定是否应该自动决策时，这是一个安全的回退选项。

这个三态模型的设计思路是：不是所有自动化都是好的，有时候最好的决策是把决定权交还给用户。它让权限管理不再是非黑即白的二元选择，而是一个可以根据上下文灵活调整的连续谱。

`decision` 字段（值为 "block"）在 PreToolUse 中已被标记为 deprecated，官方推荐使用 `hookSpecificOutput.permissionDecision` 代替 [source: system-prompt-hooks-configuration.md]。这种迁移说明 Hooks 系统的输出协议在持续演进，hookSpecificOutput 作为一个可扩展的命名空间，承载着未来功能增长的责任。

## 7. Agent Hook：用 AI Agent 评估 Hook 条件

Agent Hook 是 Hooks 系统中最复杂的类型。它不只是执行命令或评估 prompt，而是启动一个完整的 AI Agent，拥有独立的工具访问能力和决策循环。

从 Agent Hook 的 system prompt 可以看到其核心指令：

"You are verifying a stop condition in Claude Code. Your task is to verify that the agent completed the given plan. The conversation transcript is available at: ${TRANSCRIPT_PATH}. You can read this file to analyze the conversation history if needed." [source: agent-prompt-agent-hook.md]

这意味着 Agent Hook 可以：
- 读取完整的对话 transcript 文件，理解会话的完整上下文
- 使用工具检查代码库的实际状态
- 独立判断条件是否满足

Agent Hook 的输出遵循严格的二元协议：

"Your response must be a JSON object matching one of the following schemas:
1. If the condition is met, return: {"ok": true}
2. If the condition is not met, return: {"ok": false, "reason": "Reason for why it is not met"}" [source: agent-prompt-hook-condition-evaluator.md]

这个简洁的输出协议体现了 Agent Hook 的定位：它是一个条件评估器，不是一个通用的对话 Agent。它只需要回答一个二元问题，条件是否满足。

Hook condition evaluator 的 prompt 进一步强化了这个定位。Agent 被要求 "Use as few steps as possible - be efficient and direct" [source: agent-prompt-agent-hook.md]。这不是一个需要深入思考的对话场景，而是一个需要快速、准确判断的验证任务。

## 8. 实用模式分析

源文件提供了四种典型的 Hook 使用模式：

### Auto-format 模式

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write|Edit",
      "hooks": [{
        "type": "command",
        "command": "jq -r '.tool_response.filePath // .tool_input.file_path' | { read -r f; prettier --write \"$f\"; } 2>/dev/null || true"
      }]
    }]
  }
}
```

[source: system-prompt-hooks-configuration.md]

这是最常见的模式。核心设计要点：matcher 使用 `"Write|Edit"` 同时匹配两种写入工具；命令通过 `jq -r` 安全提取文件路径，使用 `{ read -r f; ... "$f"; }` 模式而非不安全的 `xargs`（7-step flow 明确警告 "NOT unquoted `| xargs` (splits on spaces)" [source: skill-update-config-7-step-verification-flow.md]）；末尾的 `2>/dev/null || true` 确保格式化失败不会阻断工作流。

### Logging 模式

```json
{
  "hooks": {
    "PreToolUse": [{
      "matcher": "Bash",
      "hooks": [{
        "type": "command",
        "command": "jq -r '.tool_input.command' >> ~/.claude/bash-log.txt"
      }]
    }]
  }
}
```

[source: system-prompt-hooks-configuration.md]

审计日志模式，在 Bash 命令执行前记录其内容。使用 PreToolUse 而非 PostToolUse 确保即使命令执行失败也能记录。日志追加到 `~/.claude/bash-log.txt`，集中管理。

### Test-trigger 模式

```json
{
  "hooks": {
    "PostToolUse": [{
      "matcher": "Write|Edit",
      "hooks": [{
        "type": "command",
        "command": "jq -r '.tool_input.file_path // .tool_response.filePath' | grep -E '\\.(ts|js)$' && npm test || true"
      }]
    }]
  }
}
```

[source: system-prompt-hooks-configuration.md]

代码变更后自动触发测试。`grep -E '\\.(ts|js)$'` 限制只对 TypeScript/JavaScript 文件生效，避免对非代码文件的不必要测试触发。`|| true` 保证即使测试失败也不会阻断 Claude Code 的后续操作。

### Notification 模式

Stop 事件可以向用户展示完成消息：

```bash
echo '{"systemMessage": "Session complete!"}'
```

[source: system-prompt-hooks-configuration.md]

这是一个轻量模式，利用 `systemMessage` 字段向用户推送状态信息。

## 9. 7-step Verification Flow 分析

7-step verification flow 是一个精心设计的 Hook 构建流程，每一步都捕获一类特定的失败模式。源文件的总结精准到位："Each step catches a different failure class -- a hook that silently does nothing is worse than no hook." [source: skill-update-config-7-step-verification-flow.md]

### Step 1: Dedup Check

读取目标配置文件，检查同一 event+matcher 组合是否已存在 Hook。如果存在，向用户展示现有命令并询问：保留、替换、还是并行添加。这避免了重复 Hook 的静默冲突。

### Step 2: Construct the Command

为当前项目构建命令。关键原则：不假设环境。源文件要求 "Invokes the underlying tool the way this project runs it (npx/bunx/yarn/pnpm? Makefile target? globally-installed?)" [source: skill-update-config-7-step-verification-flow.md]。

安全提取数据的方式被严格规定："use `jq -r` into a quoted variable or `{ read -r f; ... "$f"; }`, NOT unquoted `| xargs`" [source: skill-update-config-7-step-verification-flow.md]。这防止了文件路径中包含空格导致的分词问题。

此阶段刻意不添加错误抑制："Stays RAW for now -- no `|| true`, no stderr suppression" [source: skill-update-config-7-step-verification-flow.md]，以便下一步能看到真实的错误信息。

### Step 3: Pipe-test

构造模拟的 stdin payload 进行端到端测试：

- `Pre|PostToolUse` on `Write|Edit`：`echo '{"tool_name":"Edit","tool_input":{"file_path":"<a real file from this repo>"}}' | <cmd>`
- `Pre|PostToolUse` on `Bash`：`echo '{"tool_name":"Bash","tool_input":{"command":"ls"}}' | <cmd>`
- `Stop`/`UserPromptSubmit`/`SessionStart`：`echo '{}' | <cmd>`

[source: skill-update-config-7-step-verification-flow.md]

测试不仅验证退出码，还验证副作用（"Check exit code AND side effect (file actually formatted, test actually ran)" [source: skill-update-config-7-step-verification-flow.md]）。只有通过测试后才添加错误抑制包装。

### Step 4: Write the JSON

将 Hook 合并写入目标配置文件。关键提醒：如果是新建 `.claude/settings.local.json`，需要手动添加到 `.gitignore`，因为 "the Write tool doesn't auto-gitignore it" [source: skill-update-config-7-step-verification-flow.md]。

### Step 5: Validate Syntax + Schema

用一条 `jq` 命令同时验证语法和结构：

```
jq -e '.hooks.<event>[] | select(.matcher == "<matcher>") | .hooks[] | select(.type == "command") | .command' <target-file>
```

[source: skill-update-config-7-step-verification-flow.md]

退出码含义：Exit 0 表示正确，Exit 4 表示 matcher 不匹配，Exit 5 表示 JSON 格式错误或嵌套结构错误。源文件特别警告："A broken settings.json silently disables ALL settings from that file" [source: skill-update-config-7-step-verification-flow.md]，这是整个流程中最危险的失败模式。

### Step 6: Prove the Hook Fires

验证 Hook 确实被触发。对于格式化 Hook，方法是 "introduce a detectable violation via Edit... re-read, confirm the hook fixed it" [source: skill-update-config-7-step-verification-flow.md]。对于其他类型的 Hook，使用哨兵文件（sentinel file）模式：临时在命令前加 `echo "$(date) hook fired" >> /tmp/claude-hook-check.txt; `，触发后检查文件是否生成。

无论测试是否通过，都必须清理："Always clean up -- revert the violation, strip the sentinel prefix -- whether the proof passed or failed" [source: skill-update-config-7-step-verification-flow.md]。

### Step 7: Handoff

告知用户 Hook 已生效。如果 Step 6 失败但前几步都通过，可能是 settings watcher 的限制："the settings watcher isn't watching `.claude/` -- it only watches directories that had a settings file when this session started" [source: skill-update-config-7-step-verification-flow.md]。此时用户需要手动打开 `/hooks` 或重启 session。

## 10. 可编程的决策引擎

回顾整个 Hooks 系统，它的核心设计目标很明确：把 Claude Code 从一个封闭的 AI 助手变成一个可编程的决策引擎。

这套系统的设计分三个层次递进。Command Hook 用确定性逻辑处理明确的自动化需求，格式化、日志、测试。Prompt Hook 引入 AI 判断处理需要语义理解的场景。Agent Hook 赋予 Hook 独立使用工具和访问会话上下文的能力，处理需要深度验证的复杂条件。

输入输出协议的设计同样经过仔细考虑。stdin JSON 协议让 Hook 获得了完整的上下文感知能力，结构化的 JSON 输出协议让 Hook 可以精确控制 Claude Code 的行为，从显示消息到阻断操作，从修改工具输入到做出权限决策。

7-step verification flow 说明 Hooks 系统对可靠性的重视程度。每一步都在捕获一类特定的失败：重复配置、环境假设、管道错误、JSON 格式错误、静默失败。源文件的断言 "a hook that silently does nothing is worse than no hook" [source: skill-update-config-7-step-verification-flow.md] 既是设计原则，也是对使用者的提醒。自动化只有在可靠时才有价值，一个静默失效的 Hook 比没有 Hook 更危险。

Hooks 系统回答了一个根本问题：如何让 AI 编程助手适配千差万别的项目环境和工作流？答案不是内置更多功能，而是提供一个足够灵活的扩展点，让每个团队根据自身需求定制行为。这不是一个功能，而是一个平台。
