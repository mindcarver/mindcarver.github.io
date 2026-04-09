# 上下文快炸了怎么办：Claude Code 的压缩生存术

## 1. 为什么需要上下文压缩

大语言模型的工作方式决定了每一次交互都需要携带完整的上下文历史。随着会话推进，消息不断累积，context window 的容量逐渐被填满。当 token 数量逼近模型上限时，Claude Code 必须做出选择：要么丢弃早期消息导致信息丢失，要么将历史会话压缩为摘要以腾出空间。

上下文压缩（context compaction）做的就是后面这件事。它不是简单的文本截断，而是一套摘要系统，目标是让"未来的自己"（或另一个模型实例）仅凭摘要就能无缝继续工作。正如 Claude Code 的 system prompt 所言，摘要需要足够详尽，确保"someone reading only your summary and then the newer messages can fully understand what happened and continue the work" [source: system-prompt-partial-compaction-instructions.md]。

context window 的硬性限制之外，token 成本也是实际考量。长上下文意味着更高的 API 调用费用和更慢的响应速度。高质量的压缩能在保留关键信息的同时大幅减少 token 消耗，是一次以精度换取空间的技术权衡。

## 2. 三种压缩模式对比

Claude Code 并非采用单一的压缩策略。通过分析其 system prompt 文件，可以识别出至少三种不同的压缩模式，每种模式针对不同的场景设计。

完整压缩（Full Compaction）：当用户显式请求压缩，或系统检测到上下文即将溢出时触发。整个会话历史被总结为一个结构化摘要，替换原始消息。这是最彻底的压缩方式，对应 `agent-prompt-conversation-summarization.md` 中的完整对话摘要指令 [source: agent-prompt-conversation-summarization.md]。

部分压缩（Partial Compaction）：用户选择仅压缩会话的某一部分，保留较新的消息不动。压缩后的摘要放在会话开头，新的消息直接跟在摘要之后。这种方式在保留近期上下文的同时释放了早期历史占用的空间，对应 `system-prompt-partial-compaction-instructions.md` [source: system-prompt-partial-compaction-instructions.md]。

近期消息压缩（Recent Message Summarization）：针对会话中最新的一部分消息进行总结，而更早的已保留上下文保持原样。它的 prompt 明确指出："The earlier messages are being kept intact and do NOT need to be summarized. Focus your summary on what was discussed, learned, and accomplished in the recent messages only" [source: agent-prompt-recent-message-summarization.md]。

三种模式共享相同的 9 段式总结结构，但在字段的语义取向上存在微妙差异。部分压缩独有的 "Work Completed" 和 "Context for Continuing Work" 字段体现了其独特的定位：它记录做了什么，也明确传递"接续工作需要知道什么"。而近期消息压缩因为早期上下文仍在场，不需要这个字段，它的 "Current Work" 和 "Optional Next Step" 足以桥接已有上下文和最新进展。

## 3. 完整压缩的 `<analysis>` 标签设计

完整压缩的 prompt 采用了一个值得注意的两阶段结构。它要求模型在输出最终摘要之前，先用 `<analysis>` 标签包裹一段分析过程：

> "Before providing your final summary, wrap your analysis in <analysis> tags to organize your thoughts and ensure you've covered all necessary points." [source: agent-prompt-conversation-summarization.md]

这个设计背后有实际的工程考量。摘要的质量直接决定了后续会话能否正常工作。如果遗漏了关键的代码变更、错误修复路径或用户的偏好反馈，后续的工作将在错误的基础上展开。`<analysis>` 标签强制模型先进行一轮系统性的梳理，而不是直接跳入摘要撰写。

分析阶段的具体要求包括按时间顺序逐条审视每一条消息，识别：

- 用户的显式请求和意图
- 助手的处理方法
- 关键决策、技术概念和代码模式
- 具体细节：文件名、完整代码片段、函数签名、文件编辑
- 遇到的错误及修复方式
- 用户的特别反馈，尤其是"告诉你要换一种做法"的指示 [source: agent-prompt-conversation-summarization.md]

在完成分析之后，prompt 还要求"Double-check for technical accuracy and completeness"，相当于在摘要生成前再加一道质量检查。这种"先分析后总结"的两阶段结构，本质上是把思维链（chain-of-thought）外化为一个必经步骤，确保摘要的系统性和完整性。

同样的 `<analysis>` 结构在部分压缩和近期消息压缩中也保持一致 [source: system-prompt-partial-compaction-instructions.md] [source: system-prompt-recent-message-summarization.md]，说明 Claude Code 团队把这个两阶段流程视为压缩系统的核心保障。

## 4. 9 段式结构化总结格式

三种压缩模式最终产出的摘要都遵循一个 9 段的结构化格式。这个格式覆盖了继续开发工作所需的全部关键信息维度。

以完整压缩为例，这 9 个段落是 [source: agent-prompt-conversation-summarization.md]：

1. Primary Request and Intent -- 捕获用户的所有显式请求和意图。摘要的锚点，后续所有段落围绕这个锚点展开。

2. Key Technical Concepts -- 列出所有讨论过的重要技术概念、技术和框架。为后续工作提供技术栈的上下文背景。

3. Files and Code Sections -- 枚举所有被检查、修改或创建的文件和代码段。prompt 特别强调"Pay special attention to the most recent messages and include full code snippets where applicable"，并且要求说明每个文件读取或编辑的原因。

4. Errors and Fixes -- 列出所有遇到的错误及其修复方式。同样强调用户反馈："Pay special attention to specific user feedback that you received, especially if the user told you to do something differently."

5. Problem Solving -- 记录已解决的问题和正在进行的故障排查。确保排错思路不会因为压缩而中断。

6. All User Messages -- 列出所有非工具调用结果的用户消息。关于这个字段的重要性，下一节专门讨论。

7. Pending Tasks -- 列出所有被明确要求但尚未完成的工作。

8. Current Work -- 精确描述压缩请求前正在进行的工作。要求"paying special attention to the most recent messages from both user and assistant"，并包含文件名和代码片段。

9. Optional Next Step -- 列出与当前工作直接相关的下一步。这个字段有严格的约束条件，后文将详细分析。

9 段结构的设计逻辑是自上而下的：从用户的宏观意图出发，经过技术细节和文件变更，到错误和问题解决，再到未完成的任务和当前进展，最后落脚于下一步行动。每个段落为后续工作提供不同层面的上下文支撑。

## 5. 部分压缩的独特设计

部分压缩在 9 段结构中做了两个关键的字段替换。它将完整压缩中的第 8 段 "Current Work" 和第 9 段 "Optional Next Step" 替换为：

- 8. Work Completed -- 描述截至这个部分结束时完成了什么。
- 9. Context for Continuing Work -- 总结后续消息在理解并继续工作时所需的任何上下文、决策或状态。 [source: system-prompt-partial-compaction-instructions.md]

这一替换反映了部分压缩的独特定位。部分压缩的 prompt 开宗明义地描述了摘要的放置位置：

> "This summary will be placed at the start of a continuing session; newer messages that build on this context will follow after your summary (you do not see them here)." [source: system-prompt-partial-compaction-instructions.md]

摘要之后还会有新的消息自然衔接。因此，它不需要"Current Work"来描述正在做什么（因为新消息会自己展示），也不需要"Optional Next Step"来建议下一步（因为新消息可能已经走出了不同的方向）。它真正需要的是一份清晰的"已完成事项清单"加上"接续工作需要的背景知识"。

"Context for Continuing Work" 这个字段的设计尤为精妙。它包含技术状态（比如哪些文件被修改了），也包括决策理由（为什么选择方案 A 而不是方案 B）、未解决的约束条件、以及任何对后续工作有影响的隐含知识。这个字段的本质是一份知识传递文档，确保新消息的上下文不会被截断效应割裂。

## 6. 近期消息压缩的使用场景

近期消息压缩的 prompt 开头即定义了它的职责边界：

> "Your task is to create a detailed summary of the RECENT portion of the conversation -- the messages that follow earlier retained context. The earlier messages are being kept intact and do NOT need to be summarized." [source: agent-prompt-recent-message-summarization.md]

这个场景适用于一种特定的压缩需求：会话的前半部分已经足够重要或足够紧凑，值得原样保留，但近期的大量交互（可能是一轮密集的调试、多次文件修改、或反复的错误修复）需要被浓缩。

典型场景：用户在完成一个主要功能后进行了大量细粒度的调整和修复，这些调整的最终结果已经反映在代码中，但中间的试错过程占据了大量 token。此时保留早期的高层对话（需求讨论、架构决策），压缩近期的实现细节，是最合理的选择。

近期消息压缩保留了标准的 9 段结构，但语义范围限定在"近期消息"内。比如它的 "All user messages" 字段只列出近期部分的用户消息，"Pending Tasks" 只包含近期消息中出现的待办事项 [source: agent-prompt-recent-message-summarization.md]。这种限定避免了与早期已保留上下文的信息重复。

SDK 场景下还有一种更简洁的压缩格式，用于 context compaction summary。它使用 5 段结构：Task Overview、Current State、Important Discoveries、Next Steps、Context to Preserve [source: system-prompt-context-compaction-summary.md]。这个版本的指令更加简洁明确："Be concise but complete -- err on the side of including information that would prevent duplicate work or repeated mistakes" [source: system-prompt-context-compaction-summary.md]。它还包含一个其他版本没有的维度："Any promises made to the user"，即记录在会话中对用户做出的任何承诺。

## 7. 为什么 All User Messages 是必填字段

在 9 段结构中，第 6 段 "All User Messages" 的定位值得单独说。它的说明是：

> "List ALL user messages that are not tool results. These are critical for understanding the users' feedback and changing intent." [source: agent-prompt-conversation-summarization.md]

关键词是 "ALL" 和 "critical"。

用户反馈是最重要的上下文。在一次典型的开发会话中，用户的意图并非一成不变。他们可能在第 3 轮对话中说"用 React"，第 7 轮说"算了换成 Vue"，第 12 轮说"其实还是用 React 吧，但不要用 Redux"。这些意图变化散布在不同的用户消息中，如果摘要只记录最终结论（"使用 React，不用 Redux"）而遗漏了变化过程，后续的模型实例可能在遇到类似决策点时重蹈覆辙。

用户的反馈常常包含隐含的偏好和约束。一句"这里的命名不太清楚"可能暗示用户偏好描述性命名而非简洁命名；一句"这个太慢了"可能意味着性能优先于代码美观。这些微妙的信号只有通过完整的用户消息列表才能被保留。

prompt 还特别强调 "that are not tool results"，因为工具调用结果（比如文件内容、命令输出）已经被其他段落（Files and Code Sections、Errors and Fixes 等）捕获。用户消息段落专注的是人的意图和判断，这些是无法从技术细节中反推的。

## 8. Optional Next Step 的约束

第 9 段 "Optional Next Step" 是整个摘要结构中约束最严格的字段。它的完整指令是：

> "List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's most recent explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests or really old requests that were already completed without confirming with the user first." [source: agent-prompt-conversation-summarization.md]

这段指令包含了多层约束。首先是相关性约束："DIRECTLY in line with the user's most recent explicit requests"。注意 "explicit" 一词——模型的推断或揣测不算数，必须是用户明确说出的需求。

其次是时序约束：必须基于"the task you were working on immediately before this summary request"。即使有一个更早的、尚未完成的任务，如果当前正在做的是另一件事，Next Step 应该基于当前工作而非历史遗留。

第三是终止判断：如果上一个任务已经完成，则只有在用户明确请求时才列出下一步。这是防止"越权执行"的安全阀。

最后还有引用要求："If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation." [source: agent-prompt-conversation-summarization.md]。要求使用原文引用（verbatim），是为了防止摘要过程中的意图漂移（task drift）。模型在总结时可能无意识地美化或曲解原始请求，原文引用提供了一个不可篡改的锚点。

这些约束层层叠加，反映了一个核心设计原则：压缩系统宁可保守（不提供 Next Step），也不愿激进地替用户做决定。这是关于 AI 自主性边界的一个审慎的设计选择。

## 9. MCP 工具结果截断策略

除了会话级别的压缩，Claude Code 还有一套针对工具调用结果的细粒度截断策略。MCP（Model Context Protocol）工具可能返回大量数据，比如一个完整的 JSON 文件或数兆字节的日志输出。如果这些结果直接进入上下文，会迅速消耗 token 预算。

Claude Code 的截断策略通过 `system-prompt-mcp-tool-result-truncation.md` 实现，核心思路是根据查询类型选择不同的处理路径 [source: system-prompt-mcp-tool-result-truncation.md]：

定向查询走直接工具：如果目标是查找特定行、按字段过滤等精确操作，直接使用 `jq` 或 `grep` 处理文件。这类操作不需要完整读取文件内容，因此不会产生大量 token 消耗。

> "For targeted queries (find a row, filter by field): use jq or grep on the file directly." [source: system-prompt-mcp-tool-result-truncation.md]

全量分析走子代理：如果需要阅读完整内容进行分析或总结，则使用 Agent 工具在隔离的上下文中处理文件，确保完整输出不会进入主上下文。

> "For analysis or summarization that requires reading the full content: use the ${AGENT_TOOL_NAME} tool to process the file in an isolated context so the full output does not enter your main context." [source: system-prompt-mcp-tool-result-truncation.md]

关键设计是"隔离上下文"（isolated context）。子代理在独立的 context window 中运行，可以完整阅读大文件并返回精炼的结果，而主上下文只接收子代理的摘要输出。这是一种空间换时间的策略：消耗了一次额外的 API 调用，但保护了主上下文的 token 预算。

prompt 还对子代理的指令质量提出了明确要求：

> "Be explicit about what the subagent must return -- e.g. 'Read ALL of ${FILE_PATH}; summarize it and quote any key findings, decisions, or action items verbatim' -- a vague 'summarize this' may lose the detail you actually need. Require it to read the entire file before answering." [source: system-prompt-mcp-tool-result-truncation.md]

这段指令揭示了截断策略中一个微妙的平衡点：既要避免大文件撑爆主上下文，又要确保子代理不会因为过度压缩而丢失关键细节。"quote any key findings, decisions, or action items verbatim" 这个要求与 Next Step 的原文引用要求一脉相承，都是通过保留原始文本来防止信息在压缩过程中被曲解。

## 10. 总结：上下文压缩的设计哲学

回顾 Claude Code 的上下文压缩系统，可以提炼出几条核心设计哲学。

"总结是给未来的自己看的"。这不仅是隐喻，而是字面意义上的系统设计目标。部分压缩的 prompt 直接指出摘要的读者是"someone reading only your summary and then the newer messages" [source: system-prompt-partial-compaction-instructions.md]，SDK 版本的摘要指令更进一步，要求摘要能让你"or another instance of yourself"恢复工作 [source: system-prompt-context-compaction-summary.md]。摘要必须自包含（self-contained），不能假定读者拥有原始对话的记忆。

信息完整性优先于简洁性。三种压缩模式都要求 "full code snippets where applicable"，都强调用户反馈的逐条记录，都要求原文引用。这不是对简洁性的忽视，而是对"丢失信息的代价远大于多消耗几百 token"这一判断的体现。SDK 版本的指令说得最直白："err on the side of including information that would prevent duplicate work or repeated mistakes" [source: system-prompt-context-compaction-summary.md]。

层次化设计匹配不同场景。完整压缩、部分压缩、近期消息压缩、MCP 结果截断——四种机制覆盖了从整段会话到单次工具调用的不同粒度。每种机制的结构和字段都针对其使用场景做了定制化调整，而非套用同一个模板。

对自主性的审慎克制。Optional Next Step 的多层约束、"All User Messages" 的必填要求、子代理指令中对原文引用的强调——这些设计都指向同一个原则：压缩系统应当忠实地传递上下文，而非替用户做决策。它是一个记忆管理者，不是一个任务规划者。

这套压缩系统的存在本身也说明了一个事实：在大模型驱动的开发工具中，内存管理不再是操作系统独有的课题。AI 助手同样需要精心设计的"垃圾回收"机制，在有限的 context window 内高效地分配信息密度。Claude Code 的上下文压缩系统，是这一新范式下的一个工程实践样本。
