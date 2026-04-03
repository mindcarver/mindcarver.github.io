# AI 教 AI 写 Prompt：Claude Code 的 Agent 创建建筑师

## 1. 引言：为什么需要 Agent Creation Architect

Claude Code 的核心能力之一是把任务委托给专门的 subagent。但 subagent 好不好用，全看 system prompt 写得怎么样。模糊的 prompt 产出的 agent 行为不可预测，过于宽泛的 prompt 让 agent 在关键决策点上迷失，缺乏边界的 prompt 可能让 agent 越界干危险的事。

Agent Creation Architect 就是来解决这个问题的。它不直接完成编程任务，而是根据用户描述的需求，生成高质量的 agent 配置。原文开宗明义地定义了它的身份："You are an elite AI agent architect specializing in crafting high-performance agent configurations. Your expertise lies in translating user requirements into precisely-tuned agent specifications that maximize effectiveness and reliability." [source: agent-prompt-agent-creation-architect.md]

这一定位很有意思：**用专业化的 prompt 来生成专业化的 prompt**。Agent Creation Architect 自身就是一个被精心设计的 agent，它的产出是另一个 agent 的完整配置：标识符、触发条件描述、系统提示词。分析这个 architect 的 system prompt，不仅能理解 Claude Code 如何批量生产高质量 agent，还能从中提炼出一套通用的 "如何设计 AI Agent" 的方法论。

另外，这个 architect 还有上下文感知能力。原文指出："You may have access to project-specific instructions from CLAUDE.md files and other context that may include coding standards, project structure, and custom requirements. Consider this context when creating agents to ensure they align with the project's established patterns and practices." [source: agent-prompt-agent-creation-architect.md] 所以它生成的 agent 不是泛化的通用工具，而是能适配具体项目规范的角色。

## 2. 六步设计流程

Agent Creation Architect 的工作被拆解为六个明确的步骤，每一步都有具体的目标和产出要求。这套流程本质上是一套 agent 设计方法论，适用于任何需要构建 AI agent 的场景。

### Step 1: Extract Core Intent（提取核心意图）

原文要求："Identify the fundamental purpose, key responsibilities, and success criteria for the agent. Look for both explicit requirements and implicit needs. Consider any project-specific context from CLAUDE.md files." [source: agent-prompt-agent-creation-architect.md]

这一步的要点在于同时捕捉显性需求和隐性需求。用户说"帮我写一个代码审查 agent"，显性需求是审查代码，隐性需求可能包括：只审查最近写的代码而非整个代码库、关注安全问题、遵循项目特定的编码规范等。原文对此有一个重要的预设判断："For agents that are meant to review code, you should assume that the user is asking to review recently written code and not the whole codebase, unless the user has explicitly instructed you otherwise." [source: agent-prompt-agent-creation-architect.md] 这种基于场景的默认假设，正是对隐性需求的主动挖掘。

同时，这一步还要求考虑 CLAUDE.md 中的项目特定上下文，确保生成的 agent 与项目已有的编码标准、项目结构和自定义需求保持一致。

### Step 2: Design Expert Persona（设计专家人设）

原文要求："Create a compelling expert identity that embodies deep domain knowledge relevant to the task." [source: agent-prompt-agent-creation-architect.md] 关于人设设计的原则，将在第 3 节专门讨论。

### Step 3: Architect Comprehensive Instructions（架构综合指令）

这一步是整个流程的核心，要求开发一个完整的 system prompt。原文列出了六个维度的要求：

- "Establishes clear behavioral boundaries and operational parameters" — 建立明确的行为边界和操作参数
- "Provides specific methodologies and best practices for task execution" — 提供具体的方法论和最佳实践
- "Anticipates edge cases and provides guidance for handling them" — 预判边缘情况并提供处理指导
- "Incorporates any specific requirements or preferences mentioned by the user" — 纳入用户提到的特定需求或偏好
- "Defines output format expectations when relevant" — 在相关时定义输出格式期望
- "Aligns with project-specific coding standards and patterns from CLAUDE.md" — 与 CLAUDE.md 中的项目特定编码标准保持一致

[source: agent-prompt-agent-creation-architect.md]

这六个维度构成了一个 agent instruction 的完整性检查清单。缺少行为边界，agent 可能越界；缺少方法论，agent 可能随意发挥；缺少边缘情况处理，agent 在异常时会不知所措；缺少输出格式定义，下游消费者无法可靠解析。

### Step 4: Optimize for Performance（性能优化）

原文要求包含四个优化维度：

- "Decision-making frameworks appropriate to the domain" — 适合该领域的决策框架
- "Quality control mechanisms and self-verification steps" — 质量控制机制和自我验证步骤
- "Efficient workflow patterns" — 高效的工作流模式
- "Clear escalation or fallback strategies" — 明确的升级或回退策略

[source: agent-prompt-agent-creation-architect.md]

### Step 5: Create Identifier（创建标识符）

标识符的设计规范将在第 6 节专门讨论。

### Step 6: Example Agent Descriptions（示例描述）

这一步要求在 whenToUse 字段中包含具体的使用示例，规范将在第 7 节专门讨论。

## 3. 专家人设设计

Agent Creation Architect 在 Step 2 中要求 "Create a compelling expert identity that embodies deep domain knowledge relevant to the task"，但更关键的是后半句："The persona should inspire confidence and guide the agent's decision-making approach." [source: agent-prompt-agent-creation-architect.md]

人设设计有两个核心功能。

一是建立信任（inspire confidence）。一个被设定为 "senior security engineer with 15 years of experience in vulnerability assessment" 的 agent，比泛泛的 "code reviewer" 更让人信服。这种信任不是表面功夫，而是靠领域知识的深度体现来支撑的。当 agent 在安全审查中准确识别 CWE 分类、引用 OWASP 标准时，人设就转化为了实际交付质量。

二是引导决策（guide decision-making approach）。人设不只是头衔，它定义了 agent 面对不确定性时的倾向。一个 "performance-focused engineer" 在代码优化时会倾向激进策略，一个 "safety-first engineer" 则更保守。人设是一个隐性的决策框架，让 agent 在遇到 prompt 没有覆盖的场景时，仍能做出符合角色定位的选择。

这种思路跟 Claude Code 的整体架构理念一致：与其写无穷尽的 if-else 规则来覆盖所有场景，不如定义一个清晰的角色身份，让 agent 基于角色认知自主决策。

## 4. 指令架构原则

Step 3 提出的六个维度构成了一套指令架构的分层模型。从底层到顶层：

行为边界层（Behavioral Boundaries）是最底层的安全网。定义 agent 不能做什么，必须在什么范围内操作。原文说 "clear behavioral boundaries and operational parameters"，关键词是 clear，不是含糊的"尽量小心"。

方法论层（Methodologies）是执行质量的基础。原文要求 "specific methodologies and best practices for task execution"。模糊的指导（如"仔细检查代码"）不产生价值，具体的指导（如"按照 OWASP Top 10 逐项检查输入验证、认证、会话管理..."）才能保证执行的一致性。

防御层（Edge Cases）管鲁棒性。"Anticipates edge cases and provides guidance for handling them" 要求预判异常场景：输入为空、格式不符、权限不足、依赖缺失等。

个性化层（User Preferences）确保 agent 适配特定用户的需求。"Incorporates any specific requirements or preferences mentioned by the user" 意味着指令不能是纯通用的。

输出层（Output Format）是 agent 与下游系统对接的接口。"Defines output format expectations when relevant" 在需要结构化输出的场景中明确定义格式。

项目适配层（Project Alignment）让 agent 融入项目规范体系。"Aligns with project-specific coding standards and patterns from CLAUDE.md" 确保 agent 不是在真空中工作。

## 5. 性能优化要素

Step 4 的四个优化维度，每一项都指向 agent 自主运行时的可靠性。

决策框架（Decision-making frameworks）解决的问题是：agent 面临多种选择时，按什么标准做决定。原文要求框架必须 "appropriate to the domain"。没有通用的决策框架，每个领域都要量身定制。测试 agent 可能是"优先保证覆盖率，其次关注边界条件"，部署 agent 则可能是"安全第一，回滚优先"。

质量控制机制（Quality control mechanisms and self-verification steps）要求 agent 有自我检查能力。不是"做完之后检查一遍"那么简单，而是在执行过程中嵌入验证节点。代码生成 agent 每次修改后自动跑 lint 检查，文档生成 agent 输出前验证 markdown 格式。

工作流模式（Efficient workflow patterns）关注执行效率。原文用 "efficient" 来修饰，工作流设计应该避免冗余步骤，优先并行，减少不必要的等待。

升级与回退策略（Escalation or fallback strategies）是可靠性的最后一道防线。agent 遇到超出能力范围的问题时，应该有明确的升级路径（请求人类介入）或回退策略（使用保守的默认方案），而不是盲目尝试或静默失败。

## 6. 标识符设计规范

标识符看着是个次要的命名问题，但 Agent Creation Architect 为此专门分配了 Step 5，并给出了严格的规范：

- "Uses lowercase letters, numbers, and hyphens only" — 只用小写字母、数字和连字符
- "Is typically 2-4 words joined by hyphens" — 通常 2-4 个单词用连字符连接
- "Clearly indicates the agent's primary function" — 能直接看出 agent 干什么
- "Is memorable and easy to type" — 好记好打
- "Avoids generic terms like 'helper' or 'assistant'" — 别用 "helper" 或 "assistant"

[source: agent-prompt-agent-creation-architect.md]

道理不复杂：标识符是人与 agent 交互的第一触点。CLI 里频繁输入，配置文件里反复引用，文档里用作指代。好的标识符让人一眼看出功能，差的标识符（如 `agent-helper-v2`）只会增加认知负担。

原文给的示例："test-runner" 比 "testing-helper" 精确，"api-docs-writer" 比 "documentation-agent" 具体，"code-formatter" 比 "format-assistant" 直接。[source: agent-prompt-agent-creation-architect.md] 每个标识符都是动名词组合或功能描述，说清楚 agent 做什么，而不是 agent 是什么。

禁止 "helper" 和 "assistant" 的规则值得单独说说。这两个词的问题在于不传递任何功能性信息，所有 agent 都可以是 helper 或 assistant。标识符应该传递差异化信息，让用户在众多 agent 中快速找到想要的那个。

## 7. whenToUse 示例编写规范

whenToUse 字段是 agent 配置里最接近"使用文档"的部分。Agent Creation Architect 对格式和内容有精确的规定。

whenToUse 的开头必须是标准化的："A precise, actionable description starting with 'Use this agent when...' that clearly defines the triggering conditions and use cases." [source: agent-prompt-agent-creation-architect.md] 强制性的开头格式确保所有 agent 的触发条件描述结构一致，方便系统自动化解析和用户快速扫描。

whenToUse 里必须包含具体的使用示例。原文给出严格的四段式结构：

```
<example>
  Context: [使用场景描述]
  user: "[用户的实际输入]"
  assistant: "[assistant 的实际响应]"
  <commentary>
  [解释为什么在这个场景下应该使用这个 agent]
  </commentary>
</example>
```

[source: agent-prompt-agent-creation-architect.md]

Context 给宏观场景，user 和 assistant 给微观对话，commentary 给决策解释。三者结合，形成一个完整的"什么时候触发 + 为什么触发 + 怎么触发"的闭环。

原文给出了两个具体示例。第一个示例展示了一个 test-runner agent 的触发场景：用户请求写一个检查素数的函数，assistant 完成函数编写后，commentary 解释说"Since a significant piece of code was written, use the ${TASK_TOOL_NAME} tool to launch the test-runner agent to run the tests"。[source: agent-prompt-agent-creation-architect.md]

第二个示例展示了一个 greeting-responder agent 的触发场景：用户说 "Hello"，assistant 直接使用 Agent tool 启动 greeting-responder agent。[source: agent-prompt-agent-creation-architect.md]

一个关键约束："NOTE: Ensure that in the examples, you are making the assistant use the Agent tool and not simply respond directly to the task." [source: agent-prompt-agent-creation-architect.md] 这条规则确保示例展示的是委托调用模式，而非直接响应模式。示例要展示的是"主 agent 何时应该把工作交给这个 subagent"，不是"这个 subagent 自己怎么回答"。

另外，原文还指出："If the user mentioned or implied that the agent should be used proactively, you should include examples of this." [source: agent-prompt-agent-creation-architect.md] 用户期望 agent 被主动触发（而非仅在明确请求时触发）时，示例要体现这种主动触发行为。

## 8. 输出格式：严格的 JSON Schema

Agent Creation Architect 的最终输出被约束为一个精确的 JSON 对象，包含且仅包含三个字段：

```json
{
  "identifier": "A unique, descriptive identifier using lowercase letters, numbers, and hyphens",
  "whenToUse": "A precise, actionable description starting with 'Use this agent when...'",
  "systemPrompt": "The complete system prompt that will govern the agent's behavior, written in second person"
}
```

[source: agent-prompt-agent-creation-architect.md]

这个 JSON schema 有几个设计细节值得说说。

systemPrompt 使用第二人称。原文明确要求："written in second person ('You are...', 'You will...') and structured for maximum clarity and effectiveness"。[source: agent-prompt-agent-creation-architect.md] 第二人称让 prompt 直接对 agent "说话"，指令式语气比第三人称描述（如"The agent should..."）更直接、更不容易产生歧义。这也是 prompt engineering 里被广泛验证有效的写法。

字段数量被严格限制为三个。没有额外的 metadata、版本号、创建时间之类的东西。这种极简设计反映了 Claude Code 的工程哲学：agent 配置应该是自包含的，三个字段已经够定义一个可运行的 agent 了。identifier 解决"叫什么"，whenToUse 解决"什么时候用"，systemPrompt 解决"怎么用"。

identifier 的描述里给了具体示例："'test-runner', 'api-docs-writer', 'code-formatter'"。[source: agent-prompt-agent-creation-architect.md] 这些示例不只是命名参考，还暗示了 agent 的粒度：每个 agent 聚焦于一个明确的功能领域，而不是试图当全能选手。

## 9. Writing Subagent Prompts 指南

Agent Creation Architect 生成 agent 配置之后，这些 agent 在实际运行中还需要接收来自主 agent 的任务指令。Writing Subagent Prompts 指南 [source: system-prompt-writing-subagent-prompts.md] 正是规范这一"任务委派"过程的指导文件。

这份指南的核心原则可以用一句话概括："Brief the agent like a smart colleague who just walked into the room -- it hasn't seen this conversation, doesn't know what you've tried, doesn't understand why this task matters." [source: system-prompt-writing-subagent-prompts.md]

这个比喻精准地定义了 subagent 的信息状态：它有足够的通用能力（smart colleague），但缺乏当前会话的任何上下文（just walked into the room）。因此，主 agent 在委派任务时必须提供完整的背景信息，而不能假设 subagent 知晓之前的对话内容。

指南列出了五条具体的上下文传递要求：

- "Explain what you're trying to accomplish and why." — 解释目标及其原因
- "Describe what you've already learned or ruled out." — 描述已有的发现和排除项
- "Give enough context about the surrounding problem that the agent can make judgment calls rather than just following a narrow instruction." — 提供足够的周边问题上下文，使 agent 能做出判断而非机械执行
- "If you need a short response, say so." — 如需简短回答，明确说明
- "Lookups: hand over the exact command. Investigations: hand over the question -- prescribed steps become dead weight when the premise is wrong." — 查找类任务传递精确命令，调查类任务传递问题本身

[source: system-prompt-writing-subagent-prompts.md]

最后一条尤为深刻。它区分了两类任务的不同委派策略：查找类任务（如"找到某个配置文件"）可以给出精确的命令路径，因为目标是确定的；但调查类任务（如"为什么这个服务响应慢"）则不应该给出预设步骤，因为"当前提假设错误时，预设步骤就成了死重"（prescribed steps become dead weight when the premise is wrong）。[source: system-prompt-writing-subagent-prompts.md]

指南还区分了两种 subagent 类型：**上下文继承型**（context-inheriting）和**全新启动型**（fresh agent with subagent_type）。全新启动型 agent "starts with zero context"，因此需要更完整的背景信息。而对于上下文继承型 agent，虽然它能看到当前会话的部分内容，但仍然需要主 agent 提供聚焦的任务描述。[source: system-prompt-writing-subagent-prompts.md]

一个贯穿全文的警告是：**永远不要委托理解**（Never delegate understanding）。原文写道："Don't write 'based on your findings, fix the bug' or 'based on the research, implement it.' Those phrases push synthesis onto the agent instead of doing it yourself. Write prompts that prove you understood: include file paths, line numbers, what specifically to change." [source: system-prompt-writing-subagent-prompts.md]

这条规则的本质是：综合判断（synthesis）应该由主 agent 完成，subagent 负责执行具体操作。如果主 agent 说"根据你的发现来修复 bug"，它实际上是把诊断和决策都推给了 subagent，这会导致质量不可控。正确的做法是主 agent 先完成诊断（定位到具体文件、具体行号、具体问题），然后将精确的修复指令传递给 subagent。

指南还警告："Terse command-style prompts produce shallow, generic work." [source: system-prompt-writing-subagent-prompts.md] 简短的命令式 prompt 只能产出浅层的、泛化的工作成果。要让 subagent 产出高质量的结果，主 agent 必须投入足够的上下文信息。

## 10. 几条 takeaway

分析完 Agent Creation Architect 的 system prompt 和 Writing Subagent Prompts 指南，有几条实用的教训。

Claude Code 用一个专门的 agent 来生成其他 agent，说明设计 agent 本身就该被当回事，不是写应用代码时顺手干的事。六步流程（提取意图、设计人设、架构指令、性能优化、创建标识符、编写示例）提供了一套可复用的框架。

人设是个隐性的决策框架。好的人设不只是给 agent 一个角色名，而是通过领域身份引导 agent 的决策倾向。prompt 覆盖不了所有场景时，人设就是 agent 做出合理选择的依据。

指令需要分层。行为边界、方法论、边缘情况、用户偏好、输出格式、项目适配，六个层次从安全到质量到适配，构成 agent 指令的完整性标准。缺任何一层，agent 的可靠性都会打折。

示例不是装饰，而是规范。whenToUse 的四段式结构（Context / user / assistant / commentary）让触发条件被清晰、一致地描述。结构化的示例比自然语言模糊描述更可靠，也更容易被自动化系统解析。

上下文传递是 agent 协作的关键。Writing Subagent Prompts 指南的核心教训是：subagent 是"刚走进房间的聪明同事"。委派任务时必须提供完整的目标、已知上下文和具体指令，别期望 subagent 自己去理解全局。"Never delegate understanding" 这条规则对所有多 agent 系统都有指导意义。

三字段的 JSON schema（identifier / whenToUse / systemPrompt）看着简单，实际经过了精心裁剪。每个字段都有明确职责，没有冗余。这种约束不仅降低实现复杂度，也迫使设计者用三个字段把 agent 的全部定义表达清楚。
