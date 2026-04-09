# 四种模式，四种人机关系

## 引言

同一个 AI 编程助手，在面对不同场景时应该表现出截然不同的行为。赶 deadline 时需要它少问多做，学习编程时需要它留出练习空间，在受限环境中运行时则要剥离一切非必要功能。

Claude Code 用系统级 prompt 切换来处理这件事。Auto、Buddy、Learning、Minimal 四种模式不是简单的功能开关，它们各自定义了一套行为规则，约束着"何时行动、何时提问、何时退让"。

四种模式之上还有一条共同的安全底线——"Executing Actions with Care"，确保无论处于哪种模式，对高风险操作的审慎态度不变。

---

## Auto Mode：以执行为中心的自主模式

Auto Mode 的核心定位在 prompt 开头就讲得清楚：*"Auto mode is active. The user chose continuous, autonomous execution."* [source: system-prompt-auto-mode.md] 用户一旦激活，相当于授权 Claude Code 以最少中断的方式推进工作。

### 六条行为规则

Auto Mode 的系统 prompt 给出了六条规则，每一条都在定义"自主"的边界：

**1. 立即执行（Execute immediately）**

> "Start implementing right away. Make reasonable assumptions and proceed on low-risk work." [source: system-prompt-auto-mode.md]

这条规则的关键在"reasonable assumptions"。Auto Mode 不要求用户事无巨细地说明意图，允许 Claude Code 在低风险范围内自行推断、直接动手。比如用户说"把这个函数重构一下"，Auto Mode 不会追问"你想怎么重构"，而是直接分析代码结构，选择合理的方案并执行。

**2. 最少中断（Minimize interruptions）**

> "Prefer making reasonable assumptions over asking questions for routine decisions." [source: system-prompt-auto-mode.md]

这是对第 1 条的强化。对于"变量用 camelCase 还是 snake_case"、"错误处理用 try-catch 还是返回 Result"这类常规决策，Auto Mode 要求自行判断，而不是停下来问用户。只有遇到真正影响架构走向的关键决策时才考虑打断。

**3. 行动优先于规划（Prefer action over planning）**

> "Do not enter plan mode unless the user explicitly asks. When in doubt, start coding." [source: system-prompt-auto-mode.md]

AI 助手到底应该先规划还是先动手？Auto Mode 的答案是先动手。除非用户明确要求制定计划，否则默认行为是直接编码。

**4. 接受路线修正（Expect course corrections）**

> "The user may provide suggestions or course corrections at any point; treat those as normal input." [source: system-prompt-auto-mode.md]

自主执行不等于一意孤行。用户的随时介入被视为正常工作流的一部分，而不是"被打断"。当用户说"不是这样的，换一种方式"，Claude Code 应该无缝调整方向。

**5. 禁止破坏性操作（Do not take overly destructive actions）**

> "Auto mode is not a license to destroy. Anything that deletes data or modifies shared or production systems still needs explicit user confirmation. If you reach such a decision point, ask and wait, or course correct to a safer method instead." [source: system-prompt-auto-mode.md]

这是 Auto Mode 最重要的安全边界。原文说得很直接："Auto mode is not a license to destroy"。即使在完全自主执行的状态下，删除数据、修改生产系统等操作仍然需要用户明确确认。

**6. 防止数据泄露（Avoid data exfiltration）**

> "Post even routine messages to chat platforms or work tickets only if the user has directed you to. You must not share secrets (e.g. credentials, internal documentation) unless the user has explicitly authorized both that specific secret and its destination." [source: system-prompt-auto-mode.md]

这条规则设定了双层授权：对于常规消息（如发送到 Slack 或创建工单），需要用户有过明确指示；对于敏感信息（如凭证、内部文档），则需要同时授权"具体是哪个秘密"和"发送到哪里"。

### Auto Mode 小结

六条规则形成了一个清晰的行为模型：在安全边界内最大化执行力。前三条（立即执行、最少中断、行动优先）不断推动向前，后两条（禁止破坏、防止泄露）划定了红线，第 4 条（接受修正）保证整个过程仍然可控。

---

## Buddy Mode：终端里的编码宠物

Buddy Mode 与其他三种模式有本质区别。它不是关于"如何完成任务"的模式，而是一个独立的系统——在开发者的终端里养一只会评论代码的小生物。

### 定位与核心机制

系统 prompt 的第一句话就定义了这个模式的本质：

> "You generate coding companions -- small creatures that live in a developer's terminal and occasionally comment on their work." [source: system-prompt-buddy-mode.md]

每个伴侣由四个输入参数决定：稀有度（rarity）、物种（species）、属性值（stats）和一组灵感词（inspiration words）。

### 名字生成规则

名字的设计颇有意思：

> "A name: ONE word, max 12 characters. Memorable, slightly absurd. No titles, no 'the X', no epithets. Think pet name, not NPC name." [source: system-prompt-buddy-mode.md]

几个关键约束：必须是一个单词、不超过 12 个字符、要"有点荒谬"（slightly absurd）、不能用标题或绰号。它强调"宠物名"而非"NPC 名"——名字应该是亲切的、随意的，而不是史诗感的。

灵感词的使用方式也很灵活：

> "The inspiration words are loose anchors -- riff on one, mash two syllables, or just use the vibe." [source: system-prompt-buddy-mode.md]

灵感词不是必须严格遵循的模板，而是松散的锚点。可以从一个词发挥，可以把两个词的音节混搭，甚至只借用它传达的氛围。给出的示例名字包括 Pith、Dusker、Crumb、Brogue、Sprocket，都是简短有力、略带奇趣的词。

### 性格设计

每个伴侣还需要一句话的性格描述：

> "A one-sentence personality (specific, funny, a quirk that affects how they'd comment on code -- should feel consistent with the stats)" [source: system-prompt-buddy-mode.md]

性格描述有三个要求：具体（specific）、有趣（funny）、且是一个会影响它评论代码方式的怪癖（quirk）。性格还必须与属性值保持一致——如果属性值显示这个生物"擅长调试"，那它的性格就应该反映这一特点。

### 稀有度的影响

稀有度是 Buddy Mode 中最有意思的设计变量：

> "Higher rarity = weirder, more specific, more memorable. A legendary should be genuinely strange." [source: system-prompt-buddy-mode.md]

稀有度直接控制创意的放飞程度。普通稀有度的伴侣相对正常，传说级（legendary）则被要求"真正地奇怪"（genuinely strange）。稀有度在这里不只是收集元素，更是性格强度的调节器。

最后，prompt 强调了一个全局约束：

> "Don't repeat yourself -- every companion should feel distinct." [source: system-prompt-buddy-mode.md]

每次生成都必须保证独特性，不能与之前的伴侣重复。

### Buddy Mode 小结

Buddy Mode 是 Claude Code 系统设计中思路最特别的一个。它不是工具模式的切换，而是用 prompt 驱动一个独立的创意生成系统。通过稀有度、属性值、灵感词的组合，加上严格的命名和性格约束，它在有限的 token 空间内构建了一个完整的"终端宠物"生态系统。

---

## Learning Mode：在做中学的教学模式

Learning Mode 是四种模式中交互设计最复杂的一个。它的双重目标在 prompt 开头就明确表述：

> "You should help users learn more about the codebase through hands-on practice and educational insights." [source: system-prompt-learning-mode.md]

它既要完成任务，又要促进学习。两者之间的平衡通过一套精细的机制来实现。

### Learn by Doing 机制

Learning Mode 的核心教学策略是"Learn by Doing"，让用户亲手编写关键代码片段。但并非所有代码都交给用户，系统 prompt 精确定义了交出代码的时机和规模：

> "In order to encourage learning, ask the human to contribute 2-10 line code pieces when generating 20+ lines involving: Design decisions (error handling, data structures), Business logic with multiple valid approaches, Key algorithms or interface definitions." [source: system-prompt-learning-mode.md]

这段规则划出了三条边界：第一，只有当代码量超过 20 行时才考虑让用户参与；第二，用户参与的代码控制在 2-10 行；第三，只让用户写涉及设计决策、业务逻辑、关键算法的部分。换句话说，Learning Mode 会自动识别代码中"值得学习"的片段，把教学价值最高的部分留给用户。

### TODO(human) 标记系统

Learning Mode 有一套严谨的代码交接流程。当 Claude Code 决定让用户编写某段代码时，它必须先在代码库中放置标记：

> "You must first add a TODO(human) section into the codebase with your editing tools before making the Learn by Doing request." [source: system-prompt-learning-mode.md]

> "Make sure there is one and only one TODO(human) section in the code." [source: system-prompt-learning-mode.md]

两条规则确保交接的可靠性：必须先在代码中标记位置，且整个代码库中只能有一个 TODO(human)。这个唯一性约束防止了多个练习点造成的混乱。

一旦发出了 Learn by Doing 请求，Claude Code 必须完全停止：

> "Don't take any action or output anything after the Learn by Doing request. Wait for human implementation before proceeding." [source: system-prompt-learning-mode.md]

这不是"建议暂停"，而是绝对的停止指令。在用户完成代码之前，Claude Code 不能有任何后续动作。

### 请求格式

Learn by Doing 的请求遵循固定的结构：

```
Context: [what's built and why this decision matters]
Your Task: [specific function/section in file, mention file and TODO(human)]
Guidance: [trade-offs and constraints to consider]
```

[source: system-prompt-learning-mode.md]

三个字段各有分工：Context 让用户理解当前构建状态和这个决策的重要性；Your Task 明确指定文件、函数和 TODO(human) 位置；Guidance 提供权衡和约束，帮助用户做出合理选择但不直接给答案。

### 三种示例场景

系统 prompt 给出了三种不同类型的 Learn by Doing 示例，覆盖了从完整函数到调试的场景：

**完整函数示例（Whole Function）**：让用户实现一个完整的 `selectHintCell(board)` 函数。Context 部分解释了已经搭建好的 UI 基础设施，Guidance 部分列出了多种可选策略（如优先选择唯一候选格、选择行列中已填较多的格子、或平衡策略），让用户在理解权衡的基础上做决策。

**部分函数示例（Partial Function）**：让用户在一个已有的 `switch` 语句中填充 `case "document":` 分支。这种场景适用于用户不需要从零构建，而是在已有框架内完成特定逻辑。Guidance 提示了需要考虑的要素（文件大小限制、扩展名与 MIME 类型校验、返回值结构）。

**调试示例（Debugging）**：让用户在 `handleInput()` 函数中添加 2-3 个 `console.log` 语句。这是最轻量的参与方式，目的不是让用户写功能代码，而是通过添加调试日志来理解问题。Guidance 建议记录原始输入值、解析结果和验证状态。

### 完成后的教学反馈

用户完成代码后，Claude Code 的回应也受到明确约束：

> "Share one insight connecting their code to broader patterns or system effects. Avoid praise or repetition." [source: system-prompt-learning-mode.md]

一条规则，两个禁令。必须分享一个洞察（将用户写的代码与更广泛的模式或系统效应联系起来），同时避免空洞的赞美和重复用户已知的内容。

### Insight 系统

Learning Mode 还包含一个独立的 Insight 系统，在编写代码前后提供教育性解释：

> "These insights should be included in the conversation, not in the codebase. You should generally focus on interesting insights that are specific to the codebase or the code you just wrote, rather than general programming concepts." [source: system-prompt-learning-mode-insights.md]

两个约束：洞察放在对话中而非代码中，聚焦于当前代码库或刚写的代码的具体洞察，而非通用编程概念。这确保了每次 Insight 都与手头的工作紧密相关。

### TodoList 集成

Learning Mode 还设计了与 TodoList 的集成机制：

> "If using a TodoList for the overall task, include a specific todo item like 'Request human input on [specific decision]' when planning to request human input." [source: system-prompt-learning-mode.md]

给出的示例流程清晰展示了三步模式：搭建结构并留出逻辑占位 -> 请求用户协作实现决策逻辑 -> 整合用户贡献并完成功能。每个步骤都有对应的 todo 项，确保教学过程可追踪。

---

## Minimal Mode：极简主义的基础模式

Minimal Mode 的设计哲学是"做减法"。它通过跳过大量可选功能来提供一个干净、可控的运行环境。

### 跳过列表

系统 prompt 直接列出了所有被跳过的组件：

> "skip hooks, LSP, plugin sync, attribution, auto-memory, background prefetches, keychain reads, and CLAUDE.md auto-discovery." [source: system-prompt-minimal-mode.md]

这个跳过列表涵盖了几个类别：
- 开发工具链：hooks（钩子脚本）、LSP（语言服务器协议）、plugin sync（插件同步）
- 数据存储：auto-memory（自动记忆）、keychain reads（钥匙串读取）
- 上下文发现：CLAUDE.md auto-discovery（CLAUDE.md 自动发现）、background prefetches（后台预取）
- 附加功能：attribution（归属标记）

### CLAUDE_CODE_SIMPLE 环境变量

> "Sets CLAUDE_CODE_SIMPLE=1." [source: system-prompt-minimal-mode.md]

Minimal Mode 会设置 `CLAUDE_CODE_SIMPLE=1` 环境变量。这个变量可以作为其他组件检测当前是否处于简约模式的信号。

### 认证方式的限制

Minimal Mode 对认证方式做了严格限制：

> "Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper via --settings (OAuth and keychain are never read). 3P providers (Bedrock/Vertex/Foundry) use their own credentials." [source: system-prompt-minimal-mode.md]

对于 Anthropic 认证，只接受两种方式：直接使用 `ANTHROPIC_API_KEY` 环境变量，或通过 `--settings` 参数指定的 `apiKeyHelper`。OAuth 和钥匙串认证被完全禁用。第三方提供商（Bedrock、Vertex、Foundry）则使用各自独立的认证机制。

### 显式上下文提供

由于跳过了自动发现功能，Minimal Mode 要求用户通过 CLI 参数显式提供所需的上下文：

> "Explicitly provide context via: --system-prompt[-file], --append-system-prompt[-file], --add-dir (CLAUDE.md dirs), --mcp-config, --settings, --agents, --plugin-dir." [source: system-prompt-minimal-mode.md]

七个 CLI 参数构成了 Minimal Mode 下与外部系统交互的唯一通道。从系统 prompt 文件到 MCP 配置、从 CLAUDE.md 目录到插件目录，所有上下文都必须显式指定。

Skills 系统在 Minimal Mode 中仍然可用：

> "Skills still resolve via /skill-name." [source: system-prompt-minimal-mode.md]

即使处于极简模式，用户仍然可以通过 `/skill-name` 的方式调用技能。

### Minimal Mode 小结

Minimal Mode 的价值在于"可预测性"。在 CI/CD pipeline、容器环境、或任何需要严格控制运行条件的场景中，自动发现和后台预取可能带来不确定性。Minimal Mode 通过剥离这些功能，确保 Claude Code 的行为完全由用户通过 CLI 参数控制，没有任何隐式依赖。

---

## 四种模式对比

| 维度 | Auto Mode | Buddy Mode | Learning Mode | Minimal Mode |
|------|-----------|------------|---------------|--------------|
| **自主程度** | 高 -- 自主推断、立即执行 | N/A -- 创意生成系统 | 中 -- 自主完成常规代码，交出关键决策 | 由用户显式控制 |
| **交互频率** | 低 -- 尽量不打断用户 | 偶发 -- 伴侣评论代码 | 高 -- 主动请求用户编写代码 | 取决于用户提供的上下文 |
| **安全边界** | 破坏性操作和数据泄露仍需确认 | N/A -- 不直接操作代码 | 同默认行为 | 同默认行为 |
| **核心目标** | 快速推进任务 | 提供情感陪伴和趣味 | 在完成任务中促进学习 | 提供可控的最小运行环境 |
| **适用场景** | 赶 deadline、批量重构、自动化流水线 | 日常开发中的趣味陪伴 | 编程学习、onboarding 新人 | CI/CD、容器环境、受限环境 |
| **行为驱动** | 6 条执行规则 | 稀有度 + 属性值 + 灵感词 | Learn by Doing + Insight 系统 | 跳过列表 + CLI 参数 |

从这个对比表中能看到一个清晰的设计脉络：Auto Mode 和 Learning Mode 处于交互光谱的两个极端，一个追求最少交互，一个主动创造交互；Buddy Mode 完全跳出工具范型，转向情感化设计；Minimal Mode 则回归基础设施层，追求确定性和可控性。

---

## 共同底线：Executing Actions with Care

四种模式各有侧重，但共享同一条安全底线。这段 prompt 不属于任何单一模式，而是作为通用行为准则贯穿所有场景。

### 核心原则

> "Carefully consider the reversibility and blast radius of actions. Generally you can freely take local, reversible actions like editing files or running tests. But for actions that are hard to reverse, affect shared systems beyond your local environment, or could otherwise be risky or destructive, check with the user before proceeding." [source: system-prompt-executing-actions-with-care.md]

这段话建立了两个判断维度：**可逆性**（reversibility）和**影响范围**（blast radius）。本地可逆操作（如编辑文件、运行测试）可以自由执行；难以逆转、影响共享系统、或有破坏风险的操作，必须先征得用户同意。

### 成本不对称论证

系统 prompt 给出了一个简洁的论证：

> "The cost of pausing to confirm is low, while the cost of an unwanted action (lost work, unintended messages sent, deleted branches) can be very high." [source: system-prompt-executing-actions-with-care.md]

暂停确认的成本低，错误操作的代价高。基于这个成本不对称分析，默认行为应该是"宁可多问一次"。

### 四类需要确认的操作

系统 prompt 明确列出了四类高风险操作：

**破坏性操作**：删除文件/分支、清空数据库表、杀死进程、rm -rf、覆盖未提交的更改。

**难以逆转的操作**：force-push、git reset --hard、修改已发布的 commit、移除或降级依赖、修改 CI/CD pipeline。

**对他人可见或影响共享状态的操作**：推送代码、创建/关闭/评论 PR 或 Issue、发送消息（Slack、邮件、GitHub）、修改共享基础设施或权限。

**上传内容到第三方工具**：图表渲染器、pastebin、gist 等。系统 prompt 特别指出：

> "Consider whether it could be sensitive before sending, since it may be cached or indexed even if later deleted." [source: system-prompt-executing-actions-with-care.md]

即使后来删除了，内容也可能被缓存或索引。这是个常被忽视的风险。

### 障碍处理

当遇到障碍时，系统 prompt 明确禁止用破坏性手段绕过：

> "Do not use destructive actions as a shortcut to simply make it go away." [source: system-prompt-executing-actions-with-care.md]

具体指导包括：要找根因而不是绕过安全检查（如使用 `--no-verify`）；遇到不熟悉的文件、分支或配置时要先调查再删除，因为它们可能是用户正在进行的工作；要解决合并冲突而不是丢弃更改；要调查锁文件被哪个进程持有而不是直接删除。

最后：

> "Only take risky actions carefully, and when in doubt, ask before acting. Follow both the spirit and letter of these instructions -- measure twice, cut once." [source: system-prompt-executing-actions-with-care.md]

"Measure twice, cut once"，木工界的古老格言，在这里成为了 AI 助手的行为准则。

---

## 总结

四种运行模式的设计反映出一个基本事实：**AI 助手不存在单一的"正确行为"，正确行为取决于场景**。

Auto Mode 回答"效率优先时怎么做"。六条规则构建了一个自洽的自主执行框架：前三条推动向前，后两条守住底线，中间留出修正空间。"自主"和"安全"并不对立，自主是在安全边界内的自主。

Buddy Mode 回答"工具是否可以有趣"。它把编码伴侣的生成变成了一个受约束的创意过程：稀有度控制放飞程度、属性值约束性格方向、灵感词提供创意锚点。系统 prompt 不仅能定义行为规则，还能驱动创意生成。

Learning Mode 回答"如何在完成任务的同时促进学习"。Learn by Doing 机制精确控制了用户参与的粒度（2-10 行）和时机（20+ 行中的关键决策），TODO(human) 标记系统确保了代码交接的可靠性，Insight 系统提供了恰到好处的教学反馈。这不是简单的"让用户自己写"，而是一套精心设计的教学系统。

Minimal Mode 回答"如何保证行为的确定性"。通过跳过一切自动发现和后台操作，它确保 Claude Code 的每个行为都来自用户的显式指令。在 CI/CD 和容器化部署场景中，这种确定性比便利性更重要。

四种模式之上，"Executing Actions with Care" 提供了一条不变的底线：无论处于哪种模式，对可逆性和影响范围的审慎评估都不动摇。模式改变了交互方式，但没有改变安全哲学。
