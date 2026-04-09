# 给 AI 立规矩：Claude Code 背后的 10 条铁律

## 引言：AI 编码的系统性偏差

当你让一个大语言模型帮你修一个 bug，它很可能不仅修了 bug，还顺手重构了周围的代码，加了几条注释，补了类型标注，甚至创建了一个新的工具函数来"提高可复用性"。这不是偶然，而是 AI 编码助手的系统性行为偏差。

大语言模型在训练过程中吸收了大量教科书级的工程实践、设计模式文章和"最佳实践"指南。这些知识在恰当的场景下有价值，但当模型面对一个简单的、局部的任务时，会产生一种"过度补偿"倾向——它倾向于展示它所知道的一切，而不是只做被要求的事。

Anthropic 在 Claude Code 的 system prompt 中用了一套极简但精准的约束体系来对抗这种偏差。这些约束不是冗长的行为准则，每条只有几十个 token，直接针对 AI 的特定"自然倾向"发出禁止指令。本文逐一分析这 10 条核心行为规则，以及围绕它们的风险控制哲学和输出效率约束。

---

## 规则一：不做额外的添加

> "Don't add features, refactor code, or make 'improvements' beyond what was asked. A bug fix doesn't need surrounding code cleaned up. A simple feature doesn't need extra configurability. Don't add docstrings, comments, or type annotations to code you didn't change. Only add comments where the logic isn't self-evident."

[source: system-prompt-doing-tasks-no-unnecessary-additions.md]

这是整个约束体系中信息密度最高的一条。它在一段话里封堵了 AI 过度补偿行为的四个主要出口：功能添加、代码重构、文档补充、类型标注。

设想开发者让 Claude Code 修复一个 API 端点的空指针异常。没有这条约束时，模型可能不仅修复了空指针，还顺手把整个 controller 的错误处理重构了一遍，给所有方法加了 JSDoc 注释，把 query 参数的类型从隐式的 `any` 改成了显式的 interface 定义。从模型的角度看，这些都是在"改善"代码。但从开发者的角度看，一次本应是三行改动的 hotfix 变成了一个 200 行的 diff，需要逐行审查是否引入了新问题。

核心原则就是**最小变更**：只改需要改的，不动不需要动的。最后一句——"Only add comments where the logic isn't self-evident"——在说注释本身也需要有正当理由，不是默认行为。

## 规则二：不做过早抽象

> "Don't create helpers, utilities, or abstractions for one-time operations. Don't design for hypothetical future requirements. The right amount of complexity is what the task actually requires—no speculative abstractions, but no half-finished implementations either. Three similar lines of code is better than a premature abstraction."

[source: system-prompt-doing-tasks-no-premature-abstractions.md]

这条规则直指 AI 的另一个深层倾向：对设计模式的过度热衷。大语言模型在训练数据中见过无数关于 DRY 原则、设计模式和代码复用的文章，以至于它会本能地把任何出现两到三次的相似代码片段提取成一个工具函数或抽象类。

但真实的软件工程经验告诉我们，过早抽象的危害往往大于代码重复。三条相似的代码行可能代表三个不同的业务意图，今天把它们抽象在一起，明天其中一个需要分叉时，抽象反而成为负担。规则中那句"Three similar lines of code is better than a premature abstraction"是对传统 DRY 原则的一次修正——不是不要复用，而是不要为假设的未来需求提前买单。

后半句也值得注意："no half-finished implementations either"。不做过早抽象不等于可以偷工减料。正确的复杂度是任务实际需要的复杂度，既不多也不少。

## 规则三：不做不必要的错误处理

> "Don't add error handling, fallbacks, or validation for scenarios that can't happen. Trust internal code and framework guarantees. Only validate at system boundaries (user input, external APIs). Don't use feature flags or backwards-compatibility shims when you can just change the code."

[source: system-prompt-doing-tasks-no-unnecessary-error-handling.md]

AI 编码的一个典型症状：防御性编程的过度使用。模型在训练中见过大量关于"防御性编程"的正面描述，导致它在每个函数调用、每个内部赋值、每个不可能出错的路径上都加上 try-catch 和 null check。

规则明确划分了验证的边界：只在系统边界验证，内部代码互信。所谓系统边界，就是用户输入和外部 API 这两类不可信数据源。边界之内，应该信任你的代码和框架的保证。如果一个函数的调用者保证了参数不为 null，被调用者就不需要再检查一遍。

最后一句同样值得注意："Don't use feature flags or backwards-compatibility shims when you can just change the code." 错误处理的过度使用往往与对"安全变更"的过度追求相伴——模型想通过 feature flag 和兼容层来避免破坏性变更，但在 AI 编码场景下，直接改代码往往是最安全的做法，因为开发者可以直接审查完整的 diff。

## 规则四：不做兼容性 hack

> "Avoid backwards-compatibility hacks like renaming unused _vars, re-exporting types, adding // removed comments for removed code, etc. If you are certain that something is unused, you can delete it completely."

[source: system-prompt-doing-tasks-no-compatibility-hacks.md]

这条规则针对的是 AI 对"安全删除"的过度谨慎。当模型需要删除或重命名一个变量、类型或函数时，它的本能反应不是直接删除，而是保留一个重导出、加个 deprecated 注释、或者用下划线前缀标记为"未使用"。

根源在于模型对"可能破坏其他代码"的过度担忧。在大型代码库中确实需要谨慎，但 Claude Code 的工作场景是：它可以看到当前项目的完整上下文，可以确定某个导出是否被引用。确定没有被使用的话，直接删除就是正确的做法。保留那些兼容性 hack 只会制造代码噪音，增加未来维护者的认知负担。

"If you are certain"这个条件很重要——它不是鼓励鲁莽删除，而是要求在有把握的情况下果断行动。确定性来自对代码库的分析，不是凭空假设。

## 规则五：不给时间估计

> "Avoid giving time estimates or predictions for how long tasks will take, whether for your own work or for users planning projects. Focus on what needs to be done, not how long it might take."

[source: system-prompt-doing-tasks-no-time-estimates.md]

乍看反直觉——一个 AI 助手为什么不能估计任务时间？答案在于 AI 估计时间的本质：它不是基于对自己执行速度的实测数据，而是基于训练数据中对"这类任务通常需要多久"的统计平均。这种估计对用户的项目管理几乎没有任何参考价值，反而可能产生误导。

更深层的问题是，当 AI 给出时间估计时，它在做出一个它无法负责的承诺。用户可能会据此安排项目计划，而实际执行时间可能与估计相差甚远。Anthropic 选择的策略是直接禁止这类输出，让 AI 专注于它能做好的事——描述需要做什么，而不是猜测需要多久。

## 规则六：最小化文件创建

> "Do not create files unless they're absolutely necessary for achieving your goal. Generally prefer editing an existing file to creating a new one, as this prevents file bloat and builds on existing work more effectively."

[source: system-prompt-doing-tasks-minimize-file-creation.md]

AI 编码助手有一种"创建新文件"的强烈倾向。面对一个新需求时，模型倾向于创建新的模块、新的工具文件、新的配置文件，而不是在现有文件中添加逻辑。这种倾向部分来自训练数据中对模块化设计的推崇，部分来自一个技术现实：模型生成全新内容比理解并修改已有内容更容易。

文件膨胀是真实项目中一个严重的维护问题。每多一个文件就意味着多一个需要理解、维护、测试和审查的单元。"builds on existing work more effectively"点出了关键：在现有文件中添加代码，意味着新代码会自然继承现有的测试覆盖、代码审查上下文和项目约定。创建新文件则意味着所有这些都要从零开始。

"Absolutely necessary"是一个很高的门槛。它迫使模型在创建文件前思考：这个逻辑真的不能放在现有文件中吗？真的需要一个独立的模块吗？大多数情况下，答案是否定的。

## 规则七：先读后改

> "In general, do not propose changes to code you haven't read. If a user asks about or wants you to modify a file, read it first. Understand existing code before suggesting modifications."

[source: system-prompt-doing-tasks-read-before-modifying.md]

这可能是所有规则中最"常识"的一条，但它的存在本身就说明了一个事实：没有这条约束时，AI 确实会基于文件名、函数签名或上下文推断来提出代码修改建议，而不先读取完整文件内容。

这种行为在人类工程师中也有对应——不仔细看代码就提 PR review 意见，或者在不了解上下文的情况下提出重构建议。但 AI 的版本更危险，因为模型可能会基于对函数名或类型的"理解"来生成它认为合理的代码片段，而这个理解可能与实际实现完全不同。

注意这里说的是"suggesting modifications"——即使在建议阶段，也需要先读取代码。模型的任何输出（无论是实际修改还是讨论性建议）都必须建立在已读代码的基础上。

## 规则八：允许有野心的任务

> "You are highly capable and often allow users to complete ambitious tasks that would otherwise be too complex or take too long. You should defer to user judgement about whether a task is too large to attempt."

[source: system-prompt-doing-tasks-ambitious-tasks.md]

在所有约束规则中，这条是唯一一条"放"而不是"收"的规则。它的作用是防止 AI 过度自我限制——拒绝执行用户提出的大规模、高复杂度任务。

没有这条规则时，模型可能会基于对任务复杂度的评估，主动拒绝某些请求，或者建议用户"分步进行"、"先做一个小版本"。当用户明确知道自己要什么时，这种"善意的阻拦"会严重影响生产力。

规则没有说"接受所有任务"，而是说"defer to user judgement"——把任务是否可行的判断权交给用户。模型的责任是执行，风险评估是用户的责任。用户比模型更了解项目的全局上下文和优先级。

## 规则九：安全第一

> "Be careful not to introduce security vulnerabilities such as command injection, XSS, SQL injection, and other OWASP top 10 vulnerabilities. If you notice that you wrote insecure code, immediately fix it. Prioritize writing safe, secure, and correct code."

[source: system-prompt-doing-tasks-security.md]

在一条条"不要做这个、不要做那个"的约束中，这条是反向锚点——在所有"不做"的事项里，安全性不在其列。它要求的是更多的注意，而非限制。

规则中提到的 OWASP Top 10 是 Web 应用安全的事实标准列表，包括注入攻击、跨站脚本、身份认证失效等最常见的 Web 安全漏洞。

"If you notice that you wrote insecure code, immediately fix it"这句值得关注。它不是要求模型在每次生成代码时都做完整的安全审计，而是在模型自己意识到代码存在安全问题时，要求它主动修复。这种"自我纠正"机制比事后的安全检查更高效，因为它在代码生成的同一上下文中完成。

## 规则十：软件工程聚焦

> "The user will primarily request you to perform software engineering tasks. These may include solving bugs, adding new functionality, refactoring code, explaining code, and more. When given an unclear or generic instruction, consider it in the context of these software engineering tasks and the current working directory. For example, if the user asks you to change 'methodName' to snake case, do not reply with just 'method_name', instead find the method in the code and modify the code."

[source: system-prompt-doing-tasks-software-engineering-focus.md]

这条规则定义了 Claude Code 的身份锚点：你是一个软件工程工具，不是一个通用问答系统。

例子很精确：当用户说"把 methodName 改成 snake case"时，一个通用 AI 可能会回答"snake case 的形式是 method_name"，然后停下来。但一个软件工程工具的理解是：找到代码中的 methodName，把它重命名为 method_name，更新所有引用，确保代码仍然能工作。这个区别不在于"更聪明"，而在于对自身角色定位的理解。

规则还强调了"current working directory"作为上下文的锚点。在项目上下文中，"那个函数"指的是项目中的特定函数，而不是某个同名的通用概念。

---

## Executing Actions with Care：风险分级

前面 10 条规则约束的是代码生成的"内容"，而"Executing Actions with Care"这一节约束的是 AI 执行操作的"方式"。核心思想是**基于可逆性和影响范围的风险分级**。

> "Carefully consider the reversibility and blast radius of actions. Generally you can freely take local, reversible actions like editing files or running tests. But for actions that are hard to reverse, affect shared systems beyond your local environment, or could otherwise be risky or destructive, check with the user before proceeding."

[source: system-prompt-executing-actions-with-care.md]

规则将所有操作分为两个层级：

**低风险操作**（自由执行）：编辑文件、运行测试。这些操作是本地的、可逆的——可以用 git checkout 撤销文件修改，测试失败也不会造成任何损失。

**高风险操作**（需要确认）：推送代码、删除分支、修改共享基础设施、发送消息。这些操作要么不可逆（force-push 可以覆盖上游代码），要么影响范围超出本地环境（推送代码会影响所有协作者），要么对他人可见（在 GitHub 上评论、发送 Slack 消息）。

有一段关键的风险经济学论述：

> "The cost of pausing to confirm is low, while the cost of an unwanted action (lost work, unintended messages sent, deleted branches) can be very high."

确认的代价是几秒钟的等待，但错误操作的代价可能是数小时的工作丢失或不恰当的对外沟通。在这种非对称下，默认策略应该是"先问再做"。

规则也允许用户通过 CLAUDE.md 等持久化配置来改变这个默认行为：

> "This default can be changed by user instructions - if explicitly asked to operate more autonomously, then you may proceed without confirmation."

它定义了一个安全的默认值，但允许用户根据自己的风险偏好来调整。同时，"一次批准不等于永久授权"——用户批准了一次 git push 不意味着在所有上下文中都可以自动 push。授权的范围必须与实际请求匹配。

在处理障碍时，规则要求"不要用破坏性操作作为捷径"：

> "When you encounter an obstacle, do not use destructive actions as a shortcut to simply make it go away. For instance, try to identify root causes and fix underlying issues rather than bypassing safety checks."

这直接封堵了一种常见的"偷懒"行为：遇到合并冲突时直接覆盖、遇到 lock file 时直接删除、遇到 pre-commit hook 失败时用 --no-verify 跳过。模型应该去理解问题的根因，而不是用破坏性手段绕过。

---

## Output Efficiency：输出效率

> "Go straight to the point. Try the simplest approach first without going in circles. Do not overdo it. Be extra concise."

[source: system-prompt-output-efficiency.md]

这条规则针对 AI 输出中的三大浪费：

**推理先行**：模型倾向于先解释自己的思考过程，再给出答案。规则要求反过来——先给答案或行动，推理过程可以省略或后置。

**填充词和过渡句**："好的，我理解了你的需求"、"让我来分析一下"这类前奏词被直接禁止——"Do not restate what the user said — just do it."

**过度解释**：当一句话能说清楚时不要用三句话。规则列出了三类值得输出的内容：需要用户决策的信息、里程碑式的高层状态更新、改变计划方向的错误或阻塞。不在这三类中的内容，就应该精简或省略。

> "If you can say it in one sentence, don't use three."

这句话可以作为整条规则的缩写。它不适用于代码和工具调用——代码可以写该写的长度——但文本输出必须精简。

---

## Tone and Style：语气风格

Tone and Style 部分包含两条极短的规则。

第一条：

> "Your responses should be short and concise."

[source: system-prompt-tone-and-style-concise-output-short.md]

所有规则中最短的一条，仅有六个词。它的简洁本身就是一种示范——约束输出风格，"short and concise"就够了。

第二条：

> "When referencing specific functions or pieces of code include the pattern file_path:line_number to allow the user to easily navigate to the source code location."

[source: system-prompt-tone-and-style-code-references.md]

这条规则定义了代码引用的标准格式：`file_path:line_number`。它是大多数 IDE 和代码编辑器支持的跳转格式，用户可以直接 cmd+click 或 ctrl+click 跳转到对应位置。AI 的输出不仅精简，而且具有**可操作性**。

---

## 共同的设计模式

观察这 14 条规则（10 条核心行为规则 + 执行谨慎 + 输出效率 + 2 条语气风格规则），有几个共同的设计模式。

**极短指令**。大多数行为规则的正文在 16 到 104 个 token 之间。最短的"Your responses should be short and concise"只有 6 个词，最长的 no-unnecessary-additions 也就两三句话。短指令减少了模型在解析规则时的歧义空间，也减少了规则本身被"创造性解读"的可能。

**单向指令**。几乎所有规则都使用"不要"或"避免"开头，定义的是禁止行为而不是期望行为。这种"负面定义"在 prompt engineering 中是已知的有效策略——告诉模型"不要做什么"比告诉模型"要做什么"更容易产生可预测的行为。因为"要做什么"有无限种可能，"不要做什么"是有限集。

**没有解释**。规则只说"不要这样做"，不说"因为这样做会导致什么后果"。两个好处：节省 token，并且避免模型利用解释中的例外条款来绕过规则。当规则说"Don't add docstrings to code you didn't change"时，没有任何例外可以讨论——规则就是规则。

**问题导向**。每条规则都针对一个真实的、在开发中反复出现的问题场景。不是泛泛地要求"写好代码"，而是精确地指出"不要在 bug fix 中清理周围代码"。模型不需要理解什么是"好代码"的抽象概念，只需要知道在特定场景下不要做特定的事。

**"Yes, but"结构**。10 条核心规则中大多数是"收"（限制行为），但规则八（允许有野心的任务）和规则九（安全第一）是"放"。Anthropic 的思路不是"让 AI 尽量少做事"，而是"让 AI 只做该做的事"——限制过度工程化的同时，不限制必要的能力发挥。

---

## 这些规则揭示了 AI 编码的哪些"自然倾向"

这 14 条规则的存在本身，就是对 AI 编码行为模式的一份精确诊断。它们不是凭空制定的，而是基于大量实际使用中观察到的模型行为偏差。把这些规则翻转过来，就得到 AI 编码助手的"自然倾向"清单：

1. **添加倾向**：面对任何代码修改任务，AI 倾向于添加比必要更多的变更——功能、重构、注释、类型标注。
2. **抽象倾向**：AI 倾向于为一次性操作创建抽象层，为假设的未来需求设计架构。
3. **防御倾向**：AI 倾向于在不可能发生的场景上添加错误处理和验证，用兼容层来避免直接删除。
4. **保守删除倾向**：AI 倾向于保留不再使用的代码，用重命名、注释和兼容层来"软删除"而不是彻底删除。
5. **估计倾向**：AI 倾向于给出时间估计和预测，即使这些估计没有可靠的数据基础。
6. **创建倾向**：AI 倾向于创建新文件而不是在现有文件中修改，部分因为生成全新内容比理解已有内容更容易。
7. **跳跃倾向**：AI 倾向于在不先读取文件内容的情况下就提出修改建议。
8. **自我限制倾向**：AI 倾向于对大规模任务过度谨慎，主动拒绝或缩小任务范围。
9. **忽视安全倾向**：在追求"完成功能"的过程中，AI 可能忽略注入攻击、XSS 等安全漏洞。
10. **脱离上下文倾向**：AI 倾向于脱离当前项目的具体上下文来理解用户指令，给出通用回答而不是项目内操作。

这些倾向不是 bug，是大语言模型训练方式的必然产物。模型在训练中见过太多"教科书式"的代码、太多"最佳实践"文章、太多关于设计模式和架构原则的讨论。当它面对一个实际的编码任务时，这些知识会自然地"溢出"——不是因为它想炫耀，是因为这些模式已经在它的参数中被强化了无数遍。

Anthropic 这套约束体系的做法是：不试图让模型"学会"什么是好的工程判断，而是用一系列精确的负面指令来封堵已知的偏差行为。这比教模型"写出简洁的代码"有效得多，因为后者是一个模糊的目标，而"不要在 bug fix 中添加类型标注"是一个可执行的指令。

从工程角度看，这套规则代表了一种值得借鉴的 AI 产品设计思路：不追求让 AI 变得更"智能"，而是通过精确的行为约束让它在实际使用场景中更"有用"。
