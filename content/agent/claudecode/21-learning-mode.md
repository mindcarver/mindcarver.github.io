# 让 AI 变成你的编程私教：Claude Code Learning Mode

## 1. 为什么 Claude Code 需要教学功能

大多数 AI 编程工具只做一件事：替你把代码写完。你提需求，它输出结果，全程零交互。效率没话说，但开发者从过程里学不到任何东西。

Claude Code 的 Learning Mode 就是冲着这个问题来的。它的 system prompt 开宗明义：

> "You are an interactive CLI tool that helps users with software engineering tasks. In addition to software engineering tasks, you should help users learn more about the codebase through hands-on practice and educational insights." [source: system-prompt-learning-mode.md]

一个关键的设计决策藏在这句话里：Learning Mode 不是独立的教学工具，而是嵌入在工程任务执行过程中的学习机制。用户不需要切换模式，不需要离开当前工作流。在完成真实任务的同时，通过亲手编写关键代码片段来理解代码库。

Prompt 对 AI 的行为基调也划了线：

> "You should be collaborative and encouraging. Balance task completion with learning by requesting user input for meaningful design decisions while handling routine implementation yourself." [source: system-prompt-learning-mode.md]

"balance" 是枢纽词。AI 不把所有工作都推给用户，也不全部包揽。它自己处理 routine implementation，只在遇到 meaningful design decisions 时才邀请用户参与。这条分界线怎么划，正是 Learning Mode 的核心设计。

## 2. 核心设计：平衡任务完成与学习

Learning Mode 的架构可以拆成两个并行运行的子系统：任务执行（AI 搭建结构和常规实现）与学习介入（在关键节点暂停，邀请用户编写 2-10 行代码）。两者靠精确的触发条件协调，避免介入过于频繁或稀疏。

另外还有一个贯穿始终的辅助机制，即 Insight 分享。AI 在编写代码前后都会给出简短的教育性解释：

> "In order to encourage learning, before and after writing code, always provide brief educational explanations about implementation choices." [source: system-prompt-learning-mode-insights.md]

这些 Insight 出现在对话中而非代码中，且内容范围受到明确约束：

> "You should generally focus on interesting insights that are specific to the codebase or the code you just wrote, rather than general programming concepts." [source: system-prompt-learning-mode-insights.md]

Insight 不是泛泛的编程教程，而是跟当前代码库紧密挂钩的具体知识，保证了学习的情境性。

## 3. "Requesting Human Contributions" 触发条件

Learning Mode 的关键问题是：什么时候该让用户动手写代码？

System prompt 给出了精确的触发规则：

> "In order to encourage learning, ask the human to contribute 2-10 line code pieces when generating 20+ lines involving:
> - Design decisions (error handling, data structures)
> - Business logic with multiple valid approaches
> - Key algorithms or interface definitions" [source: system-prompt-learning-mode.md]

三个维度的约束：

代码量阈值 -- 只有当 AI 即将生成 20 行以上的代码时，才考虑请求用户参与。少于 20 行的代码片段通常不涉及足够复杂的设计决策，没有教学价值。

用户工作量边界 -- 用户被要求编写的代码限制在 2-10 行。少于 2 行没有实质学习意义，多于 10 行容易造成挫败感。

内容类型筛选 -- 不是所有代码都适合用户参与，只有三类会触发请求：涉及设计决策的代码（比如错误处理策略的选择、数据结构的设计），存在多种合理实现方式的业务逻辑，关键算法或接口定义。

这三类代码有一个共同特征：它们是整个功能模块中认知价值最高的部分。把这部分交给用户，学习效果最大化，又不让人陷入琐碎细节。

## 4. TODO(human) 机制

Learning Mode 在代码层面的实现依赖一个占位符机制：TODO(human)。

当 AI 决定请求用户参与时，它不是直接提问，而是先通过编辑工具在代码库中插入一个标记区域。System prompt 对此有硬性规定：

> "You must first add a TODO(human) section into the codebase with your editing tools before making the Learn by Doing request" [source: system-prompt-learning-mode.md]

"先编辑代码，再发出请求"这个顺序有三个设计意图：给用户提供定位锚点（搜索 TODO(human) 就能找到目标位置）；给请求提供上下文引用（AI 可以说 "look for TODO(human)" 指向具体位置）；保证代码结构完整（AI 已搭好周围基础设施，用户只需填充核心逻辑）。

System prompt 还对唯一性提出了硬性要求：

> "Make sure there is one and only one TODO(human) section in the code" [source: system-prompt-learning-mode.md]

多个 TODO(human) 会让用户困惑于该实现哪一个，也会增加 AI 集成工作的复杂度。

## 5. "Learn by Doing" 请求格式

AI 通过编辑工具放好 TODO(human) 占位符之后，会向用户发出一个结构化的学习请求。请求遵循严格的三段式格式：

> ```
> **Learn by Doing**
> **Context:** [what's built and why this decision matters]
> **Your Task:** [specific function/section in file, mention file and TODO(human) but do not include line numbers]
> **Guidance:** [trade-offs and constraints to consider]
> ``` [source: system-prompt-learning-mode.md]

Context 告诉用户目前已经构建了什么，以及为什么这个设计决策对整体功能有意义。它的作用是建立认知框架 -- 用户需要理解大图景才能做出合理的微观决策。

Your Task 明确指出用户需要在哪个文件的哪个函数或代码段中编写代码，引用 TODO(human) 作为定位标记。一个值得注意的细节：prompt 特别规定 "do not include line numbers"。因为行号会随编辑操作变化，硬编码行号反而制造混淆。

Guidance 列出用户在做设计决策时需要考虑的权衡因素和约束条件。这一段不给标准答案，而是呈现问题的多个维度，引导用户独立思考。

三段的排列顺序经过设计：先建立全局理解（Context），再明确局部任务（Your Task），最后提供思考框架（Guidance）。从宏观到微观再到方法论，构成一个完整的学习闭环。

Prompt 还对请求的措辞提出了指导：

> "Frame contributions as valuable design decisions, not busy work" [source: system-prompt-learning-mode.md]

用户被要求编写的代码不是机械的填充练习，而是真正影响系统行为的设计选择。这种 framing 直接影响用户的学习动机和投入程度。

## 6. 三种学习场景

System prompt 提供了三个示例，覆盖从完整函数实现到调试辅助的典型场景。

### 场景一：完整函数实现

> "In sudoku.js, implement the selectHintCell(board) function. Look for TODO(human). This function should analyze the board and return {row, col} for the best cell to hint, or null if the puzzle is complete." [source: system-prompt-learning-mode.md]

数独提示系统。AI 已经搭好了完整的 UI 基础设施（按钮、高亮显示、可能的值展示），唯独留下核心策略函数让用户实现。Guidance 部分列举了多种可选策略：优先处理只有一个可能值的格子（naked singles）、优先处理行列中已填入较多格子的区域、或者采用平衡策略。用户需要在多个合理方案中做选择 -- 这正是 "Business logic with multiple valid approaches" 触发条件的体现。

### 场景二：部分函数实现

> "In upload.js, inside the validateFile() function's switch statement, implement the 'case "document":' branch. Look for TODO(human). This should validate document files (pdf, doc, docx)." [source: system-prompt-learning-mode.md]

文件上传组件展示了更精细的粒度控制。AI 完成了主体验证逻辑，只留下 switch 语句中的一个分支让用户实现。Guidance 提示用户考虑文件大小限制、扩展名与 MIME 类型的匹配。这对应的是 "Design decisions (error handling, data structures)" 触发条件。

### 场景三：调试辅助

> "In calculator.js, inside the handleInput() function, add 2-3 console.log statements after the TODO(human) comment to help debug why number inputs fail." [source: system-prompt-learning-mode.md]

调试场景跟前两种有本质区别。用户不是在写功能代码，而是在添加诊断代码。Guidance 建议记录原始输入值、解析结果和验证状态。这个场景的教学目标不是让用户理解算法或设计模式，而是培养系统化的调试思维：知道该观察什么变量、在什么位置插入观测点。

三个示例的递进关系值得注意：从实现完整函数（全局设计），到实现函数片段（局部设计），再到添加调试代码（诊断思维），覆盖了软件开发中最核心的几种认知能力。

## 7. 关键规则："Don't take any action after the Learn by Doing request"

Learning Mode 中最引人注目的一条规则：

> "Don't take any action or output anything after the Learn by Doing request. Wait for human implementation before proceeding." [source: system-prompt-learning-mode.md]

在 AI 交互的常规模式中，AI 往往一次性给出完整答案，或在提出问题后紧接着提供解决方案。这种行为在教学场景中有害，用户的思考过程会被短路。

"Don't take any action" 强制创造了一个认知停顿。用户面对问题、阅读指导、定位 TODO(human)、思考方案、编写代码，这个完整的思考链条不会被打断。AI 在发出请求后必须彻底沉默，等待用户完成实现。控制权真正交给了用户：AI 不自作主张推进其他任务，不暗示"正确答案"，不输出任何额外解释。

从工程角度看，这条规则也简化了状态管理。AI 进入明确的等待态，无需处理并发编辑冲突。

## 8. 完成后的 Insight 分享原则

用户写完代码之后，AI 的回应方式同样受到精确约束：

> "Share one insight connecting their code to broader patterns or system effects. Avoid praise or repetition." [source: system-prompt-learning-mode.md]

一条简短的规则，三个设计指令：

One insight -- 只分享一个洞察，不做冗长总结。用户的注意力在刚完成编码时最集中，一个精准的洞察比十个泛泛的观点更有价值。

Connecting their code to broader patterns or system effects -- 洞察的方向限定为"连接"，把用户刚写的具体代码跟更广泛的模式或系统效应联系起来。不是在说"你写得很好"，而是在说"你写的这段代码，其实体现了 XX 模式，或者对系统的 YY 部分产生了 ZZ 影响"。这种连接性知识是最具迁移价值的学习成果。

Avoid praise or repetition -- 明确禁止两种行为。不给空洞的赞美（"Great job!"），因为这种赞美不含信息量，还会让用户质疑 AI 反馈的真实性。不重复用户已经知道的内容，重复同样不含信息量。

这个原则跟系统另一层的 Insight 机制形成互补。System prompt for insights 要求在编写代码前后都提供教育性解释，而这里的 After Contributions 规则专门针对用户亲自编写代码之后的反馈场景，更强调"连接"而非"解释"。

## 9. TodoList 集成

Learning Mode 不是孤立运行的教学机制，它需要跟任务管理流程紧密结合。System prompt 定义了 TodoList 的集成方式：

> "If using a TodoList for the overall task, include a specific todo item like 'Request human input on [specific decision]' when planning to request human input. This ensures proper task tracking. Note: TodoList is not required for all tasks." [source: system-prompt-learning-mode.md]

这段规定揭示了几个设计细节：

TodoList 是可选的 -- "Note: TodoList is not required for all tasks" 明确表示，简单任务可以直接执行，不需要项目计划。

但一旦用了 TodoList，学习介入必须作为显式条目 -- 如果任务复杂到需要 TodoList 来追踪进度，那么"请求用户参与"这个步骤不能是临时的、隐含的，必须作为独立条目出现在任务列表中。

Prompt 还给出了一个具体的 TodoList 流程示例：

> ```
>    [check] "Set up component structure with placeholder for logic"
>    [check] "Request human collaboration on decision logic implementation"
>    [check] "Integrate contribution and complete feature"
> ``` [source: system-prompt-learning-mode.md]

三步流程展示了一个完整的协作周期：AI 搭建结构并预留占位符，邀请用户实现核心决策逻辑，最后 AI 整合用户贡献并完成功能。中间的 "Request human collaboration" 步骤在 TodoList 中有明确位置，整个学习介入过程可追踪、可预期。

## 10. 补充：Teach Mode 与 Learning Mode 的互补

除了 Learning Mode，Claude Code 还通过 `request_teach_access` 工具提供了独立的 Teach Mode：

> "Request permission to guide the user through a task step-by-step with on-screen tooltips. Use this INSTEAD OF request_access when the user wants to LEARN how to do something (phrases like 'teach me', 'walk me through', 'show me how', 'help me learn')." [source: tool-description-request_teach_access-part-of-teach-mode.md]

Teach Mode 是步骤引导式的：AI 获得权限后，通过反复调用 `teach_step` 逐一展示全屏 tooltip，用户点击 Next 继续。跟 Learning Mode 形成互补：Teach Mode 适用于概念学习和流程演示，Learning Mode 适用于动手实践和设计决策。两者共同构成了 Claude Code 的教学能力。

## 11. 总结

Claude Code 的 Learning Mode 是一套经过深思熟虑的 AI 辅助编程教学方案。几个核心原则：

嵌入而非替代 -- 学习机制嵌入在真实工作流中，用户在完成实际任务的过程中自然地学习。

选择性地分配认知负荷 -- AI 处理低认知价值的常规工作，将高认知价值的设计决策留给用户，且基于明确的触发条件（代码量阈值、内容类型筛选、工作量边界）。

结构化但开放 -- 请求格式提供完整的上下文和指导框架，但绝不给出标准答案。用户面对的是真正开放的设计问题。

强制等待保护思考空间 -- "Don't take any action after the Learn by Doing request" 本质上是 AI 对用户认知自主权的尊重。在 AI 能瞬间给出答案的时代，主动选择等待本身就是一种教学行为。

连接而非评判 -- 完成后的反馈不是评判，而是把用户的代码跟更广泛的模式连接起来，提供具有迁移价值的知识。

从系统设计角度看，Learning Mode 用极少的规则实现了复杂的教学功能：三个触发条件、一个三段式请求格式、一条等待规则、一个反馈原则。这些规则的总文本不超过一段话，却足以驱动一个有效的编程学习体验。这种简约性本身值得 AI 教育产品设计者借鉴。
