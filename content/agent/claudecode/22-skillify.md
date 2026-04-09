# 把好运气变成可复用技能：Claude Code 的 Skillify 机制

## 1. 从一次成功会话到可复用技能

在日常开发中，用户跟 Claude Code 的对话往往包含大量可复用的工作流：一次完整的 cherry-pick 流程、一套从代码变更到发布上线的验证步骤、一个跨多个仓库的依赖升级模式。这些流程在第一次执行时需要人工引导和纠正，但一旦跑通，后续的每一次执行本质上都是在重复同一套逻辑。

Skillify 就是为捕捉这种重复性而设计的。它的定位很明确："You are capturing this session's repeatable process as a reusable skill." [source: system-prompt-skillify-current-session.md] 核心假设是：一次成功的会话中藏着可被提取、结构化和复用的流程知识。Skillify 的工作不是创造新东西，而是从已有的成功经验中提炼出一个可编程的规范。

Skillify 的输入是当前会话的完整上下文，包括两部分关键数据：会话记忆摘要（session memory summary）和用户在会话中的所有消息。原文对此有明确说明："Here is the session memory summary" 和 "Here are the user's messages during this session. Pay attention to how they steered the process, to help capture their detailed preferences in the skill." [source: system-prompt-skillify-current-session.md] 用户消息中的纠正和引导信息尤为重要，它们揭示了用户在流程中的真实偏好，这些偏好往往比用户口头描述的"理想流程"更可靠。

整个 Skillify 流程分四步：分析会话（Analyze）、访谈用户（Interview）、编写技能文件（Write）、确认保存（Confirm）。四步构成一条从原始会话数据到结构化技能规范的处理管线。

## 2. Step 1: Analyze the Session -- 七维分析框架

在向用户提出任何问题之前，Skillify 首先需要对会话进行独立的分析。原文要求："Before asking any questions, analyze the session to identify" 以下七个维度 [source: system-prompt-skillify-current-session.md]：

可复用流程识别 -- "What repeatable process was performed"。这是最基础的分析维度，决定了整个技能的价值基础。如果会话中没有可重复的流程模式，Skillify 就没有启动的意义。

输入参数提取 -- "What the inputs/parameters were"。识别流程中哪些是固定不变的步骤，哪些是需要每次变化的参数。例如一个 cherry-pick 流程中，源分支名称是参数，但 cherry-pick 的操作步骤是固定的。参数的识别质量直接决定了技能的通用性。

步骤序列梳理 -- "The distinct steps (in order)"。将流程拆解为有序的独立步骤。每个步骤应该是一个有明确输入和输出的原子操作，步骤之间的依赖关系需要被清晰地识别。

成功标准定义 -- "The success artifacts/criteria (e.g. not just 'writing code,' but 'an open PR with CI fully passing') for each step"。原文特别强调成功标准不能是笼统的"写了代码"，而必须是具体的、可验证的产出物，比如"一个 CI 全部通过的 PR"。这种精确性是后续步骤中 Success criteria 标注的基础。

用户纠正捕捉 -- "Where the user corrected or steered you"。用户在会话中的纠正行为是极有价值的信息。用户说"不对，应该先跑测试再提交"这种纠正，直接反映了对流程顺序的真实偏好。这些纠正会被转化为技能文件中 Rules 标注的硬约束。

工具权限识别 -- "What tools and permissions were needed"。记录流程执行过程中使用了哪些工具，以及需要什么样的权限。这直接映射到 SKILL.md frontmatter 中的 `allowed-tools` 字段。

Agent 使用记录 -- "What agents were used" 以及 "What the goals and success artifacts were"。如果流程中涉及了 subagent 的调度，需要记录每个 agent 的角色和职责。这决定了技能文件中 Execution 标注的选择。

这七个维度不是独立的，而是相互关联的。步骤序列中的每一步都有自己的成功标准，步骤之间的数据依赖构成了参数传递的依据，用户的纠正反映了步骤之间隐含的约束关系。Skillify 要求在进行任何用户交互之前先完成这七维分析，确保后续的访谈是基于充分的准备工作，而非从零开始的盲目提问。

## 3. Step 2: Interview the User -- 四轮渐进式访谈

分析完成后，Skillify 进入用户访谈阶段。访谈通过 AskUserQuestion 工具进行，原文有一条严格的规则："Use AskUserQuestion for ALL questions! Never ask questions via plain text." [source: system-prompt-skillify-current-session.md] 这确保了所有用户交互都通过统一的接口进行，而非散落在对话文本中。

访谈的另一个约束是关于选项设计："The user always has a freeform 'Other' option to type edits or feedback -- do NOT add your own 'Needs tweaking' or 'I'll provide edits' option. Just offer the substantive choices." [source: system-prompt-skillify-current-session.md] 系统不应该替用户预设"微调"之类的模糊选项，而应该提供实质性的选择，让用户在"其他"选项中表达自己的精确意图。

### Round 1: 高层确认

第一轮聚焦于技能的整体定义："Suggest a name and description for the skill based on your analysis. Ask the user to confirm or rename." 以及 "Suggest high-level goal(s) and specific success criteria for the skill." [source: system-prompt-skillify-current-session.md]

目的是确保 Skillify 对会话的理解跟用户的意图对齐。Skillify 基于七维分析的结果，提出技能的名称、描述和成功标准，由用户确认或修正。如果名称和描述在第一轮就没有对齐，后续的细节讨论就会建立在错误的基础上。

### Round 2: 细节讨论

第二轮进入流程的结构层面，包含四个议题：

步骤列表呈现 -- "Present the high-level steps you identified as a numbered list. Tell the user you will dig into the detail in the next round." [source: system-prompt-skillify-current-session.md] 步骤以编号列表形式呈现，让用户看到流程的全貌，但不深入每一步的细节。

参数建议 -- "If you think the skill will require arguments, suggest arguments based on what you observed. Make sure you understand what someone would need to provide." [source: system-prompt-skillify-current-session.md] 参数的识别基于 Step 1 中对输入参数的分析，但需要用户确认这些参数是否完整且正确。

执行模式选择 -- 原文提出了一个关键的架构决策："If it's not clear, ask if this skill should run inline (in the current conversation) or forked (as a sub-agent with its own context). Forked is better for self-contained tasks that don't need mid-process user input; inline is better when the user wants to steer mid-process." [source: system-prompt-skillify-current-session.md] Inline 和 Fork 的选择决定了技能的执行模型，影响贯穿技能的整个生命周期。

保存位置选择 -- "Ask where the skill should be saved." 提供两个选项 [source: system-prompt-skillify-current-session.md]：
- 项目级别：`.claude/skills/<name>/SKILL.md`，适用于特定项目的工作流
- 个人级别：`~/.claude/skills/<name>/SKILL.md`，跨项目通用的个人工作流

### Round 3: 逐步拆解

第三轮是访谈中最深入的部分，针对每个主要步骤逐一讨论。原文列出了六个需要确认的维度 [source: system-prompt-skillify-current-session.md]：

产出物依赖 -- "What does this step produce that later steps need? (data, artifacts, IDs)" -- 确认步骤之间的数据依赖关系。例如第一步创建的 PR 编号是第二步监控 CI 所需要的。这种依赖关系映射到 SKILL.md 中的 Artifacts 标注。

成功标准 -- "What proves that this step succeeded, and that we can move on?" -- 每一步的完成判定条件。映射到 Success criteria 标注。

人工检查点 -- "Should the user be asked to confirm before proceeding? (especially for irreversible actions like merging, sending messages, or destructive operations)" -- 对于不可逆操作，是否需要暂停等待用户确认。映射到 Human checkpoint 标注。

并行机会 -- "Are any steps independent and could run in parallel? (e.g., posting to Slack and monitoring CI at the same time)" -- 识别可以并发执行的步骤。影响步骤编号的方式（使用 3a、3b 的子编号表示并行）。

执行方式 -- "How should the skill be executed? (e.g. always use a Task agent to conduct code review, or invoke an agent team for a set of concurrent steps)" -- 确认每一步的执行模式，是直接执行、委托给 Task agent、使用 Teammate 并行，还是由人工完成。

硬约束 -- "What are the hard constraints or hard preferences? Things that must or must not happen?" -- 流程中绝对不能违反的规则。

原文特别说明，当步骤超过三个或需要较多澄清时，这一轮可以进行多轮 AskUserQuestion，每个步骤一轮。同时再次强调用户纠正的重要性："IMPORTANT: Pay special attention to places where the user corrected you during the session, to help inform your design." [source: system-prompt-skillify-current-session.md]

### Round 4: 最终确认

最后一轮确认技能的触发条件："Confirm when this skill should be invoked, and suggest/confirm trigger phrases too." 原文给出了具体的例子："For a cherrypick workflow you could say: Use when the user wants to cherry-pick a PR to a release branch. Examples: 'cherry-pick to release', 'CP this PR', 'hotfix.'" [source: system-prompt-skillify-current-session.md]

触发条件的设计需要同时满足两个需求：让 Claude Code 的 skill 匹配系统能准确识别何时应该自动调用该技能；让用户知道用什么短语可以手动触发该技能。

最后一轮还可以补充："You can also ask for any other gotchas or things to watch out for, if it's still unclear." [source: system-prompt-skillify-current-session.md]

但访谈有明确的停止条件："Stop interviewing once you have enough information. IMPORTANT: Don't over-ask for simple processes!" [source: system-prompt-skillify-current-session.md] 简单的流程不应该被过度访谈，这是对工程效率的尊重。

## 4. Step 3: Write the SKILL.md -- frontmatter 规范

访谈完成后，Skillify 将所有信息整合为一份 SKILL.md 文件。文件的 frontmatter 部分定义了技能的元数据 [source: system-prompt-skillify-current-session.md]：

```yaml
---
name: {{skill-name}}
description: {{one-line description}}
allowed-tools:
  {{list of tool permission patterns observed during session}}
when_to_use: {{detailed description of when Claude should automatically invoke this skill}}
argument-hint: "{{hint showing argument placeholders}}"
arguments:
  {{list of argument names}}
context: {{inline or fork -- omit for inline}}
---
```

每个字段都有明确的设计意图：

name 是技能的唯一标识符，也是用户通过 `/name` 触发技能时的名称。

description 是一行描述，用于在技能列表中快速展示技能的用途。

allowed-tools 采用模式匹配的格式，原文要求使用 "Minimum permissions needed"，并且用模式如 `Bash(gh:*)` 而非笼统的 `Bash`。这种最小权限原则确保技能只被授予它实际需要的工具权限。

when_to_use 是整个 frontmatter 中最关键的字段。原文用 CRITICAL 标注了它的重要性："tells the model when to auto-invoke. Start with 'Use when...' and include trigger phrases." [source: system-prompt-skillify-current-session.md] 这个字段决定了 Claude Code 的技能调度系统是否会在合适的时机自动匹配到该技能。格式要求以 "Use when..." 开头，并包含触发短语和用户消息示例。

arguments 和 argument-hint 仅在技能接受参数时包含。参数在正文体中通过 `$name` 语法进行替换。

context 仅在技能需要 fork 模式时设置为 `context: fork`，inline 模式时省略该字段。

## 5. Per-step Annotations 详解

SKILL.md 正文中的每个步骤可以附加五种标注。原文对这些标注的定义和使用条件有精确的说明 [source: system-prompt-skillify-current-session.md]：

Success criteria（必需） -- "is REQUIRED on every step. This helps the model understand what the user expects from their workflow, and when it should have the confidence to move on." 每个步骤必须有成功标准，这是模型判断"这一步是否完成、是否可以继续"的依据。没有成功标准的步骤就像没有出口条件的循环，模型不知道何时该停下来。

Execution（可选） -- 默认值为 Direct（直接执行），其他选项包括 Task agent（委托给 subagent）、Teammate（具备并行能力和 agent 间通信的 agent）、以及 `[human]`（由用户手动完成）。原文说明："Only needs specifying if not Direct." 只有在非默认执行模式时才需要显式标注。

Artifacts（可选） -- "Data this step produces that later steps need (e.g., PR number, commit SHA). Only include if later steps depend on it." 标注的是步骤间的数据依赖。只有在后续步骤确实需要该步骤的产出物时才需要标注。

Human checkpoint（可选） -- "When to pause and ask the user before proceeding. Include for irreversible actions (merging, sending messages), error judgment (merge conflicts), or output review." 人工检查点适用于三类场景：不可逆操作（合并代码、发送消息）、错误判断（合并冲突的处理方式选择）、以及输出审查。该标注的存在意味着步骤执行到某个节点时必须暂停，等待用户的明确确认后才能继续。

Rules（可选） -- "Hard rules for the workflow. User corrections during the reference session can be especially useful here." 硬规则标注记录的是流程中绝对不能违反的约束。原文特别指出，用户在原始会话中的纠正行为是 Rules 的重要来源。如果用户在会话中说"不要自动提交，让我先看看"，这条纠正就应该被转化为一条 Rule。

步骤编号还有两个结构规则："Steps that can run concurrently use sub-numbers: 3a, 3b" 和 "Steps requiring the user to act get `[human]` in the title"。并行步骤使用子编号，人工步骤在标题中标注。

最后，原文给出了一个关于复杂度的指导原则："Keep simple skills simple -- a 2-step skill doesn't need annotations on every step." [source: system-prompt-skillify-current-session.md] 简单的技能不应该被过度标注，这体现了 Skillify 的实用主义设计。

## 6. Step 4: Confirm and Save

技能文件编写完成后，Skillify 在保存之前设置了一道确认关卡："Before writing the file, output the complete SKILL.md content as a yaml code block in your response so the user can review it with proper syntax highlighting." [source: system-prompt-skillify-current-session.md]

这一步有两个值得注意的设计细节。输出格式选择 yaml code block 而非直接展示 markdown，是因为 SKILL.md 的 frontmatter 本身就是 yaml 格式，code block 提供了语法高亮，让用户更容易检查 frontmatter 的正确性。

确认方式使用 AskUserQuestion 工具，提出简洁的问题如 "Does this SKILL.md look good to save?"，并且原文特别要求："do NOT use the body field, keep the question concise." [source: system-prompt-skillify-current-session.md] 保持问题简洁，不添加额外的解释文本，让用户专注于对文件内容的审查。

保存完成后，Skillify 会向用户报告三项信息 [source: system-prompt-skillify-current-session.md]：
- 技能文件的保存路径
- 调用方式：`/{{skill-name}} [arguments]`
- 用户可以直接编辑 SKILL.md 来进一步调整

第三点尤其重要。SKILL.md 不是一次性的生成物，而是一个可持续演进的配置文件。用户可以在实际使用中发现问题，直接编辑文件来修正，无需再次运行 Skillify 流程。

## 7. SKILL.md 完整模板分析

将所有元素组合在一起，SKILL.md 的完整模板如下 [source: system-prompt-skillify-current-session.md]：

```markdown
---
name: {{skill-name}}
description: {{one-line description}}
allowed-tools:
  {{list of tool permission patterns observed during session}}
when_to_use: {{detailed description of when Claude should
auto-invoke this skill, including trigger phrases and
example user messages}}
argument-hint: "{{hint showing argument placeholders}}"
arguments:
  {{list of argument names}}
context: {{inline or fork -- omit for inline}}
---

# {{Skill Title}}
Description of skill

## Inputs
- `$arg_name`: Description of this input

## Goal
Clearly stated goal for this workflow. Best if you have
clearly defined artifacts or criteria for completion.

## Steps

### 1. Step Name
What to do in this step. Be specific and actionable.
Include commands when appropriate.

**Success criteria**: ALWAYS include this!

### 2. Step Name
...

### 3a. Parallel Step A
...

### 3b. Parallel Step B
...
```

模板的正文分三个主要部分：Inputs、Goal 和 Steps。

Inputs 部分列出技能接收的所有参数，每个参数用 `$arg_name` 的语法标记，并附上描述。这些参数在 frontmatter 的 arguments 字段中声明，在正文中通过 `$name` 语法引用。

Goal 部分是整个技能的目标声明。原文要求："Clearly stated goal for this workflow. Best if you have clearly defined artifacts or criteria for completion." [source: system-prompt-skillify-current-session.md] 目标最好有明确的完成标准，让执行技能的模型知道最终的交付物是什么。

Steps 部分是技能的核心执行流程。每个步骤是一个三级标题，包含具体的操作指令和必需的 Success criteria 标注。步骤的编号体现执行顺序，子编号（如 3a、3b）表示并行步骤。

模板的整体设计遵循"结构化知识"的原则：frontmatter 是机器可解析的元数据，正文是人类可读的流程描述。两者结合，使得 SKILL.md 既能被 Claude Code 的调度系统自动匹配和调用，又能被人类开发者阅读、理解和修改。

## 8. Inline vs Fork 的选择逻辑

Skillify 在第二轮访谈中提出了一个关键的架构决策：技能应该 inline 执行还是 forked 执行。原文对这个选择的描述是："Forked is better for self-contained tasks that don't need mid-process user input; inline is better when the user wants to steer mid-process." [source: system-prompt-skillify-current-session.md]

Inline 模式下，技能在当前对话中执行，共享当前对话的上下文窗口。这意味着技能可以访问之前的对话历史、文件状态和其他上下文信息。Inline 模式还允许技能在执行过程中与用户交互，比如在某一步骤暂停请求确认，或根据用户的实时反馈调整后续步骤。对于需要人工引导的流程，inline 模式是唯一合理的选择。

Fork 模式下，技能作为一个独立的 subagent 启动，拥有自己的上下文窗口。技能无法访问当前对话的历史，但也不受当前上下文窗口大小的限制。Fork 模式适合那些自包含的、不需要中途人工输入的任务。例如一个自动化的依赖升级流程，从头到尾可以自动完成，不需要用户在中途做决策。

在 SKILL.md 的 frontmatter 中，这个选择通过 `context` 字段体现。原文指出："Only set `context: fork` for self-contained skills that don't need mid-process user input." [source: system-prompt-skillify-current-session.md] Inline 模式是默认值，不需要显式设置。只有在确认使用 fork 模式时才设置 `context: fork`。

这个默认选择反映了一个假设：大多数用户工作流在中途都需要某种形式的人工参与或确认。只有少数高度自动化的流程才适合 fork 模式。将 inline 设为默认值，减少了大多数用户在配置时需要做的决策。

## 9. 总结：从"会话"到"技能"的元编程系统

Skillify 本质上是一个元编程系统。它的输入是一次具体的、成功的会话执行，输出是一个抽象的、可复用的技能规范。这个转化过程涉及几个层次的抽象：

第一层，从具体到抽象的提取。Step 1 的七维分析将一次具体的会话执行中的偶然细节剥离，提取出可重复的流程骨架。一个具体的"把 PR #42 cherry-pick 到 release-2.1 分支"的操作，被抽象为"将任意 PR cherry-pick 到任意目标分支"的通用流程。

第二层，从模糊到精确的确认。Step 2 的四轮访谈通过结构化的对话，将 Skillify 的自动分析跟用户的真实意图对齐。每一轮访谈都在缩小不确定性：第一轮确认整体方向，第二轮确认结构细节，第三轮确认步骤级别的执行参数，第四轮确认触发条件。这种渐进式的确认避免了在信息不充分时做出不可逆的设计决策。

第三层，从过程到规范的固化。Step 3 和 Step 4 将确认后的流程转化为一份标准化的 SKILL.md 文件。这份文件既是机器可解析的配置（frontmatter 部分），也是人类可读的文档（正文部分）。它将一次性的经验固化为可持续复用、可独立修改、可版本控制的知识资产。

Skillify 的设计还体现了一个工程原则：对简单流程的克制。"Keep simple skills simple" 和 "Don't over-ask for simple processes" 这两条指导原则反复出现在源文件中。一个两步的技能不需要每个步骤都加上五种标注，一个直观的流程不需要四轮完整的访谈。Skillify 提供了一套完整的工具集，但不要求每次都使用所有工具。这种按需使用的灵活性，使得 Skillify 在处理从简单到复杂的各类工作流时都能保持效率。

最终，Skillify 构建的是一个正向循环：用户在日常工作中积累的成功经验被转化为技能，这些技能在后续的使用中被不断调用和验证，发现的问题通过直接编辑 SKILL.md 来修复，修复后的技能在下次使用时更加可靠。每一次循环都让系统的知识库更丰富、更精确。"会话到技能"转化的真正价值就在这里：不是一次性的代码生成，而是持续的知识沉淀。
