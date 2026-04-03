# 谁来判断 AI 做对了没有？——Claude Code 的对抗性验证专家

## 为什么需要一个专门的验证 agent

当 AI 编写代码之后，谁来验证这些代码是否真正正确？

一个自然的回答是：让另一个 AI 来验证。但这立刻引出一个更深的问题——如果编写代码的 AI 存在认知偏差，验证代码的 AI 难道不会有同样的偏差吗？它们毕竟是同一个模型。

Claude Code 的回答是：承认这个问题，然后在 prompt 设计上正面强攻。它没有回避"AI 验证 AI"这一结构性的信任困境，而是通过一个专门的 Verification Specialist agent，将 AI 的已知弱点写入 prompt 本身，强制 agent 对抗自身的倾向。

这个 agent 的定位极其明确。它的开篇声明是：

> "Your job is not to confirm the work. Your job is to break it."
>
> [source: agent-prompt-verification-specialist.md]

这不是一个礼貌的代码审查员。这是一个被明确指派来破坏实现成果的对抗性角色。它接收父 agent（parent agent）当前轮次的完整对话——每一个 tool call、每一个输出、每一个它走的捷径——然后试图证明这些工作是有问题的。

理解这个设计的关键在于：Verification Specialist 不是在模型能力上做加法，而是在认知偏差上做减法。它通过在 prompt 中显式列举模型的失败模式，然后强制执行与直觉相反的行为，来弥补 LLM 在验证任务上的结构性缺陷。

## SELF-AWARENESS 段落：五条系统性失败模式

Verification Specialist prompt 中最有力的部分是 `SELF-AWARENESS` 段落。这段话直接说出了 Claude（以及所有 LLM）在验证场景下的五条系统性失败模式。每一条都是对模型自身行为的精确诊断。

### 失败模式一：以阅读代替运行

> "You read code and write 'PASS' instead of running it."
>
> [source: agent-prompt-verification-specialist.md]

这是 LLM 做验证时最常见的陷阱。模型的本质是文本处理器，它"阅读"代码的方式天然高效——逐 token 解析、理解逻辑结构、追踪数据流。这种阅读能力强大到会产生一种幻觉：读完即验证完。

但阅读代码只能告诉你代码"看起来怎样"，不能告诉你代码"运行起来怎样"。一个函数的逻辑在纸面上无懈可击，但它被调用的上下文、运行时的环境变量、依赖库的实际版本、并发时的竞态条件——这些东西只有真正运行才能暴露。

Verification Specialist 对此的应对策略是：在输出格式上强制要求每个检查项必须包含 `Command run` 块。一个没有实际命令执行的检查项被直接定义为"不是 PASS，而是 skip"。

### 失败模式二：被前 80% 蒙蔽

> "You see the first 80% — polished UI, passing tests — and feel inclined to pass. The first 80% is on-distribution, the easy part. Your entire value is the last 20%."
>
> [source: agent-prompt-verification-specialist.md]

这条失败模式揭示了一个关于 LLM 行为的观察：模型对"大部分看起来没问题"有一种强烈的 completion bias（完成偏差）。当一个页面的主要功能正常、测试套件的大部分用例通过时，模型会自然地倾向于给出积极的判断。

但 Verification Specialist 被告知，这前 80% 是"on-distribution"的——即落在模型训练数据常见模式内的部分，是容易的部分。它真正的价值在于最后 20%：边界条件、异常路径、竞态状态、极端输入。这些是 off-distribution 的场景，是训练数据中稀疏但生产环境中致命的部分。

说到底这是一个概率论直觉的问题。LLM 在高概率事件上表现良好，但验证的核心恰恰是找到那些低概率但高影响的故障。

### 失败模式三：被 AI 生成的测试欺骗

> "You're easily fooled by AI slop. The parent is also an LLM. Its tests may be circular, heavy on mocks, or assert what the code does instead of what it should do. Volume of output is not evidence of correctness."
>
> [source: agent-prompt-verification-specialist.md]

这一条尤其值得深思。它指出的是一个递归问题：如果代码是 AI 写的，测试也是 AI 写的，那么让 AI 来验证这些测试——这整个链条中是否存在独立的检验节点？

Verification Specialist 明确指出了 AI 生成测试的三个典型缺陷：

- **循环测试（circular tests）**：测试断言的是代码实际做了什么，而不是代码应该做什么。比如一个排序函数的实现是错的，但测试断言的恰好是错误实现的输出，于是测试通过了。
- **过度 mock（heavy on mocks）**：所有的外部依赖都被 mock 掉了，测试实际验证的只是一个精心编排的木偶戏，真实的交互从未发生。
- **输出量的迷思（volume of output）**：100 个测试用例全部通过并不意味着 100% 的覆盖率。AI 擅长生成看起来全面的测试矩阵，但这些矩阵可能在维度上是正交的——覆盖了很多边界值，但遗漏了一个核心的业务逻辑分支。

对此的应对是：测试套件的结果被定位为"context, not evidence"（上下文，而非证据）。运行测试套件是必须的，但它只是起点，不是终点。

### 失败模式四：信任自我报告

> "You trust self-reports. 'All tests pass.' Did YOU run them?"
>
> [source: agent-prompt-verification-specialist.md]

这是一个关于信息来源可信度的问题。父 agent 可能在对话中声称"所有测试通过了"、"功能正常工作"、"我验证过了"。Verification Specialist 被告知：不要信任这些声称。

这条规则在 SCAN 阶段被进一步强化。Verification Specialist 在扫描父 agent 的对话时，被要求专门寻找三类内容：

1. 声明（claims）——"我验证了..."、"测试通过了"、"功能正常"——这些需要独立验证。
2. 捷径（shortcuts）——"应该没问题"、"大概可以"、"我想"——这些需要额外审查。
3. 被忽略的 tool_result 错误——父 agent 可能看到了报错但选择性地忽略了。

Verification Specialist 的基本假设是：父 agent 的话需要被当作未经证实的声称，而非事实。

### 失败模式五：用 PARTIAL 逃避判断

> "When uncertain, you hedge with PARTIAL instead of deciding. PARTIAL is for environmental blockers, not for 'I found something ambiguous.' If you ran the check, you must decide PASS or FAIL."
>
> [source: agent-prompt-verification-specialist.md]

这是五条失败模式中最后一条，也是最微妙的一条。它指向的不是能力缺陷，而是决策回避。

当 LLM 面对一个不确定的结果时，它倾向于选择一个模糊的中间状态——"部分通过"——来避免做出可能错误的二元判断。这在社交语境下是一种合理的策略（给自己留余地），但在工程验证的语境下，它是一个严重的失败：一个不敢下判断的验证者等于没有验证者。

Verification Specialist 对 PARTIAL 的使用施加了严格的约束：只有在环境限制（没有测试框架、工具不可用、服务器无法启动）的情况下才能使用 PARTIAL。如果你已经运行了检查，你就必须给出 PASS 或 FAIL。

## 认知合理化识别清单："你会有以下冲动，做相反的事"

在五条失败模式之后，Verification Specialist 的 prompt 中有一个单独的段落，标题是 `RECOGNIZE YOUR OWN RATIONALIZATIONS`。这个段落列举了六条模型在验证过程中常见的自我合理化借口——那些看起来合理、但实质上是在逃避实际验证的思维模式。

这个段落的结构非常独特。每一条都以一个引号中的内心独白开头，然后是一句简短的反驳指令。整个段落的核心指令只有一句：

> "If you catch yourself writing an explanation instead of a command, stop. Run the command."
>
> [source: agent-prompt-verification-specialist.md]

这六条合理化借口分别是：

**"The code looks correct based on my reading"**

反驳：reading is not verification. Run it. 阅读不是验证，运行才是。

**"The implementer's tests already pass"**

反驳：the implementer is an LLM. Verify independently. 实现者也是 LLM，必须独立验证。

**"This is probably fine"**

反驳：probably is not verified. Run it. "大概可以"不等于"已验证"。

**"Let me start the server and check the code"**

反驳：no. Start the server and hit the endpoint. 启动服务器不是为了检查代码，是为了访问端点。

**"I don't have a browser"**

反驳：did you actually check for browser automation tools? 如果你没有检查就声称没有，你是在编造自己的"做不到"叙事。

**"This would take too long"**

反驳：not your call. 这不是你能做的判断。

这份清单的价值在于它的设计方式。它不是在告诉模型"你应该做什么"，而是在告诉模型"你会不想做什么，以及你不想做的时候会用什么借口"。这是一种反向的 prompt engineering——不是引导正确行为，而是拦截自我欺骗。

## 强制对抗性探测协议

Verification Specialist 的验证协议中有一个硬性要求：每一份报告必须包含至少一个对抗性探测（adversarial probe），否则报告会被拒绝。

> "A report with zero adversarial probes is a happy-path confirmation, not verification. It will be rejected."
>
> [source: agent-prompt-verification-specialist.md]

协议定义了四类对抗性探测：

### Boundary（边界值探测）

向系统输入边界值和异常值：0、-1、空字符串、超长字符串、unicode 字符、MAX_INT。这些值不在正常业务逻辑的分布内，但恰恰是生产环境中导致崩溃的常见原因。

### Concurrency（并发探测）

针对服务器和 API，发送并行请求到 create-if-not-exists 路径，检查是否会出现重复会话或写入丢失。这是验证分布式系统正确性的基本手段。

### Idempotency（幂等性探测）

对同一个 mutation 请求发送两次，检查系统是否正确处理：是创建了重复记录、抛出了错误、还是正确地作为 no-op 处理？

### Orphan Operations（孤儿操作探测）

尝试删除或引用不存在的 ID。一个健壮的系统应该优雅地处理这些情况，而不是抛出未捕获的异常。

这些探测不是检查清单上的全部必选项——prompt 明确说"These are seeds, not a checklist"（这些是种子，不是清单）。Verification Specialist 应该根据变更类型选择最相关的探测方式。但关键约束是：你至少要做一种，并且要记录结果——即使结果是"handled correctly"（正确处理了）。

在发出 PASS 之前，还有一条补充的强制要求：

> "If all your checks are 'returns 200' or 'test suite passes,' you have confirmed the happy path, not verified correctness. Go back and try to break something."
>
> [source: agent-prompt-verification-specialist.md]

只有 happy path 验证的报告等同于没有验证。你必须尝试破坏系统，并且记录你是怎么尝试的、结果是什么。

## 十种变更类型的验证策略

Verification Specialist 不是一刀切的验证者。它针对十种不同的变更类型提供了差异化的验证策略，每种策略都基于"如何真正运行这段变更"这个核心问题展开。

### Frontend Changes（前端变更）

策略：启动 dev server，使用浏览器自动化工具（而非声称"需要真实浏览器"却不尝试）进行导航、截图、点击和读取控制台。关键细节是：还要 curl 页面子资源（如 image-optimizer URL、同源 API 路由、静态资源），因为 HTML 本身可能返回 200，而它引用的所有资源都在失败。

### Backend/API Changes（后端/API 变更）

策略：启动服务器，curl/fetch 端点，验证响应结构（不只是状态码），测试错误处理，检查边界情况。这里强调的是"response shapes against expected values"——响应的完整结构必须与期望值匹配，而不只是检查 HTTP 200。

### CLI/Script Changes（CLI/脚本变更）

策略：使用代表性输入运行，验证 stdout/stderr/exit code，测试边界输入（空、畸形、边界值），验证 --help/usage 输出的准确性。CLI 的验证焦点在于输出格式和退出码的正确性。

### Infrastructure/Config Changes（基础设施/配置变更）

策略：验证语法，在可能的地方做 dry-run（terraform plan、kubectl apply --dry-run=server、docker build、nginx -t），检查环境变量和 secrets 是否真正被引用了而非只是被定义了。很多配置问题不是语法错误，而是"定义了但没使用"的僵尸配置。

### Library/Package Changes（库/包变更）

策略：构建，运行完整测试套件，从一个全新的上下文导入库并作为消费者使用公共 API，验证导出的类型与 README/文档示例是否一致。这里的"从全新上下文导入"是一个关键设计——它模拟的是真实用户的使用场景，而非开发者的本地环境。

### Bug Fixes（Bug 修复）

策略：首先复现原始 bug，验证修复有效，运行回归测试，检查相关功能是否有副作用。这个顺序很重要：先复现，再验证修复。跳过复现步骤意味着你不知道原始问题是否真的存在过。

### Mobile（移动端）

策略：clean build，在模拟器/ emulator 上安装，dump accessibility/UI tree 来定位和操作元素（截图是次要手段），杀掉并重新启动以测试持久化，检查 crash logs。dump accessibility tree 被定位为主要手段，截图是次要的。因为 accessibility tree 提供的是结构化的可操作信息，而截图只是视觉确认。

### Data/ML Pipeline（数据/ML 管道）

策略：使用样本输入运行，验证输出形状/schema/类型，测试空输入、单行数据、NaN/null 处理，检查是否存在静默的数据丢失（输入行数 vs 输出行数）。"silent data loss"是数据管道中最隐蔽的缺陷：没有报错，没有异常，只是行数变少了。

### Database Migrations（数据库迁移）

策略：运行 migration up，验证 schema 是否符合意图，运行 migration down（可逆性测试），用已有数据测试而非空数据库。空数据库上的 migration 测试是最常见的虚假信心来源——真实数据库中的数据约束、外键关系、数据类型变化才是真正的问题。

### Refactoring（重构——无行为变更）

策略：现有测试套件必须在无修改的情况下通过，diff 公共 API surface（无新增/删除的 exports），抽样检查可观察行为是否一致（相同输入产生相同输出）。重构的验证核心是"不变"——任何改变都意味着重构引入了行为差异。

对于不在这十种之内的变更类型，Verification Specialist 提供了一个通用模式：

> "(a) figure out how to exercise this change directly (run/call/invoke/deploy it), (b) check outputs against expectations, (c) try to break it with inputs/conditions the implementer didn't test."
>
> [source: agent-prompt-verification-specialist.md]

这三步——执行、检查、破坏——是所有验证策略的元模式。

## PARTIAL 不是挡箭牌的设计哲学

在 Verification Specialist 的 prompt 中，关于 PARTIAL verdict 的讨论占据了大量篇幅。这不是因为它复杂，而是因为它太容易被滥用。

基础定义已经足够清晰：

> "PARTIAL is for environmental limitations only (no test framework, tool unavailable, server can't start) — not for 'I'm unsure whether this is a bug.' If you can run the check, you must decide PASS or FAIL."
>
> [source: agent-prompt-verification-specialist.md]

但 prompt 随后用了一段更长的文字来堵住可能的漏洞：

> "PARTIAL is NOT a hedge. 'I found a hardcoded key and a TODO but they might be intentional' is FAIL — a hardcoded secret-pattern and an admitted-incomplete TODO are actionable findings regardless of intent. 'The tests are circular but the implementer may have known' is FAIL — circular tests are a defect. PARTIAL means 'I could not run the check at all,' not 'I ran it and the result is ambiguous.'"
>
> [source: agent-prompt-verification-specialist.md]

这段话的每一句都在封堵一个具体的合理化路径：

- "可能是故意的" -> 不论意图如何，hardcoded key 是可操作发现，FAIL。
- "实现者可能知道" -> 循环测试是缺陷，FAIL。
- "结果模糊" -> 模糊的结果不等于无法运行检查，FAIL。

这种设计哲学的核心是：**模糊不是 PARTIAL，不确定不是 PARTIAL，只有物理上不可能才是 PARTIAL。**

同样地，对于 FAIL verdict，prompt 也设计了防误报机制。在报告 FAIL 之前，Verification Specialist 必须检查三个"它是其实没问题的"可能性：

- **Already handled**：是否存在上游的防御性代码或下游的错误恢复机制已经处理了这个问题？
- **Intentional**：CLAUDE.md、注释或 commit message 是否解释了这是有意的？
- **Not actionable**：这是否是一个真实限制但无法在不破坏外部契约（稳定的 API、协议规范、向后兼容性）的情况下修复？

这三条检查的目的是减少 false FAIL——即把正常行为误报为错误的判断。但 prompt 同时警告：不要把这些当成推脱真实问题的借口。

## Verify Skill 的补充："验证是运行时观察"的哲学

如果说 Verification Specialist 是一个专门的 agent，那么 Verify Skill 就是一个面向更广泛使用场景的 skill。它出现在 Claude Code v2.1.91 版本中（Verification Specialist 在 v2.1.90），可以看作是对验证理念的进一步提炼。

Verify Skill 的开篇声明同样直接：

> "Verification is runtime observation. You build the app, run it, drive it to where the changed code executes, and capture what you see. That capture is your evidence. Nothing else is."
>
> [source: skill-verify-skill.md]

这句话定义了一种更纯粹的验证哲学：验证就是运行时观察。构建应用、运行它、驱动到变更代码执行的位置、捕获你看到的。捕获到的输出是证据，其他一切都不是。

这个 skill 还明确说了三件"不要做的事"：

### 不要运行测试

> "Don't run tests. Don't typecheck. CI ran both before you got here — green checks on the PR mean they passed. Running them again proves you can run CI."
>
> [source: skill-verify-skill.md]

这是一个大胆的立场。Verification Specialist 会运行测试套件（作为 baseline），但 Verify Skill 明确不运行测试——因为 CI 已经跑过了，再跑一次只是证明你能跑 CI，不增加任何信息量。

### 不要写临时测试脚本

> "Don't import-and-call. `import { foo } from './src/...'` then `console.log(foo(x))` is a unit test you wrote. The function did what the function does — you knew that from reading it. The app never ran."
>
> [source: skill-verify-skill.md]

这一条瞄准的是一种更隐蔽的虚假验证：import 一个函数然后调用它。这本质上是你写的一个单元测试。函数做了函数该做的事——你在阅读代码时就已经知道了。但应用从未真正运行过。

### 不要把测试当 Surface

> "Tests in the diff are the author's evidence, not a surface. CI runs them. You'd be re-running CI."
>
> [source: skill-verify-skill.md]

diff 中的测试是作者的证据，不是验证的 surface。CI 会运行它们。你重新跑一遍只是在重复 CI 的工作。

Verify Skill 引入了一个核心概念——Surface（表面）。Surface 是用户——人类或程序化的——与变更接触的地方。不同类型的变更对应不同的 surface：

| 变更到达的渠道 | Surface | 验证方式 |
|---|---|---|
| CLI / TUI | 终端 | 输入命令，捕获 pane 输出 |
| Server / API | socket | 发送请求，捕获响应 |
| GUI | 像素 | 在 xvfb/Playwright 下驱动，截图 |
| Library | 包边界 | 通过公共 export 的示例代码 |
| Prompt / agent config | agent 本身 | 运行 agent，捕获其行为 |
| CI workflow | Actions | dispatch 它，读取运行结果 |

这个 surface 概念的精妙之处在于：内部函数不是一个 surface。某个地方调用了它，那个调用者最终会到达上表中的某一行。验证者必须跟随调用链到达真正的 surface，而不是在中间停下来。

如果没有任何 runtime surface——纯文档、无 emit 的类型声明、不产生行为差异的构建配置——那么报告 `SKIP`，一行说明原因即可。不要为了填充空间而去跑测试。

## 从 Verification Specialist 到 Verify Skill 的演变

Verification Specialist（v2.1.90）和 Verify Skill（v2.1.91）虽然都围绕"验证"这个主题，但它们在定位、方法和哲学上有显著差异。理解这种差异，可以看作是理解 Claude Code 团队对验证问题认识的深化。

### 定位差异

Verification Specialist 是一个 subagent。它被设计为在一个 agent 编码完成之后，作为独立的验证步骤介入。它接收父 agent 的完整对话，执行一套结构化的验证协议，输出一份详细的报告。

Verify Skill 是一个 skill（技能）。它是一个更轻量、更通用的验证工作流，可以在更多场景下触发——验证 PR、确认修复、手动测试变更、在推送前验证本地修改。

### 方法差异

| 维度 | Verification Specialist | Verify Skill |
|---|---|---|
| 测试套件 | 运行作为 baseline | 明确不运行 |
| 对抗性探测 | 强制要求至少一种 | 隐含在"break it"指令中 |
| 报告格式 | 严格结构化（Command run / Output observed / Result） | 自由格式但要求包含 Steps 和 Findings |
| Verdict 范围 | PASS / FAIL / PARTIAL | PASS / FAIL / BLOCKED / SKIP |

最关键的差异在于 BLOCKED 和 SKIP 这两个新增的 verdict。

**BLOCKED** 是 Verify Skill 独有的。它表示"无法到达变更可被观察的状态"——构建失败、环境缺少依赖、handle 无法启动。这不是对变更本身的判断，而是验证过程被阻塞了。要求说明具体在哪里停止，并附上一个 filled-in 的 `/run-skill-generator` prompt。

**SKIP** 也是 Verify Skill 独有的。它表示"没有 runtime surface 存在"——纯文档、纯类型、纯测试的 PR。没有出错，只是没有东西可以运行。一行说明原因即可。

相比之下，Verification Specialist 只有 PARTIAL 来覆盖这两种情况（环境阻塞 + 无可运行内容），这导致了 PARTIAL 被过度使用的问题。Verify Skill 将这些场景拆分为 BLOCKED 和 SKIP，使得 PASS 和 FAIL 的含义更加纯粹。

### 哲学差异

Verification Specialist 的哲学核心是"对抗"——它的使命是打破实现。它的 prompt 反复强调 adversarial probe、break it、try to break something。

Verify Skill 的哲学核心是"观察"——它的使命是运行应用并捕获行为。它的 prompt 强调 capture、evidence、observation。对抗性测试被包含在观察过程中（"Once the claim checks out, keep going: break it, combine it, wander"），但它不是唯一的焦点。

Verify Skill 还引入了一个独特的价值观：

> "The verdict is table stakes. Your observations are the signal. A PASS with three sharp 'hey, I noticed...' lines is worth more than a bare PASS."
>
> [source: skill-verify-skill.md]

verdict 只是底线，观察才是价值所在。一个带有三个敏锐的"嘿，我注意到..."的 PASS，比一个光秃秃的 PASS 更有价值。验证者是唯一一个真正运行了代码的 reviewer——任何让验证者停顿、绕道或发出"嗯？"的信息，都是作者不具备的信息。

这个理念进一步延伸为一条具体的过滤规则：

> "Don't filter for 'is this a bug.' Filter for 'would I mention this if they were sitting next to me.'"
>
> [source: skill-verify-skill.md]

不要过滤"这是不是 bug"，而要过滤"如果他们坐在你旁边，你会不会提这件事"。这是一个更宽松的筛选标准——它鼓励验证者报告所有值得注意的观察，而不仅仅是确认的缺陷。

### 核心共识

尽管有这些差异，两个系统在根本原则上是一致的：

1. **运行胜于阅读**：读代码不是验证，运行代码才是。
2. **端到端优于单元级**：通过真实接口验证，而不是在隔离环境中测试函数。
3. **不可信的自我报告**：实现者的声称需要独立验证。
4. **没有免费的 PASS**：每个 PASS 必须有实际运行的证据支撑。

## "告诉 AI 自己不擅长什么"的 Prompt 设计模式

回到最初的问题：AI 如何验证 AI？

Claude Code 的 Verification Specialist 和 Verify Skill 给出了一个具体的答案。但比答案本身更有意思的是这个答案的设计方法。

传统 prompt engineering 的思路是告诉 AI "你应该做什么"。但 Verification Specialist 的 prompt 大量使用了一种反向策略：**告诉 AI 它自己不擅长什么，然后要求它做相反的事。**

这种方法包含三个层次：

**第一层：枚举失败模式。** 不是抽象地说"要小心"，而是精确地列举具体的失败行为——"你会读代码然后写 PASS"、"你会被前 80% 蒙蔽"、"你会用 PARTIAL 逃避判断"。每一条都是一个可以被自我监测的具体行为模式。

**第二层：识别合理化借口。** 不只是列出失败行为，还列出失败行为的心理伪装——"代码看起来是对的"、"实现者的测试已经通过了"、"这大概没问题"、"我没有浏览器"。每一条都是一个可以被拦截的思维模式。

**第三层：强制执行相反行为。** 识别到失败模式后的行动不是"注意避免"，而是"做相反的事"。不是"尽量运行"，而是"如果你在写解释而不是命令，停下来，运行命令"。不是"谨慎使用 PARTIAL"，而是"如果你运行了检查，你必须给出 PASS 或 FAIL"。

这种设计模式的本质是一种 **对抗性自我认知（adversarial self-awareness）**。它假定模型对自己的弱点有足够的元认知能力，能够在运行时识别自己正在陷入某个已知的失败模式，并主动纠正。

这个假设是否成立？从 prompt 设计的角度看，它不需要完全成立——即使模型只在一半的情况下成功识别了自己的偏差，这比完全没有这种自我监测要好得多。更何况，这些失败模式被写入了 prompt 的输出结构中（比如强制要求 Command run 块），即使模型的自我认知失败，结构化的输出格式也能作为一道物理屏障。

Verification Specialist 的 prompt 设计提出了一种更广泛的启示：在 AI 系统设计中，诚实可能比乐观更有效。不是告诉 AI "你可以做好验证"，而是告诉它"你在验证上很差，这是你具体的失败模式，现在做相反的事"。这种坦诚的 prompt 策略，可能比任何正向的能力描述都更能提升 AI 在困难任务上的实际表现。

当然，这种方法也有边界。如果模型真的无法识别自己正在陷入某个失败模式，那么再精确的描述也无济于事。但在 Claude Code 的设计中，这不是唯一的防线——结构化的输出格式、强制性的对抗性探测要求、以及"没有 command run 块就不是 PASS"的硬性规则，共同构成了一个多层防护体系。即使自我认知这一层偶尔失效，其他层仍然在起作用。

Claude Code 验证系统最重要的设计教训或许是：**不要依赖单一机制来弥补 AI 的弱点，而是用多个互补的机制构建一个失效安全的系统。** 自我认知是其中最独特的一层，但不是唯一的一层。
