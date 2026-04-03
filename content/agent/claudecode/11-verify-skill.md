# 别跑测试，去跑应用：Verify Skill 的"运行时观察"哲学

## 1. Verify Skill 与 Verification Specialist 的关系

在 Claude Code 的 skill 体系中，Verification Specialist 是一个角色的定位，而 Verify Skill 是这个角色的执行手册。Verification Specialist 定义了"谁来验证"，它不是代码审查者，不是测试运行者，而是一个亲手把应用跑起来、在实际运行环境中观察行为的人。Verify Skill 则定义了"怎么验证"，一套精确的操作流程、判断标准和报告格式。

这种角色与流程的分离并非偶然。在 `skill-verify-skill.md` 中，Verify Skill 的定位非常明确：

> "Verify that a code change actually does what it's supposed to by running the app and observing behavior. Use when asked to verify a PR, confirm a fix works, test a change manually, check that a feature works, or validate local changes before pushing." [source: skill-verify-skill.md]

注意这段描述中的关键词："running the app"、"observing behavior"。这不是静态分析，不是代码审查，不是测试执行。这是一套以"运行"为前提、以"观察"为方法的验证哲学。本文从这条核心主张出发，逐层拆解 Verify Skill 的设计逻辑。

## 2. "Verification is runtime observation" 核心主张

Verify Skill 的开篇第一句话就确立了整个体系的基石：

> "Verification is runtime observation. You build the app, run it, drive it to where the changed code executes, and capture what you see. That capture is your evidence. Nothing else is." [source: skill-verify-skill.md]

四个动作：build（构建）、run（运行）、drive（驱动到变更代码的执行路径）、capture（捕获观察结果）。每一个动作都在强化"运行时"这个限定词。验证不是阅读代码后的推理，不是对测试用例的信任，而是在真实运行环境中看到变更代码实际执行后的输出。

"That capture is your evidence. Nothing else is." 这一句尤其值得注意。"Nothing else is"排除了所有非运行时的证据来源：代码审查的直觉、测试框架的绿勾、类型检查的通过。在 Verify Skill 的世界观里，只有应用实际运行时产生的输出才是证据。

这种立场是极端的，但也是精确的。它回答了一个根本问题：当代码变更声称修复了某个 bug 或添加了某个功能时，谁能说它真的做到了？Verify Skill 的答案是：只有当变更代码在运行中被执行，且执行结果与预期一致时，验证才算完成。

## 3. Surface 概念：六种用户与变更的交汇点

要实现运行时观察，首先需要确定观察的位置。Verify Skill 引入了 "Surface" 这一核心概念：

> "The surface is where a user — human or programmatic — meets the change. That's where you observe." [source: skill-verify-skill.md]

Surface 是用户（人或者程序）与代码变更发生交互的界面。Verify Skill 定义了六种 Surface 类型：

| 变更到达的界面 | Surface | 验证方式 |
|---|---|---|
| CLI / TUI | terminal | 输入命令，捕获终端输出 |
| Server / API | socket | 发送请求，捕获响应 |
| GUI | pixels | 通过 xvfb/Playwright 驱动，截图 |
| Library | package boundary | 通过公共导出调用，而非内部路径 |
| Prompt / agent config | the agent | 运行 agent，捕获其行为 |
| CI workflow | Actions | 触发运行，读取结果 |

[source: skill-verify-skill.md]

这六种 Surface 覆盖了绝大多数代码变更的可观测面。关键在于，Surface 始终是"外部"的，是用户能触碰到的界面，不是内部的函数实现。

对于内部函数，Verify Skill 的态度很明确：

> "Internal function? Not a surface. Something in the repo calls it and that caller ends at one of the rows above. Follow it there." [source: skill-verify-skill.md]

即使变更发生在深层内部函数中，验证者也不应该直接测试该函数，而是沿着调用链向上追踪，直到找到一个用户可达的 Surface，然后通过那个 Surface 触发变更代码的执行。原文给出了一个具体的例子："A bash security gate's surface isn't the function's return value — it's the CLI prompting or auto-allowing when you type the command."

当变更完全没有运行时 Surface 时，比如纯文档变更、仅包含类型声明但没有 emit 的变更、构建配置变更但没有产生行为差异，Verify Skill 要求报告 SKIP，而不是用运行测试来填补空白：

> "No runtime surface at all — docs-only, type declarations with no emit, build config that produces no behavioral diff — report SKIP — no runtime surface: (reason). Don't run tests to fill the space." [source: skill-verify-skill.md]

对于 PR 中包含的测试文件，Verify Skill 也做了明确区分：

> "Tests in the diff are the author's evidence, not a surface. CI runs them. You'd be re-running CI. Tests-only PR → SKIP, one line. Mixed src+tests → verify the src, ignore the test files." [source: skill-verify-skill.md]

测试可以作为学习工具，"Reading a test to learn what to check is fine — it's a spec"，但验证的动作必须发生在运行的应用上，而不是测试框架中。

## 4. 三个"DON'T"：Don't run tests / Don't import-and-call / Don't typecheck

Verify Skill 用三个明确的禁令划定了验证的边界。

**Don't run tests：**

> "Don't run tests. Don't typecheck. CI ran both before you got here — green checks on the PR mean they passed. Running them again proves you can run CI. Not as a warm-up, not 'just to be sure,' not as a regression sweep after. The time goes to running the app instead." [source: skill-verify-skill.md]

逻辑很简单：CI 已经运行过测试和类型检查了。如果 PR 上有绿色的勾，说明它们通过了。重新运行测试只能证明你能运行 CI，不能证明变更在实际应用中的行为。验证者的时间应该花在运行应用上。

**Don't import-and-call：**

> "Don't import-and-call. `import { foo } from './src/...'` then `console.log(foo(x))` is a unit test you wrote. The function did what the function does — you knew that from reading it. The app never ran. Whatever calls `foo` in the real codebase ends at a CLI, a socket, or a window. Go there." [source: skill-verify-skill.md]

这条禁令针对的是一种常见的偷懒行为：直接 import 变更的函数然后调用它。Verify Skill 认为这本质上是你自己写的一个 unit test。函数做了函数该做的事，你读代码就知道了。但应用从未运行过。真正的调用链会经过 CLI、socket 或窗口，验证者应该去那个终点，而不是在中间插入一个测试点。

**Don't typecheck：** 这条已经包含在 "Don't run tests" 的禁令中。类型检查是 CI 的职责，不是运行时验证的职责。

三个禁令共同构建了一个清晰的原则：验证者不是 CI 的复制品，不是测试的替代者，不是类型检查的执行者。验证者的独特价值在于亲手运行应用，在真实的执行环境中观察变更的行为。

## 5. Find the change：diff 是 ground truth，PR description 是 claim

在开始验证之前，必须先确定验证什么。Verify Skill 要求首先建立变更的完整范围：

> "Establish the full range first — a branch may be many commits" [source: skill-verify-skill.md]

具体操作包括：

```bash
git log --oneline @{u}..              # count commits
git diff @{u}.. --stat                # full range, not HEAD~1
gh pr diff                            # if in a PR context
```

[source: skill-verify-skill.md]

注意使用 `@{u}..` 而不是 `HEAD~1`，因为一个分支可能包含多个 commit。报告中也必须说明 commit 数量。如果 diff 太大被截断，应该重定向到临时文件然后用 Read 读取。如果没有 diff，则直接报告并停止。

最关键的判断原则在这一句：

> "The diff is ground truth. The PR description is a claim about it. Read both. If they disagree, that's a finding." [source: skill-verify-skill.md]

diff 是 ground truth，它客观记录了代码发生了什么变化。PR description 是 claim，它是作者对变更意图的声明。两者都读，如果出现不一致，这就是一个 finding。这种对 diff 和 PR description 不对称关系的定义体现了 Verify Skill 的严谨性：代码本身比人的描述更可靠，但人的描述提供了验证的预期目标。

## 6. Get a handle：verifier skill 到 run skill 到 cold start

确定了变更范围之后，下一步是找到运行应用的方法。Verify Skill 定义了一个三级查找策略：

第一级，查找已有的 verifier skill：

> ".claude/skills/*verifier*/ — if one matches your surface (CLI verifier for a CLI change, etc.), route to it. It knows readiness signals and env gotchas you don't. Mismatched surface → skip that one, try the next. Stale verifier (fails on mechanics unrelated to the change) → ask the user whether to patch it; don't FAIL the change for verifier rot." [source: skill-verify-skill.md]

如果项目已经有针对特定 Surface 的 verifier skill（比如 CLI 变更有 CLI verifier），直接使用它。它知道环境中的就绪信号和常见陷阱。但如果 Surface 不匹配则跳过；如果 verifier 过时了（在无关的环节上失败），不应该把变更判定为 FAIL，而应该询问用户是否需要修复 verifier。

第二级，查找 run skill：

> ".claude/skills/run-*/ — knows how to build and launch. Use its primitives as your handle." [source: skill-verify-skill.md]

run skill 知道如何构建和启动应用，可以借用它的基础设施。

第三级，cold start：

> "Neither — cold start from README/package.json/Makefile. Timebox ~15min. Stuck → BLOCKED with exactly where, plus a filled-in /run-skill-generator prompt. Got through → mention /init-verifiers in your report so next time is faster." [source: skill-verify-skill.md]

如果两者都没有，就从 README、package.json、Makefile 冷启动。时间限制约 15 分钟。如果卡住了，报告 BLOCKED 并附上精确的卡住位置和填好的 `/run-skill-generator` 提示。如果成功通过了，在报告中建议运行 `/init-verifiers`，这样下次验证就不需要再次冷启动。

这个三级策略体现了 Verify Skill 的实用主义：优先利用已有知识，但在没有现成工具时也能从零开始，同时确保每次冷启动的痛苦都会转化为未来的便利。

## 7. Drive it：最小路径让变更代码执行

有了 handle 之后，验证的核心动作是"drive"，用最小路径让变更代码在运行中执行：

> "Smallest path that makes the changed code execute" [source: skill-verify-skill.md]

具体的策略随变更类型而定：

- 改了一个 flag？带着它运行。
- 改了一个 handler？访问那个路由。
- 改了错误处理？触发那个错误。
- 改了一个内部函数？找到能触达它的 CLI 命令/请求/渲染路径，运行那个。

Verify Skill 还要求在执行前进行一项关键的自我检查：

> "Read your plan back before running. If every step is build / typecheck / run test file — you've planned a CI rerun, not a verification. Find a step that reaches the surface or report BLOCKED." [source: skill-verify-skill.md]

如果计划的每一步都是构建、类型检查、运行测试文件，那么这不是一个验证计划，而是 CI 的重复运行。必须找到一个能触达 Surface 的步骤，否则报告 BLOCKED。

当基本的 claim 验证通过后，Verify Skill 鼓励继续探索：

> "Once the claim checks out, keep going: break it (empty input, huge input, interrupt mid-op), combine it (new thing + old thing), wander (what's adjacent? what looked off?). The PR description is what the author intended. Your job includes what they didn't." [source: skill-verify-skill.md]

尝试破坏（空输入、巨大输入、中断操作）、组合（新旧功能交叉使用）、漫游（相邻的功能有什么异常？）。PR description 是作者的意图，但验证者的工作包括作者没有想到的部分。

Verify Skill 对端到端验证的态度很坚决：

> "End-to-end, through the real interface. Pieces passing in isolation doesn't mean the flow works — seams are where bugs hide. If users click buttons, test by clicking buttons, not by curling the API underneath." [source: skill-verify-skill.md]

孤立的部分通过不代表整个流程能工作，接缝处是 bug 藏身的地方。如果用户通过点击按钮来操作，就通过点击按钮来测试，而不是绕过 UI 直接 curl 底层 API。

## 8. Capture：证据是 stdout/response/screenshots，不是记忆

验证过程中的捕获环节同样被严格定义：

> "Stdout, response bodies, screenshots, pane dumps. Captured output is evidence; your memory isn't. Something unexpected? Don't route around it — capture, note, decide if it's the change or the environment. Unrelated breakage is a finding, not noise." [source: skill-verify-skill.md]

证据的载体是 stdout、response body、截图、终端转储。被捕获的输出才是证据，验证者的记忆不是。遇到意外情况时不要绕过它，捕获它，记录它，然后判断这是变更引起的还是环境问题。与变更无关的故障也是一个 finding，不是噪音。

Verify Skill 还提醒注意进程状态的隔离：

> "Shared process state (tmux, ports, lockfiles) — isolate. `tmux -L name`, bind `:0`, `mktemp -d`. You share a namespace with your host." [source: skill-verify-skill.md]

验证者与宿主环境共享命名空间，所以需要通过 tmux 的 `-L` 参数、绑定会话编号、创建临时目录等方式来隔离状态，避免与宿主环境的进程或端口产生冲突。

## 9. Report 格式详解：Steps/Findings/Screenshot

验证完成后，Verify Skill 要求输出一份结构化的报告。报告模板如下：

```
## Verification: <one-line what changed>

**Verdict:** PASS | FAIL | BLOCKED | SKIP

**Claim:** <what it's supposed to do>

**Method:** <how you got a handle>

### Steps
1. status_icon <what you did to the running app> -> <what you observed>
   <evidence>

**Screenshot / sample:** <the one frame a reviewer looks at>

### Findings
<Things you noticed>
```

[source: skill-verify-skill.md]

其中几个关键约束：

Steps 部分只记录"对运行中的应用做了什么"以及"观察到了什么"。构建、安装、checkout 是准备工作，不算 Steps。测试运行和类型检查不属于这里，它们是 CI 的输出。

Findings 部分的门槛很低：

> "Not just bugs — friction, surprises, anything a first-time user would trip on." [source: skill-verify-skill.md]

不限于 bug，还包括摩擦、意外情况、任何初次使用者会踩到的坑。原文给出了具体例子："Took three tries to find the right flag.""Error message on typo was unhelpful.""Default seems odd for the common case.""Works, but slower than I expected." 判断标准是："if it made you pause, it goes here."

值得注意的条目用警告标记开头，它们会被提升到 PR 评论的可见区域。Findings 为空是允许的，但"nothing sticking out is itself rare"。

Verify Skill 还提出了一个重要的价值主张：

> "The verdict is table stakes. Your observations are the signal. A PASS with three sharp 'hey, I noticed...' lines is worth more than a bare PASS. You're the only reviewer who actually ran the thing — anything that made you pause, work around, or go 'huh' is information the author doesn't have." [source: skill-verify-skill.md]

verdict 只是基本要求，真正有价值的是观察。一个附带三条敏锐观察的 PASS 比一个光秃秃的 PASS 更有价值。因为你是唯一一个实际运行了应用的人，任何让你停顿、绕路或困惑的东西，都是作者没有的信息。

## 10. 四种 Verdicts 的严格定义

Verify Skill 定义了四种 verdict，每一种都有严格的判定条件：

**PASS：**

> "you ran the app, the change did what it should at its surface. Not: tests pass, builds clean, code looks right." [source: skill-verify-skill.md]

PASS 的条件是：你运行了应用，变更在其 Surface 上做到了它应该做的事。"不是：测试通过了、构建干净了、代码看起来正确。"

**FAIL：**

> "you ran it and it doesn't. Or it breaks something else. Or claim and diff disagree materially." [source: skill-verify-skill.md]

FAIL 的条件是：运行了但行为不符合预期，或者破坏了其他东西，或者 claim 和 diff 存在实质性不一致。

**BLOCKED：**

> "couldn't reach a state where the change is observable. Build broke, env missing a dep, handle wouldn't come up. Not a verdict on the change. Say exactly where it stopped + /run-skill-generator prompt." [source: skill-verify-skill.md]

BLOCKED 的条件是：无法到达变更可被观察的状态。构建失败、环境缺少依赖、应用无法启动。这不是对变更本身的评判，只是验证者无法到达验证的状态。必须说明精确的停止位置，并附上填好的 `/run-skill-generator` 提示。

**SKIP：**

> "no runtime surface exists. Docs-only, types-only, tests-only. Nothing went wrong; there's just nothing here to run. One line why." [source: skill-verify-skill.md]

SKIP 的条件是：不存在运行时 Surface。纯文档、纯类型声明、纯测试。没有出错，只是没有可以运行的东西。用一句话说明原因。

还有一个绝对规则：

> "No partial pass. '3 of 4 passed' is FAIL until 4 passes or is explained away." [source: skill-verify-skill.md]

不存在"部分通过"。"4 项中 3 项通过"就是 FAIL，直到 4 项全部通过或者第 4 项的不通过被合理排除。

## 11. "When in doubt, FAIL" 的设计哲学

在四种 verdict 之间犹豫时，Verify Skill 给出了明确的倾向：

> "When in doubt, FAIL. False PASS ships broken code; false FAIL costs one more human look. Ambiguous output is FAIL with the raw capture attached — don't interpret." [source: skill-verify-skill.md]

背后的数学很清晰：错误的 PASS 会导致破损代码被发布到生产环境；错误的 FAIL 最多让另一个人再看一眼。两种错误的代价严重不对称。因此在模糊地带，选择 FAIL。

对于模棱两可的输出，Verify Skill 要求附上原始捕获数据，而不是进行解读："don't interpret." 验证者不需要做出最终判断，把原始证据和 FAIL verdict 放在一起，让人类审查者来做解读。验证者的职责是诚实地呈现运行时观察的结果，而不是充当仲裁者。

这种哲学也体现在 "drive it" 阶段的探索性测试中。Verify Skill 鼓励验证者突破 PR description 的范围，去尝试作者没有考虑到的场景。这种探索不是在寻找 PASS 的理由，而是在寻找 FAIL 的可能性。每一个被发现的意外行为，无论是否构成 bug，都是对变更质量的额外信息。

## 12. CLI 验证示例分析

CLI 是最直接的 Surface 类型。`skill-verify-cli-changes-example-for-verify-skill.md` 提供了一个完整的验证范例。

**场景设定：** 变更内容是给 `status` 子命令添加 `--json` flag。新增了 flag 解析代码和输出分支。

**Claim（来自 commit 消息）：** "machine-readable status output."

**推断（Inference）：** Verify Skill 要求验证者从 claim 中推导出具体的验证计划。在这个例子中，推断是：`tool status --json` 现在应该存在，输出有效的 JSON，包含与人可读输出相同的字段。不带 flag 的 `tool status` 应该保持不变。

**执行过程：**

```bash
go build -o /tmp/tool ./cmd/tool

/tmp/tool status
# -> Status: healthy
# -> Uptime: 3h12m
# -> Connections: 47

/tmp/tool status --json
# -> {"status":"healthy","uptime_seconds":11520,"connections":47}

/tmp/tool status --json | jq -e .status
# -> "healthy"
# (jq -e exits nonzero if the path is null/false — cheap validity check)

echo $?
# -> 0
```

[source: skill-verify-cli-changes-example-for-verify-skill.md]

几个设计细节：构建输出到 `/tmp/` 而不是覆盖项目目录；使用 `jq -e` 作为 JSON 有效性的廉价校验（`-e` 参数在路径为 null/false 时返回非零退出码）；先验证不带 flag 的行为不变（非回归测试），再验证新 flag 的行为。

示例还列举了 FAIL 的具体表现形式：

> "unknown flag: --json → not wired up, or you're running a stale build"
> "Output isn't valid JSON (jq errors) → serialization bug"
> "tool status (no flag) changed → regression; the diff touched more than it should"
> "JSON has different field names than expected → claim/code mismatch, might be fine, note it"

[source: skill-verify-cli-changes-example-for-verify-skill.md]

对于可能产生副作用的 CLI 命令，示例建议使用管道输入测试数据、指向临时目录、使用 mock 或 dry-run flag。如果没有安全模式且变更涉及破坏性路径，"say so and verify what you can around it"，报告这个限制，然后验证你能验证的部分。

## 13. Server/API 验证示例分析

Server/API 的验证比 CLI 多了一个生命周期的管理问题。`skill-verify-serverapi-changes-example-for-verify-skill.md` 详细说明了这个流程。

**场景设定：** 变更内容是在 `rateLimit.ts` 中给 429 响应添加 `Retry-After` header。

**Claim（来自 PR body）：** "clients can now back off correctly."

**推断：** 触发速率限制后，429 响应应该包含 `Retry-After: <n>` header，之前没有。

**服务启动模式：**

```bash
<start-command> &> /tmp/server.log &
SERVER_PID=$!
for i in {1..30}; do curl -sf localhost:PORT/health >/dev/null && break; sleep 1; done
# ... your curls ...
kill $SERVER_PID
```

[source: skill-verify-serverapi-changes-example-for-verify-skill.md]

如果没有 readiness endpoint，就轮询你即将测试的路由直到它不再返回 connection-refused，"then add a beat"。

**执行过程：**

```bash
# trigger the limit — 10 fast requests, limit is 5/sec per the diff
for i in {1..10}; do curl -s -o /dev/null -w "%{http_code}\n" localhost:3000/api/thing; done
# -> 200 200 200 200 200 429 429 429 429 429

# capture the 429 headers
curl -si localhost:3000/api/thing | head -20
# -> HTTP/1.1 429 Too Many Requests
# -> Retry-After: 12
# -> ...
```

[source: skill-verify-serverapi-changes-example-for-verify-skill.md]

几个值得注意的操作细节：先用循环快速发送请求来触发速率限制（根据 diff 得知限制是 5 次/秒），用 `curl -w "%{http_code}"` 只捕获状态码以确认 429 被触发，然后再用 `curl -si` 捕获完整的响应头。这种分步验证确保了每一步的观察都有据可依。

FAIL 的情况包括：

> "Header absent → the diff didn't take effect, or you're not actually hitting the 429 path (check the status code first)"
> "Header present but value is NaN / undefined / negative → the logic is wrong"
> "You got 200s all the way through → you never triggered the changed path. Tighten the request burst or check the rate limit config."

[source: skill-verify-serverapi-changes-example-for-verify-skill.md]

这些 FAIL 场景的描述体现了 Verify Skill 的诊断思维：每一个 FAIL 都不仅说明"什么错了"，还指向"为什么可能错了"。header 缺失可能是 diff 没生效，也可能是根本没有走到 429 路径。值异常说明逻辑有问题。全部 200 说明没有触发变更路径。

## 14. 设计总结

Verify Skill 的设计思想可以提炼为四个关键词：**运行时优先、Surface 导向、证据唯一性、不对称错误代价**。

运行时优先意味着所有验证都必须基于应用的实际运行，排除了代码审查、测试运行和类型检查作为验证手段。这不是对这三种实践的否定，而是对验证角色的精确定位：CI 负责测试和类型检查，代码审查负责逻辑和风格，验证者负责在真实运行环境中观察行为。三者互补，不互相替代。

Surface 导向确保了验证总是发生在用户可达的界面，而不是内部实现细节上。六种 Surface 类型覆盖了从 CLI 到 CI 的完整变更可达面。内部函数不是 Surface，你必须沿着调用链追踪到用户可触碰的界面，这保证了验证的端到端性质。

证据唯一性，"Captured output is evidence; your memory isn't"，确保了验证结果的可审查性。stdout、response body、截图是客观的、可复现的证据，而人的记忆是主观的、会衰减的。这种对客观证据的坚持，使得验证报告能够经受住独立审查。

不对称错误代价，"When in doubt, FAIL"，是 Verify Skill 的风险偏好。在软件发布场景中，错误 PASS（放行破损代码）的代价远高于错误 FAIL（多一次人工审查）。这种偏好不是保守，而是对生产环境安全的理性计算。

Verify Skill 的整个流程构成了一个完整的闭环：Find the change（确定范围）到 Get a handle（找到运行方法）到 Drive it（驱动到变更执行）到 Capture（捕获证据）到 Report（结构化报告）。每一个环节都有明确的约束和判断标准，使得验证过程可重复、可审查、可传授。
