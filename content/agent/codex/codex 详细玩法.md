
  先给你一个总判断：

  - features 更像是 CLI 本身能力的总开关
  - -c, --config 更像是 一次运行的行为注入器
  - -p, --profile 则是 把一组常用 -c 配置命名复用

  对你这个 QROS 项目，真正最有价值的是：

  1. 用 features 识别哪些能力当前可用、稳定不稳定
  2. 用 -c 做单次任务的行为切换
  3. 用 profile 把不同工作流固化成“模式”

  ———

  ## 1. features 到底是什么

  你本机 codex features list 当前能看到很多 feature flag，
  比如：

  - multi_agent = true
  - apps = true
  - shell_tool = true
  - shell_snapshot = true
  - plugins = true
  - fast_mode = true
  - child_agents_md = true

  也有很多是 under development、experimental、removed。

  ### 这意味着什么

  这些 flag 不是你项目自己的 feature，而是 Codex CLI 自己
  的功能开关。
  它们决定的是：

  - 某个工具链是否启用
  - 某种交互模式是否启用
  - 某种实验能力是否开放
  - 某些旧能力是否已经废弃

  ### features 有三种用法

  #### 1. 查看当前能力

  codex features list

  用途：

  - 看哪些能力现在能用
  - 看哪些能力是稳定版，哪些还是实验版
  - 判断你后面设计 harness 或 AGENTS 的时候能不能依赖某能
    力

  #### 2. 持久开启某个能力

  codex features enable <feature_name>

  #### 3. 持久关闭某个能力

  codex features disable <feature_name>

  这类修改通常会写进你的 ~/.codex/config.toml，所以是 长期
  生效 的。

  ———

  ## 2. 对你项目，features 怎么用才合理

  ### 适合做什么

  - 做 受控实验
  - 评估某个 Codex 能力是否会影响：
      - AGENTS.md 的行为
      - MCP server 的调用体验
      - shell / sandbox 行为
      - multi-agent 工作流

  ### 不适合做什么

  - 不适合把项目流程直接绑死在 under development 的
    feature 上
  - 不适合为了单次任务，长期改动全局 feature

  ### 对 QROS 的实际建议

  #### 建议 1：把 features list 当成“能力盘点”

  比如你准备设计：

  - QROS 子 agent 流程
  - harness 模式
  - MCP 接 research registry

  那先看：

  codex features list

  你现在本机能确认：

  - multi_agent = true
  - child_agents_md = true
  - plugins = true
  - shell_tool = true

  这说明：

  - 多 agent / 子 agent 这条线是可用的
  - shell 工具是可用的
  - 你可以基于这些能力设计流程

  但像：

  - tool_search = false
  - memories = false
  - undo = false

  就说明这些不该被你当前 workflow 默认依赖。

  #### 建议 2：实验能力用临时开关，不要先全局 enable

  如果以后你想测某个实验能力，优先用：

  codex --enable <feature_name> ...

  而不是先：

  codex features enable <feature_name>

  原因：

  - 前者只影响这一次运行
  - 后者会污染你以后所有仓库的默认行为

  ———

  ## 3. -c, --config 到底强在哪

  这是 Codex CLI 最有工程味的能力之一。

  它允许你在 不改全局配置文件 的前提下，对某一次运行临时注
  入配置。

  例如你本机 help 里明确支持：

  -c model="gpt-5.4"
  -c 'sandbox_permissions=["disk-full-read-access"]'
  -c shell_environment_policy.inherit=all

  ### 你可以用它改什么

  按你本机帮助和配置结构，常见包括：

  - 模型
      - model="gpt-5.4"
  - 沙箱
      - sandbox_mode="workspace-write" 这类
  - feature flag
      - features.multi_agent=true
  - 环境继承策略
      - shell_environment_policy.inherit=all
  - 其他 config.toml 里已有字段

  ### 它为什么比直接改配置文件更重要

  因为它让你可以把不同任务模式变成“临时运行配置”。

  也就是说，你不用把 Codex 只有一种人格和一种执行方式。

  ———

  ## 4. -c 和 features 的关系

  这是最容易混淆的点。

  ### features enable/disable

  - 改的是长期状态
  - 更像“修改默认设置”

  ### --enable/--disable

  - 改的是单次运行的 feature
  - 更像“本次临时打开/关闭某能力”

  ### -c features.xxx=true

  - 本质上也是单次运行临时覆盖
  - 只是写法更通用

  所以优先级上，你可以这样理解：

  - 想长期改变默认行为：codex features enable/disable
  - 想某次任务临时试验：--enable/--disable 或 -c
    features.xxx=true

  ———

  ## 5. profile 是什么

  你本机 codex exec --help 明确支持：

  -p, --profile <CONFIG_PROFILE>

  这意味着 Codex 支持在 ~/.codex/config.toml 里定义命名配
  置组。

  虽然你当前配置文件里还没看到 [profiles.xxx]，但 CLI 已经
  支持这个机制。

  ### 它的意义

  就是把一组常用配置起个名字。

  比如你不用每次都写：

  codex exec \
    -m gpt-5.4 \
    -s workspace-write \
    --enable multi_agent \
    -C /path/to/repo \
    "..."

  你可以用 profile 把它固化成：

  codex exec -p qros-stage-audit "..."

  ———

  ## 6. 你提到的 4 个场景，怎么设计 profile

  下面我按你这个项目讲。

  ———

  ### A. docs-review

  适合什么

  - 改文档
  - 对齐 runtime 字段说明
  - 检查 README / SOP / experience 文档一致性

  你需要的特点

  - 模型强一点
  - 沙箱风险低
  - 一般不需要太激进的执行权限
  - 可以偏向只读或轻编辑

  思路：

  - workspace-write
  - 保持默认模型
  - 不一定需要 multi-agent
  - 强调文档一致性

  适合你项目的用途：

  - 改 stage-freeze-group-field-guide.md
  - 改 docs/experience/*
  - 补入口文档链接
  - 做字段一一对应说明

  ———

  ### B. runtime-edit

  适合什么

  - 改 tools/
  - 改 scripts/
  - 改 scaffold / build / runtime 逻辑

  你需要的特点

  - 允许写工作区
  - 更强调测试
  - 对 shell/tool 使用要求更高

  思路：

  - workspace-write
  - 强一些的模型和 reasoning
  - shell/tool 能力必须可用
  - 跑最小相关测试

  适合你项目的用途：

  - 改 tools/csf_*_runtime.py
  - 改 scripts/run_research_session.py
  - 改 artifact 命名或 freeze draft shape

  ———

  ### C. harness-test

  适合什么

  - 专门验证 Codex 自己的行为
  - 测 AGENTS.md 加载链
  - 测不同目录启动时的指令发现
  - 测 CLI 行为和 feature 影响

  你需要的特点

  - 关注 CLI 本身，不只是仓库代码
  - 需要可重复切换目录、模式、feature
  - 更适合短任务、对照实验

  思路：

  - 用 -C 在不同目录下运行
  - 用 --enable/--disable 做 feature 对照
  - 尽量 --ephemeral
  - 把输出落文件或用 --json

  适合你项目的用途：

  - 验证 root AGENTS.md 和 harness/AGENTS.md 的发现机制
  - 测某 feature 是否影响 instruction chain
  - 做你现在这种“CLI harness engineering”实验

  ———

  ### D. qros-stage-audit

  这个最值得展开。

  它适合：

  - 审某个 stage 的合同、产物、测试和文档是否一致
  - 不一定改代码，更多是做“阶段一致性核查”
  - 很适合 QROS 这种流程仓

  #### 这个 profile 应该解决什么问题

  在 QROS 里，一个 stage 经常同时有这些对象：

  - runtime draft 字段
  - skill 里的 group 名
  - docs 里的字段说明
  - tests 里的 fixture 形状
  - build/scaffold 实际生成的 artifact

  最容易出的问题是：

  - 文档字段和 runtime 字段不一致
  - skill 术语和测试里的真实 draft 名称不一致
  - artifact 名字变了，但文档没改
  - review 规则还在旧命名上

  qros-stage-audit 就是专门做这个。

  #### 它应该具备的行为

  - 在某个 stage 上做横向对账
  - 默认先读：
      - skills/qros-*-author/SKILL.md
      - 对应 tools/*runtime.py
      - 对应 tests/test_*runtime.py
      - 对应 docs/main-flow-sop/*
      - 对应 docs/experience/*
  - 输出重点不是代码修改，而是：
      - 发现不一致
      - 建议修正点
      - 必要时自动补文档或测试

  #### 很适合你项目的场景

  比如你说：

  - “审一下 csf_signal_ready 的字段、artifact、文档、测试
    是不是一套东西”
  - “检查 holdout_validation 主线和 CSF 路线命名有没有漂
    移”
  - “看看 delivery_contract 在所有 stage 里是不是表述一致”

  这就是典型的 qros-stage-audit

  ———

  ## 7. qros-stage-audit profile 我建议怎么配

  ### 运行目标

  让 Codex变成“阶段一致性审计器”，而不是普通写代码助手。

  ### 核心要求

  - 模型强
  - 允许工作区写入
  - 默认跑局部测试
  - 适合读大量文档和测试文件
  - 不一定需要子 agent，但可以保留

  ### 典型任务形式

  codex exec -p qros-stage-audit \
    -C /Users/mac08/workspace/web3qt/quant-research-os \
    "审计 csf_signal_ready：字段名、artifact、skill、
  runtime、tests、docs 是否一致。先报不一致项，再修。"

  ### 它的最佳输出

  不是“总结一下”，而是：

  1. stage 的事实来源有哪些
  2. 当前有哪些命名/契约不一致
  3. 哪些是 runtime 真值
  4. 哪些文档和测试要改
  5. 改完后跑哪些验证

  ———

  ## 8. 你现在这个项目，最值得先做哪几个 profile

  如果你真要落地，我建议先做这 3 个，不要一下配太多：

  ### docs-review

  因为你现在文档密集，而且字段说明很重要。

  ### runtime-edit

  因为你有很多 tools/*runtime.py 和 stage scaffold。

  ### qros-stage-audit

  因为这是你这个仓库最有特色、也最容易产生价值的 profile。

  harness-test 我会放第四个，因为它更偏 Codex 本身实验，不
  是你主仓日常最高频任务。

  ———

  ## 9. 一个非常实际的建议

  先不要急着长期 features enable。

  你应该先做的是：

  - 定义几个 profile
  - 每个 profile 里把模型、sandbox、偏好固定住
  - feature 只在确实需要时临时开

  也就是：

  - profiles 是工作流
  - features 是实验能力

  不要反过来。

  ———

  ## 10. 如果你要继续，我建议下一步做这个

  我可以直接帮你产出一份适合你本机的 ~/.codex/config.toml
  片段，包含：

  - docs-review
  - runtime-edit
  - harness-test
  - qros-stage-audit

  四个 profile 的具体配置草案。

  这样你下一步就不是抽象讨论，而是可以直接上手跑。
