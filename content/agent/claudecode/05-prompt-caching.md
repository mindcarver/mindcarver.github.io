# 省下的每个 token 都是钱：Claude Code 的 Prompt 缓存架构

## 1. Claude Code 内嵌的 caching 设计指南

Claude Code 的 system prompt 中嵌入了一份 "Prompt Caching -- Design & Optimization" 技术文档。这份文档不是 API 使用说明，而是面向工程实践的架构指南。它存在本身就说明了一件事：prompt caching 不是加一个 `cache_control` 标记就能生效的功能，它要求开发者在 prompt 组装的每个环节都做出正确的设计决策。

源文件开篇交代了定位：

> "This file covers how to design prompt-building code for effective caching. For language-specific syntax, see the ## Prompt Caching section in each language's README or single-file doc." [source: data-prompt-caching-design-optimization.md]

不教语法，教架构。它要解决的问题只有一个——当你构建一个会反复调用 Claude API 的应用（比如 Claude Code 本身），如何让缓存命中率从零提升到有意义的水平。

文档给出了一个完整的工作流：追踪 prompt 组装路径，将输入按稳定性分类，检查渲染顺序是否匹配稳定性顺序，在稳定性边界放置断点，然后审计静默失效器。五步构成了一套可操作的优化方法论。

读这份指南，实际上是在学如何在 LLM 应用中构建经济可行的长期运行架构。

## 2. 核心不变量：前缀匹配

整份文档的所有规则都从一个不变量推导出来：

> "Prompt caching is a prefix match. Any change anywhere in the prefix invalidates everything after it." [source: data-prompt-caching-design-optimization.md]

这个不变量的技术含义：cache key 由渲染后 prompt 中每个 `cache_control` 断点之前的精确字节序列推导而来。位置 N 处哪怕只有一个字节的差异（一个时间戳、一个重排的 JSON key、一个不同的 tool），都会使所有位置大于等于 N 的断点的缓存失效。

要理解这个不变量，得先理解 prompt 的渲染顺序：

> "Render order is: tools -> system -> messages. A breakpoint on the last system block caches both tools and system together." [source: data-prompt-caching-design-optimization.md]

整个 prompt 被拼接成一条字节流：最前面是 tools 定义，接着是 system prompt，最后是 messages。缓存基于这条字节流的前缀来匹配。如果在最后一个 system block 上放一个断点，它缓存的是 tools + system 的完整前缀。

文档用一句话总结了设计的核心原则：

> "Design the prompt-building path around this constraint. Get the ordering right and most caching works for free. Get it wrong and no amount of cache_control markers will help." [source: data-prompt-caching-design-optimization.md]

缓存不是靠标记"开启"的，而是靠顺序"设计"出来的。如果你的 prompt 组装路径本身就不符合稳定性递减的原则，放再多 `cache_control` 标记也没用。

## 3. 五种放置模式

文档定义了四种典型的放置模式，加上多轮断点模式，构成五种场景。

### 模式一：大型系统 prompt 跨请求共享

这是最简单的场景：一个大型系统 prompt 被大量请求共享。

```json
"system": [
  {"type": "text", "text": "<large shared prompt>", "cache_control": {"type": "ephemeral"}}
]
```

文档指出："Put a breakpoint on the last system text block. If there are tools, they render before system -- the marker on the last system block caches tools + system together." [source: data-prompt-caching-design-optimization.md]

关键点：由于 tools 在渲染顺序中位于 system 之前，在最后一个 system block 上放置断点，实际上缓存了 tools + system 的完整前缀。你不需要为 tools 单独放一个断点，一个标记就够了。

### 模式二：多轮对话

多轮对话场景下，缓存是增量累积的。

```json
// Last content block of the last user turn
messages[-1].content[-1].cache_control = {"type": "ephemeral"}
```

文档解释："Each subsequent request reuses the entire prior conversation prefix. Earlier breakpoints remain valid read points, so hits accrue incrementally as the conversation grows." [source: data-prompt-caching-design-optimization.md]

这里的要点是"增量命中"。每一轮对话都复用了之前所有轮次的完整前缀，而之前放置的断点仍然是有效的读取点。随着对话增长，缓存命中率逐轮提高。这不是一次性的优化，而是随着使用逐渐生效的复合收益。

### 模式三：共享前缀 + 变化后缀

这是 RAG 场景的典型模式：多个请求共享一大段固定的前导内容（few-shot 示例、检索到的文档、指令），但最后的提问各不相同。

```json
"messages": [{"role": "user", "content": [
  {"type": "text", "text": "<shared context>", "cache_control": {"type": "ephemeral"}},
  {"type": "text", "text": "<varying question>"}  // no marker -- differs every time
]}]
```

文档特别警告了常见的错误做法："Put the breakpoint at the end of the shared portion, not at the end of the whole prompt -- otherwise every request writes a distinct cache entry and nothing is ever read." [source: data-prompt-caching-design-optimization.md]

如果把断点放在整个 prompt 的末尾，每个请求都会写入一个不同的缓存条目，但没有任何请求会读取这些条目。这不是缓存，这是用 1.25 倍的写入溢价买了一堆永远不会被读取的数据。断点必须放在共享部分的末尾，变化的部分不加标记。

### 模式四：无缓存场景

文档明确承认有些场景不适合缓存：

> "Don't cache. If the first 1K tokens differ per request, there is no reusable prefix. Adding cache_control only pays the cache-write premium with zero reads. Leave it off." [source: data-prompt-caching-design-optimization.md]

知道何时不缓存，也是工程判断力。每个请求的前 1K tokens 都不同，强行加标记只会付出写入溢价，没有任何回报。

### 模式五：多轮断点与长对话

在 agentic 循环中，单轮对话可能包含大量的 `tool_use` / `tool_result` 对，导致内容块数量迅速膨胀。文档在 "20-block lookback window" 一节中指出，每个断点最多向前回溯 20 个内容块来寻找之前的缓存条目。如果单轮超过 20 个块，下一个请求的断点就找不到上一轮的缓存。

解决方案是："Place an intermediate breakpoint every ~15 blocks in long turns, or put the marker on a block that's within 20 of the previous turn's last cached block." [source: data-prompt-caching-design-optimization.md]

这是一个容易被忽略但影响很大的细节。在 agentic 工作流中，一次工具调用往返就会产生两个 block（tool_use + tool_result），10 次调用就 20 个 block。如果你不在中间插入断点，整个缓存链就会断掉。

## 4. 架构指导：三条关键规则

文档将架构指导放在 placement patterns 之后，但明确表示这些规则的优先级更高：

> "These are the decisions that matter more than marker placement. Fix these first." [source: data-prompt-caching-design-optimization.md]

### 规则一：冻结系统 prompt

> "Don't interpolate 'current date: X', 'mode: Y', 'user name: Z' into the system prompt -- those sit at the front of the prefix and invalidate everything downstream. Inject dynamic context as a user or assistant message later in messages. A message at turn 5 invalidates nothing before turn 5." [source: data-prompt-caching-design-optimization.md]

系统 prompt 位于渲染顺序的前部。如果你在系统 prompt 中插入了动态内容（当前时间、用户名、模式标识），每一次变化都会使整个前缀失效。正确的做法是将动态内容作为 message 注入。因为 messages 位于渲染顺序的末尾，第 5 轮的 message 不会使第 5 轮之前的任何内容失效。

### 规则二：不切换 tools 和 model

> "Tools render at position 0; adding, removing, or reordering a tool invalidates the entire cache. Same for switching models (caches are model-scoped)." [source: data-prompt-caching-design-optimization.md]

tools 定义位于整个 prompt 的最前端（position 0），任何变更都是毁灭性的。模型切换同理，因为缓存是按模型隔离的。文档给出了一个实用的建议：如果需要"模式切换"，不要交换 tool set，而是给 Claude 一个用于记录模式转换的 tool，或者把模式作为 message 内容传递。tools 的序列化也必须确定性——按名称排序。

### 规则三：fork 操作复用父级前缀

> "Side computations (summarization, compaction, sub-agents) often spin up a separate API call. If the fork rebuilds system / tools / model with any difference, it misses the parent's cache entirely. Copy the parent's system, tools, and model verbatim, then append fork-specific content at the end." [source: data-prompt-caching-design-optimization.md]

在 agentic 架构中，子代理、摘要压缩等操作会发起独立的 API 调用。如果 fork 重建的 system、tools 或 model 与父级有任何差异，它就完全无法命中父级的缓存。必须逐字复制父级的这三个组件，然后在末尾追加 fork 特有的内容。

## 5. 静默失效器：六种让缓存悄悄失效的模式

这是文档中最具实战价值的部分。六种"静默失效器"不会报错，不会抛异常，只是让缓存命中率归零。文档建议在审查代码时，用 grep 搜索以下模式：

### 5.1 时间戳注入

| 模式 | 失效原因 |
|------|----------|
| `datetime.now()` / `Date.now()` / `time.time()` in system prompt | 前缀每次请求都变 |

每次请求的时间戳都不同，注入到系统 prompt 中意味着每次请求的前缀都不同，缓存永远不会命中。

### 5.2 随机 ID 提前出现

| 模式 | 失效原因 |
|------|----------|
| `uuid4()` / `crypto.randomUUID()` / request IDs early in content | 同样——每个请求都唯一 |

UUID 或随机请求 ID 如果出现在前缀的早期位置，效果和时间戳一样致命。

### 5.3 非确定性序列化

| 模式 | 失效原因 |
|------|----------|
| `json.dumps(d)` without `sort_keys=True` / iterating a `set` | 非确定性序列化导致前缀字节不同 |

Python 的 dict 在 3.7+ 虽然保持插入顺序，但 `json.dumps` 不保证 key 的排序。如果你把一个没有 `sort_keys=True` 的 JSON 字符串拼入 prompt，两次调用的字节序列可能不同。遍历 `set` 的顺序更是不确定的。

### 5.4 用户/会话 ID 插值

| 模式 | 失效原因 |
|------|----------|
| f-string interpolating session/user ID into system prompt | 每用户不同的前缀；无法跨用户共享 |

将用户 ID 或会话 ID 插入系统 prompt，导致每个用户拥有不同的前缀。缓存条目只能在同一用户的后续请求中命中，失去了跨用户共享的可能性。

### 5.5 条件性系统段落

| 模式 | 失效原因 |
|------|----------|
| Conditional system sections (`if flag: system += ...`) | 每种 flag 组合都是一个不同的前缀 |

如果系统 prompt 中有条件分支，每种 flag 组合都会产生一个独立的前缀变体。两个 flag 的笛卡尔积就是四个不同的前缀，缓存被碎片化。

### 5.6 动态 tool 集

| 模式 | 失效原因 |
|------|----------|
| `tools=build_tools(user)` where set varies per user | tools 在 position 0 渲染；跨用户无缓存 |

如果 tool 列表因用户而异，因为 tools 在 position 0 渲染，整个缓存链条在最开始就断裂了。

文档给出的修复策略是三选一：将动态部分移到最后一个断点之后，使其变为确定性的，或者如果该动态内容不是必要的则直接删除。

## 6. 经济学分析：5 分钟 TTL vs 1 小时 TTL

缓存的经济模型是决定是否使用缓存以及如何选择 TTL 的关键依据。文档给出了精确的成本数据。

API 支持两种 TTL：

```json
"cache_control": {"type": "ephemeral"}              // 5-minute TTL (default)
"cache_control": {"type": "ephemeral", "ttl": "1h"} // 1-hour TTL
```

成本结构：

- 缓存读取价格约为基础输入价格的 0.1 倍
- 5 分钟 TTL 的写入价格为 1.25 倍
- 1 小时 TTL 的写入价格为 2 倍

文档给出了盈亏平衡分析：

> "Break-even depends on TTL: with 5-minute TTL, two requests break even (1.25x + 0.1x = 1.35x vs 2x uncached); with 1-hour TTL, you need at least three requests (2x + 0.2x = 2.2x vs 3x uncached)." [source: data-prompt-caching-design-optimization.md]

算一下：
- 5 分钟 TTL：第 2 个请求就能回本。第一次请求写入缓存花费 1.25x，第二次请求从缓存读取花费 0.1x，总计 1.35x。不缓存的话两次请求花费 2x。两次就赚了。
- 1 小时 TTL：需要至少 3 个请求才能回本。第一次写入 2x，后续两次读取各 0.1x，总计 2.2x。不缓存三次花费 3x。

文档还指出了 1 小时 TTL 的适用场景：

> "The 1-hour TTL keeps entries alive across gaps in bursty traffic, but the doubled write cost means it needs more reads to pay off." [source: data-prompt-caching-design-optimization.md]

对于流量有波峰波谷的应用，1 小时 TTL 能让缓存在波谷期存活下来，在下一个波峰到来时仍然有效。但翻倍的写入成本意味着你需要更多的读取次数才能摊销。

每个请求最多允许 4 个 `cache_control` 断点，这个限制约束了你能同时缓存的层级数量。

## 7. 模型最小可缓存前缀表

并非所有长度的 prompt 都能被缓存。文档指出，最小可缓存前缀是模型相关的，较短的 prompt 即使加了标记也不会被缓存，而且不会报错：

> "Shorter prefixes silently won't cache even with a marker -- no error, just cache_creation_input_tokens: 0" [source: data-prompt-caching-design-optimization.md]

你可能在不知情的情况下以为自己开启了缓存，实际上从未生效。验证方法是检查响应中的 `cache_creation_input_tokens` 字段是否为零。

以下是各模型的最小可缓存前缀：

| 模型 | 最小 tokens |
|------|------------:|
| Opus 4.6, Opus 4.5, Haiku 4.5 | 4096 |
| Sonnet 4.6, Haiku 3.5, Haiku 3 | 2048 |
| Sonnet 4.5, Sonnet 4.1, Sonnet 4, Sonnet 3.7 | 1024 |

文档用了一个具体的例子：

> "A 3K-token prompt caches on Sonnet 4.5 but silently won't on Opus 4.6." [source: data-prompt-caching-design-optimization.md]

3000 tokens 的 prompt 在 Sonnet 4.5（阈值 1024）上可以缓存，但在 Opus 4.6（阈值 4096）上却不行。如果你在模型之间切换，缓存行为会悄然改变。

## 8. 失效层级：三层缓存体系

不是所有的参数变更都会导致完全失效。文档揭示了一个三层缓存体系，变更只影响自己所在的层级及其以下层级：

| 变更类型 | Tools 缓存 | System 缓存 | Messages 缓存 |
|----------|:---------:|:-----------:|:-------------:|
| Tool 定义变更（增删重排） | 失效 | 失效 | 失效 |
| 切换模型 | 失效 | 失效 | 失效 |
| speed、web-search、citations 切换 | 保持 | 失效 | 失效 |
| System prompt 内容变更 | 保持 | 失效 | 失效 |
| tool_choice、images、thinking 开关 | 保持 | 保持 | 失效 |
| Message 内容变更 | 保持 | 保持 | 失效 |

这个层级关系有几个直接的工程含义。

tool 定义和模型位于最顶层，它们的变更导致全链失效。这再次印证了"不要切换 tools 和 model"这条规则的分量。

`tool_choice` 和 `thinking` 的开关只影响 messages 缓存层，不影响 tools 和 system 的缓存。文档的结论是：

> "You can change tool_choice per-request or toggle thinking without losing the tools+system cache. Don't over-worry about these -- only tool-definition and model changes force a full rebuild." [source: data-prompt-caching-design-optimization.md]

你可以根据每个请求的实际情况灵活调整 `tool_choice` 或开关 thinking，不必担心损失最值钱的 tools + system 缓存。

## 9. 20-block 回看窗口限制

这是一个容易被忽视但影响深远的限制：

> "Each breakpoint walks backward at most 20 content blocks to find a prior cache entry." [source: data-prompt-caching-design-optimization.md]

每个断点在寻找之前的缓存条目时，最多只能向前回溯 20 个内容块。如果单轮对话产生了超过 20 个内容块，下一个请求的断点就找不到上一轮的缓存条目，缓存被静默丢失。

在 agentic 循环中，这个阈值很容易被突破。一次 tool_use 产生一个 block，对应的 tool_result 又是一个 block。10 次工具调用就产生了 20 个 block。如果一个 agent 在一轮中执行了 15 次以上的工具调用，20-block 限制就会生效。

文档给出的修复方案是："Place an intermediate breakpoint every ~15 blocks in long turns, or put the marker on a block that's within 20 of the previous turn's last cached block." [source: data-prompt-caching-design-optimization.md]

每隔约 15 个 block 插入一个中间断点，留出 5 个 block 的安全余量。这是防御性编程：不是等缓存失效了再修复，而是在组装 prompt 时就主动规划断点位置。

## 10. 并发请求时序问题

缓存写入和读取之间存在一个微妙的时序窗口：

> "A cache entry becomes readable only after the first response begins streaming. N parallel requests with identical prefixes all pay full price -- none can read what the others are still writing." [source: data-prompt-caching-design-optimization.md]

缓存条目只有在第一个响应开始流式传输之后才变得可读。如果你同时发出 N 个具有相同前缀的并行请求，它们全都无法读取彼此正在写入的缓存，全部按全价计费。

文档给出的解决方案：

> "For fan-out patterns: send 1 request, await the first streamed token (not the full response), then fire the remaining N-1. They'll read the cache the first one just wrote." [source: data-prompt-caching-design-optimization.md]

对于扇出（fan-out）模式：先发送 1 个请求，等待第一个流式 token（不需要等完整响应），然后发出剩余的 N-1 个请求。这些后续请求会读取第一个请求刚刚写入的缓存。

注意关键细节：你只需要等待第一个 token，不是完整响应。流式传输一旦开始，缓存就已经可读了。这最小化了扇出模式的启动延迟。

这个时序问题在高并发场景下尤其重要。如果你有一个 API 后端同时处理多个用户的请求，而这些请求共享相同的前缀（比如相同的系统 prompt），你需要确保第一个请求已经开始流式传输后，后续请求才能从缓存中受益。

## 11. 验证缓存命中

文档提供了验证缓存是否生效的方法。响应的 `usage` 对象报告了缓存活动：

| 字段 | 含义 |
|------|------|
| `cache_creation_input_tokens` | 本次请求写入缓存的 tokens（支付了约 1.25 倍写入溢价） |
| `cache_read_input_tokens` | 本次请求从缓存读取的 tokens（支付了约 0.1 倍） |
| `input_tokens` | 未缓存的全价 tokens |

文档给出了关键的诊断方法：

> "If cache_read_input_tokens is zero across repeated requests with identical prefixes, a silent invalidator is at work -- diff the rendered prompt bytes between two requests to find it." [source: data-prompt-caching-design-optimization.md]

如果重复请求中 `cache_read_input_tokens` 始终为零，说明有静默失效器在起作用。诊断方法是对比两次请求的渲染 prompt 字节差异，找到那个导致前缀不同的字节。

关于 `input_tokens` 字段，文档特别澄清了一个常见误解：

> "input_tokens is the uncached remainder only. Total prompt size = input_tokens + cache_creation_input_tokens + cache_read_input_tokens." [source: data-prompt-caching-design-optimization.md]

`input_tokens` 只是未缓存的部分，不是全部。如果 agent 运行了数小时但 `input_tokens` 只有 4K，其余部分都是从缓存提供的。检查总和，而不是单个字段。

## 12. 总结

这份缓存设计指南表面上在讲一个 API 特性，实际上在讲如何构建一个经济可行的 LLM 应用。

Prompt 的渲染顺序（tools -> system -> messages）决定了缓存的脆弱性梯度。越靠前的组件越需要稳定，越靠后的组件越可以动态。组装 prompt 不遵循这个原则，缓存从一开始就不会生效。

静默失效是最难排查的问题。没有错误信息，没有警告，只有账单上的数字在涨。六个静默失效器（时间戳、随机 ID、非确定性序列化、用户 ID 插值、条件段落、动态 tool 集）应该进入每个 LLM 应用的代码审查 checklist。

三层缓存体系让你不需要对每个参数变更都如履薄冰。`tool_choice` 和 `thinking` 可以按请求切换，不影响 tools + system 缓存。了解层级关系，才能在性能和灵活度之间做出正确权衡。

缓存的时序特性意味着扇出模式不能简单地并行发射。先发一个请求，等第一个 token，再发射其余请求。5 分钟 TTL 两次请求就回本，1 小时 TTL 需要三次但能穿越流量间隙。选择哪个 TTL，取决于你的流量模式和成本预算。

在 LLM 应用中，prompt 不是一串文本，而是一个需要精心设计字节布局的数据结构。缓存命中率不是运气，是设计的结果。
