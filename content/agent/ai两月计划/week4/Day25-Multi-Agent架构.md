# Day 25：Multi-Agent 架构

## 学习目标

前两天学了单 Agent 的基础认知和工具调用，今天要跨入一个新的层次：Multi-Agent，多智能体架构。

为什么要多 Agent？因为单 Agent 处理复杂任务时会遇到能力瓶颈。一个 Agent 既要懂行业分析、又要懂岗位拆解、又要懂流程优化、又要懂 AI 方案设计、还要做风险审查，每个领域的知识深度都不够，工具集也太庞大，决策质量必然下降。Multi-Agent 的思路是"分工协作"：让不同的 Agent 各自专精一个领域，然后通过协调机制让它们合作完成复杂任务。

今天的目标是理解 Multi-Agent 的价值、掌握分工和通信的设计方法、学会处理 Agent 间的冲突和结果整合。学完这一天，你应该能设计出一个由 6 个 Agent 组成的行业研究团队，并定义每个 Agent 的职责边界和协作协议。

---

## 核心概念

### Multi-Agent 的价值

Multi-Agent 不是为了炫技而存在的。它解决的是单 Agent 处理复杂任务时的三个核心痛点。

第一个痛点是"任务太复杂，一个 Agent 搞不定"。行业研究这个任务包含行业概览、岗位分析、流程分析、AI 机会评分、方案设计、风险审查等子任务，每个子任务都有专业门槛。让一个 Agent 做所有事，就像让一个人同时当医生、律师、会计师和工程师，每个领域都只能浅尝辄止。

第二个痛点是"工具太多，选择困难"。昨天学了 8 个工具，如果把全部 8 个工具给一个 Agent，它每次决策都要从 8 个工具中选择，选错概率不低。如果分成 6 个 Agent，每个 Agent 只配备 2-3 个工具，选择就简单多了。工具少了，决策准确率自然提高。

第三个痛点是"上下文太长，注意力分散"。单 Agent 执行复杂任务时，对话历史会越来越长，State 越来越大。模型的注意力是有限的，当上下文塞了几十个步骤的结果时，模型很难专注于当前步骤。多 Agent 让每个 Agent 的上下文保持精简，只关注与自己职责相关的信息。

但 Multi-Agent 也有明显的代价。第一，系统复杂度急剧增加。6 个 Agent 的系统比 1 个 Agent 复杂得多，调试困难、成本更高、出问题的概率更大。第二，Agent 间的通信和协调是新的故障点。信息在传递过程中可能丢失或变形。第三，总 Token 消耗更大。每个 Agent 都要独立调用模型，6 个 Agent 的 Token 消耗可能是单 Agent 的 3-4 倍（不是 6 倍，因为每个 Agent 的上下文更短）。

所以 Multi-Agent 的使用有一个明确的原则：能用单 Agent 解决的，不要用 Multi-Agent。只有当任务确实需要多个领域的专业知识、且各领域之间有明确的分工边界时，才值得引入 Multi-Agent。

### 主控 Agent

Multi-Agent 系统需要一个"总指挥"，这就是主控 Agent，通常叫 Orchestrator。

Orchestrator 的职责不是亲自干活，而是协调其他 Agent 的工作。具体来说，它做四件事。

第一，任务分解。接收用户的原始请求，分析需要哪些专业技能，将大任务拆成子任务分配给对应的专家 Agent。比如收到"分析半导体行业的 AI 机会"这个请求，Orchestrator 判断需要：行业概览（交给 Industry Research Agent）、岗位分析（交给 Role Analyst Agent）、流程分析（交给 Process Analyst Agent）、AI 方案（交给 AI Solution Agent）、风险审查（交给 Risk Review Agent）。

第二，执行调度。决定子任务的执行顺序和依赖关系。有些子任务可以并行（行业概览和岗位分析可以同时开始），有些必须串行（AI 方案必须等前面三个分析完成才能开始）。Orchestrator 负责管理这些依赖关系。

第三，结果整合。收集各个 Agent 的输出，检查完整性，整合成最终结果。如果某个 Agent 的输出质量不达标，Orchestrator 可以让它重做或者补充。

第四，异常处理。如果某个 Agent 执行失败或超时，Orchestrator 决定怎么处理：跳过、重试、降级还是报告失败。

Orchestrator 不应该有自己的专业分析能力。它的角色是纯粹的"管理者"，不是"全能选手"。如果 Orchestrator 也参与具体分析，就失去了多 Agent 分工的意义。

### 专家 Agent

专家 Agent 是具体干活的 Agent。每个专家 Agent 专精一个领域，配备该领域专用的工具和 Prompt。

专家 Agent 的设计有三个关键要素。

第一，职责边界清晰。每个 Agent 只做一件事，而且这件事有明确的边界。Industry Research Agent 负责行业概览，不做岗位分析。Role Analyst Agent 负责岗位拆解，不做流程分析。边界清晰的好处是避免 Agent 间的职责重叠和冲突。

第二，工具精简。每个 Agent 只配备 2-3 个核心工具。Industry Research Agent 配备 search_web 和 query_knowledge_base 就够了，不需要 send_email 或 calculate_metrics。工具精简能减少模型的选择压力，提高决策准确性。

第三，Prompt 专注。每个 Agent 的系统 Prompt 只关注自己的领域。Industry Research Agent 的 Prompt 详细描述怎么分析行业、怎么看数据、怎么输出结构化概览。它不需要知道其他 Agent 的存在，也不需要理解全局任务。

### Agent 间通信

Multi-Agent 系统中，Agent 之间需要通信。怎么通信是架构设计的关键问题。

通信方式有两种。一种是间接通信，所有信息通过 Orchestrator 中转。Agent A 的输出给 Orchestrator，Orchestrator 再转发给 Agent B。这种方式的好处是 Orchestrator 掌控全局信息流，容易调度和监控。坏处是信息传递路径长，可能丢失或延迟。

另一种是直接通信，Agent 之间可以直接对话。Agent A 需要信息时直接问 Agent B，不需要通过 Orchestrator。这种方式的好处是通信高效。坏处是 Orchestrator 失去全局信息流视图，难以调度和监控。

实践中通常是混合模式。关键信息（如行业概览这种会被多个 Agent 使用的信息）通过 Orchestrator 分发。临时性的、局部的信息（如 Agent A 需要向 Agent B 确认一个具体数据）可以直接通信。

另一个关键问题是通信协议。Agent 之间用什么样的格式交换信息？建议用结构化的 JSON 格式，包含消息类型、发送者、接收者、内容、时间戳。结构化协议让消息可解析、可追踪、可审计。

### 冲突解决

多个 Agent 可能产生冲突的输出。比如 Industry Research Agent 说市场规模是 5000 亿，Role Analyst Agent 引用的数据说是 8000 亿。AI Solution Agent 基于哪个数据来设计方案？

冲突解决策略有几种。

第一种是优先级规则。某些 Agent 的输出优先级更高。比如 Risk Review Agent 的判断优先级最高，如果有冲突，其他 Agent 要调整。

第二种是人工仲裁。当 Agent 间出现无法自动解决的冲突时，暂停流程，提交人工审核。这会引入延迟，但能保证准确性。

第三种是协商机制。让相关 Agent 之间协商，互相提供证据，达成一致。比如 Industry Research Agent 和 Role Analyst Agent 各自说明数据来源，比较数据的新旧和可靠性，达成共识。

### 结果整合

各个 Agent 完成任务后，需要把它们的输出整合成最终结果。这是 Orchestrator 的职责，也是整个 Multi-Agent 流程的最后一步。

结果整合不是简单拼接。要做三件事：检查完整性、解决冲突、生成连贯的输出。

检查完整性。确认所有应该提交的 Agent 都已经提交，且输出格式符合要求。如果有 Agent 失败或超时，决定是等待重试还是继续处理。

解决冲突。如前所述，处理不同 Agent 间的输出冲突。

生成连贯的输出。把各个 Agent 的输出整合成一份结构化的报告。报告要有清晰的章节结构、一致的术语和风格、逻辑连贯的内容。

### Multi-Agent 的典型架构

以行业研究为例，一个典型的 Multi-Agent 架构包含：

**Orchestrator**（主控 Agent）：任务分解、执行调度、结果整合、异常处理

**Industry Research Agent**（行业研究 Agent）：分析行业概览，输出产业链结构、市场规模、主要玩家、发展趋势

**Role Analyst Agent**（岗位分析 Agent）：拆解关键岗位，输出岗位列表、职责描述、技能要求

**Process Analyst Agent**（流程分析 Agent）：梳理业务流程，输出流程图、瓶颈分析、优化建议

**AI Solution Agent**（AI 方案 Agent）：设计 AI 应用方案，输出场景列表、技术方案、优先级评分

**Risk Review Agent**（风险审查 Agent）：评估方案风险，输出风险清单、应对建议

这个架构中，前三个 Agent（Industry Research、Role Analyst、Process Analyst）可以并行执行。AI Solution Agent 等前三个完成后执行。Risk Review Agent 等方案生成后执行。Orchestrator 在整个过程中调度和监控。

## 今日总结

Multi-Agent 解决单 Agent 的三个痛点：任务太复杂、工具太多、上下文太长。

代价是系统复杂度增加、通信成为新故障点、Token 消耗更大。

使用原则：能用单 Agent 解决的不要用 Multi-Agent。

主控 Agent（Orchestrator）负责任务分解、执行调度、结果整合、异常处理。

专家 Agent 职责边界清晰、工具精简、Prompt 专注。

Agent 间通信有间接（通过 Orchestrator）和直接两种，实践中通常是混合模式。

冲突解决策略：优先级规则、人工仲裁、协商机制。

结果整合要做三件事：检查完整性、解决冲突、生成连贯输出。

## 明日预告

今天设计了一个能自动运行的 Multi-Agent 系统。但有一个问题：如果 Agent 做出了错误的判断，怎么办？如果它推荐了一个实际上不可行的方案，报告已经发出去了，怎么挽回？这就是 Human-in-the-loop 要解决的问题。明天要学如何在 Agent 系统中加入人工审核和干预机制。
