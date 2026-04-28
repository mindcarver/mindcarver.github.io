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

第一，职责边界要清晰。每个专家 Agent 只负责一个明确领域的工作，不做跨界的事。Industry Research Agent 只做行业分析，不碰岗位拆解和流程分析。边界清晰能避免 Agent 之间的职责重叠和冲突。

第二，工具集要精准。每个专家 Agent 只配备完成自己职责所需的工具。Industry Research Agent 配备 search_web 和 read_document，不需要 calculate_roi 和 send_email_draft。工具少了，选择更准确，误用概率更低。

第三，Prompt 要专业。每个专家 Agent 的 System Prompt 应该包含：角色定义（你是行业研究专家）、专业背景（你熟悉产业链分析、市场研究方法论）、输出规范（输出格式、质量标准）、边界说明（你只负责行业概览，岗位和流程不是你的职责）。

专家 Agent 的"专"体现在三个方面：专用的知识（通过 Prompt 注入领域知识）、专用的工具（只配备相关工具）、专用的输出格式（结构化的专业输出）。

### Agent 分工

分工是 Multi-Agent 设计中最考验功底的环节。分得好，各 Agent 高效协作；分得不好，要么职责重叠导致冲突，要么职责遗漏导致结果不完整。

分工的方法论可以总结为三步。

第一步，按知识领域分。分析任务涉及哪些不同的知识领域，每个领域分配一个 Agent。行业研究涉及"行业知识"、"岗位知识"、"流程知识"、"AI 方案知识"、"风险评估知识"五个领域，所以设计五个专家 Agent。这条原则叫"知识边界决定 Agent 边界"。

第二步，按工作流程分。在知识领域的基础上，根据工作流程的先后顺序调整分工。如果两个领域的工作高度耦合（必须同步进行），可以考虑合并为一个 Agent。如果一个大领域可以拆成独立的工作流，可以考虑拆成两个 Agent。

第三步，按工具集分。检查每个 Agent 的工具集是否有大量重叠。如果两个 Agent 的工具集 80% 以上重叠，说明它们的职责边界可能不够清晰，考虑合并。

以行业研究系统为例，最终的分工方案是：

Orchestrator Agent：总协调，不参与具体分析。负责分解任务、调度执行、整合结果。
Industry Research Agent：负责行业概览，包括产业链、市场规模、趋势分析。
Role Analyst Agent：负责岗位拆解，包括岗位识别、任务分析、痛点挖掘。
Process Analyst Agent：负责流程拆解，包括流程识别、步骤分析、效率评估。
AI Solution Agent：负责 AI 方案设计，包括场景识别、技术选型、ROI 估算。
Risk Review Agent：负责风险审查，包括可行性评估、风险识别、合规检查。

这 6 个 Agent 的知识领域不重叠、工具集不重叠、工作流程有清晰的先后依赖关系，是一个合理的分工方案。

### Agent 通信

Agent 之间的通信是 Multi-Agent 系统的神经系统。通信设计的好坏直接影响系统的可靠性和效率。

通信方式主要有三种。

第一种是直接消息传递。Agent A 把结果直接发给 Agent B。像同事之间发微信，点对点。这种方式简单直接，但缺点是 Agent 之间产生了耦合——Agent B 需要知道 Agent A 的存在和接口。

第二种是共享黑板模式。所有 Agent 把中间结果写入一个共享的存储空间（Blackboard），其他 Agent 从 Blackboard 读取需要的信息。像公司内部的共享文档，大家都能看到。这种方式解耦了 Agent 之间的关系，但 Blackboard 的数据管理变得复杂。

第三种是主控转发模式。所有 Agent 只和 Orchestrator 通信。Agent A 把结果发给 Orchestrator，Orchestrator 转发给 Agent B。像公司里的项目经理，所有信息都通过他中转。这种方式最可控，Orchestrator 掌握全局信息，但 Orchestrator 可能成为瓶颈。

对于行业研究系统，我推荐主控转发模式。原因有三：Orchestrator 需要掌握全局进度来做调度决策；通过 Orchestrator 中转可以在传递过程中做数据清洗和格式转换；如果某个 Agent 的输出有问题，Orchestrator 可以在转发前拦截。

通信数据格式需要标准化。不管用什么通信方式，Agent 之间传递的数据格式必须统一。建议定义一个通用的消息格式：

- sender：发送方 Agent 名称
- receiver：接收方 Agent 名称
- message_type：消息类型（task_assignment / result / error / query）
- content：消息内容（JSON 格式）
- timestamp：发送时间
- task_id：关联的任务 ID

这个格式既是通信协议，也是审计日志的数据来源。

### 中间结果整合

每个专家 Agent 完成自己的任务后，Orchestrator 需要把所有中间结果整合成最终交付物。这不是简单的拼接，而是一个需要判断力的过程。

整合工作包含三个步骤。

第一步，质量检查。逐个检查每个 Agent 的输出是否完整、格式是否正确、内容是否合理。如果 Industry Research Agent 返回的行业概览只有两行描述，明显不达标，需要让它补充。

第二步，一致性校验。检查不同 Agent 的输出之间是否存在矛盾。比如 Industry Research Agent 说这个行业有 5 个核心环节，但 Role Analyst Agent 拆出了 8 个不同环节的岗位，两者对"核心环节"的理解不一致。Orchestrator 需要发现这种矛盾并协调解决。

第三步，格式统一。不同 Agent 的输出格式可能有差异（字段命名不同、结构层级不同），需要统一成最终报告的格式。

整合过程中最常见的坑是"信息丢失"。Agent A 输出了很详细的分析，但在整合时被过度压缩，关键信息丢失了。建议在整合时保留每个 Agent 的原始输出作为附件，整合后的报告是"精炼版"，需要看细节时可以查阅原始输出。

### 冲突处理

Agent 之间的冲突在 Multi-Agent 系统中难以避免。常见的冲突类型有三种。

第一种是结论冲突。两个 Agent 对同一件事给出了不同的判断。比如 AI Solution Agent 认为某个场景的 ROI 很高，但 Risk Review Agent 认为风险很大不建议做。这种冲突需要 Orchestrator 做仲裁。仲裁方法包括：综合两方意见给出折中方案、让两个 Agent 分别提供更详细的论据后再次评估、标记为争议点由人工决定。

第二种是数据冲突。两个 Agent 使用了不同来源的数据，数据之间有出入。比如 Industry Research Agent 查到的市场规模是 500 亿，Role Analyst Agent 查到的是 380 亿。这种冲突需要回溯数据来源，判断哪个来源更可靠，然后统一数据口径。

第三种是资源冲突。两个 Agent 同时需要调用同一个有并发限制的工具或 API。比如都调用搜索工具，但 API 有每分钟调用次数限制。这种冲突需要在调度层面解决，通过任务排队或错峰执行来避免。

冲突处理的通用原则是"冲突不上交给用户"。Orchestrator 应该有能力解决大部分冲突，只在无法判断时才请求人工介入。用户看到的应该是整合后的统一结论，不是 Agent 之间的分歧。

### 多 Agent 适用边界

Multi-Agent 不是万能的。以下场景不适合用 Multi-Agent。

第一，任务简单，单 Agent 就能搞定。比如"分析某个岗位的 AI 机会"，一个 Agent 就够了，不需要拆成多个。

第二，子任务之间高度耦合，无法独立执行。如果 Agent A 的每一步都需要 Agent B 的实时反馈，那分开两个 Agent 反而增加了通信成本。

第三，对延迟敏感的场景。Multi-Agent 的通信和协调需要额外时间。如果用户要求秒级响应，单 Agent 更合适。

第四，资源受限的场景。Multi-Agent 的 Token 消耗和计算成本更高。如果预算有限，优先考虑单 Agent 方案。

判断是否需要 Multi-Agent 的一个简单标准：画出任务的所有子步骤，如果每个子步骤都能独立完成、有明确的输入输出、不需要频繁回溯其他步骤的结果，就可以考虑 Multi-Agent。否则，单 Agent 更合适。

---

## 概念关系图

```
Multi-Agent 行业研究系统架构

用户输入："分析半导体行业的 AI 应用机会"
  |
  v
Orchestrator Agent（主控）
  |-- 任务分解
  |-- 调度决策
  |-- 结果整合
  |-- 异常处理
  |
  +----> Industry Research Agent
  |      |-- 工具：search_web, read_document
  |      |-- 输入：行业名称
  |      |-- 输出：行业概览 JSON
  |
  +----> Role Analyst Agent
  |      |-- 工具：search_web, query_database
  |      |-- 输入：行业概览
  |      |-- 输出：岗位分析 JSON Array
  |
  +----> Process Analyst Agent
  |      |-- 工具：search_web, read_document
  |      |-- 输入：行业概览 + 岗位分析
  |      |-- 输出：流程分析 JSON Array
  |
  +----> AI Solution Agent
  |      |-- 工具：search_web, calculate_roi, generate_markdown
  |      |-- 输入：行业概览 + 岗位分析 + 流程分析
  |      |-- 输出：AI 方案 JSON Array
  |
  +----> Risk Review Agent
         |-- 工具：read_document, generate_markdown
         |-- 输入：以上所有分析结果
         |-- 输出：风险审查 JSON

执行顺序：
  Phase 1（并行）：Industry Research Agent
  Phase 2（并行）：Role Analyst Agent + Process Analyst Agent
  Phase 3：AI Solution Agent
  Phase 4：Risk Review Agent
  Phase 5：Orchestrator 整合 -> 最终报告
```

```
通信模式对比

直接消息：      Agent A -----> Agent B
                 （简单但耦合）

共享黑板：      Agent A ---> [Blackboard] <--- Agent B
                 （解耦但管理复杂）

主控转发：      Agent A ---> Orchestrator ---> Agent B
                 （可控但 Orchestrator 是瓶颈）
```

---

## 实战分析

### 实战任务：设计 6 个 Agent

指南要求设计 6 个 Agent，定义每个 Agent 的职责、输入和输出。下面逐一展开设计思路。

**Orchestrator Agent**

职责：接收用户请求，分解为子任务，调度专家 Agent 执行，收集和整合结果，输出最终报告。

输入：用户请求（行业名称 + 可选的分析要求）。

输出：完整的行业 AI 机会分析报告（Markdown 格式）。

工具：不需要专业工具，但需要"调用其他 Agent"的能力。这个能力可以封装为一个特殊工具，比如 call_agent，参数是 Agent 名称和输入数据。

设计要点：Orchestrator 的 System Prompt 需要包含完整的任务分解策略和调度逻辑。它要知道什么任务应该分给哪个 Agent、执行顺序怎么安排、结果怎么整合。

**Industry Research Agent**

职责：分析给定行业的产业链结构、市场规模、发展趋势、主要玩家、技术特征。

输入：行业名称 + 分析范围（可选）。

输出：结构化的行业概览对象，包含产业链（上中下游描述）、市场规模（数据和来源）、发展趋势（3-5 条趋势判断）、主要玩家（头部企业列表）、技术特征（核心技术栈）。

工具：search_web（搜索行业信息）、read_document（阅读行业报告）。

设计要点：这个 Agent 的 Prompt 需要注入行业分析的方法论知识，比如波特五力模型、价值链分析法。不要求它应用这些模型做正式分析，但要让它知道从哪些维度去收集和整理信息。

**Role Analyst Agent**

职责：基于行业概览，识别关键岗位，分析每个岗位的任务结构、痛点、AI 机会。

输入：行业概览（来自 Industry Research Agent）。

输出：岗位分析数组，每个岗位包含名称、所属环节、核心职责、高频任务、低价值高重复任务、使用工具和系统、痛点。

工具：search_web（搜索岗位相关信息）、query_database（查询岗位数据库）。

设计要点：这个 Agent 需要知道怎么从行业概览中推断关键岗位。它的推理逻辑是：行业的每个核心环节对应一批核心岗位。所以它首先从行业概览中提取核心环节，然后针对每个环节推断岗位。

**Process Analyst Agent**

职责：基于行业概览和岗位分析，识别核心业务流程，拆解每个流程的步骤和效率瓶颈。

输入：行业概览（来自 Industry Research Agent）+ 岗位分析（来自 Role Analyst Agent）。

输出：流程分析数组，每个流程包含名称、触发条件、参与角色、步骤列表（含耗时估算）、系统依赖、人工判断点、错误率、AI 优化点。

工具：search_web（搜索流程相关信息）、read_document（阅读流程文档）。

设计要点：流程拆解的粒度控制是这个 Agent 的难点。Prompt 中需要明确粒度要求：每个流程 5-10 步，太多太细就超出范围。

**AI Solution Agent**

职责：基于前面的分析结果，设计 AI 应用方案，估算 ROI，给出实施建议。

输入：行业概览 + 岗位分析 + 流程分析。

输出：AI 方案数组，每个方案包含场景名称、应用类型（Copilot/Agent/Workflow/RAG）、技术方案概述、预估 ROI、实施难度、实施周期、数据需求、风险提示。

工具：search_web（搜索 AI 应用案例）、calculate_roi（计算 ROI）、generate_markdown（生成方案描述）。

设计要点：这个 Agent 需要同时具备 AI 技术知识和行业理解。它的 Prompt 要包含常见 AI 应用类型的能力边界描述，让它不会推荐超出当前技术能力的方案。

**Risk Review Agent**

职责：审查所有 AI 方案的可行性、风险、合规性，给出审查意见。

输入：所有分析结果（行业概览、岗位分析、流程分析、AI 方案）。

输出：风险审查报告，包含每个方案的审查意见（通过/有条件通过/不通过）、风险列表（风险类型、严重程度、缓解措施）、合规性检查结果、综合建议。

工具：read_document（阅读政策和规范文档）、generate_markdown（生成审查报告）。

设计要点：这个 Agent 需要有"质疑"的思维模式。它的角色不是肯定前面的方案，而是挑毛病。Prompt 中要强调"审查者"的角色定位，让它保持独立判断，不要因为前面的分析看起来专业就默认认可。

### 方法论总结

设计 Multi-Agent 系统的方法论可以总结为五步。

第一步，明确总任务。清楚定义系统要完成什么，输入是什么，输出是什么。

第二步，分析知识领域。总任务涉及哪些不同的知识领域？这些领域之间边界是否清晰？

第三步，划分 Agent。按知识领域划分 Agent，每个 Agent 一个领域。检查工具集是否重叠，检查工作流程是否有依赖。

第四步，设计通信协议。Agent 之间怎么通信？数据格式是什么？通过 Orchestrator 中转还是直接通信？

第五步，设计整合策略。Orchestrator 怎么整合各 Agent 的结果？质量标准是什么？冲突怎么处理？

---

## 当日产物说明

### 《Multi-Agent 架构图》

这张图展示 6 个 Agent 的关系结构，包括 Orchestrator 和 5 个专家 Agent。标注每个 Agent 的职责、工具、输入来源和输出方向。标注执行阶段的先后顺序。

质量标准：看图就能理解整个系统的运作方式。Agent 之间的数据流向清晰可追溯。

### 《Agent 职责表》

一张汇总表格，每个 Agent 一行，列包括：名称、职责描述、输入、输出、工具集、依赖（依赖哪些 Agent 的输出）、执行阶段。

质量标准：任何一行拿出来都能独立理解该 Agent 的全貌。没有模糊描述。

### 《Agent 输入输出协议》

定义 Agent 之间通信的消息格式，包括字段名、类型、说明。同时定义每个 Agent 的输入 JSON Schema 和输出 JSON Schema。

质量标准：协议定义足以让两个独立开发者分别实现 Orchestrator 和专家 Agent，且两者能正确通信。

---

## 常见误区与避坑

### 误区一：Agent 越多越好

看到一个大任务，恨不得拆成 20 个 Agent。每个 Agent 做一小件事，看起来很精细。但 Agent 数量每增加一个，系统的通信复杂度、调试难度和 Token 消耗都大幅增加。6 个 Agent 是行业研究系统的一个合理上限。如果发现需要更多 Agent，先考虑是否有些职责可以合并。

### 误区二：Orchestrator 也参与分析

Orchestrator 的角色是协调者，不是分析者。如果 Orchestrator 同时做分析和协调，它会因为上下文太长而降低决策质量。让 Orchestrator 保持"管理者"的纯粹角色，分析工作交给专家 Agent。

### 误区三：所有 Agent 并行执行

并行执行确实能节省时间，但不是所有 Agent 都能并行。Role Analyst Agent 需要 Industry Research Agent 的行业概览作为输入，不能并行。正确的做法是分析依赖关系，把没有依赖关系的 Agent 并行执行，有依赖关系的串行执行。

### 误区四：忽视 Agent 间的数据格式兼容性

每个 Agent 独立设计输出格式时，可能出现字段名不一致、结构层级不匹配的问题。比如 Industry Research Agent 输出 industry_chain 字段，但 Role Analyst Agent 期望接收 value_chain 字段。这种不兼容会导致整合时出错。建议在设计阶段统一所有 Agent 的字段命名规范。

### 误区五：不处理 Agent 执行失败

Multi-Agent 系统中，一个 Agent 失败可能导致整个任务失败。如果 Role Analyst Agent 超时了，后面的 Agent 都拿不到岗位分析数据。Orchestrator 必须有处理单个 Agent 失败的策略：重试、降级（用简化版结果替代）、跳过（如果岗位分析不是必需的）、中止。

---

## 延伸思考

今天的 Multi-Agent 架构是 Week 4 整体设计的核心。把 Day 22 的 Workflow、Day 23 的 Agent 认知、Day 24 的 Tool Calling 串起来，加上今天的 Multi-Agent 架构，一个完整的行业研究系统已经初具雏形。

明天 Day 26 的 Human-in-the-loop 会讨论在这个 Multi-Agent 系统中，哪些节点需要人工介入。可以预见的是：最终报告生成后需要人工审核，AI 方案推荐时某些高风险方案需要人工确认，工具调用中涉及外部动作（发邮件、创建任务）时需要人工审批。

Day 27 的安全专题会讨论 Multi-Agent 系统面临的独特安全挑战。比如攻击者可能通过 Prompt Injection 控制某个专家 Agent，然后通过 Agent 间的通信影响其他 Agent。这种"横向攻击"是 Multi-Agent 系统特有的风险。

Day 28 的复盘会把整个系统组装起来。到那时，你应该能设计并实现一个包含 Workflow 控制、6 个专业 Agent、完整工具集、人工审核节点、安全防护机制的多 Agent 行业研究系统。

从更宏观的视角看，Multi-Agent 是当前 AI 应用架构的前沿方向。LangGraph、CrewAI、AutoGen 等框架都在解决 Multi-Agent 的编排问题。今天学的分工、通信、冲突处理等概念，是理解和使用这些框架的理论基础。掌握了底层原理，后面学任何框架都能快速上手。

---

## 自测问题

1. Multi-Agent 解决了单 Agent 的哪三个核心痛点？

2. 什么情况下应该用 Multi-Agent，什么情况下不应该？

3. Orchestrator Agent 的四个职责是什么？为什么它不应该参与具体分析？

4. 专家 Agent 的"专"体现在哪三个方面？

5. Agent 分工的三步方法论是什么？用自己的话描述每一步。

6. Agent 之间的三种通信方式是什么？各自的优缺点？

7. 中间结果整合的三个步骤是什么？整合过程中最常见的坑是什么？

8. Agent 之间的冲突有哪三种类型？各举一个具体例子。

9. 设计 Multi-Agent 系统的五步方法论是什么？

10. 为什么不能让所有 Agent 并行执行？应该怎么安排执行顺序？

---

## 关键词

- **Multi-Agent（多智能体）**：由多个专精不同领域的 Agent 协作完成复杂任务的系统架构
- **Orchestrator Agent（主控 Agent）**：负责任务分解、调度、整合和异常处理的管理型 Agent
- **专家 Agent**：专精某一领域的 Agent，配备专用工具和 Prompt
- **Agent 分工**：按知识领域、工作流程和工具集将任务划分给不同 Agent 的设计过程
- **Agent 通信**：Agent 之间传递信息和结果的机制，包括直接消息、共享黑板和主控转发
- **中间结果整合**：Orchestrator 收集各 Agent 输出并进行质量检查、一致性校验和格式统一的过程
- **冲突处理**：解决 Agent 之间结论冲突、数据冲突和资源冲突的策略
- **知识边界**：决定 Agent 职责范围的专业领域划分
- **执行阶段**：根据 Agent 之间的依赖关系安排的执行时序
- **横向攻击**：攻击者通过控制一个 Agent 向其他 Agent 传播恶意指令的安全威胁
