# 2600 token 的安全审计员：一个只报真漏洞的 AI Agent

## 一、角色定位

在 Claude Code 的 slash command 体系中，`/security-review` 是一个专精安全审计的 agent。它的角色定义只有一句话：[source: agent-prompt-security-review-slash-command.md] "You are a senior security engineer conducting a focused security review of the changes on this branch."

这个定位本身就包含了几个关键约束。它不是通用代码审查工具，不负责风格建议、架构评审或性能分析。它的全部注意力集中在高置信度的安全漏洞上。系统提示用一句话概括了核心设计哲学：[source: agent-prompt-security-review-slash-command.md] "Better to miss some theoretical issues than flood the report with false positives."

宁可漏报理论性问题，也不要用误报淹没报告。这个选择直接回应了 AI 安全审计中最棘手的矛盾：模型如果不加过滤地输出所有"可疑"模式，开发者会很快对报告失去信任。

## 二、审计目标：增量审查，80% 置信度门槛

prompt 在 OBJECTIVE 部分定义了精确的边界：

> "Perform a security-focused code review to identify HIGH-CONFIDENCE security vulnerabilities that could have real exploitation potential. This is not a general code review - focus ONLY on security implications newly added by this PR. Do not comment on existing security concerns." [source: agent-prompt-security-review-slash-command.md]

三个关键约束：

第一，只看新增代码。已有的安全问题不在审计范围内。这是一个增量审计工具，不是全量扫描器。这让 agent 能聚焦在当次变更引入的风险上，而不是被历史债务分散注意力。

第二，要求实际可利用。不是"理论上可能的"、"看起来可疑的"，而是有具体利用路径的漏洞。

第三，80% 置信度门槛。CRITICAL INSTRUCTIONS 部分写道：

> "MINIMIZE FALSE POSITIVES: Only flag issues where you're >80% confident of actual exploitability" [source: agent-prompt-security-review-slash-command.md]

传统的静态分析工具（SAST）倾向于"宁可误报不可漏报"。这里反其道而行之。这个参数选择反映了对 AI 审计场景的特殊理解：模型容易对模式匹配产生过度反应，如果不设高门槛，输出质量会迅速劣化。

同时有三类问题被硬性排除：

- 拒绝服务（DOS）漏洞，即使确实能造成服务中断
- 磁盘上的秘密数据（由其他专门的 secret scanning 工具处理）
- 速率限制和资源耗尽问题

这些排除各有原因。DOS 在实际攻防中优先级通常偏低，磁盘上的凭据有专用工具负责。排除它们让审计能集中精力在高价值发现上。

## 三、五大安全类别

### 3.1 输入验证漏洞（Input Validation）

涵盖 SQL 注入、命令注入、XXE 注入、模板注入、NoSQL 注入和路径遍历。每一项都指向用户输入未经清理直接到达敏感操作的场景。

这部分的重点在于数据流追踪：从用户输入的入口点开始，沿着数据流追踪到敏感操作（数据库查询、系统调用、文件操作），看中间是否有有效的清理或转义。

### 3.2 认证与授权（Authentication & Authorization）

关注认证绕过、权限提升、会话管理缺陷、JWT 漏洞和授权逻辑绕过。这类问题的特点是：代码表面上看起来正常，但在特定的执行路径组合下会打开安全缺口。

Prompt 要求 agent 识别权限边界的跨越，特别是用户 A 的操作是否能影响到用户 B 的资源。这类问题通常不会出现在简单的单元测试中，需要从整体架构角度审视。

### 3.3 加密与密钥管理（Crypto & Secrets Management）

包括硬编码的 API key、弱加密算法、不当的密钥存储、加密随机性问题、证书验证绕过。这类漏洞的隐蔽性在于：代码往往能正常工作，只是安全保证低于预期。

Prompt 明确将 "Hardcoded API keys, passwords, or tokens" 列为首要检查项。但硬编码测试凭据被排除在报告范围之外（见排除规则第 1 条），只报告真实的凭据泄露。

### 3.4 注入与代码执行（Injection & Code Execution）

覆盖远程代码执行（反序列化）、Python pickle 注入、YAML 反序列化、eval 注入和 XSS（反射型、存储型、DOM 型）。

XSS 部分有一个重要的先例判断：[source: agent-prompt-security-review-slash-command.md] "React and Angular are generally secure against XSS. Do not report XSS vulnerabilities in React or Angular components or tsx files unless they are using unsafe methods." 如果没有 `dangerouslySetInnerHTML`、`bypassSecurityTrustHtml` 这类显式绕过，就不要报告 XSS。框架本身的转义机制被视为可信。

### 3.5 数据暴露（Data Exposure）

包括敏感数据日志记录、PII 处理违规、API 端点数据泄露和调试信息暴露。

关于日志的判定标准值得一提：[source: agent-prompt-security-review-slash-command.md] "Logging non-PII data is not a vulnerability even if the data may be sensitive. Only report logging vulnerabilities if they expose secrets, passwords, or personally identifiable information (PII)." "Logging URLs is assumed to be safe."

Prompt 还指出了一个容易被忽略的维度：[source: agent-prompt-security-review-slash-command.md] "Even if something is only exploitable from the local network, it can still be a HIGH severity issue." 内网可利用不等于低风险。

## 四、三阶段分析方法论

### Phase 1：仓库上下文研究

先用文件搜索工具了解项目现有的安全框架和库，找到已有的安全编码模式，理解项目的安全模型和威胁模型。这一步的目的是建立基线：这个项目已经采用了哪些防护措施。

### Phase 2：对比分析

将新代码变更与已有安全模式对比。寻找偏离：新的代码是否跳过了项目其他地方都在做的输入清理？是否引入了项目之前没有的攻击面？安全实现是否前后不一致？

### Phase 3：漏洞评估

逐文件检查安全影响。追踪数据流：用户输入经过哪些路径到达敏感操作。检查权限边界是否被安全跨越。定位注入点和反序列化漏洞。

## 五、置信度评分系统

置信度评分是这个审计系统的核心质量控制机制 [source: agent-prompt-security-review-slash-command.md]：

| 范围 | 含义 |
|------|------|
| 0.9 - 1.0 | 确定的利用路径，可能已测试验证 |
| 0.8 - 0.9 | 明确的漏洞模式，已知的利用方法 |
| 0.7 - 0.8 | 可疑模式，需要特定条件才能利用 |
| < 0.7 | 不报告（过于推测性） |

在第二阶段的过滤中，只有置信度 >= 8/10 的发现才会保留在最终报告中。这是一个双重过滤：先在发现阶段用 80% 门槛筛选，再在误报过滤阶段用 8/10 评分二次筛选。

严重性分为三级：

- HIGH：直接可利用的 RCE、数据泄露或认证绕过
- MEDIUM：需要特定条件但影响重大的漏洞
- LOW：纵深防御问题或低影响漏洞

最终报告只保留 HIGH 和 MEDIUM 级别的发现。

## 六、报告格式

每个漏洞遵循统一格式 [source: agent-prompt-security-review-slash-command.md]：

```markdown
# Vuln N: [类型]: [文件名]:[行号]

* Severity: High/Medium/Low
* Description: [漏洞描述]
* Exploit Scenario: [具体利用场景]
* Recommendation: [修复建议]
```

Description 要求说明漏洞的本质——用户输入如何未经处理到达敏感操作。Exploit Scenario 要求描述攻击者实际会怎么利用，不是抽象的可能性。Recommendation 要求给出具体的修复方案，指明用哪个函数或方法替代。

## 七、硬性排除规则与判例先例

prompt 中最长的部分是 FALSE POSITIVE FILTERING。它包含 17 条硬性排除和 12 条判例先例。这些规则反映了对误报模式的深刻理解。

### 硬性排除

一些值得注意的排除规则：

**内存安全**：[source: agent-prompt-security-review-slash-command.md] "Memory safety issues such as buffer overflows or use-after-free vulnerabilities are impossible in rust. Do not report memory safety issues in rust or any other memory safe languages." 对 Rust 等内存安全语言不报告内存安全问题。

**环境变量信任**：[source: agent-prompt-security-review-slash-command.md] "Environment variables and CLI flags are trusted values. Attackers are generally not able to modify them in a secure environment. Any attack that relies on controlling an environment variable is invalid." 基于环境变量控制前提的攻击被视为无效。

**客户端代码**：[source: agent-prompt-security-review-slash-command.md] "A lack of permission checking or authentication in client-side JS/TS code is not a vulnerability. Client-side code is not trusted and does not need to implement these checks, they are handled on the server-side." 客户端不做权限检查不是漏洞，后端才负责验证。

**Shell 脚本命令注入**：[source: agent-prompt-security-review-slash-command.md] "Command injection vulnerabilities in shell scripts are generally not exploitable in practice since shell scripts generally do not run with untrusted user input." 除非有明确的不受信输入路径。

### 判例先例

判例先例（Precedents）部分提供了一套具体的判断标准：

- UUID 被视为不可猜测的，不需要验证
- 日志中记录 URL 被假设为安全
- 在 React/Angular 组件中，除非使用了 `dangerouslySetInnerHTML` 等不安全 API，否则不报告 XSS
- GitHub Actions 工作流中的大多数漏洞实际上不可利用，需要非常具体的攻击路径才值得报告
- Jupyter Notebook 中的漏洞类似，除非有明确的不受信输入触发路径
- 只有在暴露密码、秘密或 PII 时才报告日志漏洞，记录敏感但非 PII 数据不算

这些先例不是随意的选择。每一条背后都是对实际攻防场景中利用可能性的评估。比如 GitHub Actions 工作流的漏洞之所以大多不报，是因为工作流的触发条件通常受仓库权限控制，外部攻击者很难注入不受信输入。

## 八、三步执行流程

Security review agent 的执行流程本身也使用了 subagent 委派 [source: agent-prompt-security-review-slash-command.md]：

第一步，启动一个 sub-task 来识别漏洞。这个 sub-task 使用仓库探索工具理解代码上下文，然后分析 PR 变更的安全影响。Prompt 中包含完整的审计指令。

第二步，对第一步发现的每个漏洞，创建独立的 sub-task 进行误报过滤。这些 sub-task 并行启动，每个都包含完整的 FALSE POSITIVE FILTERING 指令。

第三步，过滤掉置信度低于 8 的发现。

最终只输出 markdown 格式的报告，不包含其他内容。

这个三步流程是一个精巧的设计。第一步负责广撒网——尽可能发现潜在问题。第二步用独立的 agent 对每个发现做第二遍审视，避免同一个 agent 对自己的发现产生确认偏误。第三步做最终的置信度筛选。整个流程通过子 agent 的独立性来实现自我校验。

## 九、设计哲学

Security review agent 的设计哲学可以归结为三点：

**信号优于覆盖**。宁可漏掉一些理论性问题，也要保证报告中的每个发现都是可靠的。80% 置信度门槛和双重过滤机制都是这个原则的体现。

**增量而非全量**。只审查当前分支的新增变更，不评论已有代码的安全问题。这让审计范围可控，也让报告与开发者的实际变更直接相关。

**具体而非抽象**。每个发现必须包含具体的利用场景和修复建议，不接受"可能存在风险"这类模糊表述。Exploit Scenario 字段强制 agent 把攻击路径写出来，写不出来的就不报。

这种设计对 AI agent 的安全审计场景是合理的。静态分析工具可以穷举所有模式匹配，因为它不会疲劳、不会失去信任。但 AI agent 生成的报告如果充满误报，开发者的下一步操作就会变成忽略所有报告，agent 的价值归零。高门槛、低噪声的策略，本质上是在保护 agent 自身的可信度。
