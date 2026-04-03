# push --force 被拦了：Claude Code 的 Git 安全协议

## 1. 引言：AI 编码中 git 操作的特殊风险

当 AI agent 获得在终端执行命令的能力时，git 操作成为最高风险区域之一。与人类开发者不同，AI agent 不会因为"感觉不对"而犹豫，也不会因为某个操作"看起来危险"而主动停下来确认。它只会忠实执行指令，包括那些可能造成不可逆破坏的 git 命令。

Claude Code 通过一套 Git Safety Protocol 来应对这个问题。这套协议不是笼统的安全建议，而是嵌入在 system prompt 中的硬性规则，覆盖从 commit 创建到 PR 提交的完整工作流。本文从源码层面拆解这套协议的设计逻辑，分析每条规则背后的具体事故场景。

## 2. Git Safety Protocol 四条红线

Claude Code 的 Git Safety Protocol 以 NEVER 开头的四条禁令为核心，构成不可逾越的安全边界。

### 2.1 NEVER update the git config

> "NEVER update the git config" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

这条规则看似简单，实则保护的是身份完整性。Git config 中存储着 `user.name` 和 `user.email`，一旦被篡改，所有后续 commit 的作者信息都会出错。在团队协作场景中，这意味着代码归属混乱、审计追溯失效。AI agent 可能出于"规范化"的目的修改 config，但 Claude Code 明确禁止这种行为，无论动机如何。

### 2.2 NEVER run destructive git commands

> "NEVER run destructive git commands (push --force, reset --hard, checkout ., restore ., clean -f, branch -D) unless the user explicitly requests these actions. Taking unauthorized destructive actions is unhelpful and can result in lost work, so it's best to ONLY run these commands when given direct instructions" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

被列入黑名单的命令具有共同特征：它们都可能导致工作成果的不可逆丢失。`push --force` 会覆盖远程历史，`reset --hard` 会丢弃所有未提交的修改，`checkout .` 会还原所有本地文件变动，`clean -f` 会删除未跟踪的文件，`branch -D` 会强制删除分支。

这条规则并非绝对禁止，而是要求"explicitly requests"，用户必须明确发出指令。Claude Code 不会将模糊的意图解读为授权，比如用户说"把这个分支清理一下"，AI 不会自行决定执行 `branch -D`。

在另一份补充规则中，Claude Code 进一步要求在执行破坏性操作前先考虑替代方案：

> "Before running destructive operations (e.g., git reset --hard, git push --force, git checkout --), consider whether there is a safer alternative that achieves the same goal. Only use destructive operations when they are truly the best approach." [source: tool-description-bash-git-avoid-destructive-ops]

即使用户明确要求执行破坏性操作，Claude Code 仍需先评估是否存在更安全的路径。比如用户说"我想回到上一个版本"，`git revert` 可能比 `git reset --hard` 更合适，因为前者创建新 commit 而非丢弃历史。

### 2.3 NEVER skip hooks

> "NEVER skip hooks (--no-verify, --no-gpg-sign, etc) unless the user explicitly requests it" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

单独的规则文件对此做了更严格的表述：

> "Never skip hooks (--no-verify) or bypass signing (--no-gpg-sign, -c commit.gpgsign=false) unless the user has explicitly asked for it. If a hook fails, investigate and fix the underlying issue." [source: tool-description-bash-git-never-skip-hooks]

Hooks 是代码质量的自动关卡。pre-commit hooks 可能执行 lint 检查、格式化验证、敏感信息扫描；commit-msg hooks 可能校验提交信息格式；GPG signing 确保提交的来源可信。绕过这些检查意味着让可能有问题的代码进入仓库。

关键在最后一句话："If a hook fails, investigate and fix the underlying issue." Claude Code 被要求不是绕过失败的 hook，而是追根溯源修复问题。当代码无法通过质量关卡时，正确做法是修复代码，而非移除关卡。

### 2.4 NEVER run force push to main/master

> "NEVER run force push to main/master, warn the user if they request it" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

这是四条红线中唯一一条即使用户要求也不会直接执行的规则。其他三条禁令在用户"explicitly requests"时可以放行，但 force push main/master 即使在用户请求时也需要先发出警告。原因在于 main/master 分支是整个团队的共享基础，force push 会重写所有人的公共历史，影响范围远超单个开发者。

## 3. 提交流程详解

Claude Code 的 commit 创建流程被设计为一个三阶段的严格管线，每一阶段都有明确的输入、处理和输出。

### 3.1 阶段一：并行信息收集

> "1. Run the following bash commands in parallel, each using the Bash tool:
>   - Run a git status command to see all untracked files. IMPORTANT: Never use the -uall flag as it can cause memory issues on large repos.
>   - Run a git diff command to see both staged and unstaged changes that will be committed.
>   - Run a git log command to see recent commit messages, so that you can follow this repository's commit message style." [source: tool-description-bash-git-commit-and-pr-creation-instructions]

三个命令并行执行，分别回答三个问题：有哪些文件被修改或新增？具体的代码变更内容是什么？这个仓库的 commit message 风格是怎样的？

`git status` 的 `-uall` 限制值得注意。在大型仓库中，`-uall` 会列出每一个未跟踪文件，可能导致内存溢出。Claude Code 明确要求不使用此 flag，这是对实际工程环境的务实考量。

### 3.2 阶段二：分析与决策

> "2. Analyze all staged changes (both previously staged and newly added) and draft a commit message:
>   - Summarize the nature of the changes (eg. new feature, enhancement to an existing feature, bug fix, refactoring, test, docs, etc.). Ensure the message accurately reflects the changes and their purpose (i.e. 'add' means a wholly new feature, 'update' means an enhancement to an existing feature, 'fix' means a bug fix, etc.).
>   - Do not commit files that likely contain secrets (.env, credentials.json, etc). Warn the user if they specifically request to commit those files
>   - Draft a concise (1-2 sentences) commit message that focuses on the 'why' rather than the 'what'
>   - Ensure it accurately reflects the changes and their purpose" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

这个阶段要求 Claude Code 识别变更类型（新功能、增强、修复、重构等），检查是否包含敏感文件，起草 commit message。动词选择有严格映射关系："add"对应全新功能，"update"对应功能增强，"fix"对应缺陷修复。这不是随意选择措辞，而是遵循 conventional commits 的语义规范。

### 3.3 阶段三：执行与验证

> "3. Run the following commands in parallel:
>    - Add relevant untracked files to the staging area.
>    - Create the commit with a message ending with [Co-Authored-By line]
>    - Run git status after the commit completes to verify success.
>    Note: git status depends on the commit completing, so run it sequentially after the commit." [source: tool-description-bash-git-commit-and-pr-creation-instructions]

执行阶段的关键细节是 staging 策略：

> "When staging files, prefer adding specific files by name rather than using 'git add -A' or 'git add .', which can accidentally include sensitive files (.env, credentials) or large binaries" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

`git add -A` 和 `git add .` 是危险的快捷操作，它们会将当前目录下所有变更文件加入 staging area，包括 `.env`、credentials 文件或大型二进制文件。Claude Code 被要求按文件名逐个添加，步骤更多，但避免了意外提交。

验证步骤同样不可省略。commit 完成后必须运行 `git status` 确认操作成功，且由于 git status 依赖 commit 的结果，这两个命令必须串行执行。

## 4. 为什么 "Always create NEW commits rather than amending"

> "CRITICAL: Always create NEW commits rather than amending, unless the user explicitly requests a git amend. When a pre-commit hook fails, the commit did NOT happen — so --amend would modify the PREVIOUS commit, which may result in destroying work or losing previous changes. Instead, after hook failure, fix the issue, re-stage, and create a NEW commit" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

这条规则被标记为 CRITICAL，其重要性体现在对 git 内部机制的理解上。拆解 `git commit --amend` 的工作原理就能明白：

当 pre-commit hook 执行失败时，git 不会创建新的 commit 对象。当前 HEAD 指向的仍然是上一次的 commit。此时如果执行 `git commit --amend`，git 会将 staged 的修改合并到上一个 commit 中，替换掉原来的 commit 对象。

想象一个具体场景：开发者 A 创建了一个包含重要功能代码的 commit。Claude Code 尝试创建一个新的 commit，但 pre-commit hook 因为 lint 错误而失败。此时如果 Claude Code 使用 `--amend`，它不是重试自己的 commit（因为自己的 commit 根本没有创建成功），而是修改了开发者 A 的 commit。开发者 A 的原始代码可能被部分覆盖或完全丢失。

正确做法：修复 hook 报告的问题，重新 stage 文件，创建一个全新的 commit。这样既不会影响已有的 commit 历史，也保证了每次 commit 操作的原子性和可追溯性。

补充规则文件对此做了更简洁的表述：

> "Prefer to create a new commit rather than amending an existing commit." [source: tool-description-bash-git-prefer-new-commits]

## 5. Commit message 设计

### 5.1 聚焦 why 而非 what

> "Draft a concise (1-2 sentences) commit message that focuses on the 'why' rather than the 'what'" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

这是一个经过深思熟虑的设计选择。代码的 diff 本身已经展示了"what"，哪些文件被修改、哪些行被添加或删除。Commit message 的价值在于解释"why"，为什么要做这个修改，解决什么问题，达到什么目的。

### 5.2 变更类型的动词映射

Claude Code 被要求使用精确的动词来描述变更性质：

> "Ensure the message accurately reflects the changes and their purpose (i.e. 'add' means a wholly new feature, 'update' means an enhancement to an existing feature, 'fix' means a bug fix, etc.)" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

这种动词映射遵循 conventional commits 的精神，使得 git log 的读者能够快速判断每个 commit 的性质，而无需翻阅代码变更。

### 5.3 Co-Authored-By 标记

> "Create the commit with a message ending with: [Co-Authored-By line]" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

Co-Authored-By 是 git 社区公认的多作者标记方式。Claude Code 在 commit message 末尾添加此标记，明确声明这个 commit 是人机协作的结果。这对代码审计和贡献者归属都有实际意义。

## 6. PR 创建流程

### 6.1 理解分支全貌

> "1. Run the following bash commands in parallel using the Bash tool, in order to understand the current state of the branch since it diverged from the main branch:
>    - Run a git status command to see all untracked files (never use -uall flag)
>    - Run a git diff command to see both staged and unstaged changes that will be committed
>    - Check if the current branch tracks a remote branch and is up to date with the remote, so you know if you need to push to the remote
>    - Run a git log command and `git diff [base-branch]...HEAD` to understand the full commit history for the current branch (from the time it diverged from the base branch)" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

PR 创建的第一步比 commit 创建更全面，因为它需要理解整个分支的完整历史。四个并行命令分别收集：当前文件状态、代码变更、远程同步状态、完整提交历史。其中 `git diff [base-branch]...HEAD` 是关键，它展示了从分支分叉点到当前 HEAD 的所有变更，确保 PR 描述覆盖了分支上的全部改动。

### 6.2 分析全部 commit

> "2. Analyze all changes that will be included in the pull request, making sure to look at all relevant commits (NOT just the latest commit, but ALL commits that will be included in the pull request!!!), and draft a pull request title and summary:
>    - Keep the PR title short (under 70 characters)
>    - Use the description/body for details, not the title" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

这里用了三个感叹号来强调"ALL commits"。只看最新 commit 而忽略之前的 commit，会导致 PR 描述不完整甚至误导。一个分支可能包含多个 commit，每个 commit 解决不同的问题，PR 描述必须综合所有 commit 的信息。

### 6.3 执行创建

> "3. Run the following commands in parallel:
>    - Create new branch if needed
>    - Push to remote with -u flag if needed
>    - Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting." [source: tool-description-bash-git-commit-and-pr-creation-instructions]

PR 的创建使用 `gh` CLI 工具，这是 GitHub 官方命令行工具。文档明确要求"Use the gh command via the Bash tool for ALL GitHub-related tasks including working with issues, pull requests, checks, and releases." [source: tool-description-bash-git-commit-and-pr-creation-instructions]

PR body 包含两个标准部分：Summary（1-3 个要点概括）和 Test plan（测试检查清单）。创建完成后，Claude Code 被要求返回 PR URL："Return the PR URL when you're done, so the user can see it." [source: tool-description-bash-git-commit-and-pr-creation-instructions]

## 7. HEREDOC 格式的必要性

在 commit message 和 PR body 的传递中，Claude Code 被要求使用 HEREDOC 格式：

> "In order to ensure good formatting, ALWAYS pass the commit message via a HEREDOC, a la this example:
> ```
> git commit -m "$(cat <<'EOF'
>    Commit message here.
>
>    Co-Authored-By: Claude
>    EOF
>    )"
> ```" [source: tool-description-bash-git-commit-and-pr-creation-instructions]

PR body 同样使用 HEREDOC：

> "Create PR using gh pr create with the format below. Use a HEREDOC to pass the body to ensure correct formatting." [source: tool-description-bash-git-commit-and-pr-creation-instructions]

HEREDOC 的必要性来自两个方面。

一是 shell 中的引号处理。Commit message 和 PR body 通常包含多行文本、特殊字符、markdown 格式标记。如果直接用双引号包裹，shell 会对 `$`、反引号、`\` 等字符进行解释，导致内容被意外修改。HEREDOC 中的单引号（`<<'EOF'`）会禁用变量替换和命令替换，确保文本原样传递。

二是多行内容的完整性。HEREDOC 天然支持多行文本，无需使用 `\n` 转义序列或多个 `-m` flag。这保证了 commit message 和 PR body 的格式与预期一致，不会因为 shell 的换行处理而出现格式错误。

## 8. 与 Executing Actions with Care 的风险分级联动

Git Safety Protocol 不是孤立存在的，它是一个更广泛的风险管理框架的具体应用。在 "Executing actions with care" 系统提示中，Claude Code 的风险管控被统一表述为：

> "Carefully consider the reversibility and blast radius of actions. Generally you can freely take local, reversible actions like editing files or running tests. But for actions that are hard to reverse, affect shared systems beyond your local environment, or could otherwise be risky or destructive, check with the user before proceeding." [source: system-prompt-executing-actions-with-care]

这段话建立了风险分级的底层逻辑：本地可逆操作可以自由执行；难以逆转、影响范围超出本地的操作需要用户确认。

Git 操作正好横跨了这两个层级。`git add`、`git commit` 是本地可逆操作，属于低风险层级。以下操作被明确列为高风险：

> "Examples of the kind of risky actions that warrant user confirmation:
> - Destructive operations: deleting files/branches, dropping database tables, killing processes, rm -rf, overwriting uncommitted changes
> - Hard-to-reverse operations: force-pushing (can also overwrite upstream), git reset --hard, amending published commits, removing or downgrading packages/dependencies, modifying CI/CD pipelines
> - Actions visible to others or that affect shared state: pushing code, creating/closing/commenting on PRs or issues, sending messages (Slack, email, GitHub), posting to external services, modifying shared infrastructure or permissions" [source: system-prompt-executing-actions-with-care]

将这份风险清单与 Git Safety Protocol 对照，可以看到明确的映射关系：force push 同时出现在两个文档中，reset --hard 被同时列为破坏性命令和难以逆转的操作，pushing code 被归为"影响共享状态"的操作。

"Executing actions with care" 还建立了一条授权边界原则：

> "A user approving an action (like a git push) once does NOT mean that they approve it in all contexts, so unless actions are authorized in advance in durable instructions like CLAUDE.md files, always confirm first. Authorization stands for the scope specified, not beyond. Match the scope of your actions to what was actually requested." [source: system-prompt-executing-actions-with-care]

用户在场景 A 中授权的 push 操作，不会自动延伸到场景 B。每次执行高风险操作都需要独立的授权确认，且授权的范围不能超出用户的实际请求。

当遇到障碍时，Claude Code 被要求避免使用破坏性操作作为捷径：

> "When you encounter an obstacle, do not use destructive actions as a shortcut to simply make it go away. For instance, try to identify root causes and fix underlying issues rather than bypassing safety checks (e.g. --no-verify). If you discover unexpected state like unfamiliar files, branches, or configuration, investigate before deleting or overwriting, as it may represent the user's in-progress work." [source: system-prompt-executing-actions-with-care]

这条规则与 Git Safety Protocol 中的 "NEVER skip hooks" 形成呼应。当 pre-commit hook 失败时，正确做法是调查并修复根本问题，而不是用 `--no-verify` 绕过检查。当发现陌生的文件或分支时，应该先调查其来源和用途，而不是直接删除，因为这可能是用户正在进行的工作。

## 9. 这套协议防止的具体事故类型

Claude Code 的 Git Safety Protocol 和 PR 创建流程针对 AI agent 在 git 操作中可能引发的具体事故，逐条设置了防线：

**身份篡改**：禁止修改 git config，防止 agent 修改提交者身份信息，确保 commit 的作者归属准确。

**代码丢失**：禁止未经授权的破坏性命令（reset --hard、checkout .、clean -f 等），防止 agent 因追求效率而丢弃用户未提交的工作。要求在执行前先考虑更安全的替代方案，进一步降低风险。

**质量关卡失效**：禁止跳过 hooks 和绕过签名，防止 agent 为了完成提交而绕过项目设置的质量检查。当 hook 失败时，要求调查根本原因而非走捷径。

**共享历史重写**：在 force push main/master 时发出警告（即使用户要求），保护团队公共分支的完整性。

**Commit 历史污染**：要求创建新 commit 而非 amend，防止在 pre-commit hook 失败时意外修改他人的 commit。这条规则基于对 git 内部机制的精确理解：hook 失败意味着 commit 未创建，此时 amend 修改的是上一个 commit。

**敏感信息泄露**：要求按文件名 staging 而非使用 `git add -A`，防止 `.env`、credentials 等敏感文件被意外提交。

**PR 描述不完整**：要求分析分支上的全部 commit（而非仅最新 commit），确保 PR 描述准确反映完整的变更范围。

**格式丢失**：要求使用 HEREDOC 传递 commit message 和 PR body，防止 shell 引号处理导致的内容损坏。

**授权越界**：明确"一次授权不等于永久授权"的原则，防止 agent 将用户在特定场景下的许可延伸到其他场景。

每一条规则都对应着一个真实的事故场景，而不仅仅是理论上的风险假设。这套协议的设计思路是：在每一个可能出错的环节设置防线，让 AI agent 的 git 操作在效率上接近人类开发者，但在安全性上远超随意执行命令的自动化脚本。
