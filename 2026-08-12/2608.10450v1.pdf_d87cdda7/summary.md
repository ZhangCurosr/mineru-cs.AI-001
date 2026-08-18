---
title: "Persistent Recursive Worlds Enable Autonomous Software Evolution"
source: https://arxiv.org/pdf/2608.10450v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 05:23:10"
field: "AI-assisted software engineering"
keywords: ["persistent recursive world", "long-horizon software development", "coding agent", "software evolution", "greenfield compiler construction", "scientific software redevelopment"]
innovations: ["将软件项目版本历史作为持久化对象而非 agent 会话，实现跨有限生命周期 agent 的连续开发", "recursive delegation 与 validation-gated acceptance 的分离设计，使局部探索与全局版本推进解耦", "从空仓库在 120+ 小时/$44 内自主构建通过完整 c-testsuite 的 Rust C 编译器，并在跨模型替换下保持 continuation"]
benchmarks: ["c-testsuite", "LLVM test programs", "Csmith", "MESA numerical workloads"]
---

# 论文速读：Persistent Recursive Worlds Enable Autonomous Software Evolution

## 一句话总结
论文提出 **EvoX Genesis** 框架，将软件项目本身建模为"持久化递归世界"，使长期软件开发过程跨越有限生命周期智能体的更替而持续演进；从空仓库用 DeepSeek V4 Flash 构建出通过完整 c-testsuite 的 Rust C 编译器（120+ 小时、$44 token 费用），并实现 MESA 科学代码的 Fortran→Rust 重写且保留验证数值行为。

## 研究问题与动机
- **长程软件开发中的连续性难题**：复杂软件系统的生命周期远超单个 coding agent 的存续时间，现有方案多通过扩展 agent 自身的上下文窗口、记忆机制或共享 scratchpad 来维持连续性，本质上是在延长 agent 进程而非让项目本身持久化。
- **有限 agent 与长期项目的解耦**：当活跃 agent 终止后，什么应当被继承——是 agent 的会话状态还是项目的已接受版本历史？论文主张后者才是正确的持久化对象。
- **递归委托与版本历史的管理**：如何将大规模仓库级任务分解到局部路径上，并通过验证门控决定哪些候选变更进入共享历史，同时避免技术债和接口漂移累积导致的退化。
- **科学软件重开发的数值保真需求**：除了功能正确性，AI 辅助重写还需要保留经审计的数值行为，这是现有 coding agent benchmark 未充分覆盖的场景。

## 核心贡献（创新点）
1. **提出"持久化递归世界"的形式化定义**：用最小化的 `version–path` 二元组 $w = (\nu, p)$ 刻画局部软件世界，区分本地 agent 行为、递归委托与已接受的软件事件，使"什么必须持久化"得到清晰的形式化回答。
2. **实现 EvoX Genesis 系统架构**：有限生命周期的 manager/executor agent 在 path-scoped 上下文中工作，递归委托 $(\nu, p) \to (\nu, q)$ 在不推进版本的前提下转移工作，仅通过 parent-mediated acceptance 使候选变更进入 Git commit 层级的持久历史。
3. **完成大规模 greenfield 编译器构建实验**：从零仓库起步，123.4 小时内积累 1,019 个 agent episode，产出 248,989 行的 Rust C 编译器，完整通过 c-testsuite（220/220）及大部分 LLVM/Csmith 测试，token 费用仅 $44.38。
4. **证明跨基础模型替换的 continuation 能力**：同一 GLM 5.2 生成的编译器在分别用 GLM 5.2 和 DeepSeek V4 Flash 继续开发时均能保持有效进展，验证了项目中心式持久化的模型无关性。
5. **展示科学代码重开发的数值保真**：将 139,414 行 Fortran MESA 模块重写为 89,946 行 Rust，在六项独立 workload 上中位数加速 1.55–6.87×，EOS lookup 和 Newton solve 达到 bit-exact 级别。

## 方法详解
- **局部软件世界的形式化**：$w = (\nu, p)$，其中 $\nu$ 为已接受的软件版本（含源码、context、约束、验证结果、provenance），$p$ 为仓库相对路径，agent 从此坐标出发获得 scope 内的修改权限与上下文。
- **有限生命周期 agent**：$A_i((\nu, p), g_i) \to \Delta_i$，agent 在执行中可能经过多次 model–tool turn，但其私有的对话与 scratch 状态不继承给后续 agent；后续工作从已接受版本与路径重新实例化。
- **递归委托（不推进版本）**：$(\nu, p) \to (\nu, q)$，父 agent 在同版本下于不同路径启子 agent，leaf executor 直接修改软件，root/中间 manager 负责任务分解与结果审查；委托本身不产生新 commit。
- **验证门控的接受事件**：$(\nu, p) \to (\nu', p')$，parent 根据 tests、constraints 与 integration evidence 决定 accept/reject/request-more；只有被接受的变更才推进版本历史，reject 的代码不留存但有用的失败信息可被显式存储后进入后续版本。
- **工程实现**：accepted changes 通过 Git commits 与受保护的 archive references 存储；proposed changes 隔离在 agent-specific branches 与 worktrees；directory-scoped nodes 从 version-controlled CONTEXT.md 组装 context。

## 实验与结果
| 实验 | 设置 | 关键结果 |
|---|---|---|
| **Formation（编译器从零构建）** | DeepSeek V4 Flash，150k token 压缩阈值 | 123.4 h，1,019 episodes，depth=5，248,989 行 / 750 文件，$44.38；c-testsuite 220/220，LLVM 32/36，Csmith 93/93，LZ4/SQLite 通过，2,904 Rust workspace 测试 |
| **Continuity（跨模型继续开发）** | GLM 5.2 vs DeepSeek V4 Flash，均从同一 GLM 生成的已完成 jcc 仓库出发 | GLM：98 agents，depth=4，LLVM 1,445/1,448；DeepSeek：178 agents，depth=8，LLVM 1,820/1,820；两者在相同已接受历史上继续演进，产出不同的 commit 历史与代码 churn |
| **Redevelopment（MESA Fortran→Rust）** | DeepSeek V4 Flash，139,414 行 Fortran → 89,946 行 Rust | 33.22 h，272 agents，$10.64；1,052 测试全通过；六项 workload 中 EOS lookup 与 Newton solve bit-exact，其余四组相对 checksum 差 $5.1\times10^{-15}$ 至 $3.1\times10^{-9}$；中位数加速 1.55–6.87× |

最强结果：编译器 formation 在 $44 费用下产出通过完整外部 c-testsuite 且规模近 25 万行的 C 编译器，展现了长程自主开发的经济性与可行性。

## 相关工作脉络
- **SWE-bench / SWE-agent / OpenHands**（Jimenez 等, 2024; Wang 等, 2025; Yang 等, 2024）：建立 issue-level 评估与通用软件 agent 平台，关注单 patch 修复；Genesis 聚焦多个 episode 间同一项目的连续可发展性。
- **NL2Repo-Bench / SWE-EVO / RoadmapBench**（Ding 等, 2025; Thai 等, 2025; Xu 等, 2026; Shastry 等, 2026）：将 horizon 扩展到 release/upgrade/序列变更；本文与之重叠于"长期"维度，但独特贡献是将持久化重心从 agent 转移到项目版本历史。
- **MetaGPT**（Hong 等, 2024）与多 agent 角色协调框架：通过 workflow 和共享 scratchpad 维持连续性；Genesis 不依赖持久共享 memory，而是依赖已接受 commit 作为共同继承物。
- **EvoGit**（Huang 等, 2025）：去中心化 Git 图上的多 agent 代码演化，无需中央协调或显式消息传递；Genesis 与其共享"版本历史即基础设施"理念，但额外引入 path-situated 委托与 parent-mediated acceptance 的门控机制。
- **FunSearch / AlphaEvolve / ERA**（Romera-Paredes 等, 2024; Novikov 等, 2025; Aygun 等, 2026）：基于 evaluator 的 program search 与进化；本文跟踪的是"单个项目的连续历史"而非候选程序种群。
- **MESA 科学软件工程**（Paxton 等, 2011）与 research-software stewardship（Barker 等, 2022）：强调数值行为保留与 FAIR 原则；Genesis 的 MESA 重开发实验回应了 O'Brien（2025）关于 AI 辅助改动可能威胁科学软件质量的担忧。

## 局限性与未来方向
- **因果分解尚未完成**：递归机制的实验性使用深度被记录（formation depth=5、continuation depth=4/8、MESA depth=4），但未证明递归比 flat 组织或替代架构更具因果必要性；缺失对照消融实验。
- **必需持久记录的识别不足**：continuation 结果说明跨模型替换可行，但未隔离哪些 persistent records 是关键（如 CONTEXT.md vs. git commits vs. test 记录）；需固定代码而改变非代码开发记录的反事实测试。
- **零人工介入声明的边界**：人类提供了初始任务规范、工具集、验证源与 controller 限制；"autonomous"在此是有界概念而非绝对无外部目标。存档也未完整记录每一处人工干预。
- **成功率与泛化估计缺失**：仅报告单一 DeepSeek / GLM 运行证据，未给出 run-to-run 成功率的统计估计；不同任务类型、仓库规模下的稳定性未知。
- **未来方向**：(1) 控制性消融（fresh vs. persistent agent、flat vs. hierarchical acceptance、递归 vs. 非递归）；(2) 扩展至更大规模仓库与更多语言对（如 C++↔Rust、Python↔Rust）；(3) 将持久化记录的最小必要集形式化；(4) 探索 open-ended 的自我修正与架构自演化。

## 研究启发与可借鉴点
- **版本历史作为 agent 的共享"长期记忆"**：将 accepted changes 通过 Git commit 作为继承物，比维护 agent 端持久会话更稳健；可直接迁移到任何需要跨会话延续上下文的多 agent 开发流程。
- **path-scoped 递归委托的分离式设计**：delegation $(\nu,p)\to(\nu,q)$ 与 acceptance $(\nu,p)\to(\nu',p')$ 的分离，使得局部探索可与全局版本推进解耦；适合用于微服务仓库的并行重构或多分支 feature development。
- **数值行为保留作为科学软件重写的 hard constraint**：在 MESA 实验中，bit-exact 与相对 checksum 差作为验收标准；可将此思路推广到 HPC / 仿真代码的 AI 辅助 porting，强制要求数值等价性而非仅功能等价。
- **成本控制的可复制范式**：120+ 小时长程开发仅 $44 token 费用，得益于 150k token 压缩阈值 + 有限 episode 粒度 + 隔离 worktree；可为团队制定 long-horizon agent run 的预算规划提供参考。
- **创新机会**：与本团队方向结合，可将"project-centered continuity"引入代码仓库迁移、多版本 LTS 分支维护、或跨语言 API 兼容层自动生成等场景，利用 path-scoped 委托实现大仓库的渐进式重写。

## 关键术语表
**Persistent Recursive World（持久化递归世界）**：以已接受软件版本与仓库路径为坐标组织的开发结构，agent 局部有限存活，但项目历史在版本层面持续积累。
**Local Software World $w=(\nu,p)$（局部软件世界）**：由已接受版本 $\nu$ 与相对路径 $p$ 定义的 agent 工作坐标，$\nu$ 固定继承状态，$p$ 划定责任范围。
**Recursive Delegation（递归委托）**：父 agent 在同版本下于另一路径 $(\nu,p)\to(\nu,q)$ 启子 agent，不立即推进版本历史，仅将工作局部化。
**Accepted Software Event（已接受软件事件）**：$(\nu,p)\to(\nu',p')$，parent 经测试/约束/集成证据采纳候选变更，推进持久版本历史。
**Finite-lived Agent（有限生命周期 agent）**：单次 episode 内有界运行的 agent，其私有对话与 scratch 不继承给后续 agent。
**Validation-gated Acceptance（验证门控接受）**：通过 tests、constraints 与 provenance 决定是否 accept/reject/request-more，reject 的代码不留存但失败信息可被显式存储后进入后续版本。
**Greenfield Formation（绿场构建）**：从无实现的空仓库开始，通过多 episode 递归积累形成复杂系统的首次构建任务。
**Numerical Fidelity（数值保真度）**：在代码重写后保留原始科学软件经审计的数值行为程度，以 bit-exact 或相对 checksum 差度量。

## 可复现要素
- **数据集/仓库**：C 编译器任务从空仓库（仅 .gitignore + genesis.toml）起步；MESA 使用 commit 461dcba94f33 的轻微修改 fork 作为只读参考；两个 continuation 分支均从 GLM 5.2 生成的 jcc 仓库 commit 37216cfa254a 起步。
- **代码/权重开源**：论文声明"Full protocol, archive and measurement details are provided in the Supplementary Information"；EvoX Genesis 系统与实验存档应在项目主页或 supplementary 中公开（论文未提供具体 URL）。
- **关键超参**：
  - Context compression threshold：150,000 tokens
  - Compiler formation：maximum delegation depth 5，1,019 archived episodes
  - Continuation：max depth 8，retry limit 15，2,048 root turns，128 delegated turns
  - MESA redevelopment：33.22 h，272 agents，median of 25 post-warm-up measurements per workload
- **评估基线**：c-testsuite（220 用例）、LLVM test programs（36 用例）、Csmith（93 程序）、LZ4 / SQLite、Rust workspace tests（2,904 / 1,052）、六项 MESA numerical workloads。
- **费用口径**：仅 foundation-model token charges，不含硬件/存储/网络/人工。
