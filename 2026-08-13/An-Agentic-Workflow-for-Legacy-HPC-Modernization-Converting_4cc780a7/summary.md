---
title: "An-Agentic-Workflow-for-Legacy-HPC-Modernization-Converting"
source: https://arxiv.org/pdf/2608.12249v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:37:57"
field: "科学计算软件现代化与Agent驱动代码转换"
keywords: ["agentic AI", "legacy code modernization", "Fortran", "GAMESS", "bit-for-bit verification", "scientific computing", "LLM agents", "HPC"]
innovations: ["三路提示专用Agent在版本控制规范下协作完成遗留科学代码现代化，规范本身由Agent撰写和修订", "将验证责任主要由Agent承担而非仅对Agent输出判罚，实现了Agent主导的self-audit和二分缺陷定位", "在零容差bit-for-bit等价标准下完成56,448行Fortran 77到F2008的转换，612次测试运行零化学差异"]
benchmarks: ["GAMESS 49-standard tests", "Jenkins CI suite", "cc-pVQZ water calculation", "scalar-relativistic DFT on HCl"]
---

# 论文速读：An-Agentic-Workflow-for-Legacy-HPC-Modernization-Converting

## 一句话总结
本文提出了一种基于 AI Agent 的遗留科学代码现代化工作流，以 GAMESS 量子化学软件的双电子积分核（12 个源文件、56,448 行 Fortran 77 代码）为案例，成功将其从固定格式 Fortran 77 转换为自由格式 Fortran 2008，在 612 次测试运行中实现了与原始行为逐位一致的零偏差验证。

## 研究问题与动机
- **遗留 Fortran 代码现代化是规模问题而非难度问题**：科学计算中大量关键代码（如气候模型、量子化学、线性代数库）仍运行在固定格式 Fortran 77 上，个体变换虽简单但代码体量巨大，导致现代化工作长期搁置。
- **现有方法无法同时满足自动化规模与零容忍正确性**：规则化工具（如 plusFORT/SPAG）可处理机械性变换，但无法进行语义理解、跨文件架构推理和缺陷检测；人工重写成本过高。
- **正确性验证是委托自动化的真正瓶颈**：对于科学计算内核，错误不一定是崩溃，可能是 SCF 能量第 12 位小数的 silently different digit，年后可影响已发表结果；问题的核心不在于 Agent 能写多少，而在于能被检查多少。
- **科学软件领域天然具备精确验证条件**：GAMESS 等成熟包拥有社区认可的测试套件和作为"金标准"的印刷能量输出，为 bit-for-bit 验证提供了可行基础。

## 核心贡献（创新点）
- **设计了三路 Agent 协作的监督式现代化工作流**：三个提示专用角色（转换、测试、审查）在版本控制规范下独立运行，规范本身由 Agent 撰写和修订，人类仅保留少量 gate（合并审批、命令执行批准）。**与已有工作的区别**：不同于将规范作为外部脚手架的做法，本文的规范是系统自身演化的产物， competence 被持久化在可 diff、可审查、可回滚的版本控制文本中。
- **明确区分了机械变换与需理解的转换任务，并将 Agent 价值定位于后者**：700+ 个不可重结构的 GO TO、类型推断、意图注解、跨文件循环依赖处理等需要全局理解的决策是 Agent 的核心贡献。**与已有工作的区别**：规则化工具在此类任务上会失败，而 Agent 能完成，本文通过实验证据显式刻画了这一边界。
- **展示了由 Agent 主导验证而非仅对 Agent 进行验证的工作流设计**：Agent 编写测试驱动、捕获 golden reference、运行 style audit、甚至通过二分法定位回归缺陷并自我修复。**与已有工作的区别**：此前工作（如 SWE-bench 系基准）的评估范式是外部对 Agent 输出的判罚，本文让 Agent 承担大部分验证责任。
- **实证刻画了自动化的可达边界**：明确了 Agent 独立完成的工作（阅读理解 56,448 行代码、生成合规代码、撰写测试用例、二分法定位缺陷）、Agent 无法处理且必须人工介入的任务（合并审批、构建配置调和），以及验证机制的结构性盲区（差分检查对共模错误盲视）。

## 方法详解
- **工作流架构**：三个角色（Conversion、Testing、Review）分别在独立的 git worktree 中运行，通过共享仓库协调，互不干扰。每个角色由不同的 prompt specialization 驱动，本质上是同一工具（Claude Code）的不同上下文加载，而非独立进程的多 Agent 系统。
- **版本控制规范（Agent 自撰）**：三份文档 govern 整个转换过程——`CLAUDE.md`（工作流契约，自动加载到上下文）、`STYLE_GUIDE.md`（转换规则集合）、`AGENT_GUIDE.md`（角色 prompt 和编译器约束）。规范由 Agent 在运行过程中自行修订，例如为减少上下文负担将 role prompts 从 CLAUDE.md 迁移到 AGENT_GUIDE.md。
- **转换信封（Conversion Envelope）**：目标为 Fortran 2008 标准——自由格式 132 列源、全局 IMPLICIT NONE、`REAL(real64)` 从 `iso_fortran_env`、所有 dummy 参数标注 INTENT、结构化控制流替代 GO TO、每文件一个 MODULE、COMMON 块原样保留并加 deferral 注解。**两个刻意排除**：COMMON 块迁移（不安全，跨文件共享）和算法重构（禁止浮点重排，否则会破坏 bit-for-bit 等价性）。
- **四工件契约**：每个文件转换后提交四个 artifact——现代源代码、变更日志（记录每类变换）、单元测试驱动、从 F77 构建捕获的 golden output。这使单文件审计成为可能。
- **合成单元测试**：每个文件附带一个 driver（图 1），用固定种子和固定宽度格式编译两次（一次针对 F77 原文，一次针对现代模块），比较输出。约 255,000 个合成 shell quartets 被测试，无差异。
- **集成测试**：替换单个文件后构建完整 GAMESS 二进制，运行 51 项验证（49 个标准测试 + 2 个额外计算），并与 Jenkins CI 套件交叉验证。bit-for-bit 检查过滤掉时间戳/主机名，仅比较化学相关内容（能量、迭代追踪、核排斥、virial ratio 等）。
- **Agent 自审机制**：每个文件需按审计条款检查 ISHFT 压缩标签构造，Agent 通过差分计数与原始代码对比，将比较结果提交供人工审查。
- **二分法定位缺陷**：当发现回归时，Agent 编写了一个 113 行的 Python 二分定位工具，逐个还原子例程，五轮迭代将 38 个子程序缩小至问题根源。

## 实验与结果
- **数据集/目标**：GAMESS 双电子积分核，12 个源文件（8 个 int2 文件 + 4 个 eftei 家族文件），56,448 行 F77 代码，225 个子程序。
- **转换结果**：总行数变化 -0.94%（56,448 → 55,916），单文件变化幅度从 -42.7%（eftei_eric 角动量情况折叠）到 +33.1%（eftei_genr70 COMMON 间接变为显式参数）。子程序数保持不变。1,066 个 COMMON 语句全部保留。718 个 GO TO 保留并标注。
- **验证结果**：12 个文件全部通过 51 项标准测试（含 49 个 GAMESS 标准测试 + cc-pVQZ 水分子计算 + HCl 标量相对论 DFT）和 Jenkins CI 测试。612 次测试运行中化学相关内容差异为零，唯一差异仅为 wall-clock 时间。合并构建（12 文件同时替换）也全部通过。
- **效率**：12 文件验证共耗时 7 小时 29 分钟（约 37 分钟/文件），时间随文件数线性增长而非文件大小。
- **首次成功率**：12 个文件中 11 个首次转换算术正确，11 个首次编译干净，9 个首次成功捕获 golden。
- **缺陷分析**：发现三类缺陷——（1）编译器捕获两类（标量 vs rank-1 实参、Hollerith 字面量）；（2）一个逃逸缺陷：`IX` 被错误地声明为 `REAL(real64)` 而非 `INTEGER(int64)`，导致 bit-shift 高位错位，在 exam01 中以 -61.29 Ha vs 基线 -37.17 Ha 暴露；（3）三个验证工具自身的缺陷（正则匹配、csh 脚本、路径处理）。
- **最强结果**：612 次测试运行零化学差异，bit-for-bit 精确复现 GAMESS 标准测试输出。

## 相关工作脉络
- **Psi4/NWChem/CP2K/Quantum-ESPRESSO**：这些量子化学/材料软件包各自经历了 F77 现代化路径（Psi4 重写为 C++，NWChem 增量迁移，CP2K 从一开始就用自由格式），但与本文的区别在于执行方式是代理式且接受标准为零容差逐位等价，而非 toleranced 比较。
- **plusFORT/SPAG 等规则化工具**：几十年历史的 Fortran 重构工具已能处理固定到自由格式、GO TO 重构等机械变换，但无法处理语义理解、循环依赖推理和审计规则编写——本文显式分离了机械变换与理解依赖任务。
- **SWE-bench（Jimenez et al., 2024）**：代表性 Agent 代码改造基准，评估小规模独立程序的函数正确性。本文与之的区别在于：在语言内语义-preserving 转换、零容差数值等价标准、以及在百万行调用图中的一个文件内工作。
- **Fortran2CPP（Chen et al., 2024）与 LLM-assisted Fortran-to-C++（Ranasinghe et al., 2025）**：跨语言迁移工作。本文的关键区别是在语言内的语义保持转换（F77 → F2008），原始数值行为必须精确保留，而非重新表达。
- **Kokkos 端口工作（Gupta et al., 2025）**：自主 Agent 工作流将 Fortran 移植到 portable Kokkos。本文与之的区别在于目标不是跨平台可移植性而是精确数值等价保留。
- **Trilinos（Heroux et al., 2005）**：HPC 中 closest 的先例，但使用 tolerance-based 比较而非 zero-tolerance。本文的逐位等价标准在此类场景中更为严格。

## 局限性与未来方向
- **仅在一个代码库（GAMESS）上验证**：GAMESS 的特殊性在于其测试套件提供了精确的正确性检查，缺乏此类检查的代码库可能不支持相同工作流。
- **仅使用 Anthropic Claude 模型和 Claude Code 运行时**：跨模型家族和 Agent runtimes 的可移植性尚未实证，作者将其视为 conjecture 而非 result。
- **仅在桌面工作站上运行，未扩展至 HPC 集群**。
- **未整合入 GAMESS 生产版本**，且仅验证了标准测试集和 Jenkins 测试集，GAMESS 提供的扩展测试套件尚未全面覆盖。
- **验证工具的结构性盲区**：差分检查对共模错误（即 golden 和被测二进制使用相同构建标志时）盲视，需要至少一层验证锚定到外部固定参考。
- **未来方向**：扩展至 GAMESS 剩余约百万行代码、集成 COMMON 块到 MODULE、在 HPC 集群上运行、验证扩展测试套件、跨其他 Agent 运行时和模型家族移植工作流。

## 研究启发与可借鉴点
- **"规范即持久工件"的设计哲学**：将系统积累的 competence 存储在版本控制的文本规范中（而非任何模型或会话记忆中），使得跨模型代际切换无需 rework，且规范可 diff、可审查、可回滚——这对科学软件的溯源性至关重要，可迁移到任何需要长期维护的 Agent 驱动工程流程。
- **Agent 自我审计与二分法定位缺陷**：Agent 不仅能生成代码，还能编写二分定位工具、设计审计规则并写入规范以防止同类缺陷重现——这种 self-repair 机制可推广到其他需要高正确性保证的代码转换场景。
- **验证边界的实证刻画方法**：本文提出的"自动化边界由验证能力而非生成能力决定"这一洞见具有普遍意义，可指导未来工作在设计 Agent 工作流时优先投资验证基础设施而非仅关注生成能力。
- **四工件契约的可复用模式**：现代源代码 + 变更日志 + 单元测试驱动 + golden output 的四件套设计，使得单文件审计成为可能且不影响其他文件的审查——这是大规模代码转换项目的可复用组织模式。
- **团队结合机会**：本团队若从事科学计算软件现代化或 Agent 辅助代码转换研究，可直接借鉴其 verification-by-agents 范式和 agent-authored specification 机制；若关注 HPC 代码并行化/GPU 卸载，本文的工作可作为前提（先完成 bit-for-bit 现代化，再进行 tolerance-based 优化）。

## 关键术语表
- **GAMESS**：General Atomic and Molecular Electronic Structure System，由 Iowa State University Gordon 团队维护的开源免费量子化学计算软件包，超过百万行 Fortran 77 代码，48 年开发历史。
- **Agentic Workflow**：由 LLM 驱动的自动化循环工作流，Agent 在规范指导下自主读取文件、重写代码、编译、运行测试并迭代直至通过，无需每步人工干预。
- **Bit-for-bit Reproduction**：零容差逐位等价验证，要求现代代码生成的输出与原始 F77 代码完全一致（第 12 位小数差异即算失败），区别于常见的 tolerance-based 比较。
- **COMMON Block**：Fortran 77 中的全局变量共享块，跨文件共享状态，阻碍线程化和 GPU 卸载；本文严格保留原样以避免跨文件不安全迁移。
- **Golden Output（Golden）**：从原始 F77 构建捕获的标准测试输出，作为 bit-for-bit 比较的参考基准，由 GAMESS 用户社区视为权威。
- **Rys Quadrature**：GAMESS 中用于计算四中心电子排斥积分的一种数值积分方法，包含不可约的 GO TO dispatch tables，需保留原结构。
- **Shell Quartet**：量子化学中由四个基函数壳层组成的积分组，表示为 (µν|λσ)，是双电子积分计算的基本单元。
- **ISHFT-based Packed Label**：将多个 shell 索引打包进 64 位字的技术，通过 bit-shift 实现；本文发现 Agent 在此类构造的类型转换上存在系统性盲点。

## 可复现要素
- **数据集**：GAMESS 双电子积分核 12 个源文件（56,448 行），GAMESS 自带的 49 个标准测试 + 2 个额外测试（cc-pVQZ 水分子、HCl 标量相对论 DFT）—— GAMESS 为 source-available freeware，代码不可自由分发但可下载。
- **代码/权重**：转换结果和相关工具在两个 GitHub 仓库中提供（论文中引用为脚注 1,2），但现代源代码因 GAMESS 许可限制不能公开分发。
- **关键超参**：未明确报告传统 ML 超参；关键工程参数包括：Fortran 编译器标志 `-std=f2008`、`-fdefault-integer-8`、浮点优化级别 `-O1`（禁止浮点重排）、测试种子固定为 17、输出格式固定宽度无时间戳。
- **模型**：Claude Opus 4.6、Claude Sonnet 4.6、Claude Opus 4.7、Claude Opus 4.8（四代模型），Claude Code 命令行 Agent 运行时。
- **硬件**：桌面工作站（具体规格论文未提及）。
