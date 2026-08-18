---
title: "How-Organizations-Use-AI-Evidence-from-ChatGPT"
source: https://arxiv.org/pdf/2608.12236v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:24:56"
field: "企业级 AI 采用与扩散"
keywords: ["企业 AI 采用", "生成式 AI", "遥测数据", "通用目的技术", "任务分类", "无形资产互补", "ChatGPT Enterprise"]
innovations: ["将 ChatGPT Enterprise 管理遥测数据与 Compustat 财务数据链接，首次在企业层面同时分析采用决策、使用强度和任务构成的异质性", "构建 60 类工作任务自动分类器，实现大规模消息级知识工作分类", "用永续盘存法量化 SG&A/R&D/软件存量作为互补性资本，论证其对 AI 采用概率的正向影响"]
benchmarks: ["Compustat 上市公司财务面板", "ChatGPT Enterprise 组织-周使用面板", "60 类工作任务分类体系"]
---

# 论文速读：How-Organizations-Use-AI-Evidence-from-ChatGPT

## 一句话总结
本文利用 ChatGPT Enterprise 的行政数据与 Compustat 上市公司财务数据的链接，记录了企业级生成式 AI 采用与使用的四个经验事实：采用增长迅速、集中在大型和无形资产密集型企业、在员工角色和职级上分布广泛但强度不均、任务类型覆盖广泛的知识工作领域。

## 研究问题与动机
- **核心问题**： firms 内部谁在使用 AI、使用多高频、用于哪些任务？现有对企业 AI 采用的理解不足。
- **现有方法不足**：绝大多数企业 AI 采用证据来自问卷调查（如 McElheran et al. 2024; Bick et al. 2026a），虽然覆盖面广且能捕捉非采用情况，但自我报告的使用数据细节有限，且可能存在回忆偏差和报告偏差。
- **理论动机**：通用目的技术（GPT）的价值实现需要实验、互补性投资和組織变革，采用不等于有效部署，需要理解扩散的内部微观结构。
- **政策/经济意义**：理解企业 AI 采用模式对解释近期关于 AI 对生产率和就业影响的发现至关重要（Brynjolfsson et al. 2025a, 2025b）。

## 核心贡献（创新点）
1. **大规模管理遥测数据的构建与链接**：将 ChatGPT Enterprise 组织级周面板数据（2024.1–2026.3）与 Compustat 财务数据通过 ticker bridge 安全链接，突破了以往调查数据的局限。
2. **区分采用与部署的微观结构分析**：不仅记录企业是否采用，还分析采用后六个月内部署在不同职能、职级和任务上的分布，揭示了采用后的扩散动态。
3. **任务级自动分类框架**：开发了基于消息内容的 60 类工作任务分类器（task classifier），使大规模消息级任务分析成为可能，无需人工逐条审阅。
4. **四个经验事实的系统性记录**：首次在企业层面同时记录采用规模、财务相关性、worker 异质性和任务广度，描绘了企业 AI 采用的全景图。
5. **互补性资本的量化证据**：将 SG&A、R&D 和资本化软件存量作为组织互补性投资的代理变量，论证了无形资产投资与企业 AI 采用概率的正向关联。

## 方法详解
- **数据结构**：构建四个样本——(1) 聚合企业使用样本（organization-week panel）；(2) 带员工职位标题和行业信息的子样本（1,764 个组织，17,446,551 条消息）；(3) 任务分类子样本（973 个组织，8,696,657 条已分类消息）；(4) 链接 Compustat 的上市公司样本（521 个 ticker-year 观测）。
- **职位分类**：使用 `gpt-5-mini` 模型（minimal reasoning effort）对行政职位标题字符串进行分类，输出 department、seniority level、people manager status 和 cleaned job title class 四类结构化标签（详见附录 C prompt 定义）。
- **任务分类**：每条消息被分配到 60 个任务类别中的唯一一类，形成两级分类法（第一层为 broad domain，第二层为 specific task type），分类器在 2025.10.30 之后可用（详见附录 D 和 Table A5）。
- **财务链接**：通过 curated + LLM-assisted 的 account-to-ticker crosswalk 将企业账户映射到 Compustat，定义 adopters 为有匹配ticker 的公司，non-adopters 为无匹配的美国上市公司（11,784 个 ticker）。
- **计量模型**：
  - 采用决策：线性概率模型（LPM），DV = 是否采用，控制 lagged 财务变量 + 年份 FE + NAICS2/NAICS4 行业 FE。
  - 使用强度：OLS 回归，DV 为 log(1 + 强度指标)，强度指标包括 msgs/active-week/emp、WAU/emp、tokens/emp、msgs/WAU。
  - 互补性资本：构建永续盘存法计算的 SG&A 存量（年折旧 20%）、R&D 存量（年折旧 15%）和 capitalized software 存量，测量 FY2021 水平。
  - 企业规模相对度量：按 Autor et al. (2020) 方法，在 NAICS2-by-year cell 内计算相对收入分位。

## 实验与结果
- **数据集规模**：6 个月采用 horizon 下的 worker-level 样本包含超过 1,500 个组织、超过 1,700 万条消息；任务分类子样本包含 973 个组织、约 870 万条已分类消息；上市公司样本含 417 个 ticker、11,784 个 non-adopter ticker。
- **Fact 1 — 快速增长**：2025.6–2026.3 期间 ChatGPT Enterprise 输出 token 总量增长约 **7 倍**；在 2024.1–2025.6 间采用的一致 cohort 内，token 增长约 **4 倍**，即约一半增长来自已采用企业的深度使用。2026 年初所有 cohort 同时加速。
- **Fact 2 — 早期采用者特征**：采用者中位数营收 $2,275.1M vs 非采用者 $209.6M；中位数员工 2,934 vs 424；中位数 R&D 支出 $113.1M vs $9.9M。Table 1 显示 lagged log(revenue/emp) 系数 0.004–0.009（p<0.05），log(employment) 系数 0.013–0.019（p<0.01）。Table 3 显示收入前 5% 企业采用概率高 9.8 个百分点（全样本）/11.3 个百分点（行业年内）。
- **Fact 3 — 使用分布与强度异质性**：六个月内，工程师占活跃用户约 11%，高管 9%，市场/传播 5%，财务/会计 5%。职级上管理者/总监占 24%，IC 占 15%，高管 10%，早期职业员工仅 7%。但强度上，早期职业员工比同企业平均活跃用户每周多发 **8–9 条**消息；分析师和 marketing/comm 用户消息量高于平均，高管低于平均。Table 2 显示 log(employment) 对 per-employee 使用强度有显著负向影响（-0.266 至 -0.667，p<0.01）。
- **Fact 4 — 任务广度**：超过半数活跃用户执行过 documentation/technical writing 任务，近半数执行过 technical digital work，大量用户用于 messaging、research、data analysis、legal、finance 等。消息量集中在 writing、technical digital work 和 message drafting，而 topic overview、research、legal 等任务用户覆盖面广但消息量占比小。行业间差异主要在 extensive margin（是否尝试），intensive margin（消息份额）上较相似。

## 相关工作脉络
1. **Bonney et al. (2026)**：区分企业采用与跨业务功能/任务的部署，本文推进之处在于同时分析财务预测因素和采用后部署的 worker/task 分布。
2. **Counts et al. (2026)**：使用 Microsoft 365 Copilot 遥测数据刻画职场使用，本文使用 ChatGPT Enterprise 数据，更侧重知识工作而非编码场景。
3. **Johnston et al. (2026)**：使用 OpenAI 遥测数据研究从 conversational 到 agentic AI 的转变（Codex），本文聚焦 ChatGPT Enterprise 的通用知识工作应用。
4. **Brynjolfsson et al. (2025b)**：基于实验证据研究 GenAI 对工作者的影响，本文提供观察性宏观证据，填补了从个体实验到企业级扩散的证据链条。
5. **Eloundou et al. (2024)**：基于任务描述估算 AI 暴露潜力，本文直接观测实际使用行为， bridging capability 与 realized use 的差距。
6. **Handa et al. (2025) / Chatterji et al. (2025)**：使用 Claude/ChatGPT 消费端数据研究任务构成，本文扩展到企业级产品中带有职位和财务链接的维度。

## 局限性与未来方向
- **数据范围局限**：仅涵盖 OpenAI ChatGPT Enterprise 产品，不包括 API 使用、自建工具或其他 AI 系统的使用。
- **职位覆盖不完整**：active users 中部分无可用 job title，分析中视为 missing/unclassified，无法获得各职能的员工基数分母。
- **无下游产出度量**：任务分类基于消息内容，不衡量实际工作产出、生产率效应或组织流程变化。
- **样本代表性**：上市公司样本仅限美国大型企业，不能推断中小企业、初创公司或非上市公司的采用模式。
- **因果识别**：财务特征与采用的关系为条件关联，非因果估计；互补性资本的结果也存在内生性风险。
- **未来方向**：链接 telemetry 到产出和组织变革的长期绩效数据，研究采用、使用强度、worker 构成和任务组合如何随 AI 普及而演化。

## 研究启发与可借鉴点
1. **遥测数据与财务数据链接的研究设计**：通过安全的 account-to-ticker bridge 将产品使用数据与 Compustat 财务面板链接，为研究技术采用与企业绩效的关系提供了可复用的数据管道范式。
2. **采用强度 vs 采用决策的边际分离**：同时将分析分为 extensive margin（是否采用）和 intensive margin（采用后多高频/多广泛使用），并分别建模，避免了单一指标的遗漏变量偏误。
3. **任务分类器的自动化 pipeline**：使用 LLM 做消息级任务分类（60 类 taxonomy），在隐私保护（不人工审阅消息）前提下实现大规模任务分析，为其他 AI 使用研究提供了可迁移的分类框架。
4. **互补性资本的永续盘存度量**：用历史支出流构建 SG&A/R&D/software 存量（而非单年支出），捕捉企业在 AI 采用前的能力积累，为 GPT 互补投资文献提供了新的量化方法。
5. **Within-firm fixed effects 的设计**：在分析 worker 异质性时加入 firm FE，分离了企业间差异与企业内部不同角色间的差异，更准确地识别了 early-career 高强度的内部模式。

## 关键术语表
**ChatGPT Enterprise**：OpenAI 面向企业客户的 centrally administered 付费产品，支持组织级账户管理和使用追踪。
**General Purpose Technology (GPT)**：通用目的技术，具有广泛适用性、持续改进和触发互补性创新的特征（如蒸汽机、电力）。
**Intangible Complements**：无形资产互补投资，包括组织流程、管理软件、人力资本等，与 GPT 协同产生价值。
**Output Tokens**：模型生成的 token 数量，衡量 AI 使用量的核心产出指标（含 ChatGPT 和 Codex）。
**Perpetual Inventory Method**：永续盘存法，将历年支出流按折旧率累积为存量，用于衡量资本积累。
**Extensive Margin vs Intensive Margin**：广泛边际指"是否采用/是否使用"，强度边际指"使用频率/深度"。
**Task Classifier**：自动将消息分类到 60 个工作任务类别的 LLM 驱动分类器。
**Week-indexed Adoption Panel**：以各组织采用周为时间原点的组织-周面板数据。

## 可复现要素
- **数据集**：ChatGPT Enterprise 行政数据和 Compustat 财务数据；**未公开**（论文声明使用 de-identified 数据，仅报告聚合结果）。
- **代码**：**未开源**（论文未提及代码/权重开源）。
- **关键超参**：SG&A 折旧率 20%/年、R&D 折旧率 15%/年；职位分类使用 gpt-5-mini minimal reasoning；任务分类器于 2025.10.30 上线。
- **样本筛选**：worker characteristics sample 要求 NAICS 行业分类 + 有效 job title + 26 周 horizon 活跃；task classification sample 额外要求 26 周时有分类数据。
