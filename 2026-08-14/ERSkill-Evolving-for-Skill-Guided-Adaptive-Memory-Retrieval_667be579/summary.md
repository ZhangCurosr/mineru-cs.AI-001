---
title: "ERSkill-Evolving-for-Skill-Guided-Adaptive-Memory-Retrieval"
source: https://arxiv.org/pdf/2608.12720v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:26:42"
---

# 论文速读：ERSkill-Evolving-for-Skill-Guided-Adaptive-Memory-Retrieval

## 一句话总结
本文提出ERSkill，一种以检索为中心的Agent记忆自进化框架，将记忆访问建模为由固定检索原语组合而成的可执行技能程序，并通过经验字典树与双前沿机制协同进化检索技能与路由选择器，实现针对不同查询异构信息需求的动态证据构建。

## 研究问题与动机
1. **检索行为静态化瓶颈**：现有自进化Agent工作主要利用历史轨迹改进推理策略或记忆内容提取，但查询时的记忆访问机制仍依赖预定义的固定检索策略，缺乏对检索行为本身的进化能力。
2. **异构查询的证据构建差异**：Agent长期交互中的查询信息诉求差异显著（如定位单一实体事件 vs. 追溯因果发展链），单一检索路径无法同时满足多类证据组织需求。
3. **探索效率与部署稳定性的矛盾**：盲目扩展检索技能易导致组合爆炸，且新技能若不能被路由稳定选中则无实际收益；需在能力边界探索与路由部署可靠性之间建立安全解耦机制。

## 核心贡献（创新点）
1. **检索中心化的自进化范式**：将记忆访问从“固定检索策略”转为“查询自适应的证据构建过程”；与MemSkill等方法仅进化记忆提取技能形成本质区别，首次将进化目标明确锚定在检索路径编排上。
2. **技能化可执行检索程序**：提出基于固定原语库（检索/扩展/处理）的可组合技能表示，检索行为被封装为可读、可复用、可优化的可执行程序，兼顾表达力与可解释性。
3. **经验字典树与双前沿协同进化机制**：设计经验字典树记录已探索的原语路径以去重和引导候选生成；提出能力前沿（Oracle-side最优覆盖）与部署前沿（路由验证稳定）解耦的Pareto风格更新机制，保障能力稳步扩张的同时避免部署震荡。
4. **全面的基准验证与成本优势**：在三个Agent记忆基准上显著超越强非自进化与自进化基线，展现优异的跨数据集迁移能力，并在内存构建与推理Token开销上实现最优成本-性能权衡。

## 方法详解
1. **结构化记忆存储**：将交互历史$D$编译为$M(D)=(\mathcal{A}, \mathcal{I}, \mathcal{G})$。$\mathcal{A}$为原子级记录（含atom_id、text、timestamp、entity_set）；$\mathcal{I}$提供检索入口（稠密向量索引、BM25词项索引、实体-原子倒排索引、时间键索引）；$\mathcal{G}$包含相似性边与LLM抽取的语义关系边（如Cause, Changed, React等）。
2. **固定原语库**：暴露三类算子供技能编排：搜索原语（`dense_search`, `lexical_search`, `entity_search`）、扩展原语（`similarity_expand`, `relation_expand`, `temporal_focus_expand`）、处理原语（`llm_process`用于查询重写、证据过滤、控制变量生成）。
3. **检索技能表示**：技能$\kappa$为可执行程序$(c_\kappa, \rho_\kappa)$，其中$\rho_\kappa=(p_{\kappa,1}, \dots, p_{\kappa,L_\kappa})$为原语序列。给定查询$q$，依次执行状态转移$s_j=p_{\kappa,j}(q,s_{j-1},M(D))$，最终输出证据视图$s_{L_\kappa}$供LLM生成答案。
4. **技能路由模型**：使用共享文本编码器Enc(·)将$q$与$\kappa$编码为$h_q, h_\kappa$，经线性投影与MLP计算打分$u_\theta(q,\kappa)$，路由概率为$R_\theta(\kappa|q,\mathcal{K})=\frac{\exp(u_\theta(q,\kappa))}{\sum_{\kappa'\in\mathcal{K}}\exp(u_\theta(q,\kappa'))}$。文本编码器冻结，仅优化路由参数。
5. **技能-路由协同进化流程**：
   - **经验字典树**：按原语前缀组织路径，每个节点记录train/validation rollout统计与前沿状态，生成候选时排除已存在路径。
   - **能力前沿$\mathcal{C}_t$更新**：在训练批次上做临时过滤，再在验证集上用$\Phi(\cdot)$算子保留能提升或维持oracle覆盖率$g_\mathcal{K}(q)=\max_{\kappa\in\mathcal{K}}r(q,\kappa)$的技能。
   - **路由训练**：基于rollout得分构造软目标分布$\tilde{p}(\kappa|q,\mathcal{K}_q)$，以软标签交叉熵$\mathcal{L}_{\text{router}}=-\sum\tilde{p}\log R_\theta$更新路由器。
   - **部署前沿$B_t$更新**：仅当能力前沿变化时触发；候选集需满足路由增益阈值$\gamma_{\text{route}}$，或在性能小幅下降（容忍$\xi_{\text{drop}}$）且技能集更紧凑时被接受，否则保持原部署集。

## 实验与结果
- **数据集**：LoCoMo（多会话对话，含单跳/多跳/时间/开放域四类问题）、LongMemEval（长期交互式记忆）、PerLTQA（异构个人记忆，含profile/relationship/event/dialogue）。
- **基线**：非自进化（A-Mem, MemoryOS, LightMem）；自进化（Dynamic Cheatsheet, ReasoningBank, GEPA, MemSkill）。
- **模型与评测**：Qwen3-Next-80B-A3B-Instruct 与 GPT-5.4-nano 作为生成backbone；GPT-4o-mini 作为LLM-as-a-Judge；指标为F1、BLEU-1、L-J。
- **主要结果**：ERSkill在两项backbone下均取得最佳整体平均。对比最强基线，F1/BLEU-1/L-J综合均值分别提升**31.3%**（Qwen3）与**28.1%**（GPT-5.4-nano）。在依赖精准
