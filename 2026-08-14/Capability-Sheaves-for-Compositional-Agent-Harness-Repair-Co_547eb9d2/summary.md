---
title: "Capability-Sheaves-for-Compositional-Agent-Harness-Repair-Co"
source: https://arxiv.org/pdf/2608.13228v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:22:04"
field: "AI agent harness optimization"
keywords: ["capability sheaf", "agent harness repair", "sheaf cohomology", "constraint satisfaction", "SWE-bench", "patch fusion", "hidden-state quotient"]
innovations: ["构造五需求有限capability sheaf并将精确CSP与线性上同调诊断分离", "证明隐藏内态商化可消除nuisance并减少50%搜索预算", "发现全pool上同调类对patch选择恒同的缺陷并提出candidate-indexed修复"]
benchmarks: ["SWE-bench Multilingual Pool (PatchFusion)", "Generated JSON migration task (20 clusters)"]
---

# 论文速读：Capability-Sheaves-for-Compositional-Agent-Harness-Repair-Co

## 一句话总结
本文利用有限capability sheaf和相对上同调理论，对AI agent harness中各子系统的状态一致性进行建模与修复；控制实验验证了商化（quotient）机制可消除隐藏内态噪声并减少50%搜索预算，但在真实SWE-bench压力测试中，提出的同调方法未能超越强锚点基线，证实了当前构造的局限性。

## 研究问题与动机
- **局部可用但全局不一致**：agent harness中各子模块（定位、契约、排序、保存、验证）各自成功，但共享字段（文件路径、API版本、提交哈希等）无法粘合，导致执行失败。
- **现有方法缺乏结构诊断**：当前prompt/程序/workflow优化依赖外层标量评分搜索，无法解释"每个能力局部存在但全局不可组合"这一故障模式。
- **sheaf理论提供精确语言**：层论天然刻画局部数据、共享约束、粘合条件，可将精确CSP与线性上同调诊断分离，为agent harness repair提供结构化推理通道。
- **真实评测验证缺口**：已有sheaf方法（如知识层、科学理论迁移检测）未直接作用于可执行的agent edit操作与公共测试验证闭环。

## 核心贡献（创新点）
1. **构造了五需求有限capability sheaf**：将定位、契约、排序、保存、验证建模为typed stalk，通过字面投影restriction map连接；与已有工作的本质区别在于对象是可执行行为痕迹而非几何测量或静态知识图谱。
2. **提出精确CSP + 相对上同调两类判据**：精确CSP决定语义可行性，相对类提供线性搜索评分且可独立使用；区别在于前者是决定性语义规则，后者只是单侧证书（one-sided certificate）。
3. **证明隐藏内态商化不变性**：插入隐藏中介顶点后，商化去除 nuisance interior coboundary，使秩搜索预算从2,000降至1,000；与已有方法的差异在于直接验证了"invariance to stalk representatives"而非泛化的同调优势。
4. **在SWE-bench多语言池上进行real-patch-fusion压力测试**：发现全pool类对所有候选相同（coker中恒为零），修复为candidate-indexed complex后获得非平凡分数但未通过开发门控；定位了当前方法的边界而非宣称通用优势。

## 方法详解
- **有限行为stalk与限制映射**：定义五类注册需求$L, C, O, P, V$，每类具有typed signature fields（如localization含file path/namespace alias/symbol）；六组overlap通过字面field projection构成restriction map $\rho_{\nu \to e}$。
- **精确可行性谓词**：候选$harness\ c$被接受当且仅当所有local predicate为真且所有registered overlap的restriction相等：
  $$\Phi(c) = \bigwedge_{\nu} [s_\nu(c) \in G_\nu] \land \bigwedge_{e=\nu w} [\rho_{\nu \to e} s_\nu(c) = \rho_{w \to e} s_w(c)]$$
- **相对连接障碍**：将one-hot值编码为$\mathbb{F}_2$向量，胞腔coboundary $\delta^0$ 定义分歧向量；相对上同调类$[\delta^0 s_A] \in H^1(X,A;\mathcal{F})$在相邻typed restriction一致时消失。
- **隐藏内态商化**：在每对public顶点间插入hidden mediator，raw residual $q_j(h_j)=(\ell_j+h_j,\ h_j+r_j)$，商化映射$[q_j(h_j)] \mapsto \ell_j + r_j$消除隐藏状态依赖，仅保留端点分歧。
- **线性化不充分的反例**：商类消失仅说明目标signature在线性span内，但不保证存在精确候选实现它；需followed by exact membership check。
- **稳定性定理**：Thm.4给出rank-truncated residual在噪声扰动下的误差界$T=\tau+\rho B$，Cor.5给出有限trace估计的样本复杂度下界。
- **Repair搜索策略**：outcome-blind filler枚举pool（25个bundle），优先minimize relative obstruction weight、maximize exact-lift fraction、minimize filler cost。

## 实验与结果
- **控制实验1（基线搜索）**：4 task replicas × 2 endpoints (DeepSeek-V4-Flash, GLM-5-FP8) × 25 candidates；full relative class与exact CSP均100%成功，search evaluations=1.0；NSGA-II需2.391 evaluations。
- **控制实验2（隐藏内态商化）**：20独立task clusters × 2 endpoints × 25 candidates = 1,000 outcomes；stale-raw需2.0 evaluations，quotient policy需1.0 evaluations，配对差+1.0贯穿所有20 clusters，$p=2^{-20}=9.54\times10^{-7}$；aligned-left raw与quotient完全一致，验证gaintied到nuisance而非泛化优势。
- **Token效率**：full-class相对NSGA-II，DeepSeek减67.5% tokens，GLM减71.6% tokens。
- **真实仓库压力测试**：SWE-bench Multilingual pool，160 issues / 20 repos / 875 candidate patches / 2,579 source-aware edit atoms；full-pool relative router解决104 issues，matched router解决109，差异$p=0.28$不显著；candidate-indexed quotient较matched selector多2 issues（848/875非平凡），但$p=0.75$不支持；leave-one-repository-out abstention达127/160，与强锚点持平（$\hat{p}=1.0$）；开发门控要求≥4 issue gains且$p\leq0.2$，均未通过，confirmatory split保持sealed。

## 相关工作脉络
1. **DSPy/AFlow/ADAS**：优化prompt/program/workflow/agent，依赖task objective与optimizer state，未声明capability sheaf结构。
2. **Meta-Harness**：端到端设计可执行harness代码，proposer拥有source/prior scores/traces访问权，验证在application benchmarks。
3. **VeRO**：versioned target agents优化，基于versions/rewards/observations/budget的可重复评估。
4. **HarnessFix**：利用failed trajectories定位harness flaw并验证scoped repairs，使用ledger trace IR与provenance。
5. **PatchFusion**：确定性重复atom证据融合patches，报告236/300 SWE-bench Multilingual resolved；本文复用其fixed candidate pool但问不同问题——隐藏状态商化是否超越匹配语义证据。
6. **Olivieri & Hernandez (2026)**：sheaf-theoretic transport用于scientific theory transition ranking，干预对象为transition candidates而非repository edit operations。
7. **知识层（Knowledge Sheaves, Gebhart et al. 2023）**：表达schema-constrained knowledge graph embedding，非可执行行为修复。

## 局限性与未来方向
- **控制实验规模有限**：baseline仅4独立task clusters，hidden-state study有20 clusters但stale mediator是人工干预而非真实系统状态噪声统计。
- **真实仓库单候选池**：seven-source SWE-bench Multilingual pool的issue分布、strong anchor及GLM-5单模型atom evidence可能不具代表性；早期独立GLM-5与Qwen hunk研究显示evidence-model依赖性。
- **同调构造强度不足**：当前obligation-and-risk complex太弱，未能编码跨edit的实际语义接口；需更强构造（如edit-aware semantic interfaces）方可在真实patch fusion中体现优势。
- **未开放confirmatory repos**：因development gate失败，21个确认仓库保持sealed，未来需预注册更强方法后重新测试。
- **未 claim 计算速度优势**：HiGHS solver在3.45秒内解完所有2,579 binary atom variables，朴素网格大小不构成速度断言。

## 研究启发与可借鉴点
1. **结构性故障分离框架**：将"局部可用性"与"全局粘合性"分离为精确CSP与线性障碍两类判据，可作为通用agent harness诊断模板。
2. **隐藏状态商化技术**：通过插入mediator顶点并在coboundary商空间中计算，消除nuisance interior代表元影响，适用于任何存在latent中间状态的search problem。
3. **预注册与门控协议**：development gate预设四重标准（≥4 issue gains、positive repository-macro effect、≥6 nonzero repos、$p\leq0.2$），失败即sealed confirmatory split；该协议可迁移至其他AI系统benchmark验证。
4. **Candidate-indexed complex修复**：当全局class对全部选择恒同时，改为按候选动作索引complex（$D_S$仅含候选相关列），使score函数$q(S)$在不同候选间变化；此技巧可用于任何span-based ranking失效场景。
5. **稳定估计定理的工程价值**：Thm.4与Cor.5给出rank-truncated residual的误差界与样本复杂度，可直接指导从fuzzy trace中学习linear coboundary的实际部署。

## 关键术语表
- **Capability Sheaf（能力层）**：将agent harness的五类注册需求建模为有限typed stalk与literal projection restriction map的结构，用于刻画局部行为与共享状态。
- **Exact CSP（精确约束满足问题）**：由local predicate合取与registered restriction等式构成的可行性判定，是semantic decision rule。
- **Relative Coboundary Class（相对上同调类）**：$[\delta^0 s_A] \in H^1(X,A;\mathcal{F})$，衡量边界section的粘合障碍，为零当且仅当相邻typed restriction一致。
- **Hidden Interior Quotient（隐藏内态商）**：在重叠坐标间插入mediator顶点后，对interior coboundary像取商空间，使表征与隐藏状态无关而仅依赖端点分歧。
- **One-Sided Portfolio Certificate（单侧投资组合证书）**：若商类非零则证明无候选可实现目标signature，但逆命题不成立。
- **PatchFusion Benchmark（补丁融合基准）**：基于SWE-bench Multilingual pool的real-repository stress test，包含875候选patches与2,579 source-aware edit atoms。
- **Development Gate（开发门控）**：预注册的实证通过标准（≥4 issue gains、positive effect、≥6 nonzero repos、$p\leq0.2$），用于决定是否开启confirmatory split。
- **Outcome-Blind Filler（结果盲填器）**：在未见真实outcome的情况下枚举filler bundle并minimize obstruction weight的repair搜索策略。

## 可复现要素
- **数据集**：SWE-bench Multilingual pool（来自PatchFusion），20 repos + 160 issues用于development split，21 repos + 140 issues用于sealed confirmatory split；控制实验生成JSON task（deepseek-ai/DeepSeek-V4-Flash与zai-org/GLM-5-FP8端点）。
- **代码/权重**：论文声明配套artifact含独立repository，包含`src/capability_sheaves`（task generation、exact CSP、quotient construction、analysis）、`configs`（protocols、restriction maps）、`data/raw`（200 baseline pairs）、`data/latent_quotient/raw`（1,000 hidden-state pairs）、`results/swebench_real`（development split results）；无endpoint credential。
- **关键超参**：filled bundle pool大小≤3的子集共25个；hidden-state experiment使用20 independent clusters；token计量仅在endpoint内比较；sign-test为exact one-sided paired。
