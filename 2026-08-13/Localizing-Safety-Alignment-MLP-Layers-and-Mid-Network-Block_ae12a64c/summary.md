---
title: "Localizing-Safety-Alignment-MLP-Layers-and-Mid-Network-Block"
source: https://arxiv.org/pdf/2608.11583v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:45:27"
field: "LLM安全性与机制可解释性"
keywords: ["safety alignment", "mechanistic interpretability", "weight transplantation", "refusal behavior", "MLP layers", "model localization"]
innovations: ["发现可转移拒绝行为主要由MLP权重介导而非Attention（至少2.7倍优势）", "定位到中间深度Block 3 (layers 8-11)为拒绝行为最集中区域，跨两种模型架构一致", "揭示MLP块组合的非加性交互，6块子集优于全MLP移植"]
benchmarks: ["TwinPrompt", "SGXSTest", "AdvBench", "OR-Bench"]
---

# 论文速读：Localizing-Safety-Alignment-MLP-Layers-and-Mid-Network-Block

## 一句话总结
本文通过从安全对齐模型向对应未对齐基础模型的**选择性权重移植**实验，定位了LLM中拒绝行为在权重空间中的集中区域：**可转移的拒绝行为主要由MLP层编码，且高度集中在网络中间深度（layers 8–11的Block 3）；不同MLP块的组合呈现非单调、非加性交互**。

## 研究问题与动机
- **核心问题**：安全对齐的拒绝行为在模型权重空间中具体编码于哪些参数、位于哪些深度？现有机制解释工作多在激活空间或编辑层面讨论，缺乏直接的**权重空间定位**研究。
- **现有方法不足**：RLHF/指令微调等对齐方法虽能提升安全性，但拒绝行为在实践中表现出**脆性**——容易被针对性参数更新削弱或被jailbreak攻击绕过；同时存在**过度拒绝（over-refusal）**问题，在无害提示上误拒。现有研究未从权重层面回答"哪些参数真正携带了可转移的拒绝信号"。

## 核心贡献（创新点）
- **MLP路径主导**：系统性地证明可转移的拒绝行为主要由MLP权重而非Attention权重介导，MLP移植的拒绝效果在全部6组实验中均超越Attention移植至少2.7倍。
- **中间深度集中定位**：在所有6次贪心搜索中，Block 3（layers 8–11）始终被最先选中，且在单个块实验中表现最强，表明拒绝相关参数在MLP堆栈中存在一致的中间深度集中现象。
- **非加性块组合发现**：揭示对齐MLP块之间存在非单调交互——多数情况下添加更多对齐块反而降低拒绝性能，5/6次贪心轨迹中出现负边际效应，且**选定的6块子集在多项指标上优于完整MLP移植**。
- **跨基准迁移性分析**：将不同恶意基准派生的贪心排序迁移到OR-Bench评估，发现基准选择显著影响精度-覆盖权衡，AdvBench派生顺序更保守但超拒率高，SGXSTest派生顺序更宽松。
- **更大样本验证**：在100条恶意+100条良性提示的配对子集上重复实验，验证了上述三个核心模式的稳健性，排除了小样本假阳性可能。

## 方法详解
- **匹配模型对**：使用两组架构相同的安全对齐/未对齐模型对：① RealSafe-R1-7B（28层，Qwen2架构）vs DeepSeek-R1-Distill-Qwen-7B；② saferlhf_ultra_sft（32层，Llama架构）vs Llama-3.1-8B。因架构一致，行为差异可归因于移植权重。
- **多层级粒度移植**：
  - **组件级**：attn（替换所有层的全部attention投影：q/k/v/o proj）vs mlp（替换所有层的gate/up/down proj）
  - **连续区域级**：first5/mid5/last5（各5层连续的attention+MLP）vs first_half/second_half
  - **MLP块级**：将MLP权重按每4层划分为块（28层→7个块B1–B7，32层→8个块B1–B8），依次移植单块及组合
- **贪心前向选择**：从基础模型出发，迭代选取使拒绝数提升最大的MLP块加入，直到全部移植完毕；当拒绝数相同时优先选含更多"可接受安全回答"的块。该过程为每个模型-数据集对产生一个重要性排序和每个预算k下的最优块子集。
- **评估子集构建**：对恶意基准，保留"对齐模型拒绝但基础模型顺从"的提示（每条件最多30条）；对良性基准同理筛选。另构建了100条的大验证子集进行稳健性检验。
- **评估指标**：MR（malicious refusal，恶意拒绝数，越高越好）和BOR（benign over-refusal，良性过度拒绝数，越低越好）。答案经人工审核。
- **基准**：TwinPrompt、SGXSTest、AdvBench（恶意）和OR-Bench（良性过拒）。

## 实验与结果
- **组件级实验（Table 2）**：
  - RealSafe-R1-7B：mlp配置在TwinPrompt/SGXSTest/AdvBench上MR分别为19/17/27（满分30），attn配置仅为5/2/10——**MLP超越Attn至少2.7倍**
  - saferlhf_ultra_sft：mlp配置MR为21/4/17，同样远超attn（5/0/0）
  - MLP移植同时带来过拒风险：RealSafe-R1-7B的mlp在TwinPrompt上BOR=11/30，而attn为0

- **单块实验（Tables 3–4）**：
  - Block 3（layers 8–11）在两种模型对中均为最强单块：RealSafe-R1-7B在TwinPrompt上MR=7（最高）、AdvBench上MR=10（最高）；saferlhf_ultra_sft在TwinPrompt上MR=2（最高）
  - 同一绝对深度范围在Qwen2-7B和Llama-3.1-8B中均出现，说明跨架构稳定性

- **贪心块组合（Tables 6–7, Figure 1）**：
  - Block 3在所有6次搜索中均最先被选中
  - RealSafe-R1-7B TwinPrompt：k=6时MR=24最高，k=7（全MLP）降至MR=19；BOR也从9增至11
  - RealSafe-R1-7B SGXSTest：k=6时MR=18最高，k=7降至17
  - saferlhf_ultra_sft SGXSTest：k=6时MR=8/BOR=1，全8块时降至MR=4/BOR=3
  - **5/6次轨迹出现负边际效应**（添加某块后MR下降）

- **OR-Bench迁移（Table 8）**：
  - SGXSTest派生顺序在RealSafe-R1-7B上5/7个预算点取得最低或并列最低BOR
  - saferlhf_ultra_sft的SGXSTest顺序在大部分k值保持零过拒
  - 存在明显的精度-覆盖权衡：低过拒顺序通常伴随低恶意拒绝

- **Block 3内部细化（Table 9）**：
  - Layer 8在三个基准中均表现最强：TwinPrompt MR=5、SGXSTest MR=5、AdvBench MR=6
  - Layer 8的单层移植即可达到全Block约半数以上的MR

- **大样本验证（Tables 10–12）**：
  - mlp hybrid在100条恶意提示上MR=61，attn仅8条；MLP优势再次确认
  - mlp hybrid在100条良性提示上BOR=33（vs attn的2），精度代价同步显现
  - Block 3以13/100 MR仍是最高单块，BOR仅1/100
  - 贪心轨迹重现非单调性：k=6时MR=64最高，k=7（全MLP）降至61

- **最强结果**：RealSafe-R1-7B MLP全移植在AdvBench上达到MR=27/30（90%），但BOR代价较高；选定的6块子集在多项任务上实现了更好的综合平衡。

## 相关工作脉络
- **Arditi et al. (2024)**：提出拒绝由单一方向介导，消融该方向抑制拒绝。本文在其激活空间发现基础上，将问题推进到权重空间，定位MLP层和中间深度。
- **Wollschlager et al. (2025)**：主张拒绝由多维多锥体组织。本文与之互补，从参数移植角度独立验证拒绝的非均匀分布。
- **Wei et al. (2024)**：通过剪枝和低秩修改发现安全关键区域稀疏（约3%参数），且安全与效用更分化于MLP层。本文以移植而非剪枝的方式独立复现了MLP主导结论，并进一步定位到具体深度范围。
- **Zhao et al. (2024) / Lee et al. (2024)**：前者通过层剪枝识别"safety layers"并提出LED编辑方法；后者提出CAST进行条件激活引导。本文的实验结果为这类干预方法提供了更精细的权重级目标区域（Block 3/layers 8–11）。
- **安全评估基准**：TwinPrompt/SGXSTest/AdvBench/OR-Bench/XSTest等基准的引用，为本文的双轴评估框架（恶意拒绝+良性过拒）提供支撑，并凸显了选择源基准对下游精度影响的实践意义。

## 局限性与未来方向
- **模型规模与架构覆盖有限**：仅测试了7B–8B量级的两个模型对（Qwen2和Llama），跨尺度（更大模型）、跨架构（MoE、稀疏架构）和不同对齐流程的泛化性尚未验证。
- **评估子集较小**：主要实验基于最多30条/条件的过滤子集，虽有大样本验证部分结果，但整体分布外泛化仍需更多工作。
- **贪心选择仅为近似**：前向贪心不能保证找到全局最优块组合，可能存在更优的子集组合未被探索。
- **因果机制未深入**：本文定位了"在哪里"，但未揭示MLP块中"如何"编码拒绝——需要结合激活 patching、因果追踪或神经元探针等机制分析方法。
- **对齐过程差异**：saferlhf_ultra_sft的拒绝转移效果弱于RealSafe-R1-7B，可能与训练数据量和多样性有关，但未充分讨论。

## 研究启发与可借鉴点
- **权重移植实验范式**：利用架构相同的对齐/基础模型对进行选择性参数移植，是一种干净分离参数贡献的高效方法，可迁移到其他行为（如推理能力、工具使用）的定位研究。
- **MLP中间层定位的发现**：Block 3（layers 8–11）作为拒绝行为的关键区域，为后续的精 sure干预（如定向编辑、安全层保护、对抗防御）提供了明确的参数级目标，可与团队现有的编辑/干预工作结合。
- **非加性交互的实用价值**：发现"全MLP移植不如部分移植"的结论，提示在实际安全增强中应做**精选子集**而非全盘移植，可指导更高效的模型安全微调策略。
- **基准选择的影响量化**：本文展示了源基准对OR-Bench迁移效果的显著影响，为安全评估实验设计提供了方法论教训——单一基准上的贪心顺序不能直接泛化，需要在部署场景对应的基准上验证。
- **大样本配对子集构建**：通过保留对齐-基础模型的分歧样本构建过滤子集的方法，是一种控制混淆变量的有效实验设计，可用于其他需要对比特性的研究中。

## 关键术语表
- **Malicious Refusal (MR)**：模型对恶意提示的拒绝次数，越高表示安全防御越强。
- **Benign Over-Refusal (BOR)**：模型对良性提示的误拒绝次数，越低表示安全性越精确。
- **Weight Transplantation**：将对齐模型的指定参数子集移植到架构匹配的基础模型中，以评估这些参数对特定行为的贡献。
- **Greedy Forward Selection**：从空集出发，每次选择能使目标指标增益最大的块加入的迭代选择算法。
- **Negative Marginal Effect**：添加某个对齐块后，模型的恶意拒绝性能反而下降的现象，反映块间非加性交互。
- **Block 3**：layers 8–11对应的MLP块，在所有贪心搜索中均最先被选中，是拒绝行为最集中的参数区域。
- **Precision-Coverage Trade-off**：恶意拒绝率（覆盖度）与良性误拒率（精度）之间的此消彼长关系。
- **Alignment Brittleness**：对齐后的安全行为在后续微调或参数修改下容易退化的脆弱性。

## 可复现要素
- **数据集**：TwinPrompt、SGXSTest、AdvBench、OR-Bench——论文未明确说明开源状态，但均为公开发布的benchmark
- **模型权重**：RealSafe-R1-7B、DeepSeek-R1-Distill-Qwen-7B、saferlhf_ultra_sft、Llama-3.1-8B——均在Hugging Face公开
- **代码**：论文未提及代码/仓库开源声明
- **关键超参**：Temperature=0.6；max new tokens=1200（子集过滤）、2000（响应生成）；MLP块大小=4层；每子集最多30条提示
- **评估生成配置**：见Appendix Section 9
