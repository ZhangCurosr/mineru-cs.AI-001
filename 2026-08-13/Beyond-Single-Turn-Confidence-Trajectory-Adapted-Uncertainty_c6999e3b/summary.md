---
title: "Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty"
source: https://arxiv.org/pdf/2608.11552v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:41:45"
field: "大语言模型 Agent 不确定性量化"
keywords: ["uncertainty quantification", "LLM agents", "trajectory-level confidence", "white-box scorer", "black-box consistency", "reflexive scoring", "tool-use agents"]
innovations: ["系统评估三类单轮UQ方法在多轮Agent轨迹上的迁移性能并揭示其失败模式的异质性", "提出跨轮聚合策略和新的一致性度量（TER/ASC/AEC等）以适配轨迹级不确定性估计"]
benchmarks: ["BFCL-v4", "τ²-bench (airline/telecom/retail)"]
---

# 论文速读：Beyond-Single-Turn-Confidence-Trajectory-Adapted-Uncertainty

## 一句话总结
本文系统评估了三种单轮UQ方法家族（白盒token概率、黑盒自洽性、反射性自我评估）向LLM Agent多轮交互轨迹层面的迁移效果，发现迁移性能参差不齐，**黑盒自洽性（TER/ASC）**通常最强，**反射性**是低成本基线，但没有任何单一方法在所有模型和任务域上均可靠。

## 研究问题与动机
- **核心问题**：LLM Agent以交互轨迹为单元，不确定性在多个决策步骤中累积传播，早期错误会传导至后续动作，单轮UQ方法是否仍能给出有效的失败信号？
- **现有方法不足**：
  1. 绝大多数单轮UQ工作仅评估静态输出，未考虑轨迹层面跨轮聚合的复杂性；
  2. 已有Agent UQ工作提出SAUP/UProp等传播方法，但缺乏在多模型、多数据集上的统一对比；
  3. 单轮白盒/黑盒/反射性方法直接迁移到轨迹上时，其判别力、校准性和选择性预测表现缺乏系统实证。

## 核心贡献（创新点）
- **受控实证比较**：在5个LLM×4个多轮工具使用数据集上，对比三类UQ家族的区分度、校准与选择性预测，揭示迁移性能的结构性差异。
- **轨迹级适应性改造**：将白盒评分扩展为跨轮聚合形式（first/mean/min/last及位置加权），并提出新的黑盒一致性度量（TER基于LLM judge的轨迹等价判断、ASC/AEC等动作结构化度量）。
- **发现三类方法失败的异质性模式**：白盒性能高度依赖聚合器选择（同一基础分数切换聚合规则可致AUROC从0.73降至0.23）；反射性在大多数模型-数据集对上提供稳健的低成本基线；黑盒自洽性整体最强但成本高，且TER/ASC在多数场景下排名最高。

## 方法详解

### 白盒评分器（White-Box）
- **基础单轮分数**：
  - 序列概率 $SP(y)=\prod_{j=1}^{L}p_j$
  - 长度归一化序列概率 $LNSP(y)=(\prod_{j=1}^{L}p_j)^{1/L}$
  - 平均token负熵 $ATN@K(y)=\frac{1}{L}\sum_{j=1}^{L}(1-TE@K(t_j)/\log K)$，其中$TE@K$为截断top-K熵
- **跨轮聚合**：对每轮动作span计算$s_i$，然后以$g_{\text{first}}=s_1$、$g_{\text{mean}}=\frac{1}{T}\sum s_i$、$g_{\text{min}}=\min_i s_i$、$g_{\text{last}}=s_T$或位置加权聚合得到轨迹级分数。

### 黑盒一致性评分器（Black-Box Consistency）
- 在温度0.7下采样$m=3$条候选轨迹$\tilde{\mathbf{F}}$，与贪婪参考轨迹$F$比较：
  - **NCP**（最终消息一致性）：使用DeBERTa-large-MNLI测量不矛盾概率
  - **FAC**（首动作一致性）：首步动作类型匹配率
  - **ASC**（动作集一致性）：各轨迹动作类型集合的Jaccard相似度均值
  - **ADC**（动作分布一致性）：$1-\mathrm{JSD}(q_F\|q_{\tilde{F}_j})$
  - **AEC**（动作编辑一致性）：$1-\mathrm{ED}(\sigma_F,\sigma_{\tilde{F}_j})/\max(T_F,T_{\tilde{F}_j})$
  - **TER**（轨迹等价率）：使用gemini-flash-lite作为judge，判断采样轨迹与参考轨迹是否达到相同任务相关结果（仅输入轨迹transcript，不泄露奖励标签）。

### 反射性评分器（Reflexive）
- 仅读取截至最后一轮的agent动作前缀，请求模型自我评估：
  - **P(True)**：输出"True"/"False"，置信度取"True" token概率
  - **VC**：输出Yes/No及0-1概率，置信度为声称概率或其补数

## 实验与结果
- **数据集**：BFCL-v4多轮子集（200任务）+ τ²-bench的airline（50）、telecom（114）、retail（114）三个文本域，共4个多轮工具使用数据集。
- **模型**：Qwen2.5-7B、gpt-oss-20b、Qwen3.5-9B、MiniMax-M3、gpt-4o-mini。
- **主要指标**：AUROC（primary）、AUPRC、PRR、ECE。
- **关键结果**：
  - **白盒**：BFCL-v4上表现较好（mean/min ATN@5可达0.65–0.71）；但τ²-bench上极不稳定，如Qwen3.5-9B在telecom上SP_mean AUROC=0.73，切换为SP_last仅0.62，而 airline上mean聚合几乎在机会水平以下；240个白盒单元格中91个低于0.5。
  - **反射性**：BFCL-v4上P(True)达0.75–0.85，VC达0.75；τ²-bench上retail最高0.71、telecom最高0.852，但airline跨模型差异大（P(True)从0.225到0.659）。
  - **黑盒自洽性**：整体最强家族。airline上Qwen3.5-9B的TER达**0.868**（峰值），ASC达0.819；BFCL-v4上TER为0.74–0.77，ASC为0.69–0.77。但TER在gpt-4o-mini telecom上骤降至0.342（成功率仅11%）。
  - **无统一最优**：没有任何单个家族或评分器在所有模型-数据集组合上均可靠。
- **成本对比**（Table 2）：白盒AUROC均值0.628、峰值0.725；反射性均值0.691、峰值0.885；action一致性均值0.705、峰值0.849；message一致性均值0.686、峰值0.868。

## 相关工作脉络
- **单轮UQ基础工作**（Kadavath et al. 2022, Kuhn et al. 2023, Manakul et al. 2023, Farquhar et al. 2024, Lin et al. 2024）：本文将其作为三类基线方法家族，并在轨迹层面重新评估其迁移性。
- **Agent UQ传播方法**（SAUP Zhao et al. 2025; UProp Duan et al. 2025）：本文将其adapt为action-level propagation controls并对比，发现其在τ²-bench上平均仅0.434 AUROC，低于固定朴素聚合。
- **轨迹级形式化**（Oh et al. 2026）：本文在其agent-UQ框架基础上扩展了聚合器和一致性度量，并做系统性实证。
- **多轮confidence estimation**（Zhang et al. 2026a）：关注per-turn calibration和单调性；本文则聚焦trajectory-level成功预测。
- **工具使用校准**（UA LA Han et al. 2024; ProbeCal Liu et al. 2024; Xuan et al. 2026）：本文定位为其在统一基准上的横向对比补充。
- **结构化澄清Agent**（Suri et al. 2026）：使用结构化工具调用参数不确定性；本文与之互补，提供通用UQ方法对比。

## 局限性与未来方向
- **基准范围有限**：仅覆盖多轮工具使用数据集（BFCL-v4 + τ²-bench三个文本域），未验证于web browsing或embodied agent等开放环境。
- **标签噪声**：成功标签由数据库状态、环境断言、LLM judge等复合构成，噪声上限了判别力；模拟器故障也可能混入失败样本。
- **接口局限**：白盒评分依赖文本action接口暴露token logprobs；native tool-calling ablation仅在gpt-4o-mini+retail上验证，未覆盖全部模型。
- **统计效力不足**：airline仅50任务，bootstrap区间宽（中位半宽0.10），部分列被标注为"minority n<20"需谨慎解读。
- **Judge依赖**：TER使用单一外部judge（gemini-flash-lite），虽经数据库ground truth验证（precision 0.88/recall 0.93）且替换为大模型后AUROC变化≤0.062，但未测试更多judge变体。
- **未来方向**：扩展到step-level label以按动作类型/工具族评估；探索在执行过程中利用per-turn分数指导abstain/clarify/escalate；在其他基准（如web browsing）上验证泛化性。

## 研究启发与可借鉴点
- **聚合器选择是关键超参**：白盒方法不能"plug-and-play"，跨轮聚合策略需针对部署域和模型单独验证，避免反相关风险。
- **黑盒自洽性在高预算场景具有吸引力**：当任务已有pass@k式采样需求或高 stakes部署可承受额外轨迹成本时，TER/ASC是更强的失败排序信号。
- **反射性作为低成本强基线**：仅需一次额外self-evaluation pass，在多数模型-数据集对上提供稳定AUROC 0.75+，适合快速迭代阶段的默认选项。
- **一致性信号的"双刃剑"效应**：附录G揭示一致性既可能反映重复成功也可能反映"stuck policy"（重复失败），需结合outcome-aware诊断（如success-success vs failure-failure分解）才能正确解读。
- **接口选择应明确声明**：白盒UQ在native tool-calling下的可用性因模型/provider而异，实验设计需显式控制接口变量以避免混淆。

## 关键术语表
- **Uncertainty Quantification (UQ)**：为模型输出提供不确定性/置信度评分，以区分正确与错误回答。
- **Trajectory**：LLM Agent在多轮交互中产生的完整动作-观察序列 $F_{\leq T} = \{(A_1,O_1),\ldots,(A_T,O_T)\}$。
- **White-box Scorer**：基于模型内部token概率计算的置信度分数（如SP、LNSP、ATN@K）。
- **Black-box Consistency Scorer**：通过多次采样轨迹并比较输出一致性来估计不确定性（如TER、ASC、NCP）。
- **Reflexive Scorer**：要求模型对自身已完成轨迹进行自我评估（如P(True)、VC）。
- **Trajectory Equivalence Rate (TER)**：使用辅助LLM judge判断采样轨迹与参考轨迹是否达到相同任务结果的比率。
- **Action-Set Consistency (ASC)**：基于动作类型集合Jaccard相似度衡量跨轨迹一致性，对顺序和重复次数不变。
- **Prediction Rejection Ratio (PRR)**：选择性预测指标，衡量拒绝最低置信度预测后提升的成功率，归一化至[0,1]。

## 可复现要素
- **数据集**：BFCL-v4（Apache-2.0）+ τ²-bench（MIT）；多轮工具使用benchmark，数据集公开。
- **代码/权重**：论文未明确声明代码开源，但附录中提到numeric grid随发布；模型通过Together AI和OpenAI API访问。
- **关键超参**：采样轨迹数$m=3$，采样温度$T=0.7$，参考轨迹温度$T=0$（greedy），ATN@K中$K=5$，NLI模型使用microsoft/deberta-large-mnli，TER judge使用gemini-flash-lite（T=0）。
