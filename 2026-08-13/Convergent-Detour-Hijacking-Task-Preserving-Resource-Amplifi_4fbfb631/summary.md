---
title: "Convergent-Detour-Hijacking-Task-Preserving-Resource-Amplifi"
source: https://arxiv.org/pdf/2608.12273v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:57:27"
field: "LLM Agent 安全与鲁棒性"
keywords: ["LLM Agent", "Skill Registry", "Progressive Disclosure", "Security Attack", "Resource Amplification", "Trajectory Necessity", "Tool Selection"]
innovations: ["提出跨阶段协同攻击 CDH，耦合 selection 与 planning 阶段的静态文本操控", "构造 pilot-guided 黑盒 description 优化与 bounded runbook body 生成协议", "定义轨迹必要性作为新安全指标，揭示任务完成正确性不等于执行安全"]
benchmarks: ["OpenClaw 53-Skill Benchmark", "491 Held-out Multi-skill Tasks", "30 Independently Authored Generalization Tasks"]
---

# 论文速读：Convergent-Detour-Hijacking-Task-Preserving-Resource-Amplifi

## 一句话总结
本文提出了 **Convergent Detour Hijacking (CDH)** 攻击，针对基于技能的 LLM Agent 中常见的渐进式披露（progressive-disclosure）设计，构造一个仅由发布者可控的静态协调者技能，通过描述层获取路由共选、指令体层诱导冗余技能调用链，最终在保持任务完成率的前提下显著放大资源消耗与执行延迟。

## 研究问题与动机
- **核心问题**：渐进式披露架构将技能选择（description）与执行规划（body）分离为两个独立阶段，攻击者只需控制一个静态技能，即可在两端施加影响，形成跨阶段的协同攻击。
- **现有方法的不足**：
  1. 已有工作分别研究元数据-based 的选择操纵、恶意技能指令和运行时工具链资源放大，但未揭示三者端到端组合后的复合风险。
  2. 传统安全评估仅关注最终输出正确性，忽略了任务执行轨迹的必要性与成本安全性。
  3. 渐进式披露的上下文效率优化反而引入了不可见的轨迹级攻击面，而现有的技能审计框架未覆盖此类跨阶段依赖。
  4. 现有后门攻击通常依赖可执行载荷或运行时响应控制，CDH 在更严格的黑盒发布场景下仍能实现资源放大。

## 核心贡献（创新点）
1. **跨阶段攻击形式化**：首次将选择阶段路由偏向与规划阶段轨迹操控统一为"吸引-绕道-汇聚"范式，定义了对比干净执行与注入执行轨迹必要性的安全属性。
2. **无执行载荷的静态协调者构造方法**：提出 pilot-guided 黑盒筛选流程生成 description，并结合语义对齐的 bounded runbook 生成 body，两者共享同一协调公理 $\rho_g$，确保跨阶段一致性。
3. **491 任务基准与配对评估协议**：在 OpenClaw 53 技能 registry 上构建九组功能分类的 held-out benchmark，通过 clean-vs-injected 配对执行分离路由行为与端到端执行，量化 CDH 成功率与资源放大幅度。
4. **轨迹必要性作为新安全指标**：论证了"任务完成正确 ≠ 执行轨迹必要"，为可扩展 Agent 系统的安全审计提出 trajectory-level 评估标准。

## 方法详解
### 威胁模型
- 攻击者向 registry 注入一个静态协调者技能 $h = (d_h, b_h, f_h)$，其中 $f_h$ 为空或无害 stub。
- 攻击者无法访问模型内部、受害者 prompt、运行时响应或执行状态，仅在发布前一次性构造 $d_h$ 与 $b_h$。

### 共享协调公理 $\rho_g$
对每个功能组 $g$，定义 $\rho_g = (\mathcal{T}_g, c_g, \mathcal{L}_g, q_g)$：
- $\mathcal{T}_g$：目标域概念与意图
- $c_g$：非执行型协调角色
- $\mathcal{L}_g$：技能间合理依赖关系
- $q_g$：非替换边界与回归条件

### Description 吸引（选择阶段）
采用 generate–evaluate–retain 循环优化 $d_g$：
$$\mathcal{C}_g^{(r+1)} = \mathrm{Gen}\Big(\rho_g, \mathrm{TopK}\{(d, A_g(d)) : d \in \mathcal{C}_g^{(r)}\}\Big)$$
其中 $A_g(d)$ 为协调者在 pilot 任务上的共选率。经过两轮反馈后固定描述，不进行任务完成/延迟/ token 反馈。

### Body 构造（规划阶段）
按六条规则生成 bounded runbook $b_g$：
1. **依赖拓扑**：在功能组内构造有向执行路径
2. **边合理化**：每条依赖边关联省略后果（如数据丢失、状态不一致）
3. **强制性表述**：将可选步骤包装为程序要求
4. **证据事件分解**：将技能责任拆分为 pre-/execution-/post-operation 阶段
5. **后置验证**：允许最多一次验证重访，设置每技能及总调用上限
6. **结构化打包**：Markdown runbook 含激活范围、依赖路径、执行协议、验证边界与回归条件 $R_g$

### CDH 形式化定义
$$\mathrm{CDH}_h(x) = \mathbf{1}\left[\begin{array}{l} h \in \Lambda_h(x),\; \mathrm{Skills}(\Gamma_0(x)) \subseteq \mathrm{Skills}(\Gamma_h(x)), \\ \left(\mathrm{Skills}(\Gamma_h(x)) \setminus \mathrm{Skills}(\Gamma_0(x))\right) \cap S_0 \neq \emptyset, \\ \mathrm{Eval}(x, y_0(x)) = \mathrm{Eval}(x, y_h(x)) = 1 \end{array}\right]$$

攻击成功率 $\mathrm{ASR}_{\mathrm{CDH}}(h) = \frac{1}{|\mathcal{D}^+|} \sum_{x \in \mathcal{D}^+} \mathrm{CDH}_h(x)$。

## 实验与结果
- **平台**：OpenClaw v2026.5.7，53 个原生技能，划分为九组功能类别
- **基准**：536 任务（45 pilot + 491 held-out），独立作者验证 30 任务
- **模型后端**：Claude-Haiku-4.5、DeepSeek-V4-Pro、Qwen3.7-Max/Plus、DeepSeek-V4-Flash、MiniMax-M3
- **核心结果（DeepSeek-V4-Pro）**：
  - 隔离路由吸引率：14.20%
  - 端到端协调者命中率：80.02%（单任务）/ 88.98%（多轮）
  - CDH 成功率：73.82%（单任务）/ 82.81%（多轮）
  - Token 消耗增幅：+66.91%（单任务）/ +91.20%（多轮）
  - 缓存 token 增幅：+54.33% / +80.71%
  - 端到端执行时间增幅：+92.45% / +20.57%
  - 平均调用增量：+2.20 / +2.03
  - 任务完成率：93.6%（干净）vs 94.3%（注入），差距 ≤1.5pp
- **最强结果**：MiniMax-M3 单任务协调者命中率达 96.60%，Token 增幅 +80.81%，Time 增幅 +26.99%
- **泛化验证**：30 个独立 authored 任务中协调者命中率 33.33%，全部触发 CDH；跨组任务亦有效

## 相关工作脉络
1. **Attractive Metadata Attack**（Mo et al., 2025；He et al., 2026）：聚焦 description 编辑导致的路由偏向，但未研究 body 阶段的轨迹操控与资源放大。
2. **SkillTrojan / BadSkill**（Tie et al., 2026；Feng et al., 2026a）：研究 skill 内部后门注入，依赖可执行载荷或模型中毒，CDH 仅需静态文本且不替换原生技能。
3. **Clawdrain**（Dong, Feng, and Wang, 2026）：利用工具调用链实现 token 耗尽，但需配合 companion script 与运行时进度信号；CDH 在无执行伴侣、无运行时控制的更弱假设下生效。
4. **Sponge Tool Attack**（Li and Wang, 2026）：通过重写受害者 query 诱导低效推理，CDH 不修改 query，仅靠静态技能嵌入改变执行轨迹。
5. **STARS / RouteGuard**（Zhang et al., 2026；Xiao et al., 2026）：提出技能触发审计与内部信号检测，但未覆盖跨 progressive-disclosure 边界的协同攻击形态。
6. **CLAW Drain 等 DoS 后门**（Zhang et al., 2025b；Luo et al., 2026）：研究可用性攻击放大，CDH 的独特定位在于"结果保留 + 轨迹扩张"的隐蔽性。

## 局限性与未来方向
- **平台单一性**：仅在 OpenClaw registry 与九组技能上验证，未覆盖 MCP、LangChain 等异构工具生态。
- **组匹配限制**：当前协调者仅针对同功能组任务设计，跨组激活率显著下降（独立任务命中 33.33%）。
- **Mock 后端替代**：部分技能使用本地 mock 实现，可能无法完全反映真实 API 调用的行为差异。
- **防御评估缺失**：论文提出预安装审查与运行时预算监控两个方向，但未提供系统性防御实验。
- **多轮累积效应**：未充分探索长时间会话中协调者技能的累积路由偏向与上下文污染。
- **未来方向**：扩展到跨功能组通用协调者、研究 trajectory necessity 的自动化审计方法、构建 invocation budget 执行框架。

## 研究启发与可借鉴点
1. **Pilot-guided 黑盒优化范式**：通过 generate–evaluate–retain 循环迭代 description 候选，仅依赖二元选择反馈而非梯度，可迁移至其他工具选择优化场景。
2. **Shared rationale 跨阶段一致性设计**：以 $\rho_g$ 为锚点同时指导 description 与 body 构造，避免后验对齐成本，适用于任何需跨模块语义协同的系统。
3. **Bounded runbook 模式**：将依赖关系显式化为带上限的有向路径 + 终止条件，既保证攻击有效性又避免无限循环，可作为安全审计的参考模板。
4. **Clean-vs-Injected 配对评估协议**：通过严格控制单一变量（协调者存在与否）分离路由与执行影响，值得推广至 Agent 安全基准构建。
5. **轨迹必要性作为防御信号**：提出"任务完成 ≠ 安全"的核心洞察，可启发团队在后续工作中引入 execution path analysis 而非仅依赖 output verification。

## 关键术语表
- **Progressive Disclosure**：渐进式披露，Agent 平台仅在技能被选中后才加载其完整指令体的设计，以提升上下文效率。
- **Coordinator Skill**：协调者技能，攻击者发布的静态技能，负责诱导额外技能调用但不直接完成任务。
- **CDH Success Rate (ASR)**：收敛绕道劫持成功率，衡量攻击任务中满足四条件（命中、保留原技能、招募新技能、双端完成）的比例。
- **Attract–Detour–Converge**：吸引-绕道-汇聚路径，CDH 攻击的三段式执行流，描述层吸引共选，body 层诱导绕道，回归条件触发汇聚。
- **Isolated Routing Evaluation**：隔离路由评估，仅测试 router 对技能的排序选择，不含实际执行，用于分离路由偏向与执行影响。
- **Bounded Runbook**：有界操作手册，body 的六规则构造协议，通过预设依赖路径与调用上限控制绕道规模。
- **Trajectory Necessity**：轨迹必要性，指执行路径中每个技能调用对任务完成的不可或缺性，CDH 破坏此属性但保持输出正确性。
- **Pilot–Evaluation Separation**： pilot-评估分离，45 个 pilot 任务仅用于 description 优化，491 个 held-out 任务用于最终评估，防止数据泄露。

## 可复现要素
- **数据集**：491 个 held-out 多技能任务 + 45 个 pilot 任务，基于 OpenClaw 默认 53 技能 registry 构建，**已开源**（含 task JSON 与 fixture 配置）
- **代码/权重**：Benchmark、mock 实现、任务本地化接口与执行脚本均已在 Supplementary Materials 中**公开**
- **关键超参**：
  - TF-IDF 词级特征上限：800；字符级上限：300
  - Action-domain 权重 $\alpha = 0.5$；字符权重 $\lambda = 0.10$
  - 对比修正强度 $\beta = 0.3$
  - 执行超时：600s
  - Description 候选生成轮数：2–3 轮
- **模型后端**：六种 LLM（Claude-Haiku-4.5、DeepSeek-V4-Pro/Flash、Qwen3.7-Max/Plus、MiniMax-M3），temperature=0
- **环境**：Ubuntu VM + VirtualBox 隔离，本地 mock backend 替代外部依赖技能
