---
title: "Convergent-Detour-Hijacking-Task-Preserving-Resource-Amplifi"
source: https://arxiv.org/pdf/2608.12273v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:58:38"
field: "LLM Agent 安全与鲁棒性"
keywords: ["LLM Agent", "Skill Registry", "Convergent Detour Hijacking", "Progressive Disclosure", "Supply-Chain Attack", "Resource Amplification", "Trajectory Necessity"]
innovations: ["提出 CDH 跨阶段攻击形式化与 Attract–Detour–Converge 构造方法", "构建 491 任务 OpenClaw 配对评测基准并证明任务正确不蕴含轨迹必要", "在无运行时反馈、payload-less 条件下实现最高 80.51% ASR 与 ~90% 资源放大"]
benchmarks: ["OpenClaw default registry 53 skills", "491 held-out tasks (9 functional groups)", "30 independently authored generalization tasks"]
---

# 论文速读：Convergent-Detour-Hijacking-Task-Preserving-Resource-Amplifi

## 一句话总结
本文提出 **Convergent Detour Hijacking (CDH)**，一种针对基于技能（skill-based）LLM Agent 的"只控静态技能"供应链攻击，通过渐进式披露机制耦合**选择阶段**的元数据操控与**规划阶段**的指令体依赖构造，使攻击者发布的协调者技能被选中并加载后，诱使代理执行一段额外调用多个合法技能的"绕道轨迹"，最终仍恢复原任务路径并给出正确答案。

## 研究问题与动机
1. **渐进式披露的跨阶段风险未被形式化**：现有 skill-based Agent 为节省上下文，只在选阶段暴露描述（description），在加载后才展示指令体（body），两类数据均由发布者控制，但其端到端组合效应未知。
2. **已有工作割裂研究**：选择操纵、恶意技能指令、运行时工具链资源放大三类攻击分别独立研究，未覆盖"一个静态技能同时驱动两轮影响"的串联场景。
3. **任务正确性不等于轨迹必要性**：最终输出正确并不能排除执行轨迹被恶意拉长、消耗更多 token 与时间。
4. **黑盒条件下构造可靠协调器难以保证**：需要在无运行时反馈、无法访问模型内部的情况下，仅凭元数据与指令体静态文本，实现跨阶段一致诱导。

## 核心贡献（创新点）
1. **跨阶段攻击形式化（CDH 定义）**：首次将"选择 + 规划"两阶段影响耦合，提出以成对干净/注入执行对比定义的 Convergent Detour Hijacking 概念，明确 hijacking、clean-route 保留、native-skill 绕道、convergence 四个条件。
2. **Attract–Detour–Converge 构造方法**：提出 pilot-guided 黑盒优化描述以获取 co-selection，同时设计共享协调原则（shared coordination rationale）$\rho_g$ 确保描述与 body 语义一致；body 采用有限制的 prerequisite/verification 规则构造有界绕道并显式 return 原任务。
3. **首个公开的多技能基线与配对评测协议**：在 OpenClaw 默认 registry 的 53 个技能上构建 9 个功能组与 491 个 held-out 任务集，提供 clean/injected 配对协议、独立人工标注的完成评测、缓存 token/墙钟时间等多维资源指标。
4. **攻击有效性实证**：在 DeepSeek-V4-Pro 等 6 个 LLM 后端上实现 70.82%–80.51% 单任务 ASR、最高 96.60% 协调器命中，并在不降低任务完成率的前提下显著放大资源开销。

## 方法详解
- **威胁模型**：攻击者只能向干净 registry $S_0$ 追加一个静态 coordinator $h=(d_h,b_h,f_h)$，无法接触模型内部、运行时响应、其他技能或系统提示，发布后不再交互。
- **共享协调原则 $\rho_g=(T_g,c_g,\mathcal{L}_g,q_g)$**：$T_g$ 为目标域概念；$c_g$ 定义非执行型协调角色；$\mathcal{L}_g$ 记录合理跨技能关系；$q_g$ 规定非替换边界与返回纪律。
- **描述优化**：候选描述含两子句——activation clause（建立路由相关）与 coordination clause（以 Organizer 身份强调协同、不取代原生技能）。通过 45 个 pilot 任务的两阶段反馈迭代：先 isolated routing 筛选（指标 $A_g(d)$），再 full-agent runtime 激活精炼（指标 $E_g(d)$）。
- **指令体构造（Runbook）**：按 6 条原则把 $\mathcal{L}_g$ 转为有界规划规则：①dependency topology 构建 DAG；②edge justification 每条边配"遗漏后果"；③mandatory framing 用程序性措辞防止跳步；④evidence-event decomposition 拆分为 pre/op/post 事件；⑤post-execution verification 至多一次回看；⑥structured packaging 含返回条件 $R_g$。
- **任务保留评估**：通过 paired clean/injected 执行对比 $\text{Eval}(x,y_0)=\text{Eval}(x,y_h)=1$，资源增幅仅在 coordinator-hit 且双端成功时统计。

## 实验与结果
- **数据集/平台**：OpenClaw v2026.5.7，53 个 native skills，9 个功能组；491 个 held-out 任务（438 个 2-skill、49 个 3-skill、4 个 4-skill）+ 45 个 pilot 任务 + 30 个独立作者任务。
- **LLM 后端**：Claude-Haiku-4.5、DeepSeek-V4-Pro、Qwen3.7-Max、Qwen3.7-Plus、DeepSeek-V4-Flash、MiniMax-M3。
- **最强路由激活**：MiniMax-M3 单任务 96.60%、多轮 94.69%；DeepSeek-V4-Pro 单任务 80.02%、多轮 88.98%。
- **最强资源放大（DeepSeek-V4-Pro，coordinator-hit 子集）**：token 增长 +66.91%、cached-token +54.33%、wall-clock time +92.45%、Avg. ∆Calls +2.20；多轮下 token +91.20%。
- **CDH ASR 最高**：MiniMax-M3 单任务 80.51%；DeepSeek-V4-Pro 多轮 82.81%。
- **泛化任务**：30 个独立任务命中 33.33%，命中均满足 CDH；tokens +45.3%、cache +39.8%、time +228.9%、calls +2.6。
- **消融**：Attract-only 命中与 Full CDH 一致但仅 +0.22 calls；Detour-only 命中仅 3/89。Full CDH 显著高于两者。
- **任务完成率变化**：所有模型/条件下 clean vs injected 差异 ≤ 1.5pp，任务基本无损。

## 相关工作脉络
1. **工具/技能元数据选择攻击**（Mo 等 2025; He 等 2026; Faghih 等 2025）：关注 description 改写偏向单一工具，CDH 在此基础上叠加 body 驱动的规划阶段再放大。
2. **间接提示注入/跨工具投毒**（Li 等 2026; Shi 等 2026b）：强调 planner-visible 描述的隐性重定向；CDH 聚焦同一发布者静态技能在两个披露阶段的连贯影响。
3. **可用性/资源耗尽攻击**（Dong 等 2026; Zhou 等 2026; Zhang 等 2025b）：部分依赖恶意服务器或脚本反馈；CDH 无需任何外部服务，仅靠静态文本完成。
4. **技能供应链后门/payload**（Feng 等 2026a; Liu 等 2026; Tie 等 2026）：常含可执行载荷或模型内污染；CDH 为 payload-less，仅影响轨迹长度与成本。
5. **终止/循环类攻击**（Xu 等 2026; Luo 等 2026）：利用坏终止维持运行；CDH 故意设计有界返回 $R_g$，避免卡死，保证任务完成。
6. **轨迹审计与防御**（Zhang 等 2026; Badhe 等 2026; Xiao 等 2026）：本文定位与这些 work 互补，主张将"轨迹必要性"纳入安全审计维度。

## 局限性与未来方向
1. **单一平台与 mock backend**：仅在 OpenClaw 默认 registry 与本地模拟环境验证，跨平台与真实外部工具的泛化待考。
2. **同组协调器设定**：实验中每个功能组仅部署一个 $h_g$，真实生态中多攻击者并发场景更复杂。
3. **黑盒限制下的命中率上限**：在强模型调度/缓存策略下，部分模型（如 DeepSeek-V4-Flash）出现负 time delta，说明 latency 不可预测。
4. **防御尚未落地**：仅提出预安装审查与运行时 token/call 预算两条思路，未给出可部署方案与 overhead 量化。
5. **未覆盖对抗性描述防御**：如 Isolated Planning、RouteGuard 等现有防御在 CDH 跨阶段场景下的效果未评估。

## 研究启发与可借鉴点
1. **"描述 + body 语义一致性"的构造范式**可用于防御侧：审计框架只需检查两者是否共享同一 $\rho_g$ 即可识别潜在跨阶段诱导。
2. **Pilot → Held-out 严格隔离机制**值得移植：45 个 pilot 仅用于描述优化、491 个 held-out 零接触，避免评测泄漏。
3. **成对 clean/injected 对比 + 人工盲评完成度**是可复现的安全评测标配设计。
4. **有界 return 纪律 $R_g$ 的设计思想**可转化为防御方的"预算+回退"策略：限定绕道深度并强制回到主链。
5. **独立作者泛化任务（30 例）**作为补充评估手段，能更稳健反映分布外攻击效果，建议纳入后续基准。

## 关键术语表
**Progressive Disclosure**：渐进式披露——Agent 先只加载技能描述做路由选择，命中后再展开完整指令体，以节省初始上下文。
**Convergent Detour Hijacking (CDH)**：收敛绕道劫持——攻击者通过一个静态协调技能，在同一执行链上制造"被选中→绕道调用多个合法技能→回归原任务"的轨迹。
**Coordinator Skill / 协调者技能**：被攻击者植入的额外技能，描述负责吸引选择，body 负责在规划期插入有界绕道规则。
**Shared Coordination Rationale ($\rho_g$)**：共享协调原则——描述与 body 共同遵循的任务域触发、角色定位与返回边界规范。
**ASR (Attack Success Rate)**：CDH 攻击成功率，定义为满足 Definition 1 四个条件的任务占比。
**Coordinated Hit**：协调器命中——执行中 coordinator 被选中且被加载。
**Isolated Routing Evaluation**：孤立路由评测——仅提供技能元数据让模型输出有序技能列表，不含实际执行。
**Detour Body / Runbook**：绕道指令体——以结构化 Markdown 呈现的依赖路径、省略后果、验证上限与返回规则。

## 可复现要素
- **数据集**：OpenClaw 默认 registry 的 53 技能 + 491 held-out 任务（JSON 与 fixtures 随 Supplementary 提供）；45 个 pilot 任务 + 30 个独立作者泛化任务。
- **代码/权重**：平台版本 OpenClaw v2026.5.7；mock 实现、task localization 接口、执行脚本与协调器 skill 文件在 Supplementary ZIP 中开源；VM 镜像提供。
- **关键超参**：执行超时 600s；temperature=0（模型支持时）；词/字符级 TF-IDF 特征上限 800/300；action/domain 权重 $\alpha=0.5$、char 权重 $\lambda=0.10$、对比校正强度 $\beta=0.3$；光谱分区候选 $k \in [4,12]$、随机种子 42；路由评分权重 gold 0.75、顺序 0.25；描述优化一般 2–3 轮。
