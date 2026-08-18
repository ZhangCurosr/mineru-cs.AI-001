---
title: "Agent-Skills-Can-Be-Harmful-An-Empirical-Study-of-Skill-Indu"
source: https://arxiv.org/pdf/2608.11888v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 09:32:06"
field: "LLM Agent可靠性与评估"
keywords: ["LLM Agent", "Agent Skills", "Skill-Induced Failures", "Differential Testing", "Agent Debugging", "Cost Regression", "Empirical Study"]
innovations: ["差分对比分析框架：通过无技能/语义匹配技能参考运行将失败归因于具体加载技能", "首个技能引发失败的全面实证研究：构建307案例数据集并分类功能失败与效率退化根因", "SKILLTRIAGE自动归因工具：基于分类体系的差异化证据提取与LLM推理实现自动化根因标注"]
benchmarks: ["SkillsBench", "SWE-Skills-Bench"]
---

# 论文速读：Agent-Skills-Can-Be-Harmful-An-Empirical-Study-of-Skill-Indu

## 一句话总结
本文对LLM Agent技能复用机制进行了系统性实证研究，通过差分分析框架在SkillsBench和SWE-Skills-Bench上构建了307个确认的技能引发失败案例（125个功能失败+182个高置信度效率退化），揭示了技能导致失败的根因机制，并开发了自动化归因工具SKILLTRIAGE。

## 研究问题与动机
1. **技能引发的失败难以隔离归因**：一个失败运行无法区分是技能导致还是基座模型局限、验证器范围不足、普通Agent方差或环境因素所致，需要对比证据
2. **现有基准侧重效用而非失败机制**：SkillsBench和SWE-Skills-Bench主要衡量技能效用，缺乏用于机制分析的负样本和语义匹配的替代技能
3. **聚合指标掩盖过程变化**：通过率与成本比率仅提供结果层度量，无法解释技能如何改变Agent执行轨迹
4. **人工归因不可扩展**：随着技能平台与市场增长，需要持续筛选新技能和技能更新，但手动分析无法规模化

## 核心贡献（创新点）
1. **差分对比分析框架**：通过配对目标技能引导执行与无技能/语义匹配技能参考执行，将失败归因于具体加载的技能，区别于已有工作仅报告技能效用
2. **首个技能引发失败的综合实证研究**：构建了307个确认案例数据集（125功能失败+182效率退化），系统分类了功能失败与效率退化的根因，填补了领域空白
3. **SKILLTRIAGE自动归因工具**：将人工分类体系操作化为归因流程，通过提取差异化证据实现自动化根因标注，在功能失败上达到88.8%子类别准确率，效率退化上达72.5%

## 方法详解
1. **差分测试-inspired对比设计**：保持任务、验证器、Agent框架、模型、仓库/容器状态固定，仅改变技能设置；目标运行审计，参考运行（无技能或语义匹配技能）作为伪oracle
2. **失败定义**：功能失败=目标运行FAIL而参考运行PASS；效率退化=双方均PASS但目标运行的token使用量与执行时间至少一项较参考增加超过阈值T=2.0
3. **数据集扩展**：从smithery.ai和skillsmp.com检索语义匹配公开技能（使用all-MiniLM-L6-v2嵌入，余弦相似度≥0.7，Top-5候选），将比较空间从826扩展至20,664对（约25倍）
4. **对比执行配置**：使用OpenCode 1.15.1 + Claude Opus 4.6，记录加载技能、轨迹、验证器结果、token用量、执行时间
5. **效率退化判定公式**：min(r_tok, r_time) > 1.0 ∧ max(r_tok, r_time) > T（T=2.0为主阈值），两者均超过1.0以排除token-时间权衡情况
6. **SKILLTRIAGE归因流程**：三阶段——输入构建（归一化配对案例）、差分证据提取（DS1-DS5差异化信号+相位/动作标签成本证据）、归因推理（LLM基于分类体系选择最佳根因）

## 实验与结果
- **数据集**：SkillsBench（84任务，11领域）和SWE-Skills-Bench（490软件工程项目）
- **最终分析案例**：307个确认失败（125功能失败+182效率退化），从665初始候选中经人工审查去重得到
- **功能失败根因分布**：Task-Implementation Fault (TIF) 占68.8%（86/125），其中Incorrect Required-Element Fill (IRF) 36.8%（46例）、Required-Element Omission (RRO) 28.8%（36例）；Artifact Misplacement (AM) 19.2%（24例）；Environment Mismatch (EM) 10.4%（13例）；Applicability Mismatch (APM) 仅1.6%（2例）
- **效率退化根因分布**：Excessive Procedure (EP) 占62.6%（114/182），其中Excessive Verification (EV) 36.8%（67例）、Heavy Implementation Pipeline (HIP) 16.5%（30例）；Context Bloat (CO) 25.3%（46例），其中Skill-Body Context Bloat占43/46；Dependency Resolution (DO) 12.1%（22例）
- **SKILLTRIAGE归因结果**：功能失败类别准确率93.6%（117/125），子类别准确率88.8%（111/125）；效率退化类别准确率79.7%（145/182），子类别准确率72.5%（132/182）；TIF子类别准确率达88.4%（76/86）
- **关键发现**：看似相关的技能更常导致实现错误而非明显无关技能；效率退化主要由过多验证和重型实现管道驱动，而非单纯提示长度

## 相关工作脉络
1. **SkillsBench/SWE-Skills-Bench**：本文与其定位差异在于后者衡量技能效用，本文聚焦技能引发的失败机制与根因分析
2. **长上下文与无关上下文研究**（如"Lost in the Middle"、"distracted by irrelevant context"）：解释广义上下文效应，但不识别具体技能文档如何导致验证器可见的功能失败或隐蔽的效率退化
3. **Agent失败分析**（如MAST、Agentless、SWE-agent）：归因于基座Agent或任务本身，本文归因于加载技能在受控对比条件下的影响
4. **性能bug/配置bug软件工程研究**：本文为agent技能场景构建了新的分类体系，将传统bug视角迁移到skill-induced failures
5. **RAG与提示工程研究**：关注模型对prompt格式的敏感性，本文关注skill作为结构化过程指导对Agent可执行轨迹的具体影响

## 局限性与未来方向
1. **研究依赖特定基准与模型**：结论可能受限于SkillsBench和SWE-Skills-Bench的任务分布、OpenCode框架及Claude Opus 4.6模型，未必直接适用于所有设置
2. **人工归因存在主观性**：判定验证器是否过窄、成本变化是否属于普通Agent方差等需要人工判断，尽管已通过小组共识缓解
3. **SKILLTRIAGE存在边界误差**：部分错误集中在分类体系边界（如IRF/RRO、EM/WAL附近），同一证据可能支持相邻标签
4. **论文建议的未来方向**：技能-任务兼容性检查、成本感知的技能打包与选择、预算感知的执行策略（动态调整验证范围和实现管道深度）

## 研究启发与可借鉴点
1. **差分对比实验设计可迁移**：固定任务/模型/环境、仅改变单一变量（技能加载状态）的对比范式，适用于其他Agent组件的影响归因研究
2. **SKILLTRIAGE的"分类体系操作化为证据检查清单"思路**：将手动标注体系转化为可计算的差分信号（DS1-DS5、相位/动作标签），为自动化Agent失败分析提供方法论参考
3. **效率退化的双指标联合判定**：min/max联合阈值（r_tok>1 ∧ r_time>1 ∧ max>T）有效排除token-时间权衡噪声，可借鉴于其他Agent效率评估
4. **语义匹配公开技能扩展比较空间**：通过外部技能库检索语义相似技能扩充对比组，显著扩大失败发现空间，可为技能生态研究提供模板
5. **可结合本团队方向**：若团队研究Agent评测/调试/安全，可将此差分框架与现有Agent平台结合，建立技能质量门禁；若研究长上下文优化，可验证"强制技能主体文本是上下文膨胀主因"的发现

## 关键术语表
**Agent Skills**：LLM Agent的可复用过程性知识包（通常为SKILL.md文件），在不修改模型参数的前提下通过加载到Agent上下文中影响规划、工具使用、代码编辑和停止决策
**差分分析框架（Differential Analysis Framework）**：通过对比目标技能引导运行与无技能/语义匹配技能参考运行来归因失败的分析方法，参考运行作为伪oracle证明任务可解或更低价可解
**功能失败（Functional Failure）**：目标技能运行验证器FAIL而参考运行PASS的情况，表明技能导致任务无法完成
**效率退化（Efficiency Regression）**：目标与参考运行均PASS，但目标运行token用量或执行时间显著增加（至少一项>2倍）的情况
**Task-Implementation Fault (TIF)**：技能导致Agent错误实现或遗漏任务要求的实现元素（如字段、API、计算公式、输出格式）的功能失败类型，占比最高
**SKILLTRIAGE**：论文开发的基于LLM的自动化归因工具，将分类体系操作化为证据提取流程，对已确认失败案例进行根因分类
**语义匹配公开技能（Semantically Matched Public Skills）**：从smithery.ai和skillsmp.com检索的与基准技能语义相似（cosine similarity≥0.7）的公开技能，用于扩大多技能对比空间

## 可复现要素
- **数据集**：基于SkillsBench和SWE-Skills-Bench构建，公开技能来源为smithery.ai和skillsmp.com（论文未明确说明新数据集是否开源，需查看论文补充材料或项目页面）
- **代码**：SKILLTRIAGE工具论文未明确说明是否开源
- **模型**：Claude Opus 4.6 + OpenCode 1.15.1
- **关键超参**：语义匹配阈值cosine similarity≥0.7，Top-5候选；效率退化阈值T=2.0；SKILLTRIAGE评估使用GPT-5.5，3次独立运行2-of-3多数投票
- **嵌入模型**：all-MiniLM-L6-v2用于技能元数据语义匹配
