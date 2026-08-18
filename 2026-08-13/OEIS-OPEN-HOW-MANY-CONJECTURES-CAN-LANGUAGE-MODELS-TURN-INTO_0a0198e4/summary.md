---
title: "OEIS-OPEN-HOW-MANY-CONJECTURES-CAN-LANGUAGE-MODELS-TURN-INTO"
source: https://arxiv.org/pdf/2608.11941v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 13:49:18"
---

# 论文速读：OEIS-OPEN-HOW-MANY-CONJECTURES-CAN-LANGUAGE-MODELS-TURN-INTO

## 一句话总结
本文构建了基于 OEIS 492 个开放整数序列猜想（Lean 形式化）的开源评测基准 OEIS OPEN，证明仅需极简工具链的语言模型智能体即可在 $50/猜想预算下自主解决 30% 的猜想，三倍超越此前依赖重型进化搜索的 AlphaProof Nexus 系统。

## 研究问题与动机
- **现有 AI 解开放问题报道缺乏系统性**：成功破解数论/几何猜想的新闻案例未披露尝试全集、成功率与平均成本，且人类专家提示与交互细节未公开，难以评估 AI 的真实前沿能力。
- **既有开放问题基准依赖“生成器-验证器间隙”**：HorizonMath 与 FrontierMath: Open Problems 仅能表达存在可计算验证对象的构造类问题，无法覆盖以证明一般性陈述为主体的研究数学。
- **计算验证的局限性**：现有基准接受高精度数值匹配或“强证据”而非严格证明，且验证器需手工编写、易出错（2026年7月已有两题因验证保真度不足被撤），同时无法有效处理猜想为假的情形。
- **形式化证明是更严谨的评测载体**：Lean 内核验证提供数学上 sound 的裁决，允许模型证明猜想或其否定，且未解决的猜想答案不会泄漏至训练语料，适合做长周期、可复现的 AI 能力基准。

## 核心贡献（创新点）
- **开源抗作弊形式化猜想基准与评估管道**：发布 OEIS OPEN（492 个猜想）与 OEIS OPEN LITE（100 个随机子集），配套三容器隔离架构与 SafeVerify 核查器，支持任意通用 LLM 接入。*与已有工作本质区别：此前同类数据集仅能用作者定制 Agent 封闭运行，本文开放全栈评测代码并通过环境隔离切断模型篡改验证器的攻击面。*
- **极简 Agent 架构大幅超越重型基线**：基础 ReAct 智能体仅提供 bash、文本编辑器、资源监控三个工具，在相同成本量级下以 30% 解决率三倍于 AlphaProof Nexus 的 9%。*与已有工作本质区别：摒弃前人 AlphaEvolve 进化搜索、子目标树遍历与多级代理排名反馈的复杂流水线，以“Bitter Lesson”式简单工具路线取得胜利。*
- **系统量化预算-性能缩放律并澄清模块价值**：证明解决率随单题花费呈对数线性增长（约每 10 倍预算提升 10 个百分点），并对照实验表明外挂 47.6 万篇 arXiv 文献库与 DeepAgent（子任务委派/持久记忆）均无性能增益。*与已有工作本质区别：首次在严格形式化基准上隔离评估“文献检索增强”与“复杂 Agent 循环”的边际收益，给出反直觉的负向消融结论。*

## 方法详解
- **猜想筛选与形式化**：源于 Tsoukalas 等人从 OEIS 2649 个开放猜想中筛选的 500 个非平凡、非著名、适合自动定理证明的猜想，经 8 个技术性排除后剩余 492 个。整数序列猜想在形式化上风险较低，因其仅依赖整数与初等运算，无需冗长 Mathlib 定义链。
- **三段式安全验证协议**：提交物必须通过 SafeVerify（基于 lean4checker 扩展），要求目标声明与提交声明同名、同 kind、同 kernel type，且仅允许 `propext`、`Quot.sound`、`Classical.choice` 三个公理。为防止攻击，全流程拆分为三个无网络 Docker 容器：① **Agent 容器**（模型思考、写 Lean/脚本）；② **Compile 容器**（干净 Lean 工具链，将源码编译为 `.olean` 二进制，隔离编译期 `#eval` 恶意代码）；③ **Scorer 容器**（独立运行 SafeVerify 裁决）。该设计封堵了环境篡改、命题偷换、额外公理注入、元编程绕过内核、`decide` 作弊等六类攻击。
- **基础 Agent 控制流**：基于 Inspect 库的 ReAct 循环，迭代直至证明/证伪猜想或触及硬性上限。单题上限为预算（完整集 $50、LITE 集 $200）与 72 小时工作时长。环境预装 Lean 4 + Mathlib、SageMath、Python（`sympy`/`mpmath`/`numpy`/`pantograph`）。
- **变体设计**：
  - 文献访问（§2.4.1）：离线挂载截至 2022 年的 476,000 篇纯数学 arXiv 论文 LaTeX 源码树。
  - DeepAgent（§2.4.2）：替换为 Inspect 的 `deepagent`，增加子代理委派、持久记忆、todo-list 工具与更长的指令式系统提示词。

## 实验与结果
- **数据集与基线**：OEIS OPEN（492 猜想，$50/题）与 OEIS OPEN LITE（100 猜想，$200/题）。对照基线为 Tsoukalas 等人的 AlphaProof Nexus，在相同成本量级下仅解决 44/492（9%）。
- **主结果**：
  - 完整集（$50）：Claude Opus 4.8 解出 30%（147 题），GPT-5.5 解出 26%，Gemini 3.5 Flash 解出 22%。
  - LITE 集（$200）：Claude Fable 5 达最高 44%，Gemini 3.5 Flash 为 29%；Claude Fable 5 与 GPT-5.6 Sol 仅在 LITE 上评测。
  - 预算-性能缩放：解决率与单题花费呈对数线性关系，约每 10 倍预算提升 10 个百分点，曲线未现明显饱和平台。
  - 变体消融：文献访问与 DeepAgent 均未在 LITE 上提升准确率（Appendix A.2 图 3）。
- **外推估计**：按 LITE 比例，当前最优模型在完整集 $200 预算下预计可解决约 216/492 个猜想，显著高于 $50 下的 147 个。

## 相关工作脉络
- **Tsoukalas 等 (AlphaProof Nexus, 2026)**：在 OEIS 猜想上首次取得批量化突破，采用 AlphaEvolve 进化种群+AlphaProof 子目标搜索+Gemini 评注代理的复杂架构。本文在其数据集上开放评估，证明简化架构可达成更高效率。
- **HorizonMath (Wang et al., 202
