---
title: "DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?"
source: https://arxiv.org/pdf/2608.10366v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:55:26"
---

# 论文速读：DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?

## 一句话总结
本文提出 **DSAgentBench**，首个在真实操作系统环境中评估智能体完成端到端数据科学工作流的基准测试；包含 275 个覆盖数据获取、探索分析、建模与可视化的多阶段任务，揭示当前最强 Agent（Claude-4.6-Sonnet）仅达 56.70% 成功率，暴露出 grounding、工具编排与长程推理的显著能力缺口。

## 研究问题与动机
- 真实数据科学工作流跨越数据获取、清洗、探索、建模、可视化与报告生成，需在 IDE、终端、浏览器、数据库等多工具间持续切换，并依赖中间输出进行迭代推理。
- 现有数据科学基准（如 DS-1000、DABStep、MLAgentBench、DSBench、DSEval）仅在静态沙盒中评测孤立代码片段，缺乏真实系统交互与跨工具状态管理。
- 现有桌面操作基准（如 OSWorld、WebArena、VisualWebArena、OfficeBench）侧重通用 GUI 控制与网页导航，未嵌入专业数据分析推理与完整工作流验证。
- 工业界报告指出当前 AI 系统难以可靠执行多步骤、跨工具的复杂分析流程，亟需统一基准量化 Agent 在真实计算环境中的自主数据科学能力。

## 核心贡献（创新点）
- **提出 DSAgentBench 基准**：首个支持在真实 OS 内评测端到端数据科学工作流的基准，覆盖完整生命周期。与已有工作仅验证代码语法/静态执行的本质区别在于强调跨工具协同与长程状态维护。
- **扩展 OSWorld 数据科学环境**：集成 Jupyter Notebook、VS Code、Chrome 及 Kaggle/OpenML/SQLite 数据源，提供统一 pyautogui 动作空间与截图/A11y 双模态观测，区别于传统沙盒 Notebook 环境。
- **设计确定性+语义化混合评估体系**：结合内置 Python 评估器（数值容差、模型指标阈值、图表元数据校验）与限流 LLM 视觉裁判，从“代码可运行”升级为“分析正确性与产出质量”双重验证。
- **大规模评测与归因诊断**：对 15 个闭源/开源 Agent 进行系统评估，输出失败根因分布、首错步态（FF Step）、预算耗尽率等指标，明确指出 grounding、工具编排、长程推理三大瓶颈。

## 方法详解
- **环境架构**：基于 OSWorld 构建 Ubuntu 虚拟桌面，预装 Python 数据科学栈、VS Code、Jupyter、Chrome。观测空间为 1920×1080 桌面截图，可叠加 AT-SPI 提取的 A11y Tree（含控件角色、边界框、交互状态）。动作空间统一为鼠标点击/拖拽/滚动、键盘输入/HOTKEY、以及 WAIT/DONE/FAIL 控制指令，所有动作经 PyAutoGUI 原子化执行。
- **任务构建 Pipeline**：三阶段流程：① 从 Kaggle、OpenML、GitHub、SQLite、Web API 采集异构数据；② 由 4 名专家标注员基于 100 份高分 Kaggle Notebook 归纳六类工作流 taxonomy，人工编写 275 个任务指令与确定性评估函数（LLM 仅辅助润色）；③ 双标注员独立审核（初始一致率 86%），全部任务达到互认。
- **形式化定义**：任务集 $\mathcal{T} = \{(\mathcal{C}_i, \mathcal{T}_i, \mathcal{V}_i)\}_{i=1}^N$，其中 $\mathcal{C}_i$ 为初始系统配置（数据集路径、依赖库、可用应用），$\mathcal{T}_i$ 为自然语言指令，$\mathcal{V}_i$ 为确定性 Python 评估器。Agent 通过感知-行动循环交互，默认最大步数 15，任务超时 1800s。
- **评估机制**：任务成功阈值设为综合得分 $\geq 0.95$。评估分为两类：(a) 确定性验证：检查输出文件是否存在、数值误差（$\epsilon = 0.01$）、模型指标阈值（如 accuracy/F1 $\geq 0.7$）、图表轴标签/图例/数据映射完整性；(b) LLM 视觉裁判：仅覆盖约 10% 任务且在确定性门禁通过后触发，采用 GPT-4o 与 Gemini-2.5-Pro 交叉评判避免自判偏倚，评估语义对齐与设计可读性。
- **Prompt 设计**：针对不同模型架构提供 Standard / Few-Shot (CoT) / UITARS (含/不含 Thought) / JEDI 等差异化系统提示，统一约束输出格式与地面坐标计算方式。

## 实验与结果
- **数据集规模**：275 个任务，Tabular 95.3%、Image 3.6%、Text 1.1%；来源覆盖 GitHub (37.1%)、Kaggle (29.8%)、OpenML (18.9%)、SQLite (7.6%)。难度分布 Hard 47.6% / Medium 46.9% / Easy 5.5%；多阶段任务占 56.7%，平均 4–5 个分析步骤。
- **评测基线**：15 个 Agent，包括 GPT-4o、GPT-5-mini、GPT-5、O4-mini、Gemini-2.5-Pro、Claude 4/4.5/4.6 Sonnet、OpenAI CUA，以及 UI-TARS 2B/7B、GUI-OWL-7B、OpenCUA-72B、Jedi-3B/7B。
- **主要结果**：
  - 最强模型 **Claude-4.6-Sonnet** 在 Screenshot+A11y Tree 设定下取得 **56.70%** 整体成功率，次强 GPT-5 为 29.81%；GPT-4o 为 24.54%，Gemini-2.5-Pro 为 20.81%。
  - 所有开源模型在 Screenshot-only 设定下成功率均 **<1%**，人类对照组达成 **85.09%**。
  - 各类任务表现差异显著：EDA 完成率最高（Claude-4.6-Sonnet 达 64.88% A11y），数据获取 (DA) 与评估部署 (Eval) 最弱。
- **关键结论**：
  - A11y Tree 整体带来正向增益，但各模型利用效率差异显著。
  - Jupyter 环境表现优于 VS Code；单阶段任务显著高于多阶段任务；难度越高成功率单调下降。
  - 步数预算从 15 增至 50 仅提升约 1.3%（24.54% → 25.81%），说明瓶颈不在步数限制而在 grounding、规划
