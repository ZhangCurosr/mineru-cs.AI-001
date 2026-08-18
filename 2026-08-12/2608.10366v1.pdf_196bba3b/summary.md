---
title: "DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?"
source: https://arxiv.org/pdf/2608.10366v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:53:08"
field: "数据科学Agent评测基准"
keywords: ["Data Science Agent", "Benchmark", "GUI Agent", "Computer Environment", "End-to-End Workflow", "OS-level Evaluation"]
innovations: ["首个在真实操作系统中评估端到端数据科学工作流的基准，覆盖完整数据科学生命周期", "确定性执行评估体系：数值容差验证+可视化元数据检查+LLM视觉评判员两阶段设计", "基于OSWorld增强的数据科学执行环境，集成多工具（VS Code/Jupyter/终端/Chrome）协同"]
benchmarks: ["DSAgentBench"]
---

# 论文速读：DSAgentBench: Can Agents Automate End-to-End Data-Science Workflows in Real Computer Environments?

## 一句话总结
本文提出了首个在真实操作系统环境中评估智能体能否自动化端到端数据科学工作流的基准测试 DSAgentBench，包含 275 个覆盖数据采集、EDA、特征工程、建模、评估与可视化的长周期任务，结果表明即使最强模型 Claude-4.6-Sonnet 成功率也仅为 56.70%，开源模型全部低于 1%。

## 研究问题与动机
- **现有数据科学基准缺乏真实计算机交互**：DS-1000、DABStep、MLAgentBench 等仅评估孤立代码片段的正确性，要求智能体在沙箱环境中生成可执行代码，不涉及文件系统导航、工具协调或错误调试等真实工作流要素。
- **OS-level 基准不评估数据分析推理能力**：OSWorld、WebArena、VisualWebArena 等聚焦通用桌面操作（打开应用、导航界面），未涵盖数据分析推理、模型训练、可视化生成等数据科学核心任务。
- **代码正确性 ≠ 真实工作流执行能力**：生成语法正确的 Python 代码与在真实计算机环境中自主导航、协调多工具（VS Code、Jupyter、终端、浏览器）、基于中间输出迭代调试是本质不同的能力。
- **工业报告佐证差距存在**：OpenAI（2024）报告指出当前 AI 系统虽在孤立任务上表现良好，但难以可靠执行多步骤、跨工具的分析工作流。

## 核心贡献（创新点）
1. **首个面向真实 OS 环境的端到端数据科学工作流基准**：DSAgentBench 首次要求智能体在具备完整文件系统和多应用（VS Code、Jupyter、Chrome、终端）的 Ubuntu 环境中自主完成从数据采集到可视化报告的全流程，区别于先前仅评估代码生成的基准。
2. **基于 OSWorld 的增强型数据科学执行环境**：扩展 OSWorld 框架，集成 Kaggle API、OpenML、SQLite 数据库访问能力，并预装 Jupyter Notebook 和 VS Code，支持任务级自动化配置与数据集预置。
3. **确定性执行评估体系，超越代码语法检查**：每个任务配备确定性 Python 评估器，验证分析正确性（数值容差）、可视化元数据（轴标签、图例）和模型性能阈值，而非仅检查代码能否运行；约 10% 任务在确定性通过后启用 LLM 视觉评判员评估可视化质量。
4. **15 个模型的全面评测与深入错误分析**：覆盖闭源（GPT-4o、Claude-4.6-Sonnet 等）、混合（Jedi）和开源（UI-TARS、GUI-OWL-7B 等）模型，从地面定位、规划、推理、工具编排等多维度诊断失败模式。

## 方法详解
- **任务形式化**：基准由 N 个任务组成 $\mathcal{T} = \{(\mathcal{C}_i, \mathcal{T}_i, \mathcal{V}_i)\}_{i=1}^{N}$，其中 $\mathcal{C}_i$ 为任务配置（初始 OS 状态、数据集、库、应用），$\mathcal{T}_i$ 为自然语言指令，$\mathcal{V}_i$ 为确定性 Python 评估器。智能体通过多模态观测与 GUI 动作与系统交互，输出满足 $\mathcal{V}_i$ 即视为成功。
- **环境架构**：基于 OSWorld 构建，运行 Ubuntu OS，分辨率 1920×1080，预装 Python 及常用数据科学库。观测空间支持两种模态：(i) 纯截图；(ii) 截图 + 可访问性树（A11y Tree，通过 AT-SPI 提取 UI 元素的角色、名称、边界框等结构化元数据）。动作空间包含鼠标操作（移动、点击、拖拽、滚动）、键盘输入（打字、快捷键）、控制动作（WAIT、DONE、FAIL）。
- **任务构建流程**：三阶段管线——①从 Kaggle、OpenML、SQLite、GitHub、网页 API 收集异构真实数据集；②两位具 5 年以上经验的数据科学家基于 100 个高排名 Kaggle notebook 建立任务分类体系，再用 LLM（GPT-5、Claude 4.5 Sonnet、Gemini 3）辅助完善，最终人工定义任务逻辑、预期输出与评估器；③双标注员独立验证，初始一致率 86%，全部 275 个任务最终达成互认。
- **评估器设计**：分为数值验证（检查输出文件存在、计算结果在容差 $\epsilon = 0.01$ 内匹配）、可视化评估（检查轴标签、标题、图例、数据映射，确定性检查 + LLM 视觉评判）、模型性能评估（如 accuracy/F1 ≥ 0.7）。主要指标为任务成功率（得分 ≥ 0.95 的比例）和平均得分。
- **任务分类**：按数据科学生命周期分为六类——数据采集（Data Acquisition, DA, 23 题）、探索性数据分析（EDA, 119 题）、特征工程（FE, 37 题）、建模（41 题）、评估与部署（12 题）、可视化与报告（33 题）。难度分级：Easy 5.5%、Medium 46.9%、Hard 47.6%，多阶段任务占 56.7%。

## 实验与结果
- **评测设置**：15 个模型，包括 GPT-4o、GPT-5-mini、GPT-5、O4-mini、Claude-4/4.5/4.6-Sonnet、Gemini-2.5-Pro、OpenAI CUA（闭源）；Jedi-3B/7B（混合）；UI-TARS-2B/7B、GUI-OWL-7B、OpenCUA-72B（开源）。两种观测设置：Screenshot 和 Screenshot + A11y Tree。人类基线：3 位参与者（2 名应用科学家 + 1 名硕士毕业生）平均成功率 85.09%。
- **最强结果**：Claude-4.6-Sonnet 在 Screenshot + A11y Tree 设置下取得 **56.70%** 最高总分（DA: 47.82%, EDA: 64.88%, FE: 56.75%, Model: 46.34%, Vis: 42.42%, Eval: 66.67%），较次强的 GPT-5（29.81%）提升近 27 个百分点。
- **开源模型极弱**：所有开源模型在 Screenshot-only 设置下得分均 ≤ 1%，UI-TARS 和 GUI-OWL-7B 几乎全败（多因地面定位错误），OpenCUA-72B 最高仅 0.73%。
- **任务类型差异显著**：EDA 和可视化任务相对容易，而数据采集、模型验证和评估任务最难；Jupyter Notebook 任务的完成率普遍高于 VS Code 任务；单阶段任务显著优于多阶段任务。
- **A11y Tree 增益有限但不均匀**：引入 A11y 信息后整体提升约 5-6 个百分点，但不同模型增益差异大，部分模型（如 Claude-4-Sonnet）几乎无法利用 A11y 信号。
- **步数预算消融**：将步数从 15 增至 50，Claude-4.6-Sonnet 仅从 56.70% 微升至约 57%，表明瓶颈不在行动预算，而在底层推理与工具编排能力。
- **效率对比**：Gemini-2.5-Pro 成功任务平均仅需 6.76 步，是最有效率模型；Claude-4.6-Sonnet 平均 10.93 步。

## 相关工作脉络
1. **代码生成基准（HumanEval、DS-1000、DSEval）**：评估 isolated code correctness via unit tests，不要求 OS 交互或多工具协调；DSAgentBench 在此基础上增加真实桌面操作与端到端工作流要求。
2. **数据科学 Agent 基准（MLAgentBench、DABStep、DA-CODE）**：MLAgentBench 聚焦 ML 实验管理，DA-CODE 评估多文件分析但仍限于沙箱 notebook 环境；DSAgentBench 扩展至真实 OS，支持跨应用协作（VS Code + Jupyter + Chrome + 终端）。
3. **GUI/计算机控制基准（OSWorld、WebArena、VisualWebArena、ScreenSpot-Pro）**：这些基准评估通用桌面交互能力，但不包含数据分析推理任务；DSAgentBench 填补了"能用电脑"与"能像数据科学家一样工作"之间的空白。
4. **可视化基准（ChartQA、Text2Vis、VisEval）**：仅评估图表生成和视觉问答，不涉及数据获取、清洗、建模等上游环节；DSAgentBench 的可视化任务嵌入完整工作流中，且评估器验证语义正确性。
5. **专用数据 Agent 工作（Data Interpreter、AutoKaggle）**：以 case study 形式展示，缺乏大规模标准化基准评估；DSAgentBench 提供可扩展、可复现的系统性评测框架。
6. **GUI 地面定位增强方法（CogAgent、ShowUI、OS-ATLAS、JEDI）**：这些工作改进 UI  grounding 精度，但 DSAGENTBench 揭示即使最强 grounding 也无法解决多步推理和工具编排的根本挑战。

## 局限性与未来方向
- **开源模型不支持 A11y Tree**：当前评估中开源模型仅在 Screenshot-only 设置下测试，无法评估结构化 UI 元数据对grounding的增益，限制了对其真实能力的全面了解。
- **错误分析仅覆盖子集**：手动审查的 604 条闭源模型轨迹和 150 条开源模型轨迹仅为全部 275 个任务的部分采样，罕见失败模式可能未被捕捉。
- **可视化评估侧重最终产物**：对可视化任务的评价主要关注最终输出质量，对分析过程中的迭代探索和多视图交互涉及有限。
- **未来方向**：需要开发更强的多步骤规划与错误恢复机制；探索 A11y Tree 与视觉信号的深度融合；构建更具挑战性的长周期 multi-tool 协作场景。

## 研究启发与可借鉴点
1. **任务构建的人机协作管线值得复用**：专家人工定义任务逻辑与评估器 + LLM 辅助润色措辞和发现 edge cases 的模式，既能保证任务真实性又能提高构建效率，可迁移至其他 Agent 基准建设。
2. **确定性评估器 + LLM 视觉评判员的两阶段设计**：先用规则/数值检查过滤明显错误，再对小比例样本启用 LLM judge，兼顾评估的稳定性与主观质量判断，避免循环评估偏差（使用 Gemini 评判 GPT-4o 输出，反之亦然）。
3. **A11y Tree 增强的双模态观测设置可作为标准实验协议**：同时报告 Screenshot-only 和 Screenshot+A11y 结果，有助于区分 grounding 能力与推理能力的贡献，建议在 GUI Agent 评测中推广。
4. **错误根因分类框架（Grounding/Terminal/Code/Logic）可直接借用**：该四维度分类清晰划分了不同层级的失败原因，便于后续工作定位改进方向。
5. **与数据科学 Agent 团队的结合机会**：可在 DSAGENTBench 上测试本团队的数据分析 Agent 框架，识别在多工具协调和长周期状态管理上的薄弱环节，尤其是多阶段任务中中间输出的错误累积问题。

## 关键术语表
- **DSAgentBench**：首个在真实操作系统环境中评估智能体能否自动化端到端数据科学工作流的基准测试，包含 275 个任务。
- **OSWorld**：用于构建数据科学执行环境的底层框架，提供 Ubuntu/Linux、Windows、macOS 上的桌面自动化能力。
- **A11y Tree（Accessibility Tree）**：通过 AT-SPI 提取的结构化 UI 元数据，包含元素角色、名称、边界框等信息，与截图联合使用可提升 grounding 精度。
- **Deterministic Evaluator**：基于规则和数值容差的自动化评估函数，验证输出文件、计算结果和分析正确性，而非依赖 LLM 评分。
- **LLM Visual Judge**：在确定性评估通过后用于评估可视化质量和语义对齐的 LLM 评判员，采用跨模型配对以避免评估循环性。
- **Task Configuration ($\mathcal{C}_i$)**：定义任务初始系统状态的 JSON 配置，包含数据集路径、文件系统结构、已安装库和应用等。
- **Multi-Stage Workflow**：需要多步迭代执行（如 load–transform–visualize 或 model–evaluate–refine）的任务结构，占总任务的 56.7%。
- **Grounding Error**：智能体未能正确识别和操作界面元素导致的失败，在开源模型中占比高达 97-98%。

## 可复现要素
- **数据集**：275 个任务，数据来源包括 Kaggle、OpenML、SQLite、GitHub，均为开源许可数据集；数据集已预下载并将随基准一起发布。
- **代码/权重**：基准代码和评估脚本将在 GitHub（https://github.com/vis-nlp/DSAgentBench）开源；闭源模型通过 API 调用，开源模型使用 vLLM 本地推理。
- **关键超参**：分辨率 1920×1080；温度 temperature=0.1，top-p=0.9；最大输出 tokens 2000；任务超时 1800 秒；最大步数 15（消融实验中测试 30、50）；动作后延迟 2.0 秒；操作空间为 PyAutoGUI。
- **运行环境**：Ubuntu 22.04 LTS，VMware 虚拟机（闭源模型）/ GCP n1-standard-4 + Docker（开源模型）；预装 VS Code、Jupyter Notebook、Chrome、Kaggle API、OpenML 客户端等。
