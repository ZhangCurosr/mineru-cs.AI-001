---
title: "Diagram-MMU-A-Multi-Modal-Benchmark-for-Scientific-Diagrams"
source: https://arxiv.org/pdf/2608.12262v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 10:16:01"
---

# 论文速读：Diagram-MMU-A-Multi-Modal-Benchmark-for-Scientific-Diagrams

## 一句话总结
提出 **Diagram-MMU**，首个面向科学图表的多模态大模型统一基准，覆盖图表解析、编辑与问答三大任务，首次以 TikZ 作为代码表示形式，并同时评测 MLLMs 的**基础能力**与**Agentic 能力**。

## 研究问题与动机
- 科学图表的理解依赖**符号感知**（类型、路径、空间关系）而非像素级匹配，现有 MLLMs 在自然图像上表现优异，但在图表解析与代码合成上存在显著短板。
- 科研写作正转向 **vibe writing** 范式，用户仅需提供自然语言意图，模型需自主完成结构推断、TikZ 代码生成与视觉渲染，现有基准缺乏对这一工作流的系统评测。
- 传统图表评测（如 MathVerse、图表细节分析）仅关注视觉识别或单一任务，缺少对**代码级结构正确性**、**编辑可操作性**以及**多智能体协作维度**（上下文、工具、状态、规划）的统一衡量标准。

## 核心贡献（创新点）
1. **首个 TikZ 原生科学图表基准**：覆盖 6 类图表（Charts、Planar Geom.、3D Shapes、Graph、Chemistry、Circuit），首次将 Chemistry 与 Circuit 纳入多模态评测范畴。
2. **三大任务 × 16 个评估设置**：统一 D2C-P（解析）、D2C-E（编辑）、DQA（问答），并系统解耦上下文利用、工具使用、状态管理、规划四大 Agentic 维度。
3. **SOM 语义对象模型与细粒度指标**：提出基于 XML 标签注入与 SVG DOM 解析的自动 Ground Truth 提取流水线，实现 Object-level F1（Type/Text/Color/BBox）的精确度量，突破传统 BLEU/图像相似度无法刻画结构正确性的局限。
4. **MCP-based TikZ 领域知识库**：设计基于 Model Context Protocol 的轻量检索服务器，为 Agentic 评测提供按需查询的权威文档，有效缓解长上下文噪声问题。

## 方法详解
- **数据集规模与来源**：共 3,744 张独特图表（18,305 个样本），源自 PGFPlots、CircuitTikZ、TKZ-Euclide、ChemFig、TikZ-Network 官方手册及 texample.net、GitHub 社区，总计 6,849 份 TikZ 源码，经 13 名研究生人工审核与交叉验证。
- **任务生成管道**：
  - D2C-P 采用固定模板直接实例化，原始 TikZ 源码即为 Ground Truth。
  - D2C-E 与 DQA 由 Gemini-Flash 智能体生成，经 GPT-5.2 + Gemini-3 Pro 双 Judge 交叉验证，并通过 pdflatex → lualatex → xelatex 多引擎编译验证；失败案例最多迭代 3 次，最终由研究生人工校验。
- **SOM Pipeline**：通过 `semantic_spy.sty` 预处理注入 XML 标签 → `latex → dvisvgm`（参数 `--font-format=svg --precision=6 --zoom=1`）→ lxml 递归提取 Type/Text/Color/BBox；文本按 x 坐标排序还原，颜色采用 CIE2000 感知相似度最优匹配，边界框采用贪心 IoU 匹配（阈值 τ=0.3）。
- **评估指标体系**：
  - Object-level：F1_type、F1_text、F1_color、F1_bbox（取平均 F1_avg）
  - Code-level：CrystalBLEU
  - Image-level：SSIM、CLIP Score、LPIPS、FID
  - DQA：Qwen3-Next-80B-A3B-Instruct 提取答案并打分准确率

## 实验与结果
- **评测范围**：12 个 MLLMs（6 闭源：Gemini-3.1 Pro、Gemini-3.0 Pro、Gemini-3.0 Flash、GPT-5.2、Claude-4.6 Opus、Seed-2.0 Pro；6 开源：Qwen3.5-397B-A17B、Qwen3-VL-235B-A22B、Kimi-K2.5、Qwen3-VL-8B、InternVL3-38B、TikZero+ 10B）。Agentic 评测使用 300 张图的 Mini split。
- **基础能力**：DQA 最高准确率 **86%**，D2C-P 编码 F1_avg 仅 **31–57%**；闭源模型 F1_avg（47.57–54.64）显著优于开源（31.47–57.48），TikZero+ 仅 15.43；空间定位（bbox）最弱（GPT-5.2 8.0，Claude-4.6 Opus 9.2）；3D Shapes 因训练数据稀缺在各任务中均垫底；成功渲染与失败渲染的模型 F1_avg 差距达 **20–40 分**。
- **Agentic 能力**：Context utilization 在 D2C-E 上全面提升（F1_avg +4.1~10.6），但 DQA 上 GPT-5.2 骤降 **-7.3**；Tool use 普遍不佳，仅 **Claude-4.6 Opus** 稳定获益（D2C-P +2.2，D2C-E +4.4/+2.2），Gemini-3.1 Pro 因过度检索引发 context rot（**-4.3**）；State management 多数模型退化；Planning 任务表现分化。**Gemini-3.0 Pro** 三项任务最均衡。

## 相关工作脉络
- MathVerse / Nodes are early, edges are late / Chain-of-Region：聚焦自然图像或图表的结构探针与细节分析，未涉及代码生成与 Agentic 工作流评测。
- DiagramAgent / VAREdit：提供多智能体或自回归图表编辑框架，但缺乏统一的多维度基准与细粒度 SOM 指标。
- TikZero / DeTikZify / TikZilla / MathCoder-VL：专注 TikZ 代码生成，但未覆盖编辑任务、化学/电路领域及 Agentic 能力解耦。
- Davinci / XML-driven / Ontology-driven：偏向单点解析或特定领域方案，本文首次以 TikZ 为原生代码表示构建端到端评测基准。
- CrystalBleU / Agentic AI 综述：本文引入 CrystalBleU 作为代码级评估标准，并将 Agentic 四维度系统化引入图表理解评测。

## 局限性与未来方向
- 3D Shapes、Chemistry、Circuit 三类领域数据量偏少（合计 777 张），制约模型在复杂空间与专业符号上的上限。
- 当前 MLLMs 的**空间定位（bbox）与 3D 空间推理**仍是显著瓶颈，现有评测主要依赖 2D 拓扑与静态属性。
- Agentic 评测暴露工具滥用（context rot）与状态管理退化问题，提示未来需设计更鲁棒的 agent 编排机制与长程记忆管理。
- 未来可扩展至生物学通路、材料晶体结构等更多科学子领域，并探索模型反馈驱动的自适应难度基准。

## 研究启发与可借鉴点
1. **SOM 语义注入流水线**：通过 LaTeX 宏包预处理嵌入结构化标签，结合多引擎编译容错与 SVG DOM 解析，为其他“代码↔图像”对齐任务提供了可复用的 Ground Truth 自动提取范式。
2. **正交编辑维度拆解**：将 D2C-E 划分为 Text/Color/Scope/Layout 四个维度，使评测具备细粒度诊断能力，可迁移至代码生成/编辑类基准的任务设计。
3. **MCP-based 领域知识库**：用精选 TikZ 文档构建轻量 MCP Server，有效规避整本 PDF 检索带来的 context rot，对 Agentic 评测的数据工程与检索架构具有直接参考价值。
4. **双 Judge + 多引擎验证闭环**：“模型生成→双模型交叉验证→多 LaTeX 引擎编译确认→人工抽检”机制大幅降低自动化基准噪声，可
