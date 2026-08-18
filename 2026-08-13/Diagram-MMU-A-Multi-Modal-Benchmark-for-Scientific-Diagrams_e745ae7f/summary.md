---
title: "Diagram-MMU-A-Multi-Modal-Benchmark-for-Scientific-Diagrams"
source: https://arxiv.org/pdf/2608.12262v1.pdf
model: agnes-2.5-flash
chunks: 4
summarized_at: "2026-08-18 10:32:25"
field: "多模态大模型评测基准"
keywords: ["scientific diagrams", "multimodal benchmark", "TikZ", "code generation", "MLLM evaluation", "semantic object model"]
innovations: ["首个覆盖6大科学领域且原生支持TikZ/LaTeX生态的综合基准", "提出SOM管道实现从LaTeX源码到SVG的对象级自动评测", "揭示MLLM在科学图表理解中的感知-编码鸿沟"]
benchmarks: ["Diagram-MMU", "D2C-P", "D2C-E", "DQA"]
---

# 论文速读：Diagram-MMU: A Multi-Modal Benchmark for Scientific Diagrams

## 一句话总结
本文提出了 **Diagram-MMU**，首个面向科学图表的多模态综合基准，涵盖 TikZ/LaTeX 原生编码与多维度评测，揭示了当前 MLLM 在科学图表理解中存在"推理强、感知与编码弱"的感知-编码鸿沟。

## 研究问题与动机
1. **感知-编码鸿沟**：现有 MLLM 在 DQA 等推理任务上表现良好（最高 86%），但在视觉感知与 TikZ 代码生成上能力薄弱（D2C-P 的 object-level F1 仅 31–57%），二者之间存在显著差距。
2. **基准缺失**：现有基准多聚焦单一任务或仅覆盖部分图表类型，缺乏对 TikZ/LaTeX 绘图生态的原生支持，也未评估 agentic 能力。
3. **科研工作流需求**：现代科学写作工具（如 OpenAI Prism 的 vibe writing workspace）要求 MLLM 同时具备基础能力（感知、编码、领域知识、推理）和 agentic 能力（上下文利用、工具调用、状态管理、规划），而现有评测无法全面刻画这一需求。

## 核心贡献（创新点）
1. **首个覆盖 6 大科学领域的综合基准**：Diagram-MMU 包含 3,744 个独立图表、18,305 个人工验证样本，首次将 Chemistry 和 Circuit 纳入 diagram-to-code 评测，填补了领域覆盖空白。
2. **三重任务定义（D2C-P / D2C-E / DQA）**：统一建模图表到代码解析、代码编辑和问答推理，实现从视觉感知到文本生成再到逻辑推理的完整评测链条。
3. **Semantic Object Model (SOM) 多粒度评测框架**：提出基于 LaTeX 语义注入 → SVG 编译转换 → DOM 解析的自动化对象级度量管道，支持 Type/Text/Color/BBox 四维度 F1 评估，解决了 TikZ 代码与视觉对象对齐的评测难题。
4. **LLM-as-a-Judge 结合规则匹配的双层 DQA 评测**：针对开放式 DQA 答案，采用规则匹配 + Qwen3-Next-80B-A3B-Instruct judge 的两阶段流程，兼顾效率与准确性。

## 方法详解
### SOM（Semantic Object Model）评测管道
- **阶段一：语义注入**  
  通过自定义 LaTeX 包 `semantic_spy.sty`，在编译前将 TikZ/pgfplots/CircuiTikZ 命令钩入 `<g>` 标签，注入 `data-id`、`data-shape`、`data-text` 等属性；遇到兼容性问题时回退至纯几何提取。
- **阶段二：编译与 SVG 转换**  
  优先 DVI 模式（保留语义标签），失败时依次回退 `pdflatex → lualatex → xelatex`，再通过 `dvisvgm`（参数 `--font-format=svg --precision=6 --zoom=1`）转换为 SVG。
- **阶段三：DOM 解析提取**  
  使用 `lxml` 遍历 SVG，提取四类对象属性：
  - **Type**：细粒度类型（如 `node:circle`）
  - **Text**：优先级为 `data-text → metadata → glyph 重建 → fallback`
  - **Color**：fill/stroke 颜色
  - **BBox**：轴对齐包围盒

### Glyph 文本重建
处理 `dvisvgm` 将文字转为 `<use>` 引用 `<defs>` 中 glyph path 的情况，按 x 位置排序、解码字符码、拼接还原文本。

### 域映射（关键 TikZ 包与 SOM 元素）
| 域 | 核心 TikZ 包 | 关键 SOM 元素 |
|---|---|---|
| Charts | pgfplots | AXIS, DATASERIES, DATAPOINT, NODE, TEXT |
| Graph | tikz | NODE (circle/rect/diamond), PATH, TEXT |
| Planar Geom. | tikz/tkz-euclide | NODE, PATH (edges/arcs), TEXT |
| Circuit | circuitikz | COMPONENT (R/C/L/V), NODE, TEXT |
| 3D Shapes | pgfplots/tikz | AXIS, PATH, NODE, FILL/FILLDRAW |
| Chemistry | chemfig | NODE (atoms/groups), PATH (bonds), TEXT |

### 评估指标体系
- **对象级 F1**：对 Type/Text/Color/BBox 四个维度分别计算 Precision、Recall、F1，取均值 $F1_{avg} = \frac{1}{4}\sum F1_*$
  - Type：基于计数的多重集匹配
  - Text：贪心一对一精确字符串匹配
  - Color：排列最优匹配，用 CIE2000 感知相似度 $\sin(\hat{k}, k) = \max(0, 1 - \Delta E_{00}/100)$
  - BBox：贪心一对一 IoU 匹配（阈值 $\tau=0.3$）
- **代码级**：CrystalBLEU（过滤高频无信息 token）
- **图像级**：SSIM、CLIP Score（ViT-B/32）、LPIPS（AlexNet）、FID（Inception v3 2048-dim）；复合分 $SC=(SSIM+CLIP)/2$，$FL=(FID+LPIPS)/2$
- **DQA**：LLM-as-a-Judge 两阶段流程（规则匹配 → 低置信度时 LLM judge）

### Preserve/Edit 拆分
- **preserve-only**：评估未变元素的保留质量
- **edit-only**：从 GT 和生成代码中减去与源代码共有的元素后计算 F1

## 实验与结果
- **数据集规模**：3,744 个独立图表、18,305 个人工验证样本，覆盖 6 大科学领域
- **DQA 问题类型**：
  - **描述性问题**：涵盖六类图表的信息提取，如识别节点度数、角度符号、三角形类型、化学键类型、电路串联/并联等
  - **推理问题**：涉及公式计算与假设性修改，如求面积/体积、最短路径、欧姆定律计算、连通分量数等
- **核心发现**：MLLM 在 DQA 推理任务上表现较强（最高 86%），但在 D2C-P 的 object-level F1 上仅 31–57%，揭示了显著的"感知-编码鸿沟"
- **评测基线**：论文未明确列出具体的基线模型名称与数字（分段笔记第 2/4 段内容缺失）
- **最强结果**：需参考原文完整版以获取具体模型对比数字

## 相关工作脉络
1. **现有图表理解基准**：多数聚焦单一任务（如仅 DQA 或仅图像生成），缺乏对 TikZ 代码生态的原生支持；本文首次统一评估编码+推理+agentic 能力。
2. **TikZ/LaTeX 生成研究**：已有工作多关注文本到 TikZ 的代码生成，本文反向关注图表到 TikZ 的解析（D2C-P）与编辑（D2C-E）。
3. **化学/电路图基准缺失**：Chemistry 和 Circuit 领域此前未被纳入 diagram-to-code 评测，本文首次覆盖这两个高价值科学领域。
4. **对象级评测方法**：传统评估依赖图像级指标（SSIM/FID），本文提出 SOM 管道实现像素级→对象级的语义对齐，更贴合科学图表的结构化特性。
5. **LLM-as-a-Judge 在科学 QA 中的应用**：本文采用 Qwen3-Next-80B-A3B-Instruct 作为 judge，结合规则匹配实现高效准确的双层评估。

## 局限性与未来方向
1. **基线模型覆盖有限**：受限于篇幅，论文可能未评测所有主流 MLLM，未来可扩展至更多模型架构。
2. **TikZ 兼容性挑战**：部分 TikZ 命令在语义注入时存在兼容性问题，需回退至纯几何提取，可能影响评测精度。
3. **Agentic 能力尚未充分评测**：虽提出 agentic 需求，但当前基准主要评估基础能力，工具调用、状态管理等 agentic 维度有待完善。
4. **DQA judge 依赖**：LLM-as-a-Judge 可能引入主观偏差，需进一步研究更可靠的自动评分方法。

## 研究启发与可借鉴点
1. **SOM 评测管道可迁移**：语义注入→编译转换→DOM 解析的三阶段设计可用于其他矢量图形（如 Mermaid、Graphviz）的自动评测。
2. **Preserve/Edit 拆分思路**：将编辑任务拆分为"保留部分"和"修改部分"分别评估，可推广至任何 code generation 任务的细粒度分析。
3. **CIE2000 颜色相似度**：在对象级评测中使用感知均匀的颜色距离（而非简单 RGB 距离），值得在视觉生成评测中借鉴。
4. **领域专用 TikZ 包映射**：本文建立的 6 大领域 TikZ 包与 SOM 元素映射表（Table C.1）可作为后续扩展新领域的参考模板。
5. **混合评估策略**：规则匹配 + LLM judge 的两阶段 DQA 评测兼顾效率与覆盖，可为开放域科学 QA 提供评估范式。

## 关键术语表
**Diagram-MMU**：首个面向科学图表的多模态综合基准，涵盖 TikZ 编码与多维度评测。
**D2C-P（Diagram-to-Code Parsing）**：将输入图表解析为完整可执行 TikZ 代码的任务。
**D2C-E（Diagram-to-Code Editing）**：根据修改指令生成新 TikZ 代码的任务，支持 text/color/scope/layout 四维编辑。
**DQA（Diagram Question Answering）**：回答图表相关问题的任务，含描述性与推理性问题。
**SOM（Semantic Object Model）**：基于 LaTeX 语义注入→SVG 转换→DOM 解析的对象级评测管道。
**CrystalBLEU**：过滤高频无信息 token 后的 BLEU 变体，用于评估 TikZ 代码语法正确性。
**CIE2000**：感知均匀的颜色距离公式，用于对象级颜色相似度匹配。
**LLM-as-a-Judge**：使用大语言模型作为裁判对开放式答案进行自动评分的方法。

## 可复现要素
- **数据集**：Diagram-MMU，数据来源为官方包手册（PGFPlots、CircuitikZ、TKZ-Euclide、ChemFig、TikZ-Network）+ 社区资源（texample.net、TeX Stack Exchange、GitHub tikz_favorites）
- **代码开源**：论文未明确提及代码仓库地址，需查阅原文补充声明
- **权重开源**：未提及
- **关键超参**：BBox IoU 阈值 τ=0.3；SVG 转换参数 `--font-format=svg --precision=6 --zoom=1`
- **评测工具**：自定义 LaTeX 包 `semantic_spy.sty`、`dvisvgm`、`lxml`
