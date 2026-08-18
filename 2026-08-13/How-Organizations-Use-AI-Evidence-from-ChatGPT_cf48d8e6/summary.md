---
title: "How-Organizations-Use-AI-Evidence-from-ChatGPT"
source: https://arxiv.org/pdf/2608.12236v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 12:25:37"
field: "企业生成式 AI 采纳与组织内扩散"
keywords: ["enterprise AI adoption", "generative AI telemetry", "ChatGPT Enterprise", "organizational diffusion", "intangible complements", "task classification"]
innovations: ["首次将 ChatGPT Enterprise 日志链接到 Compustat 美股财务面板，刻画 adoption 的规模与无形资本决定因素", "在六个月采纳窗口后分解 composition vs. intensity 双边际，发现 early-career 员工人均消息比同 firm 均值高 8–9 条", "构建 60 类两层任务 taxonomy 与 prevalence/message-share 双指标框架，证实企业 AI 使用呈长尾分布而非单场景主导"]
---

# 论文速读：How Organizations Use AI — Evidence from ChatGPT Enterprise

## 一句话总结
本文利用 ChatGPT Enterprise 企业内部日志数据（覆盖超过 1,500 家组织、约 1,700 万条消息），首次从**企业采纳**与**组织内使用**两个层面刻画了生成式 AI 在知识工作中的扩散图景：早期采纳集中于规模大、无形资本密集的美国上市公司，而采纳后的使用跨越岗位与层级但强度极不均衡——早期职业员工与分析师/市场沟通岗位为最密集用户群，任务分布呈现广泛的"长尾"而非单一场景主导。

## 研究问题与动机
- **核心问题**：企业获得 ChatGPT Enterprise 账号后，哪些员工在用、用得多深、用在哪些任务上？现有证据多来自问卷调查或消费者侧 telemetry，难以刻画**组织内部异质性**。
- **调查数据的局限**：McElheran et al. (2024)、Bick et al. (2026a) 等 surveys 能捕捉 adoption barriers 与组织背景，但自报使用通常粒度粗糙、易受回忆与报告偏差影响。
- **技术遥测的缺口**：Counts et al. (2026)（M365 Copilot）与 Johnston et al. (2026)（Codex）已提供高质量 telemetry，但分别聚焦微软生态或代码场景；本文聚焦**通用对话型 ChatGPT Enterprise** 在企业内的全职能扩散。
- **GPT 理论预测待验**：Bresnahan & Trajtenberg (1995)、Brynjolfsson et al. (2021) 指出 GPT 的价值实现依赖互补性投资与组织协同， adoption ≠ effective deployment——本文用微观数据直接检验这一命题。

## 核心贡献（创新点）
1. **首次链接 ChatGPT Enterprise 日志到 Compustat 财务面板**，以 410 家美股上市公司的采用/不使用身份与年度 tokens/messages 构建因果识别之外的"条件关联"证据，回答"谁先采纳、为何采纳"。
2. **在六个月采纳窗口后分解组织内的使用异质性**：同时报告"活跃用户构成"（extensive margin）与"人均消息强度"（intensive margin），揭示 composition ≠ intensity 的维度分离——这与 Bonney et al. (2026) 仅关注 firm-level adoption 形成本质区别。
3. **构建并公开任务分类器（60 类两层 taxonomy）**，将每条消息映射到 work task；提出"任务 prevalence（触及多少用户）"与"message share（产生多少 tokens）"双指标框架，区分 diffusion 的广度与深度。
4. **发现早期职业员工的"高强度-低话语权"悖论**：early-career workers/trainees 人均周消息比同firm平均高 8–9 条，而 executives 反而更低——对 Brynjolfsson et al. (2025a) "AI 就业效应" 提供组织内部的 microfoundation。
5. **识别 intangible complements（SG&A/R&D/资本化软件存量的折旧累积）对 adoption 的稳健正向预测力**，将 GPT 文献从"规模效应"推进到"互补资产效应"。

## 方法详解
- **数据样本构造**：四重样本分层（图 1）。① aggregate enterprise usage panel（2024.01–2026.03，组织周面板）；② worker-characteristics sample（1,764 组织、1,744.6 万消息，含 NAICS + 标准化 job title）；③ task-classification subsample（973 组织、869.7 万消息，基于 2025.10.30 起的任务分类器）；④ public-company financial sample（521 ticker-year、410 tickers，桥接到 Compustat）。
- **采用定义**：组织在当财政年度首次出现 ChatGPT Enterprise 账户即视为 adopter；non-adopter 为Compustat 中无 ticker-bridge 匹配的美股上市公司。采用 LPM（线性概率模型），聚类标准误至 gvkey。
- **使用强度指标**：weekly messages per active usage week per employee、WAU per employee、weekly output tokens per employee、messages per WAU；对计数变量做 log(1+x) 变换后 OLS。
- **无形资产存量构造**：SG&A 以 20% 年折旧率永续累积、R&D 以 15% 年折旧率永续累积（基于 FY2021 流量），资本化软件直接取 Compustat capsft 水平值；全部除以 employee 后作 log(1+·)。
- **Job title 分类器**：gpt-5-mini（minimal reasoning）输出 JSON，含 inferred_department / role_level / people_manager_signal / job_title_class 四类结构化标签；经 top-5 规则与 cross-org 频率抑制（<2 组织出现则 redact）以保护隐私。
- **任务分类器**：两层 taxonomy（12 一级类、共 60 二级类），经内部人工+模型基准评估；每条 turn 归属唯一 label，Panel A/B 分别以 user-reach 与 message-volume 为权重选 top-12 展示。
- **因果边界声明**：作者反复强调估计为**条件关联**（conditional association），不解释为 causal effect；adopter 非随机分配，存在 selection on observables（scale、R&D intensity）与不可观测 complementarity。

## 实验与结果
- **增长事实**：2025.06→2026.03 总 output tokens 增长约 **7 倍**；在 2025.06 前已采用的 cohort 内部又增长约 **4 倍**——即约一半增量来自**既有客户的深度使用**而非新户流入。2026 年初各 cohort 同步加速，指向供给侧/模型能力跃迁而非单纯 adoption expansion。
- **财务截面**：adopter 中位数 revenue $2,275.1M vs non-adopter $209.6M；median employment 2,934 vs 424；median R&D expense $113.1M vs $9.9M（图 2）。Table 1：lagged log(rev/emp) 每增 1 log-point，adopt 概率上升 **0.4–0.9 pp**（p<0.01），控制 assets/emp、PP&E/emp、year/NAICS4 FE 后仍显著；PP&E/emp 系数为负，说明物理资本强度与 adoption 负相关。
- **规模尾部**：Table 3 显示 top-25% revenue  firms adopt 概率高 **6.9 pp**，top-5% 高 **9.8 pp**；行业同 cell 内 top-5% 高 **11.3 pp**（Autor et al. 2020 相对规模法）。
- **互补资产**：Table 4 中 SG&A stock/emp 系数 **0.020***（全样本）、R&D stock/emp **0.004***、资本化软件 **0.008***；排除 tech/high-R&D 后 SG&A 仍正向（0.010，边际显著），其余不显著——**组织运维类无形资本是 adoption 的最稳健 predictor**。
- **使用强度财务关联**：Table 2 显示 firm scale（log emp）对 per-employee intensity 为显著**负向**（-0.266*** msgs/active-wk/emp，-0.667*** tokens/emp），符合"大厂每人用得更少"的 scaling 规律；rev/emp 对 intensity 点估计正但不显著。
- **岗位构成 vs 强度分离**：图 5 显示执行层占活跃用户 ~10%，早期职业 ~7%；但图 6 中 early-career 人均周消息比同 firm 均值高 **8–9 条**，executives 反而为负——**高渗透 ≠ 高强度**。
- **任务分布**：图 7 Panel A 超 50% 活跃用户做过 documentation/technical writing；Panel B 该类别亦占消息量最大份额。60 类 taxonomy 中 long tail 明显，"other" 类别占相当比例，支持"通用目的技术"假说。
- **行业/岗位/层级差异**：图 8–10 显示 finance/insurance 的 financial & tax 任务 prevalence 显著更高；engineering 偏向 technical digital work & debugging；sales/marketing 偏向 sales & marketing tasks——**共性任务（writing/communication/tech digital）跨群体稳定，特异性任务随职能分化**。

## 相关工作脉络
- **Bonney et al. (2026)** NBER WP 35141：区分 firm adoption 与 business-function/task deployment，本文在其基础上进一步在**同采纳周期时点**上比较 worker-role 与 seniority 维度的使用强度。
- **Counts et al. (2026)** arXiv:2605.23958（M365 Copilot telemetry）：聚焦微软办公栈的企业使用，本文聚焦**通用对话型 ChatGPT Enterprise**，两者互为补充而非替代。
- **Johnston et al. (2026)** arXiv:2606.26959（Codex 转向 agentic AI）：研究代码场景下的 adoption/intensity/task shift，本文把"代码工作"推广到**全职能知识工作**。
- **Handa et al. (2025)** Anthropic Economic Index：Claude 对话的 task 分类与职业覆盖，本文任务 taxonomy（60 类）与评估基准与之可交叉验证。
- **Brynjolfsson et al. (2025b)** QJE "Generative AI at Work"：基于员工-任务随机实验的 productivity 效应，本文的 telemetry 描述扩散现状，为其提供**前置的 adoption 画像**。
- **Bresnahan & Trajtenberg (1995) / Brynjolfsson et al. (2021)** GPT 理论：complementarity + co-invention 假说，本文用 SG&A/R&D stock 的实证结果为其提供**企业层面的量化证据**。

## 局限性与未来方向
- **仅覆盖 ChatGPT Enterprise 单产品**，未包含个人账号、API 调用、内部自建应用与竞品（Claude/ Copilot）使用，**总 AI 渗透被低估**。
- **Job title 覆盖率不完整**，且缺少各岗位在岗人数分母，"活跃用户占比"不能转化为"岗位采纳率"；作者坦承无法判断某类岗位是否 over/under-represented。
- **任务分类基于消息内容**，不捕捉 downstream work products 与真实 productivity 变化——high message intensity 可能意味着"高辅助"也可能意味着"高试错/低产出"。
- **财务样本局限于美股上市公司**，排除 SME、私营企业与非美企业， adoption 的规模效应结论不可外推至中小型企业。
- **识别为条件关联非因果**，未做 DiD / IV / regression discontinuity，adopter 与 non-adopter 的系统差异可能源于不可观测的管理质量或数字化成熟度。
- **六个月窗口偏短**，难以捕捉 organizational learning curve 后期（Bresnahan et al. 2002 的 co-invention 阶段）。

## 研究启发与可借鉴点
1. **双 margin 框架（prevalence × message share）可直接迁移**：后续研究评估任意 GPT 工具时，分别报告"触及多少用户"与"贡献多少 tokens"，避免单一指标的误导性。
2. **SG&A/R&D 永续折旧存量的构造方法**（20%/15% 年折旧）可作为"GPT 互补资产"的可复用代理变量，嵌入企业 adoption 的计量模型。
3. **job-title → department/seniority/manager-signal 的 JSON 结构化分类 + 频率抑制**是一套兼顾研究粒度与隐私保护的方法论模板，可直接移植到其他企业 telemetry 项目。
4. **"composition ≠ intensity"的发现提醒实验设计**：评估 AI 干预时，应同时报告 participation rate 与 per-capita usage depth，否则可能把"全员浅用"误判为"重度采纳"。
5. **early-career 高强度使用与 Brynjolfsson et al. (2025a) canary 发现的衔接**：可进一步用本文的 task 分类对接 productivity 实验，检验"谁用得多 = 谁受益多/受损多"的假设。

## 关键术语表
- **General Purpose Technology (GPT)**：具有广泛渗透性、依赖互补性投资并在长周期内通过 co-invention 释放生产率的底层技术范式（Bresnahan & Trajtenberg 1995）。
- **Extensive margin vs. Intensive margin**：前者指"多少用户/岗位首次使用"，后者指"已有用户的使用强度"；本文核心发现在两边际上方向不一致。
- **Adopter vs. Non-adopter（Compustat 口径）**：以 ticker-bridge 匹配定义，属条件关联分组，非随机对照。
- **Output tokens**：ChatGPT/Codex 向用户返回的 token 总数，本文用作 usage intensity 的核心度量。
- **WAU（Weekly Active Users）**：当周至少发送一条消息的独特用户数，衡量参与广度。
- **Intangible complements（SG&A/R&D/资本化软件存量）**：以永续折旧法从历史流量累积的无形资本，代表组织的流程、培训、软件基础设施等 GPT 落地所需配套能力。
- **Task taxonomy（60 类两层）**：一级域（如 writing、analysis、coding）× 二级具体任务（如 documentation、debug、data analysis），用于归因每条消息的工作性质。
- **Co-invention**：Bresnahan & Greenstein (1996) 概念，指采用者在使用中不断重新设计工作流、发现新用途的动态过程。

## 可复现要素
- **数据集**：ChatGPT Enterprise 组织周面板（2024.01–2026.03）、Compustat 财务面板、NAICS 行业分类；Compustat 桥接使用 curated + LLM-assisted ticker bridge，**原始消息与 job title 明细未公开**，仅发布去标识聚合统计与回归系数。
- **代码/权重**：论文未开源分析代码；job title 分类器基于 gpt-5-mini（OpenAI 内部模型，**不可复现**）；任务分类器细节见 Appendix D 但未公开权重。
- **关键超参**：SG&A 年折旧率 20%、R&D 年折旧率 15%；task classifier 阈值与 prompt 见 Appendix C/D；job title 发布规则为"跨 ≥2 组织才展示"。
- **复现路径建议**：可用 Compustat + 其他企业 AI 订阅数据（如 Anthropic/ Microsoft 的机构级匿名指标）重建 adoption 回归；任务分类可用开源 LLM（gpt-4o / claude-3.5）复刻两层 taxonomy 并做 human audit。
