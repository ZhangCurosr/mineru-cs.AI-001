---
title: "MazzikaAI: A knowledge-based performance-to-prompt compiler for real-time Arabic maqam accompaniment with a streaming text-to-music model"
source: https://arxiv.org/pdf/2608.10360v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:50:36"
field: "交互式音乐生成与非西方音乐传统"
keywords: ["实时音乐生成", "知识系统", "提示工程", "阿拉伯木卡姆", "流式文本转音乐", "人机共创", "微分音"]
innovations: ["将自然语言提示编译实时化为反馈控制律，驱动未微调流式生成器实现响应式伴奏", "构建可检查的手工知识基（6个maqam半降音拼写+114条角色描述+135个参数）替代缺失的API控制", "通过短语回声和乐器抑制两个提示级机制实现call-and-response与分轨控制"]
benchmarks: ["Maqam grounding ablation (quarter-tone frame %: 22.4% vs 14.9%)", "Latency decomposition (median 263ms, p99 945ms)", "Gating stability (0 reconnects/failures at 166 pushes/min)", "Instrument suppression ablation (negative prompting ineffective)"]
---

# 论文速读：MazzikaAI: A knowledge-based performance-to-prompt compiler for real-time Arabic maqam accompaniment with a streaming text-to-music model

## 一句话总结
论文提出 MazzikaAI，一个基于知识规则的实时伴奏系统：将演奏者的 MIDI 实时状态编译为自然语言提示，驱动未微调的流式文本转音乐模型（Google Lyria RealTime），实现对阿拉伯木卡姆（maqam）微分音体系的响应式伴奏，无需任何模型训练。

## 研究问题与动机
- 生成式音乐模型（如 MusicLM、MusicGen）的训练数据以西方十二平均律为主，对阿拉伯木卡姆的微分音、装饰音和调式结构覆盖严重不足。
- 流式文本转音乐模型（Lyria RealTime）支持实时生成和提示更新，但缺乏对演奏者行为的动态响应控制接口——给定静态提示只能输出通用伴奏，不会"倾听"独奏者。
- 现有交互式音乐系统（基于规则或神经网络）要么生成的是符号化音频缺乏音色真实感，要么需要大量领域数据微调才能适配新音乐传统。
- 实时伴奏要求系统能在亚秒级延迟内完成"倾听—分析—响应"闭环，而现有方案在控制粒度和非西方调式适配上均存在明显缺口。

## 核心贡献（创新点）
- **将自然语言提示编译实时化为反馈控制律**：把持续生成的自然语言提示视为知识控制系统的执行器，而非一次性查询，使未修改的流式模型具备响应式伴奏能力，无需微调。
- **构建面向阿拉伯木卡姆的可检查知识基**：手工编写了 6 个核心 maqam（含半降二音quarter-tone拼写）、装饰音、休息音和负面引导，以及 114 条乐器角色描述和 135 个生成参数，所有知识均为可读文本而非 learned weights。
- **提出提示级控制机制替代缺失的 API 控制**：通过有序约束子句实现乐器抑制（替代 stem-level 混音），通过短语回声（literal echo of last_phrase_pitches）实现 call-and-response，两个机制均有消融实验量化。
- **构建了完整的实时评估体系**：包括端到端延迟分解（中位数 263 ms）、门控稳定性测量（0 次连接失败/队列溢出）、输入相同回放消融（微分音含量显著提升）以及专家演奏感知评估。
- **证明知识层开销近乎免费**：状态估计+提示编译总计不到 1 ms（p99），相比生成模型 2 s 的 chunk 周期低三个数量级，符号控制流式基础模型在经济上可行。

## 方法详解
**系统架构（三层）**：浏览器前端（React/TypeScript）→ Python 后端（FastAPI）→ 云端生成模型（Lyria RealTime），通过单一双向 WebSocket 连接。前端采集四类输入：MIDI 音符事件（note_on/note_off、速度、CC64 sustain）、手势（MediaPipe Hands）、语音命令（Web Speech API）、学习模式（MusicXML/PDF OCR）。

**性能状态估计**：维护一个大小为 128 的滚动窗口 deque，计算特征包括：音区（mean pitch <50/50–68/>68）、纹理（同时按键数）、运动方向（5 个 onset 的 pitch 差符号）、乐句分割（0.35s 以内=连奏，>0.4s 静音=乐句结束）、和弦推断（模板匹配 + 12小节 blues 隐式和弦打分，公式：score_A7 = 3×#A + #E + #G 等）、和声指纹（2s 内 pitch-class 量化为 2 半音桶）、风格描述（每 2 个乐段从 30s 历史生成 NL 句子）。

**四状态伴奏策略**：Supporting（独奏者 playing 时安静伴奏）、Responding（乐句结束后 0.3–2.5s 应答窗口，列出刚弹奏的音高并指示回声）、Sustain（单音持续 >2.5s 时极简）、LongIdle（静音 >5s 时乐队接管）。有两个门控：warmup gate（段落接入后需 2 个音符才允许发声）和 silence stop（>4s 静音停止旋律流）。

**确定性提示编译器** $C: (s_t, q_t) \mapsto p_t$：按固定顺序拼接子句——[乐器约束/静音] → [无打击乐] → [特定标题] → [会话阶段] → [木卡姆描述] → [和声上下文/回声] → [角色分配] → [音区] → [动态] → [空间子句]。示例中 bayati 的 microtone 描述为："maqam bayati on D - scale: D E-half-flat F G A Bb C D, the half-flat second degree (E koron) is the expressive soul of bayati…"，并以负面引导 "avoid western major or minor tonality" 结尾。

**重提示门控**：定义粗粒度控制签名 $\sigma = (\text{genre}, \text{maqam}, \text{LongIdle/Responding flags}, \text{harmony fingerprint}, \text{first 4 echo pitches}, \text{active sections}, \text{verse stage}, \text{density/brightness buckets}, \text{BPM}, \text{detected/implied chords})$，仅当 $\sigma$ 变化时才推送新提示。

**双引擎分离**：旋律引擎（mute_drums=true，无固定 BPM，随和声/音区/乐句重提示）与打击乐引擎（tempo-locked 4/4 bed，独立运行，tighter 生成参数）完全解耦，因两者控制信号本质冲突。

**生成参数表** $\theta_t = \Theta(q_t, s_t)$：每个流派×4 状态×4 旋钮（temperature, top-k, guidance, density）+ 3 个音区亮度值 = 共 135 个人工调参值。

## 实验与结果
**延迟分解**（8.5 min 主会话，1,516 音符事件）：知识层中位数 0.5 ms（状态更新 0.3 ms + 编译 0.2 ms），API 往返 0.3 ms；端到端中位数 263 ms（note-triggered），p95=818 ms，p99=945 ms，落在 0.3–2.5s 应答窗口前端。chunk 周期约 2.0 s（模型原生速率）。

**门控与流稳定性**：1,517 次门控触发 → 1,401 次 API 推送（166/min），0 次重连/流中断/队列溢出。门控吸收 43.1% 决策（heartbeat 驱动 74.4%、note 驱动 15.1%）。即使关闭门控（202 pushes/min）仍 0 失败，证明稳定性非瓶颈，API 经济性才是门控的主要贡献。

**木卡姆接地消融**（A vs B，3 arms/条件，共 ~18 min 音频）：有接地时 tonic D 为最高能量音级（pooling share=0.140），无接地时 D 从未排第一（0.094，排第四）。≥35 cents 偏离 12-TET 的帧比例：接地 22.4% vs 无接地 14.9%（95% CI [21.7,23.2] vs [14.0,15.7]）。D 上方二度区域：接地条件下 59.4% 的帧落在半降 band（125–175 cents, E koron），无接地仅 24.1%（n=650 vs 1,530 frames）。但 aggregate energy profile 未显示集中 150-cent 峰值——微分音被"引导"而非"稳定调准"。

**乐器抑制消融**（A vs C）：PANNs Cnn14 审计发现，抑制子句未能减少排除乐器的检测；含抑制子句时 cello 类检测反而更频繁，符合 negative prompting 的" evoke what it names" 效应。

**静态提示消融**：仅编译一次不重提示时，触发→音频延迟坍缩至 ~35 s（中位数），证明编译器带来的响应性价值巨大。

**专家演奏评估**（2 位 expert musicians）：最一致的反馈是伴奏"保持恒定 tempo 而非随演奏者呼吸"——系统控制了"弹什么"但未控制"何时弹"，beat-level entrainment 缺失是主要感知缺口。

## 相关工作脉络
- **CHORAL（Ebcioğlu, 1988）**：手工规则库 + 确定性推理的巴赫四部和声专家系统，MazzikaAI 保留其"可检查知识+确定推理"二特性，仅将 symbolic effector 替换为自然语言驱动的流式生成器。
- **Voyager（Lewis, 2000）/ Continuator（Pachet, 2003）/ BoB（Thom, 2000）**：交互式即兴系统，通过 bespoke composer 实现 call-and-response，MazzikaAI 做相同的倾听-应答循环但耦合的是大型预训练音频模型而非定制符号作曲家。
- **MusicLM（Agostinelli et al., 2023）/ MusicGen（Copet et al., 2023）**：文本条件音频生成模型，支持高质量离线生成但缺乏实时响应控制；MazzikaAI 在其上层构建了实时编译控制层。
- **Lyria RealTime（Google DeepMind, 2025）**：最近的流式文本转音乐模型，支持 mid-session prompt update；本文填补了其"如何自动从演奏状态产生和更新提示"的控制策略空白。
- **Bozkurt et al. (2014) 计算 makam 综述**：指出微分音调音问题是十二音工具的failure point；Shahriar & Tariq (2021) 做 maqam 分类但未涉及生成；MazzikaAI 首次通过 prompt grounding 而非 retraining 实现微分音模态生成。
- **ReaLJam（Scarlatos et al., 2025）**：RL 调优 transformer 的实时 human-AI jamming，需大量训练数据；MazzikaAI 零数据、零微调即可适配新流派（仅编写新 prompt fragments）。

## 局限性与未来方向
- **无大规模用户研究**：仅有 2 位专家音乐家的定性评估，缺乏控制性听觉实验。
- **约束语言未被尊重**：负面提示无法替代 stem-level API 控制，排除乐器检测反而增加。
- **依赖专有托管模型**：Lyria RealTime 闭源、有配额限制、2 s chunk 粒度设下延迟下界，且不支持流中改 BPM。
- **知识覆盖有限**：和声推断部分为 blues 专用（12小节 A7/D7/E7 推断），不泛化到任意调性；状态估计仅建模单个键盘演奏者，无多人协作感知；语音命令仅支持英语。
- **未来方向**：beat tracking + tempo following（从 onset 流估计 tempo 并 re-anchor 双引擎）、重估门控粒度（coarsen harmony fingerprint 或加 hysteresis）、扩展到开源 weights 流式模型（移除网络延迟和 API 依赖）、per-stem API 控制（当未来 API 提供时替代 prompt-level 抑制）。

## 研究启发与可借鉴点
- **"Prompt as control law"范式可迁移至其他多模态生成系统**：将 NL 提示视为实时控制器的执行器而非一次性查询，该模式可用于实时视觉生成（pose→prompt→视频）、代码辅助（编辑流→prompt→代码补全）等场景。
- **手工知识基 + 未微调基础模型的组合在低资源/非主流领域极具价值**：木卡姆微分音体系在训练数据中几乎不存在，但通过显式 quarter-tone 拼写和负面引导实现了显著的方向性偏移（22.4% vs 14.9% off-grid 帧），证明 prompt engineering 可作为零样本领域适配手段。
- **输入相同回放消融（replay harness）是评估随机生成系统的有效方法**：通过 replay 同一份表演输入流对比不同消融条件，分离了"编译器设计"与"模型随机性"的贡献，值得在类似研究中采用。
- **知识层开销近乎免费（<1ms）的结论强化了神经-符号架构的经济性论证**：在 2s 级生成 chunk 面前，符号控制层成本可忽略，设计焦点应完全放在提示工程和门控策略上。
- **四状态策略（Supporting/Responding/Sustain/LongIdle）的简化设计可直接复用于其他交互式音乐生成任务**，配合 per-genre 参数表的 135 个手工调参模式展示了"少量规则覆盖多样情境"的设计哲学。

## 关键术语表
**Maqam（木卡姆）**：阿拉伯音乐调式体系，以微分音（半降音）、特征装饰音和休息音为结构核心，本文聚焦 6 个核心 maqam（bayati, rast, hijaz, nahawand, saba, kurd）。
**Half-flat second degree（E koron）**：bayati maqam 的特征音级，位于 D 和 E 之间约 150 cents 处，是微分音接地实验的核心验证对象。
**Call-and-response（呼应）**：阿拉伯及非洲 diaspora 音乐中的核心互动结构，本文通过 last_phrase_pitches 回声机制在提示层实现。
**Harmony fingerprint（和声指纹）**：2s 窗口内 pitch-class 量化为 2 半音桶的 coarse 表示，用于重提示门控以避免旋律 run 期间的频繁重提示。
**Knowledge-based system（知识系统）**：经典 AI 范式，由手工知识基+工作记忆+确定性推理层组成，本文保留前二者并将 symbolic effector 替换为 NL 驱动的生成器。
**Stem-level mixing（分轨混音）**：专业 DAW 中独立控制各乐器轨音量的能力，Lyria API 未暴露此接口，本文以 prompt-level 乐器抑制作为 best-effort 替代。
**Negative prompting（负面提示）**：通过 "avoid X / X must not appear" 类表述引导模型避开某些内容，本文发现其效果有限甚至可能反向诱发目标内容。
**Re-prompt gating（重提示门控）**：仅在粗粒度控制签名 $\sigma$ 变化时才推送新提示的机制，吸收 43.1% 决策以平衡响应性与流稳定性。

## 可复现要素
- **数据集**：无传统训练数据集；使用 8.5 min 专家演奏录音作为主会话输入（1,516 MIDI 事件），数据已公开于 figshare（DOI: 10.6084/m9.figshare.XXXXXXX）。
- **代码**：系统源代码已公开于 figshare，含 replay harness、条件标志、每 arm 会话日志、分析脚本及 JSON 输出。
- **权重**：未微调任何模型；使用 Google Lyria RealTime 公开 API（models/lyria-realtime-exp endpoint）。
- **关键超参**：rolling window=128 音符；phrase gap 分割阈值=0.35s（legato）/0.4s（end）/2.5s（idle）；和声指纹桶宽=2 semitones；heartbeat=250 ms；debounce=400 ms；gesture hold=700 ms + 2–3s cooldown；voice confidence threshold=0.62；bounded queue maxsize=8；负重叠=−40 ms（旋律）/−20 ms（打击乐）；衰减=0.45×（旋律）/0.38×（打击乐）。
