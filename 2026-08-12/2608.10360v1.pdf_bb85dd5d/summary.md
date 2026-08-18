---
title: "MazzikaAI: A knowledge-based performance-to-prompt compiler for real-time Arabic maqam accompaniment with a streaming text-to-music model"
source: https://arxiv.org/pdf/2608.10360v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 02:50:16"
field: "交互式生成音乐"
keywords: ["real-time music accompaniment", "Arabic maqam", "prompt compilation", "knowledge-based system", "streaming text-to-music", "microtonal generation", "human-AI co-creation"]
innovations: ["将自然语言提示编译定义为实时反馈控制律，驱动未微调流式音频生成器", "微音程调式的手工知识-base 构建与提示级锚定机制替代模型微调", "确定性编译器+随机生成器的神经符号分工架构"]
benchmarks: ["Maqam grounding ablation (quarter-tone frame percentage)", "Gating ablation (on/off/static push rate comparison)", "Latency decomposition (key-press to audible)"]
---

# 论文速读：MazzikaAI: A knowledge-based performance-to-prompt compiler for real-time Arabic maqam accompaniment with a streaming text-to-music model

## 一句话总结
本文提出 MazzikaAI，一种基于知识的实时伴奏系统，通过自然语言作为执行器构建闭环控制，将演奏者实时 MIDI/手势/和声状态编译为动态文本提示，驱动未微调的流式文本到音乐模型（Google Lyria RealTime）实现阿拉伯马卡姆微音程风格的交互式伴奏，无需任何模型微调。

## 研究问题与动机

1. **微音程非西方音乐传统的生成模型缺失**：现有生成音乐模型的训练框架以西方十二平均律为主，阿拉伯马卡姆音乐中的微音程、装饰音、调式 resting tones 等核心特征未被充分建模。

2. **流式生成模型缺乏细粒度实时控制接口**：Lyria RealTime 等流式模型支持增量音频生成与提示更新，但仅提供"给定提示输出音频"的表面控制，缺少从演奏状态到提示的自动化映射策略。

3. **传统符号交互系统音色真实感不足**：符号条件模型（如 MusicLM、MusicGen）提供精确控制但缺乏音频音色真实性；而音频生成模型虽能合成逼真音色，却难以在节拍粒度上实现实时响应。

4. **适应非西方音乐传统的高成本问题**：将现有模型适配到新音乐传统通常需要大规模数据收集与微调，对小众文化音乐而言不可行。

## 核心贡献（创新点）

1. **将自然语言提示编译定义为实时控制律**：提出确定性编译器 $C: (s_t, q_t) \mapsto p_t$，将估计的演奏状态与伴奏策略映射为连续更新的文本提示，使未微调的流式模型成为可响应的人类伴奏者。与已有工作的本质区别在于将提示工程从"一次性查询"扩展为"闭环反馈控制律"。

2. **面向马卡姆的微音程知识-base 构建**：手写包含六个核心 maqamat（bayati、rast、hijaz、nahawand、saba、kurd）的调式描述库，以自然语言拼写微音程度（如 "E-half-flat"、"E koron"），并提供负向引导防止模型回退到西方大小调体系。区别于训练驱动方法，此方案无需微调即可锚定微音程行为。

3. **两层解耦流架构分离旋律与打击乐控制**：旋律引擎与节拍锁定打击乐引擎独立运行，避免因控制信号冲突（旋律需自由呼吸 vs. 节奏需严格稳态）导致生成质量下降。

4. **提示级乐器约束机制替代缺失的分轨混合 API**：通过提示首部的强制乐器规则（明确列出活跃/禁声音器）弥补模型 API 不提供 stem-level mixing 的缺陷，实现演奏者主导的乐队编制切换。

5. **输入相同回放消融框架**：构建 replay harness 对同一演奏录音进行多臂回放，分离 maqam grounding、instrument suppression、gating 各设计选择对输出质量的独立贡献。

## 方法详解

**系统架构（三端两层）**：
- **客户端**：React/TypeScript SPA，通过 Web MIDI API 捕获 note_on/note_off/velocity/CC64 sustain，MediaPipe Hands 识别手势，Web Speech API 解析语音命令；本地采样钢琴渲染人类声音避免网络延迟。
- **后端**：FastAPI 服务维护 `PlayerSession`，工作记忆为最近 128 个 MIDI 事件的滑动窗口；控制器映射状态到四种伴奏行为之一。
- **生成层**：两个并发 Lyria RealTime 会话（旋律引擎 + 打击乐引擎），返回 48kHz 立体声 PCM，经 bounded queue（maxsize=8）后送客户端 Web Audio API 调度播放。

**性能状态估计 $s_t$**：
$$s_t = \Phi(\{e_i : t - W \leq \tau_i \leq t\})$$
关键特征及 horizon：
- Register（均值音高）、Texture（同时按键数）、Motion（最近 5 个 onsets 的 pitch 差分符号）
- Phrase segmentation：inter-onset gap < 0.35s 为连奏延续；> 0.4s 且静音标记乐句结束；> 2.5s 标记 idle
- Harmony fingerprint：最近 2s 的音级集合量化到 2-semitone bucket（解决相邻半音导致频繁重提示问题）
- Implied chord（十二小节 blues 区域推断）：
$$\text{score}_{A7(I)} = 3\#(A) + \#(E) + \#(G)$$
$$\text{score}_{D7(IV)} = 3\#(D) + \#(A) + \#(C)$$
$$\text{score}_{E7(V)} = 3\#(E) + \#(B) + \#(D)$$

**四状态伴奏策略 $\pi$**：
- **Supporting**：独奏者演奏时，乐队低调 comping，最高 prompt-adherence，最低密度
- **Responding**：乐句结束后打开 ~0.3–2.5s 回答窗口，提示列出独奏者最后音高并指示回声
- **Sustain**：单音持续 > 2.5s 视为 held tone，伴奏保持极简
- **LongIdle**：> 5s 静音且 section active，乐队接管并自由即兴，最高密度

**生成参数 $\theta_t$**：每genre × 每state 的 (temperature, top-k, guidance, density) 四元组 + 三档 brightness，共 135 个手工调优值。LongIdle 用高 temperature/density（乐队自由即兴），Supporting 用低 temperature/high guidance（乐队严格遵循提示）。

**重提示门控**：
$$\text{re-prompt at } t \Longleftrightarrow \sigma(s_t, q_t, \theta_t) \neq \sigma(s_{t^-}, q_{t^-}, \theta_{t^-})$$
$\sigma$ 为粗粒度控制签名元组（genre, maqam, flags, harmony fingerprint, 前四个回声音高, active section set, verse stage, 粗 bucket 化的 density/brightness, BPM, detected/implied chords），避免旋律 run 期间每个音符触发重编译。

**提示编译公式**：
$$p_t = [\text{instrument rule}] \parallel [\text{no-percussion}] \parallel [\text{header}] \parallel [\text{stage}] \parallel [\text{maqam line}] \parallel [\text{harmonic/echo}] \parallel [\text{role assignment}] \parallel [\text{register}] \parallel [\text{dynamics}] \parallel [\text{space}]$$

**马卡姆锚定提示模板**（以 bayati 为例）：
```
maqam bayati on D - scale: D E-half-flat F G A Bb C D,
the half-flat second degree (E koron) is the expressive soul of bayati,
ornaments: shimmer on E-half-flat, slides D->E-half-flat, 
descending resolution G->F->E-half-flat->D,
resting tones: D (tonic), G (dominant),
color: melancholic, warm, yearning - the most expressive Arabic maqam,
avoid western major or minor tonality, stay in bayati modal world
```

## 实验与结果

**实验设置**：
- 主会话：8.5 分钟连续演奏（1,516 个 note events，758 个 onsets，两引擎均激活，352 个 audio chunk 交付）
- 评估工具：Chromium 浏览器 + MIDI 键盘，FastAPI 后端同主机，Lyria RealTime 通过公开 API
- 延迟测量：全链路时间戳 JSONL 记录，end-to-end 直接测量而非累加

**关键结果**：

| 指标 | 中位数 | p95 | p99 |
|------|--------|-----|-----|
| 后端状态更新 | 0.3 ms | 0.5 ms | 0.6 ms |
| 提示编译 | 0.2 ms | 0.3 ms | 0.4 ms |
| 重提示推送（API 往返） | 0.3 ms | 0.8 ms | 0.9 ms |
| 端到端（note-triggered） | 263 ms | 818 ms | 945 ms |
| 重提示率 | 179/min | — | — |
| 门控吸收率 | 43.1% | — | — |
| 流失败数 | 0 | — | — |

**消融实验**：

1. **Maqam grounding（有/无）**：
   - 有 grounding：主音 D 能量占比 0.140（pool），12-TET 网格外 ≥35 cents 帧占比 22.4% vs. 无 grounding 的 14.9%（95% CI [21.7, 23.2] vs. [14.0, 15.7]）
   - 第二度区域：grounded 时 59.4% 帧落入 half-flat band（125–175 cents），ablated 仅 24.1%
   - 结论：提示级锚定显著提升微音程内容，但未能稳定调准半平第二度为独立 scale degree

2. **Instrument suppression（有/无）**：
   - PANNs Cnn14 审计显示排除乐器检测率在有抑制clause时反而更高
   - 结论：提示级否定指令无法替代 stem-level API 控制，已知 negative prompting 的" evoke what it names"效应

3. **Gating（on/off/static）**：
   - off 模式：推送率 202/min（vs. on 的 173/min），仍零失败
   - static 模式：触发→音频延迟从 470ms 骤增至 34,828ms（~35s），证明编译器对实时响应性的核心价值

**感知评估**：两位专家音乐家独立演奏反馈——伴奏 tempo 恒定不随独奏者呼吸变化，定位主要 Gap 在 beat-level entrainment 缺失。

## 相关工作脉络

1. **CHORAL (Ebcioğlu, 1988)**：手写规则库和声化巴赫四部和声，MazzikaAI 继承可 inspected 的知识表示与确定性推理，但用流式音频生成器替代符号效应的器。

2. **The Continuator (Pachet, 2003)**：在线学习演奏者风格并延伸乐句，exploit turn-taking dynamics；MazzikaAI 执行相同 listen-then-respond 循环但通过语言中介连接大型预训练音频模型而非符号作曲家。

3. **MusicLM / MusicGen (Agostinelli et al., 2023; Copet et al., 2023)**：文本条件音频生成模型，离线 one-shot 合成；MazzikaAI 利用其流式变体 Lyria RealTime 并自动闭合控制回路。

4. **Live Music Models (Lyria Team, Google DeepMind, 2025)**：描述 audio injection（直接音频 conditioning）与手动 prompt steering；MazzikaAI 填补自动结构化状态估计→提示编译的缺口。

5. **Computational Makam 综述 (Bozkurt et al., 2014)**：聚焦音高/调律分析挑战；MazzikaAI 转向生成侧，以提示锚定而非重新训练应对微音程建模。

6. **ReaLJam (Scarlatos et al., 2025)**：RL-tuned transformer 实时人机 jamming；MazzikaAI 走专家系统路线，无需训练数据即可适配新音乐传统。

## 局限性与未来方向

**自述局限**：
1. 无大规模用户研究，仅两位专家感知评估
2. 提示级乐器抑制失败（negative prompting 反向效应）
3. 依赖闭源托管模型 Lyria RealTime，受制于可用性/配额/行为变更
4. 和声推断部分 blues-specific，泛化至任意调性需额外规则
5. 单演奏者建模，无多演奏者概念
6. 语音命令仅英文，与阿拉伯音乐焦点不匹配
7. 手势控制依赖光照/摄像头位置，未在舞台条件验证

**未来方向**：
1. **节拍跟踪与 tempo following**：从 onsets 流估计 tempo 与 beat phase，在乐句结束/long idle 等安全边界 re-anchor
2. **门控重校准**：粗化 harmony fingerprint 或添加 hysteresis， sweep re-prompt frequency vs. perceived coherence
3. **感知验证规模化**：扩展为 controlled listening study
4. **开源模型迁移**：retarget 到 open-weights 流式模型实现 on-device 部署
5. **混合编译器**：从表演数据 tune phrasing/timing 同时保留可 inspected 文本接口

## 研究启发与可借鉴点

1. **"Prompt as Control Law" 范式可迁移**：将自然语言提示从一次性查询扩展为实时闭环反馈执行器，该抽象可迁移至其他流式生成模态（实时视觉、代码生成、文档辅助）。

2. **知识-base 轻量化适配非西方传统**：通过手写调式描述+负向引导替代微调，为低资源音乐传统（如 Indian raga、Persian dastgah）提供可扩展模板，适配成本仅为新 prompt fragment 编写。

3. **确定性编译器 + 随机生成器 的神经符号分工**：可 inspected 的规则层负责 idiom 与 moment-to-moment intent，foundation model 负责 raw musical competence——此分工在需可解释性的 Creative AI 系统中具参考价值。

4. **Replay harness 消融方法论**：输入相同回放控制 stochaticity，分离设计选择贡献——对评估基于 API 的端到端系统具通用参考价值。

5. **两层解耦流架构**：旋律/节奏控制信号冲突时物理分离生成会话，避免单一 prompt 中的 trade-off；可借鉴于多轨道实时交互系统。

## 关键术语表

**Maqam（马卡姆）**：阿拉伯音乐调式体系，以微音程、特征装饰音、resting tones 为组织原则，不同于西方大小调体系。

**Quarter-tone（微音程/四分之一音）**：小于半音的音高间隔，阿拉伯音乐核心特征，12-TET 网格无法精确表达。

**E koron（半平E）**：bayati maqam 的第二度音，位于 E 与 E♭ 之间约 150 cents，是该调式的"表达灵魂"。

**Takht（塔赫特）**：传统阿拉伯室内乐合奏编制，通常包含 cello、oud、strings、qanun、nay 等乐器。

**Call-and-response（呼叫-回应）**：阿拉伯音乐核心互动形式，独奏者与伴奏者交替演奏，MazzikaAI 通过 echo clause 实现。

**Harmony fingerprint（和声指纹）**：最近 2s 音级集合量化到 2-semitone bucket 的粗粒度快照，用于稳定重提示门控避免音符级颤动。

**Negative guidance（负向引导）**：提示中明确禁止 Western major/minor tonality 的条款，作为 repulsion term 将生成轨迹约束在目标调式内。

**Re-prompt gate（重提示门控）**：粗粒度控制签名 $\sigma$ 比较机制，仅当 musically meaningful determinant 变化时才触发提示更新。

## 可复现要素

- **数据集**：未使用公开数据集；主会话由第一作者亲自演奏（8.5 min, 1,516 note events）
- **代码**：figshare  dépôt (DOI: 10.6084/m9.figshare.XXXXXXX)，含系统源码、replay harness、消融日志、分析脚本、生成音频（~420 MB）
- **模型**：Google Lyria RealTime（closed API，需 API key）
- **关键超参**：
  - Rolling window: 128 note events
  - Phrase boundary gap: < 0.35s legato, > 0.4s phrase end, > 2.5s idle
  - Answer window: 0.3–2.5s post-phrase
  - Harmony fingerprint bucket: 2-semitone
  - Bounded queue maxsize: 8 chunks
  - Negative overlap: −40ms melody, −20ms percussion
  - Attenuation: 0.45× melody, 0.38× percussion
  - Cursor reset threshold: 600ms melody, 800ms percussion
- **环境**：React 19/TypeScript + FastAPI/Uvicorn + Web MIDI API + MediaPipe Hands + Web Speech API
