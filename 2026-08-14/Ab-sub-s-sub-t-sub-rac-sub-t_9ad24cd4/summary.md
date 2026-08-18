---
title: "Ab-sub-s-sub-t-sub-rac-sub-t"
source: https://arxiv.org/pdf/2608.13263v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:48:10"
field: "大模型推理系统"
keywords: ["LLM Inference", "KV Cache", "Memory Management", "PagedAttention", "Token Eviction", "System Optimization"]
innovations: ["提出 token 级虚拟化层解耦逻辑存活与物理放置，解决 intra-block 碎片", "Token table + 异步 compaction 实现物理回收，保持 CUDA Graph 兼容", "集成代码从 500+ 行降至 <50 行，支持 H2O/Scissorhands/Random 等多种驱逐策略"]
benchmarks: ["ShareGPT", "LongBench"]
---

# 论文速读：vToken: Token-level Virtualization for Reclaimable KV Caches

## 一句话总结
论文针对 LLM 推理中 token 级 KV 驱逐算法与 block 级 PagedAttention 管理之间的粒度不匹配问题，提出 vToken——一个轻量级 token 级虚拟化层，通过 token-table 间接寻址与异步 compaction 物理回收，将已被驱逐的 token 从物理 block 中释放，显著降低 intra-block 碎片并提升服务吞吐与并发能力。

## 研究问题与动机
- **粒度不匹配导致大量 intra-block 碎片**：现有 token 级驱逐策略（如 H2O、Scissorhands）在 token 粒度决策驱逐，但底层运行系统（vLLM/PagedAttention）以 block（通常 16 token）为单位分配/回收 KV 内存，导致部分存活 block 内部出现空洞，无法被其他请求复用。
- **碎片量化严重**：初步实验显示，在 16K context 下运行 token 级驱逐后，大多数已分配 block 利用率不超过 50%，intra-block 浪费比例 $F$ 达 40%-60%，大量 GPU 显存被困在"半存活"物理 block 中。
- **新算法集成成本高**：将 token 级驱逐策略接入 block 级运行时需深入理解 block allocator 内部，手动将 token 级语义翻译为 block 级操作；直接集成 H2O 等策略需修改 500+ 行代码，且每个新策略均需重复工程。
- **简单方案不可行**：缩小 block 尺寸会增加元数据开销与地址映射复杂度，降低 HBM 带宽效率；纯 token 级分配则因 GPU 内存分配粒度与元数据开销过大而不可行。

## 核心贡献（创新点）
- **提出 token 级虚拟化抽象层**：在驱逐策略与 PagedAttention 底层之间插入 vToken 层，向上暴露稳定的 token 级逻辑视图，向下负责物理 reclamation，解耦逻辑存活判断与物理 placement。
- **Token Table 间接寻址机制**：维护 per-request 的 token-table，记录每个逻辑 token 的物理位置 (block_id, offset) 及 liveness bit，使驱逐策略无需感知 block 结构即可标记 token 为 evicted。
- **惰性压缩（Lazy Compaction）异步回收后端**：基于 per-block liveness 元数据判断回收 eligibility，在 eviction headroom 约束下执行异步 KV copy，复用低利用率 block 中的 live tokens，将 emptied blocks 返回给分配器。
- **Stage-aware 异步复制避免 HBM 争用**：将 KV 复制安排在 forward 完成后、在独立 relocation stream 上执行，与 sampling/scheduling 阶段重叠，通过 CUDA event 依赖保证下一次 attention kernel 的一致性，避免显式 CPU 同步 stall。
- **保持生产级特性兼容**：不修改 PagedAttention kernels，保留 CUDA Graph 执行；支持 prefix caching（保守排除 shared-prefix blocks），集成代码从 500+ 行降至 <50 行。

## 方法详解

### 3.1 设计边界与目标
vToken 定义了一条清晰的语义边界：上层驱逐策略仅决定哪些 token 应被驱逐，不感知物理 block/attention slot；下层运行时维护 token 级逻辑地址空间，负责 logical-to-physical 映射更新、slot remapping 与异步 physical reclamation。

### 3.2 Token Table 与逻辑地址空间
Token table 是 vToken 的核心元数据结构，per-request 维护，记录每个逻辑 token ID 对应的物理位置 `(block_id, offset)` 与 liveness bit。暴露三个接口：
- `evict_token(req_id, token_id)`：标记 token 为逻辑无效，KV 条目物理保留直到 reclamation。
- `sync_new_tokens(req_id, block_ids, total_len)`：注册新生成的 token，分配顺序逻辑 ID 并记录物理位置。
- `apply_moves(req_id, moves)`：reclamation 完成后原子更新 token-table，确保后续 slot lookup 看到新布局。

Token table 主表驻留 CPU，GPU 侧维护 lookup cache 加速 steady-state 路径；仅在结构性变化（eviction/relocation/layout reconciliation）时触发 cache rebuild。

### 3.3 物理回收后端（惰性压缩）
回收后端分为四个阶段：

1. **Reclamation Eligibility**：基于 per-block liveness 元数据计算 intra-block waste ratio $F = 1 - \frac{1}{N}\sum u_i$，当 $F$ 超过阈值 $\theta_F$ 且 free block 数接近 low watermark 时触发。

2. **Headroom-Aware Admission**：回收需要 destination blocks，vToken 在调度器中预留 bounded evacuation headroom（从同一 KV block budget 中划出），仅当计划能净释放 block 时才 admit。

3. **Relocation Planning**：对每个 request，选择低利用率 source blocks，计算 live token 总数 $T_{live}$，目标 destination blocks 数为 $\lceil T_{live}/S \rceil$；move list 尽量保持逻辑 token 顺序以维持 cache locality。

4. **Stage-Aware Asynchronous Copy**：
   - 当前 step 的 forward 完成后，在独立 relocation stream 上启动 KV copy。
   - Copy 与 post-forward 阶段（sampling/scheduling/next-step preprocessing）重叠，避免与 attention 争用 HBM。
   - 下次 forward 前插入 CUDA event dependency，若 copy 已完成则无额外开销。

### 3.4 Scheduler 集成
- **Slot mapping 修改**：`slot = token_table[token_id].block_id × S + token_table[token_id].offset`，在 input preparation 阶段一次 lookup 完成。
- **Scheduler-side hook**：预留 evacuation headroom。
- **Worker-side hook**：评估回收机会、启动异步 copy、更新 layout 元数据。
- **Pre-attention sync hook**：插入 stream-level CUDA event dependency。

### 3.5 正确性不变量
- **I1 (Token conservation)**：relocation 前后每个 request 的逻辑 token ID 集合与 K/V 张量内容保持不变。
- **I2 (Unique mapping)**：每个活跃逻辑 token 唯一映射到一个 (block, offset)，每个 occupied slot 至多被一个逻辑 token 引用。
- **I3 (Pre-attention visibility)**： relocated KV entry 仅在 copy 完成后被 attention 读取。
- **I4 (Layout-aware planning)**：relocation plan 仅从一致的快照提交，不重叠 in-flight 计划，且要求 projected post-relocation block count 严格下降。

## 实验与结果

### 实验设置
- **平台**：单卡 NVIDIA H100 80GB
- **模型**：Mistral-7B、Llama-3.1-8B、Qwen2.5-14B
- **数据集**：ShareGPT、LongBench
- **驱逐策略**：H2O、Scissorhands、Random
- **基线对比**：Native vLLM（无驱逐）、Naive-Evict（token 级驱逐但无物理回收）、vToken（完整方案）
- **CUDA Graph** 全程启用

### 主要结果
| 指标 | 结果 |
|------|------|
| **内存利用率提升** | Llama-3.1-8B 提升 21.88%，Mistral-7B 提升 21.67% |
| **保留 block 数减少** | 相对 Naive-Evict 减少 27.2%-72.3% |
| **SLA 约束吞吐提升** | Mistral-7B 提升 9.9%-37.3%，Llama-3.1-8B 平均提升 18.9%；Scissorhands 策略下提升 33.3%-103.7% |
| **p95 延迟降低** | 最高降低 27.5%-47.8% |
| **最大可行并发扩展** | gpu_mem_util=0.35 时从 C=5 扩展至 C=8（+60%）；gpu_mem_util=0.50 时从 C=11 扩展至 C=22（2×）；Qwen2.5-14B 从 C=3 扩展至 C=6（2×） |
| **集成代码量** | 从 500+ LOC 降至 <50 LOC |

### 开销分析
- **Indirection-only ablation**：仅安装 vToken hooks 但不启用驱逐/回收，吞吐与 p95 相对 Native vLLM 变化 <1.0%，说明 steady-state 间接寻址开销可忽略。
- **异步复制有效性**：Force-sync ablation 暴露 CPU blocking，async 路径避免显式同步 stall。
- **Planner overhead 主导**：CPU 观测开销主要来自 planner 侧的机会检查，而非 KV relocation 记账。

### 正确性与生成稳定性
- **不变量验证**：所有 workload-policy 组合下，每轮 relocation 的 token hash 与 K/V 内容一致性均通过。
- **生成质量**：ROUGE-L F1 均值差异 -0.0016，93.1% 的 pair 差异 ≤0.01；输出长度中位数比 0.99，无系统性膨胀。

### Prefix Cache 兼容性
- 保守策略排除 shared-prefix blocks，仅回收 private suffix blocks。
- Sharing degree 8 时 retained blocks 减少 42.2%，吞吐提升 146.5%。

## 相关工作脉络
- **vLLM / PagedAttention**：block 级 KV 缓存管理基础系统，提供物理 substrate；vToken 在其上层插入 token 级虚拟化边界。
- **H2O、StreamingLLM、Scissorhands、FastGen**：token 级 KV 驱逐策略；本文不改进策略本身，而是解决这些策略在 block 级运行时中的物理回收问题。
- **PagedEviction**：将驱逐页对齐到 block 边界，限制策略灵活性；vToken 保持 policy-agnostic，在运行时解决对齐。
- **Zipage**：为 reasoning workload 设计的 bounded-cache pipeline，固定 per-request block 预算；vToken 面向通用 token 级驱逐策略，边界更抽象。
- **vAttention**：基于 CUDA VMM 在 page 粒度虚拟化地址空间，无法回收 intra-page holes；vToken 在 token 粒度虚拟化，两层可堆叠。
- **DiffKV / CacheGen / Quest**：改变 KV 表示或压缩方式；vToken 假设 uniform fp16 KV，与表征层正交。

## 局限性与未来方向
- **Prefix caching 保守处理**：当前实现排除 shared-prefix blocks 不参与回收，可能遗留可回收容量；copy-on-write 扩展可进一步优化。
- **单 GPU 单 KV cache group**：未评估多 GPU tensor parallelism 或多 KV cache group 场景下的跨 shard 协调开销。
- **Planner overhead 主导**：CPU 侧机会检查占主导，可通过 batching/indexing 进一步优化。
- **Block size 下限限制**：当前实现依赖 vLLM block size ≥16，不支持更小 block 探索。
- ** eviction ratio 敏感度**：激进驱逐可释放更多容量，但精度-性能权衡仍由上层策略决定，vToken 未自动调节。

## 研究启发与可借鉴点
- **虚拟化层抽象设计**：将"决策"（驱逐策略）与"执行"（物理回收）解耦的设计模式可迁移至其他粒度不匹配场景（如 MoE routing、KV cache offloading）。
- **Stage-aware 异步流水线**：将数据移动与 compute 阶段分离、通过 event dependency 避免 CPU stall 的模式，可应用于其他 GPU 显存管理问题。
- **轻量集成模式**：通过 scheduler/worker hook 而非修改核心模块实现新功能，50 行代码即可接入新驱逐策略，为快速原型验证提供范式。
- **Invariant 驱动的正确性验证**：I1-I4 不变量设计清晰，可直接作为测试用例，适用于系统级正确性验证。
- **Headroom-based admission control**：在回收前预留 bounded workspace 的机制，可推广至其他资源受限的 compact/repack 场景。

## 关键术语表
- **PagedAttention**：vLLM 引入的 block 级 KV 缓存管理原语，将 KV 缓存划分为固定大小 block，通过 block table 实现逻辑到物理的非连续映射。
- **Intra-block fragmentation**：token 级驱逐后 block 内部出现空洞，导致已分配物理 block 利用率低下、无法复用的碎片现象。
- **Token Table**：vToken 核心元数据结构，per-request 维护逻辑 token ID 到物理 (block_id, offset) 的映射及 liveness 位。
- **Lazy Compaction**：vToken 的物理回收机制，异步将 live tokens 从低利用率 block 重打包至紧凑 block，释放 emptied blocks 回分配池。
- **Stage-aware Asynchronous Copy**：将 KV 复制安排在 forward 完成后、与 post-forward 阶段重叠的执行策略，通过 CUDA event 依赖保证一致性。
- **SLA-constrained Throughput**：在 p95 延迟不超过基线 1.05× 约束下可达到的最大请求吞吐量。
- **Active-KV Capacity Frontier**：在固定 KV block 预算下，系统能稳定支持的最大并发请求数边界。
- **Evacuation Headroom**：为物理回收预留的临时 destination block 空间，从同一 KV block budget 中划出而非额外分配。

## 可复现要素
- **数据集**：ShareGPT、LongBench（公开）
- **代码开源**：论文未明确声明开源链接，但提到基于 vLLM v0.18.0 实现，源码应随论文发布
- **模型权重**：Mistral-7B、Llama-3.1-8B、Qwen2.5-14B（需自行获取）
- **关键超参**：
  - Block size：默认 16（vLLM 最小支持值），敏感性实验覆盖 16/32/48
  - 碎片阈值 $\theta_F$：默认 0.25
  - gpu_mem_util：0.90（吞吐实验）、0.35/0.50（capacity frontier 实验）
  - Decoding temperature：0（确定性解码）
  - Eviction ratio：0.3-0.7 扫参
- **硬件**：单卡 NVIDIA H100 80GB
