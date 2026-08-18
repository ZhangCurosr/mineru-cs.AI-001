---
title: "Ab-sub-s-sub-t-sub-rac-sub-t"
source: https://arxiv.org/pdf/2608.13263v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 14:46:40"
field: "LLM 推理系统内存管理"
keywords: ["KV Cache", "PagedAttention", "Token-level Eviction", "Memory Virtualization", "LLM Serving", "Intra-block Fragmentation", "vLLM"]
innovations: ["提出 token-level 虚拟化层解耦逻辑存活与物理块放置", "阶段感知异步重打包回收块内碎片并避免 HBM 争用", "以 <50 行策略适配器使驱逐算法零 block 知识接入块级运行时"]
benchmarks: ["ShareGPT", "LongBench", "Mistral-7B", "Llama-3.1-8B", "Qwen2.5-14B"]
---

# 论文速读：vToken: Token-level Virtualization for Reclaimable KV Caches

## 一句话总结
vToken 提出了一种轻量级的 token 级 KV 缓存虚拟化层，通过 token-table 间接寻址解耦逻辑 token 存活状态与物理块放置，利用异步重打包实现块内碎片回收，使现有的 token 级 KV 驱逐算法能在块管理运行时中真正释放物理内存。

## 研究问题与动机
- **块级管理与 token 级驱逐的粒度不匹配**：PagedAttention 以固定大小块管理 KV 缓存，而 H2O、StreamingLLM 等驱逐算法在 token 粒度决策，导致"部分存活"块无法释放，内部碎片率高达 40%–60%。
- **未被回收的 KV 容量限制并发**：token 级驱逐虽减少逻辑 KV 需求，但缺乏运行时抽象将其转化为可复用物理容量，GPU 内存仍被大量低利用率块占用。
- **算法集成成本高**：将新驱逐策略接入块级运行时需深入修改 block manager，通常需 500+ 行代码；等待整块为空会牺牲算法效果，手动迁移 token 则紧耦合策略与运行时。
- **减小块尺寸并非根本方案**：将块从 16 减至 8 或 4 虽可降低碎片，但会放大元数据开销、寻址复杂度和细粒度非连续 KV 传输，损害带宽效率。

## 核心贡献（创新点）
1. **识别并量化 token/block 粒度不匹配的运行时缺失层**：形式化定义块内碎片率 $F$，并通过实验证明在 16K 上下文下多数分配块利用率 ≤50%，内部碎片超过 40%。
2. **提出 vToken 令牌级虚拟化层**：在驱逐策略与 PagedAttention 物理层之间插入薄抽象边界，策略仅做 token 存活决策，运行时负责 slot 重映射与异步物理回收，保持块管理子strate 不变。
3. **实现 token-table 间接寻址与稳定逻辑视图**：每请求维护 per-sequence token table，记录 (block_id, offset) 映射与 liveness bit，暴露 evict_token / sync_new_tokens / apply_moves 三个 block-agnostic 接口。
4. **设计阶段感知的异步回收后端（lazy compaction）**：四阶段流水线（回收资格判定→headroom-aware 准入→请求局部重定位规划→异步拷贝），在解码三个阶段间错峰重叠，避免 HBM 争用与全局同步屏障。
5. **在 vLLM 中落地验证**：对 H2O/Scissorhands/Random 三种策略，vToken 将保留 KV 块减少 27.2%–72.3%，SLA 约束吞吐量提升最高 1.37×，受限 active-KV 预算下可行并发最高扩展 2×，单策略集成从 500+ 行降至 <50 行。

## 方法详解
- **Token Table 与逻辑地址空间**：per-request 维护 token table，每项含 (block_id, offset) 与 liveness bit；CPU 存储主表，GPU 维护查表缓存加速稳态路径；新增 token 时 `sync_new_tokens` 分配逻辑 ID，驱逐时 `evict_token` 仅更新元数据。
- **Slot 重映射**：注意力 slot 由直接计算 `slot = block_table_id × S + offset` 改为查表 `slot = token_table[token_id].block_id × S + token_table[token_id].offset`，仅在 token table 结构变化时刷新。
- **回收四阶段**：
  1. **资格判定**：基于每块存活 token 数 $n_b$ 与块容量 $S$ 计算利用率 $u_b = n_b/S$，聚合得到整体碎片率 $F = 1 - \frac{1}{N}\sum u_i$；仅在 $F > \theta_F$（默认 0.25）且空闲块低于低水位时触发。
  2. **Headroom-aware 准入**：从有界 evacuation headroom（同一 KV 预算内预留临时工作空间）与空闲池分配目的块；计划仅在所有源块释放量严格大于目的块消耗量时提交。
  3. **重定位规划**：每请求局部选择低利用率源块，按逻辑顺序打包 live tokens 到目的块，减少跨块跳跃以提升缓存局部性。
  4. **阶段感知异步拷贝**：在 forward 返回后、sampling/scheduling 阶段期间于独立 relocation stream 发起 KV 拷贝；下一轮 attention 前插入 CUDA event 依赖，避免 HBM 带宽竞争与 host 同步阻塞。
- **正确性不变量**：I1 token 守恒（逻辑 ID 与 KV 内容不变）、I2 唯一映射（原子清源写目）、I3 注意力前可见（event fence 保证）、I4 布局感知规划（snapshot 一致、盈利性检查）。
- **CUDA Graph 兼容**：KV 缓存张量与图捕获输入 buffer 稳定，仅 mutable slot-mapping buffer 在 replay 前按 token table 更新；relocation 在捕获图外独立 stream 执行。

## 实验与结果
- **数据集/模型**：ShareGPT、LongBench；Mistral-7B、Llama-3.1-8B、Qwen2.5-14B；H2O / Scissorhands / Random 三种驱逐策略。
- **基线**：Native vLLM（全保留无驱逐）、Naive-Evict（相同 token 决策但关闭 vToken 回收）。
- **主要结果**：
  - 内存利用率：vs Naive-Evict，Llama-3.1-8B 提升 21.88%，Mistral-7B 提升 21.67%。
  - 保留块数：减少 27.2%–72.3%。
  - SLA 约束吞吐（Mistral-7B）：提升 9.9%–37.3%，p95 延迟降低 9.9%–27.5%；Llama-3.1-8B 平均吞吐提升 18.9%、p95 降 14.7%。
  - Scissorhands 增益最强：吞吐提升 33.3%–103.7%，p95 降 21.8%–33.0%。
  - 容量前沿：gpu_mem_util=0.35 时 Llama-3.1-8B 最大可行并发从 5 扩至 8（+60%）；gpu_mem_util=0.50 时从 11 扩至 22（×2）；Qwen2.5-14B 从 3 扩至 6（×2）。
  - 开销：indirection-only 消融相对 Native vLLM 吞吐/p95 变化 <1%；async 路径无显式同步等待。
  - 生成稳定性：ROUGE-L F1 均值差 -0.0016，93.1% 配对差异 ≤0.01；输出长度中位比 0.99，无通胀。
  - Prefix cache 兼容：共享前缀块跳过回收，私有后缀回收块减少 28.6%–42.2%。
  - 集成成本：从 500+ LOC 降至 <50 LOC。

## 相关工作脉络
- **PagedAttention/vLLM [17]**：块级 KV 管理基石；vToken 在其上方插入 token 虚拟化层，不改动内核与分配器。
- **H2O [36] / StreamingLLM [33] / Scissorhands [22]**：token 级驱逐策略；vToken 使其决策能真正转化为物理块回收，解决"策略有效但 runtime 无法兑现"的缺口。
- **vAttention [28]**：基于 CUDA VMM 的页级地址空间虚拟化；可作 vToken 的下层后端，两者 stackable。
- **DiffKV [35]**：混合精度 K/V 与 per-head 布局压缩；正交于 vToken，后者处理 uniform fp16 上的碎片回收。
- **PagedEviction [9]**：将对齐负担推给策略，要求 eviction page-aligned；vToken 保持策略页无关、在 runtime 侧消解对齐。
- **Zipage [18]**：面向 reasoning 的有界缓存 pipeline，固定 per-request 块预算并重定位；vToken 提供通用策略中立边界。

## 局限性与未来方向
- **当前保守排除共享前缀块**：保留 prefix cache 正确性而跳过其回收，潜在 capacity 未充分挖掘；未来可用 copy-on-write 按需回收。
- **单 KV cache group 实现**：多 group 场景下逻辑存活状态共享、组内 placement 数组独立，尚待工程验证。
- **规划端 CPU 开销主导**：机会检查占主要 overhead，可通过 batching/indexing 优化。
- **异步拷贝与 decode 存在 HBM 争用**：高压力下 overlap 仍有代价；需更细粒度资源调度。
- **TP / 跨设备扩展未实现**：TP shard 本地独立 block pool，跨 shard 协调待研究；跨设备显式 transfer cost 正交但未整合。
- **块尺寸下限为 16**：受 vLLM PagedAttention 限制，更小块探索受限；非 vToken 本身瓶颈。

## 研究启发与可借鉴点
- **"运行时边界"设计范式**：将策略语义（决定 what）与物理实现（决定 how）彻底分离，可为其他 token/块粒度 mismatch 问题（如 quantization、offload）提供复用模板。
- **阶段感知异步重叠技术**：利用解码三阶段（attention 读 HBM → FFN 计算 → sampling/scheduling 低 GPU 利用）错峰 launch 拷贝，避免显式 fence；可迁移至其他 KV 移动场景（prefix 拷贝、多 GPU 迁移）。
- **Headroom-aware 准入控制**：以有界工作空间而非额外内存池支撑 out-of-place 重定位，保持总预算不变的前提下实现自愈碎片；适用于其他"回收需临时空间"的系统。
- **<50 行策略适配器**：三层接口（sync_new_tokens / evict_token / apply_moves）使新驱逐策略零 block 知识接入，极大降低生态集成成本，可作为社区插件标准。
- **正确性不变量驱动验证**：I1–I4 四个结构化不变量分别对应守恒、唯一、可见、盈利性，为后续系统提供可直接复用的测试契约。

## 关键术语表
- **PagedAttention**：vLLM 引入的块级 KV 缓存管理，类比 OS 虚拟内存分页，将 KV 连续逻辑块映射到非连续物理块以降低外部碎片。
- **Intra-block fragmentation**：块内碎片，指块内部分 token 被驱逐后留下的"空洞"，因块粒度无法部分释放而导致内存不可复用。
- **Token table**：vToken 核心元数据结构，per-request 维护逻辑 token ID 到物理 (block_id, offset) 的映射及存活位。
- **Lazy compaction**：惰性紧凑化，vToken 物理回收机制，仅在碎片率超过阈值且空闲块不足时触发异步重打包。
- **Eviction policy**：KV 驱逐策略，如 H2O/StreamingLLM/Scissorhands，基于注意力分数、sink token 或持久性模式决定保留哪些 token。
- **Headroom-aware admission**：vToken 回收准入控制，仅在保留有界 evacuation headroom 时允许新请求占用空闲块，保证重定位有目的地。
- **Stage-aware asynchronous copy**：阶段感知异步拷贝，将 KV 迁移置于 forward 之后、与 sampling/scheduling 重叠，避免与 attention 争用 HBM。
- **CUDA Graph 兼容**：vToken 保持捕获图稳定，仅动态更新 slot-mapping buffer 并在 replay 前插入 event 依赖，无需重新捕获。

## 可复现要素
- **数据集**：ShareGPT、LongBench；论文未声明公开链接但为常用开源数据集。
- **代码/权重**：原型基于 vLLM v0.18.0 + PyTorch v2.10.0；论文未提供独立 GitHub 仓库声明，vLLM 本身开源。
- **关键超参**：块大小默认 ≥16（受 vLLM 限制）；碎片阈值 θ_F = 0.25；gpu_mem_util 设为 0.90（容量前沿实验用 0.35/0.50）；decoding temperature = 0。
- **硬件**：单卡 NVIDIA H100 80GB。
