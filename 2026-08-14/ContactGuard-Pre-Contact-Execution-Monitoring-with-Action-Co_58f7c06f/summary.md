---
title: "ContactGuard-Pre-Contact-Execution-Monitoring-with-Action-Co"
source: https://arxiv.org/pdf/2608.13438v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 16:22:37"
field: "机器人操作与失败预测"
keywords: ["robotic manipulation", "failure prediction", "latent world model", "JEPA", "pre-contact monitoring", "visuomotor policy"]
innovations: ["Action-conditioned latent world model for pre-contact failure prediction", "Policy-decoupled execution monitor that rolls out future latent consequences", "Demonstrates predicted future latents provide failure info beyond current observations"]
benchmarks: ["Cup pick-and-place", "Box pick-and-place", "Pencil grasping", "Towel folding"]
---

# 论文速读：ContactGuard: Pre-Contact Execution Monitoring with Action-Conditioned Latent World Models

## 一句话总结
ContactGuard 提出了一种**接触前执行监控器**，利用 action-conditioned latent world model 预测即将执行的 action chunk 的短期后果，在机械臂实际接触物体之前预测失败风险并中止执行，无需修改底层策略。

## 研究问题与动机
- **接触密集任务失败发现太晚**：现有方法通常在机器人已接触物体后才检测到失败（如推偏、滑脱、未抓稳），此时已无法补救。
- **wrist-camera 视角的矛盾**：近距离摄像头能清晰观察接触过程，但等到看到问题发生，机器人可能已经错误推压或扰乱物体。
- **直接预测当前状态或动作不够**：仅凭当前观察或计划动作不足以区分"即将成功的接触"和"即将失败的接触"，需要模拟行动的后果。
- **策略耦合部署困难**：已有失败检测方法通常需要与策略联合训练或修改策略内部结构，难以作为即插即用模块。

## 核心贡献（创新点）
- **预接触执行监控范式**：将失败检测时机从"接触后"提前到"接触前"，利用 chunked visuomotor policy 中蕴含的接触事件（如夹爪闭合）作为触发信号。
- **Action-conditioned latent world model**：采用 JEPA 架构从**无标签**轨迹中学习多视图视觉嵌入的短期动态演化，而非像素级视频生成，计算高效且表征决策友好。
- **Policy-decoupled 部署接口**：ContactGuard 将底层策略视为黑盒提议者，仅接收其输出 action chunk 进行独立验证，无需修改或重新训练策略。
- **预测未来 latent 提供额外失败信息**：通过 ablation 证明 predicted future latent 比当前 latent 或 corrupted action 提供更强失败区分度。

## 方法详解
- **整体架构**：ContactGuard 由两部分组成——(1) 从**无标签轨迹**训练的 latent world model（冻结）；(2) 从**少量有标签预接触片段**训练的轻量 logistic regression failure probe。
- **多视图编码器**：V 个摄像头各自通过共享 ViT-Tiny 编码，per-view embedding 经 mean-pooling 和线性投影融合为 $z_t \in \mathbb{R}^D$（D=192）。
- **Action-conditioned 预测器**：采用 4 层 AdaLN-zero conditioned causal Transformer blocks，通过 teacher-forced 方式训练 next-latent 预测：$\hat{z}_{i+C} = P_\theta(z_{i:i+C-1}, u_i)$，损失函数为 MSE + SIGReg 正则化（$\lambda=0.09$）。
- **Failure probe**：冻结 encoder 和 predictor 后，在 $t = T_g - k_{pre}$ 处锚定（$k_{pre}=15$ 帧，即接触前 0.5s），向前 rollout K=30 步得到 $\hat{z}_{t+K}$，用 $\ell_2$ 正则化 class-balanced logistic regression 计算 $P(\text{fail}|\hat{z}_{t+K})$。
- **在线监控流程**：① Trigger：扫描 chunk 寻找 gripper open-to-close 过渡；② Anchor：距闭合事件 $\leq k_{pre}$ 帧时激活；③ Rollout & Score：encode 历史观测 → rollout K 步 → probe 打分；④ Abort：若 $P(\text{fail}) > \tau$ 则中止，否则继续。

## 实验与结果
- **实验设置**：AgileX Piper 双臂机器人（14-DoF）+ 3 个同步 RGB 摄像头；4 个任务：cup pick-and-place、box pick-and-place、pencil grasping（pencil-and-notebook）、towel grasping（towel-fold）。
- **基线对比**：
  - **Table 1（实时 rollouts）**：Multi-view ContactGuard 在 Box（AUC=0.946）和 Cup（AUC=0.992）上显著优于单视图 LeWM 和 Direct-linear。
  - **Table 2（AUC 对比）**：ContactGuard 在所有 4 个任务上 AUC 最高（Cup: 0.982, Box: 0.984, Pencil: 0.992, Towel: 0.978），超越 SAFE（次优）、FAIL-Detect、RND 及 Current latent ablation。
  - **Table 3（信息溯源）**：Action shuffle 使 AUC 坍缩至接近随机（~0.5），证明监控器响应的是具体行动而非静态视觉风险。
- **关键数字**：ContactGuard 在 Pencil 任务 AUC 达 0.992，为所有任务最高；Towel 任务 FAR 最低（0.240）。推理耗时：K=30 时 Full 耗时 19.18ms（RTX 5090），远小于预接触窗口。
- **Proprioceptive ablation**：添加 28-d 机器人状态反而降低 AUC（Table 5），说明小数据下纯视觉动态信息更可靠。

## 相关工作脉络
- **Latent world models**：Dreamer/TD-MPC 用于规划与策略学习；LeWorldModel 采用相同 next-embedding 预测框架；本文将其迁移至**执行监控**而非规划。
- **Failure prediction**：FAIL-Detect/RND（成功数据 novelty 信号）、SAFE（监督失败检测）、SIRIUS/Sirius-Fleet（联合训练策略与动态模型）；本文区别在于**策略解耦**、无需联合训练。
- **Chunked visuomotor policies**：ACT、Diffusion Policy 产生含交互事件的 action chunk；本文利用 chunk 自然包含的接触事件作为监控时机。
- **Test-time verification**：RoVer 等搜索候选动作评分；本文**不搜索**，仅验证策略输出的单一 chunk。

## 局限性与未来方向
- **中止后无恢复**：ContactGuard 仅能中止失败，不提供任务恢复或重试策略。
- **仅适用于 imminent contact**：当前设计针对短 horizon 接触事件（如夹爪闭合），扩展到长 horizon 技能需分层或重复监控。
- **Trigger 依赖任务特定信号**：本文用 gripper open-to-close 作为 trigger，通用接触事件需额外设计。
- **小数据集下 proprioceptive fusion 有害**：当前架构下机器人状态融合存在 domain shortcut 问题。

## 研究启发与可借鉴点
- **JEPA-style 表征用于执行监控**：将 latent world model 从"规划工具"拓展为"执行验证器"的思路可迁移至其他需要安全保证的机器人场景。
- **Prediction before commitment 范式**：在关键动作执行前预测后果并决策，适用于手术机器人、无人机避障等高风险领域。
- **Policy-decoupled 即插即用**：不修改底层策略的监控设计便于与不同 policy 架构（ACT、Diffusion Policy、VLA）组合使用。
- **Action-swap counterfactual 实验**：固定观测替换 action 以验证监控器响应行动而非静态状态的实验设计值得借鉴。

## 关键术语表
- **Chunked visuomotor policy**：输出短时 action sequence（chunk）的端到端视觉-运动策略，如 ACT、Diffusion Policy。
- **JEPA (Joint-Embedding Predictive Architecture)**：不重建像素，而是预测目标信号嵌入表示的自监督架构，避免视频生成的模糊性。
- **SIGReg**：防止 latent 坍塌的正则化项，通过随机投影鼓励 latent 分布匹配标准高斯。
- **AdaLN-zero**：自适应层归一化，通过零初始化调制将 action 注入 transformer 层。
- **Pre-contact monitor**：在机械臂实际接触物体之前预测失败并中止的监控系统。
- **Failure probe**：基于少量有标签数据训练的轻量分类器，用于从 latent 映射到失败概率。
- **Action-conditioned latent world model**：以 action 为条件预测未来 latent embedding 的动态模型。

## 可复现要素
- **数据集**：作者使用自有真实机器人数据（AgileX Piper），论文未公开数据集。
- **代码**：论文未明确提及代码开源，需查看作者主页或仓库。
- **权重**：未公开。
- **关键超参**：ViT-Tiny patch=16，D=192，Transformer 4层 8 heads dim=64 FFN=1024，C=3，L=5，K=30，k_pre=15，λ=0.09，ρ=1，lr=5e-5，batch=64，训练 100 epochs AdamW。
