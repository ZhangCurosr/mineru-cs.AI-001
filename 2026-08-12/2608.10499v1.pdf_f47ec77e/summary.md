---
title: "Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation"
source: https://arxiv.org/pdf/2608.10499v1.pdf
model: agnes-2.5-flash
chunks: 1
summarized_at: "2026-08-18 06:42:21"
field: "联邦强化学习"
keywords: ["federated reinforcement learning", "personalized federated learning", "intrinsic motivation", "exploration", "random network distillation", "sparse reward"]
innovations: ["将RND内在动机与联邦探索协调机制结合，实现隐私保护下的跨客户端协作探索", "设计低维探索统计摘要与全局新颖性先验，以采样偏置方式引导客户端聚焦未探索状态"]
benchmarks: ["MountainCar-v0", "CartPole-sparse"]
---

# 论文速读：Exploration-Driven Personalized Federated Reinforcement Learning via Intrinsic Motivation

## 一句话总结
本文提出了 EDPFRL-IM 框架，将内在好奇心驱动（RND）与联邦协调机制相结合，用于个性化联邦强化学习；通过在客户端本地使用 RND 内在奖励、服务器端聚合压缩探索统计并广播全局新颖性先验，实现隐私保护下的跨客户端协作探索，显著改善了稀疏奖励环境中的样本效率和策略个性化表现。

## 研究问题与动机
- **现有 PFRL 方法忽视探索**：当前 PFRL 框架（如 FedRL、FedAvg-RL、pFedMe）侧重于策略优化与聚合，但在稀疏/延迟奖励环境下依赖外部奖励信号，缺乏有效探索机制。
- **标准 RL 探索策略不适于联邦场景**：ε-greedy 或熵正则化等独立探索方法在客户端各自运行、隐私约束限制协调的联邦场景下效果显著下降。
- **内在动机方法在联邦场景下未被充分研究**：RND、ICM、count-based exploration 等在集中式 RL 中有效，但鲜有工作将其与联邦个性化学习结合。
- **冷启动与非 IID 条件下的个性化挑战**：客户端环境异质性高（重力、摩擦力、奖励塑形各异），冷启动客户端难以快速适应，需要协作探索机制。

## 核心贡献（创新点）
- **首次将内在动机引入个性化联邦强化学习**：提出 EDPFRL-IM 框架，在去中心化环境中实现客户端本地好奇心驱动的协作探索，与仅依赖外部奖励的 PFRL 基线形成本质区别。
- **隐私保真的探索协调协议**：客户端仅发送低维探索统计（访问计数/直方图/top-k 状态哈希），服务器聚合后广播全局新颖性先验，无需访问原始经验或本地梯度——区别于 FedRL+RND 等无协调机制的方法。
- **基于全局新颖性先验的经验采样偏置**：引入可调节的采样偏置 β，使客户端有偏重采样全局新颖状态，实现跨客户端探索方向对齐，相比纯本地 RND 方法（FedRL+RND）提升显著。

## 方法详解
- **问题设定**：N=10 个客户端，每个客户端 i 拥有独立的 MDP $\mathcal{M}_i = (S_i, \mathcal{A}, \mathcal{P}_i, R_i, \gamma)$，状态空间和转移/奖励函数异构（$\mathcal{A}$ 和 $\gamma$ 共享）。策略通过梯度上升更新：$\theta_i \leftarrow \theta_i + \eta \nabla_{\theta_i} J_i(\pi_{\theta_i})$。
- **局部 RND 内在奖励**：每个客户端维护固定的随机目标网络 $f_{\text{tgt}}$ 和可训练预测网络 $f_{\text{pred}}$，内在奖励为下一状态预测误差：$r_t^{\text{int}} = \| f_{\text{tgt}}(s_{t+1}) - f_{\text{pred}}(s_{t+1}) \|_2^2$。总奖励：$r_t^{\text{total}} = r_t^{\text{ext}} + \alpha_i \cdot r_t^{\text{int}}$，其中 $\alpha_i=0.1$ 为探索-利用权衡系数。
- **联邦探索协调机制**：
  - 客户端生成压缩探索统计 $\mathcal{E}_i$（如状态簇访问计数、频率直方图、top-k 新颖状态哈希），发送至服务器。
  - 服务器聚合为全局探索分布：$v_{\text{global}}[k] = \sum_{i=1}^{N} v_i[k]$。
  - 计算全局新颖性先验：$\mathcal{P}_{\text{novel}}(s) = \frac{1}{1 + v_{\text{global}}[\text{cluster}(s)]}$，广播至所有客户端。
- **偏置经验采样**：客户端以 $p_i(s) \propto (1 + \beta \cdot \mathcal{P}_{\text{novel}}(s)) \cdot p_i^{\text{uniform}}(s)$ 采样经验缓冲 $\mathcal{D}_i$，其中 $\beta=0.5$ 控制全局先验影响。该采样策略引导客户端聚焦全局未探索状态，同时维持本地个性化学习。

## 实验与结果
- **环境设置**：MountainCar-v0（稀疏奖励、难度高）和 CartPole-sparse（仅平衡足够长时间才给奖励）；N=10 客户端，各客户端环境参数（重力 0.0025–0.006、摩擦力、 pole 质量、奖励塑形）异构。
- **训练配置**：T=100 轮全局通信，每轮 E=10 次局部 PPO 更新；策略网络为两层全连接（64 隐藏单元 ReLU），RND 目标/预测网络各一层 128 隐藏单元；Adam 优化器，学习率 $3\times10^{-4}$，$\gamma=0.99$。
- **主要结果（最终平均回报）**：
  - MountainCar-v0：EDPFRL-IM 达 **0.80**（vs. FedRL+RND 0.48，FedRL 0.40，Local RL 更低）。
  - CartPole-sparse：EDPFRL-IM 达 **0.76**（vs. FedRL+RND 0.52，FedRL 0.35）。
  - 各客户端均优于 FedRL 基线（Table I）。
- **消融实验（Table II，最终10轮均值）**：FedRL（无RND）→ 0.40/0.35；FedRL+RND（无协调）→ 0.60/0.54；EDPFRL-IM（β=0，无全局先验）→ 0.63/0.58；EDPFRL-IM（完整）→ **0.80/0.76**。协调探索贡献显著。
- **探索覆盖率**：EDPFRL-IM 的跨客户端状态空间探索覆盖率显著高于 FedRL 和 FedRL+RND。
- **冷启动鲁棒性**：第5轮加入冷启动客户端时，EDPFRL-IM 借助全局探索先验快速适应，显著优于基线。
- **Table III 对比**：EDPFRL-IM 全面超越 RND [6]（0.31/0.32）、ICM [7]（0.34/0.36）、CBE [8]（0.39/0.49）、FedAvg-RL [4]（0.38/0.42）、pFedMe [5]（0.40/0.45）、FedPer++ [12]（0.41/0.44）。

## 相关工作脉络
- **FedRL [2]**：联邦强化学习通用框架，以全局策略聚合为核心，但未考虑内在探索，非个性化。
- **FedAvg-RL [4] / pFedMe [5] / FedPer++ [12]**：个性化联邦学习算法，侧重模型聚合/正则化策略，缺乏对稀疏奖励场景下探索效率的系统设计。
- **RND [6]**：集中式 RL 中通过随机网络蒸馏构造内在奖励，本文将其扩展至联邦场景并引入跨客户端协调。
- **ICM [7]**：基于逆动力学模型的内在动机，强调状态转移预测；本文采用更轻量的 RND 方案，更适合低通信开销场景。
- **Count-based Exploration (CBE) [8]**：基于后继表示的状态计数探索，本文对比发现 RND+全局先验的组合更有效。
- **定位差异**：本文首次将内在动机（RND）与联邦探索协调机制整合进 PFRL，填补了"稀疏奖励+隐私保护+跨客户端协作探索"三者的交叉空白。

## 局限性与未来方向
- 内在奖励权重 α 和采样偏置 β 需手动设定，不同环境和客户端数量下的超参敏感性未系统分析。
- 探索统计摘要（状态簇访问计数/直方图/top-k 哈希）在连续高维状态空间中的可扩展性有待验证。
- 隐私保护仅通过"压缩统计"实现，未提供严格的差分隐私保证或安全性证明。
- 实验限于经典控制任务（MountainCar、CartPole），尚未在更复杂连续控制或视觉输入环境（如 Atari、MuJoCo）中验证。
- 固定目标网络 $f_{\text{tgt}}$ 不更新，长期可能限制内在奖励的判别能力。
- 未来方向：探索自适应超参调节、集成差分隐私机制、扩展至高维视觉/连续动作空间、结合 ICM 等替代内在动机方案。

## 研究启发与可借鉴点
- **RND 内在奖励与联邦协调解耦设计**：本地 RND 负责个体探索多样性，全局先验负责跨客户端方向对齐——此"本地内在+全局协调"的两层架构可直接迁移至多智能体 RL 或分布式离线 RL 场景。
- **压缩探索统计的隐私-效用权衡**：状态簇访问计数/直方图等低维摘要的设计思路，可推广至其他联邦 RL 场景（如策略蒸馏、元学习联邦），为"少通信、保隐私、高协作"提供范式。
- **全局新颖性先验的采样偏置机制**：将聚合信息转化为经验采样的概率偏置（$p(s) \propto (1+\beta \cdot \mathcal{P}_{\text{novel}}(s))$），避免直接共享策略权重，这一采样侧干预策略可复用于批量强化学习（Batch RL）的去偏移探索。
- **冷启动客户端快速适应**：全局探索先验对新增客户端的热启作用，对动态加入节点的联邦系统（如移动端 IoT）具有实用价值，可结合联邦元学习进一步研究。

## 关键术语表
- **Personalized Federated Reinforcement Learning (PFRL)**：在联邦学习框架下，各客户端学习个性化策略的同时进行协作训练的强化学习范式。
- **Intrinsic Motivation（内在动机）**：智能体基于内部好奇心/预测误差产生的自驱动探索信号，用于缓解稀疏奖励困境。
- **Random Network Distillation (RND)**：通过固定随机目标网络与可训练预测网络的输出误差，构造状态新颖性度量以生成内在奖励。
- **Global Novelty Prior（全局新颖性先验）**：服务器聚合各客户端探索统计后生成的跨客户端状态访问分布倒数，用于指导各客户端聚焦未探索区域。
- **Non-IID Environments（非独立同分布环境）**：各客户端的马尔可夫决策过程在转移概率、奖励函数或状态空间上存在异质性的联邦设置。
- **Cold-start Client（冷启动客户端）**：在联邦训练中途加入的新客户端，缺乏历史交互经验，需快速适应全局学习进程。

## 可复现要素
- **数据集/环境**：MountainCar-v0、CartPole-sparse（OpenAI Gym 基准环境，开源）；论文未提供自定义环境代码。
- **代码开源情况**：论文未声明代码开源。
- **关键超参**：客户端数 N=10，全局轮数 T=100，本地更新轮数 E=10；RND 隐藏层大小 128；策略网络两层 64 隐藏 ReLU；$\alpha_i=0.1$，$\beta=0.5$，$\gamma=0.99$，学习率 $3\times10^{-4}$（Adam）。
