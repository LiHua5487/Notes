---
date: 2025-05-19
tags:
  - AI
---
# 基本概念

相比于传统的搜索，强化学习有以下优势
- 可以应对更大的搜索图
- 可以应对变化的甚至预先未知的环境

强化理论：通过奖励和惩罚的方式改变智能体的行为方式

![[AI引论/AIimg/img7/image.png|431x212]]

强化学习可以解决一类特殊的决策问题：马尔可夫决策问题 MDP
- 未来的状态由当前状态和行动确定，与过去无关
- 决策目标和代价由奖励函数给出
- 不假定环境有确定和已知的转移模型

强化学习有以下要素
- 状态 $s \in S$：智能体和环境的状态
- 动作 $a \in A$：智能体可以采取的行动/决策
- 转移模型 $p(s'|s,a)$ 或 $s' = f(s,a)$：当前状态随着动作会怎么变化，可以建模为概率分布，比如选择向上移动，有一定概率到达左侧/右侧，而非上面
- 奖励函数 $r(s,a,s')$：状态是否与目标相符，动作需要多大代价

转移方程和奖励函数都可以建模成概率分布

最后想学习出一个策略
- 策略 $\pi$：根据不同的当前状态，应分别选择什么行动以获得最大回报
	- 确定性策略 $a = \pi(s)$
	- 随机策略（关于状态的概率分布）  $a \sim \pi(a|s)$

![[AI引论/AIimg/img7/image-1.png]]

轨迹（亦称episode/rollout） ：状态、行动、奖励的序列
$$\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_t, r_t, ...)$$
- 不断利用某一策略行动的结果

在一条轨迹上，有一个累积收益
$$G(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots = \sum_{i=0}^{T} \gamma^i r_i$$
折扣因子 $\gamma \ (0 \leq \gamma \leq 1)$ ：描述未来收益的重要程度
-  若 $\gamma = 1$，则近的收益和远的收益一样重要
-  若 $\gamma = 0$，则只看下一步的收益（最贪心）
- 一般要求 $\gamma < 1$ 以便最后收敛

对于一个策略是好是坏，可以用价值函数 $V_{\pi}$ 判断
状态价值 $V_{\pi}(s)$ ：从 $s$ 出发执行策略 $\pi$ 能获得的累积收益
$$V_{\pi}(s) = \mathbb{E}_{\tau \sim \pi}[G(\tau) | s_0 = s] = \mathbb{E}_{\tau \sim \pi}\left[\sum_t \gamma^t r_t \mid s_0 = s\right]$$
- $\mathbb{E}$ 表示期望
- 从同一个状态出发，状态价值越大，策略越好
- 使得 $V$ 最大的策略就是最优策略，记作 $\pi*$，根据最优策略得到的状态价值就是最优状态价值，记作 $V*$ 

在 MDP 问题中，根据不同状态得到的最优策略是相同的，即存在一个全局的最优策略

动作价值 $Q_{\pi}(s,a)$ ：从 $s$ 出发并做动作 $a$，之后再执行策略 $\pi$ 能获得的累积收益
$$Q_{\pi}(s,a) = \mathbb{E}_{\tau \sim \pi}[G(\tau) | s_0 = s, a_0 = a] = \mathbb{E}_{\tau \sim \pi}\left[\sum_t \gamma^t r_t \mid s_0 = s, a_0 = a\right]$$
- 有些时候计算动作价值更方便，因为可以直接取 argmax 得到最优动作

# Bellman 方程

对于当前状态，可以先根据策略走一步达到下一个状态
当前状态的状态价值就是走一步得到的奖励加上之后状态的状态价值
$$
\begin{align}
V_{\pi}(s) &= \mathbb{E}_{\tau \sim \pi}[G(\tau)|s_0 = s] \\ 
&= \mathbb{E}_{\tau \sim \pi}[r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots |s_0 = s] \\ 
&= \mathbb{E}_{(a_0, r_0, s_1, a_1, r_1, s_2, \dots) \sim \pi}[r_0 + \gamma(r_1 + \gamma r_2 + \cdots)] \\ 
&= \mathbb{E}_{(a_0,r_0,s_1) \sim \pi}[r_0 + \gamma \mathbb{E}_{(a_1,r_1,s_2, \dots) \sim \pi}[r_1 + \gamma r_2 + \cdots]] \\ 
&= \mathbb{E}_{(a_0,r_0,s_1) \sim \pi}[r_0 + \gamma V_{\pi}(s_1)] \\ 
\end{align}
$$
而先走一步有不同的走法，通过状态转移方程，达到不同的下一个状态，所以可以变成这样
$$
V_{\pi}(s) = \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a) [r + \gamma V_{\pi}(s')].
$$
这就得到了 **Bellman 期望方程**
对于动作价值，第一步 $a$ 是固定下来的
$$
\begin{align}
V_{\pi}(s) &= \sum_{a} \pi(a|s) \sum_{s',r} p(s',r|s,a) [r + \gamma V_{\pi}(s')] \\
Q_{\pi}(s,a) &= \sum_{s',r} p(s',r|s,a)[r + \gamma V_{\pi}(s')] \\
&\forall s \in S, a \in A(s), s' \in S^+ \\[1em]
\end{align}
$$

要得到最优价值，就是对于不同的策略，选择对应价值最大的
$$
\begin{align}
V^*(s) &= \max_{\pi} V_{\pi}(s) \\
Q^*(s,a) &= \max_{\pi} Q_{\pi}(s,a)
\end{align}
$$
最优的策略，就是走动作价值最大的一个动作，那最优价值就是这样
$$
V^*(s) = \max_{a} Q^*(s,a) 
$$
可以得到 **Bellman 最优方程**
$$
\begin{align}
V^*(s) &= \max_{a} \mathbb{E}[r_t + \gamma V^*(S_{t+1}) | S_t = s, A_t = a] \\
&= \max_{a} \sum_{s',r} p(s',r|s,a)[r + \gamma V^*(s')] \\[1em]
Q^*(s,a) &= \mathbb{E}[r_t + \gamma \max_{a'} Q^*(S_{t+1},a') | S_t = s, A_t = a] \\
&= \sum_{s',r} p(s',r|s,a)[r + \gamma \max_{a'} Q^*(s',a')] \\
&\forall s \in S, a \in A(s), s' \in S^+
\end{align}
$$

![[AI引论/AIimg/img7/image-2.png]]

![[AI引论/AIimg/img7/image-3.png]]

## 策略估值

对于一个策略 $\pi$，可以利用 Bellman 方程得到状态价值函数 $V_\pi$

Bellman期望方程：
$$
V_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V_\pi(s')]
$$
先初始化价值函数，给所有状态的 $V(s)$ 初始化为任意值（通常为 0）
利用这个式子进行迭代更新：
$$
V_{k+1}(s) = \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V_k(s')]
$$
- $k$ 代表迭代步数

$\max_{s \in S} |V_{k+1}(s) - V_k(s)|$ 足够小时停止迭代，代表 $V_\pi(s)$ 几乎没变化
可以近似认为 $V'_\pi(s) = V_\pi(s)$ ，满足
$$
\begin{align}
V'_\pi(s) &= \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V_\pi(s')] \\
&= \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)[r + \gamma V'_\pi(s')]
\end{align}
$$
这正好符合 Bellman 期望方程，可以认为得到的就是最优状态价值函数

## 策略提升

还可以利用 Bellman 方法改进策略

根据状态估值计算动作估值：
$$
Q_\pi(s, a) = \sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')]
$$
贪心选择动作产生新策略
$$
\begin{align}
\pi'(s) &= \arg\max_a Q_\pi(s, a) \\
&= \arg\max_a \sum_{s',r} p(s',r|s,a)[r + \gamma v_\pi(s')]
\end{align}
$$
可以证明，新的策略一定不会比原策略差，即状态价值不低于原策略的

![[AI引论/AIimg/img7/image-4.png]]

## 策略迭代

$$
\pi_0 \xrightarrow{\text{E}} v_{\pi_0} \xrightarrow{\text{I}} \pi_1 \xrightarrow{\text{E}} v_{\pi_1} \xrightarrow{\text{I}} \pi_2 \xrightarrow{\text{E}} \cdots \xrightarrow{\text{I}} \pi_* \xrightarrow{\text{E}} v_*
$$
-  $\xrightarrow{\text{E}}$ 表示策略估值，$\xrightarrow{\text{I}}$ 表示策略提升

有限MDP问题中：$S, A, R$ 的取值都有有限个
- 只有有限种策略，最后一定能收敛得到最优策略与最优值函数
- 局部最优解就是全局最优解

问题在于收敛很慢
可以利用 Bellman 最优方程迭代，即**值迭代**
在值函数收敛之后，根据最优价值函数 $V^*(s)$ 得到对应的最优策略 
$$
\pi^*(s) = \arg\max_a \sum_{s'} P(s' \mid s, a) \Big[R(s, a, s') + \gamma V^*(s')\Big]
$$

# Q-Learning

值迭代是一种模型依赖型算法，它需要完全知道环境的状态转移方程
Q-Learning 无需知道环境的状态转移方程，仅通过与环境交互学习

而且，值迭代需要遍历所有的状态，收敛速度慢，Q-Learning 对状态进行采样

在强化学习中，策略分为两种
- 行为策略 $\mu$
	- 利用行为策略与环境交互，生成轨迹
	- 行为策略通常是带有探索性的，以确保算法能充分探索状态-动作空间

- 目标策略 $\pi$
	- 智能体最终希望学习和优化的策略
	- 用来指导如何更新 $Q(s,a)$ ，使其向最优价值函数逼近
	- 目标策略通常是贪心策略，选择使 $Q(s,a)$ 最大的动作

Q-Learning 中，行为策略是 $\epsilon$-greedy 策略
$$
a_t =
\begin{cases} 
\text{随机选择一个动作 } & \text{概率 } \epsilon \\ 
\arg\max_a Q(s_t, a) \,  & \text{概率 } 1 - \epsilon 
\end{cases}
$$
而目标策略是完全贪心策略
$$
\pi(s_{t+1}) = \arg\max_{a'} Q(s_{t+1}, a')
$$

核心思想是利用 Bellman 最优方程的最优动作价值函数更新
$$
Q^*(s,a) = \mathbb{E}[r_t + \gamma \max_{a'} Q^*(S_{t+1},a') | S_t = s, A_t = a]
$$
希望通过更新 Q 

在实际应用中，期望被基于采样的经验替代，更新公式为
$$
Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t) \right]
$$
- 学习率 $\alpha \in (0, 1]$：用于控制新经验对更新的影响
- $r_t + \gamma \max_{a'} Q(s_{t+1}, a')$：TD 目标值，根据与当前值的差距逐渐向其逼近
- 更新公式称为 Q 函数，$Q(s,a)$ 称为 Q 值

通过反复更新 $Q(s, a)$，Q-Learning 最终会逼近最佳动作价值函数 $Q^*(s, a)$
而根据最佳动作价值函数，就能得到最优的目标策略 $\pi^*$

比如以下情形，对于一个迷宫，建立 Q-Table 
每个格子分为4块，有4个Q值，分别对应从这个格子向上下左右移动

![[AI引论/AIimg/img7/image-5.png|502x265]]

首先，从起点开始，利用行为策略，根据当前状态逐步进行移动
每走一步，更新当前格子（移动之前的）的 Q 值

比如当前处于 $s_{33}$ ，此处选择向右移动，到了右边格子 $s_{34}$
根据下一个格子 $s_{34}$ 的信息更新当前格子 $s_{33}$ 的 Q 值

![[AI引论/AIimg/img7/image-6.png|493x308]]

重复进行上述过程，如果碰到死路，受到惩罚（如当前格子 Q 值减少）
如果到达终点，当前格子 Q 值增加，而后重新从起点开始
还可以设置最大步数限制，如果一直没到终点，就重新从起点开始

# DQN

在之前的方法中，$Q(s,a)$ 的获取方式相当于查表，但是如果状态 s 有很多，内存开销太大了

可以用用神经网络来近似 Q 表（Q 值函数），从而解决了高维状态空间下无法用传统 Q 表显式存储的问题

这种将神经网络与 Q-Learning 结合的网络称为 DQN

目的是要学习一个含有参数 $\theta$ 的函数 $Q_\theta(s,a)$

目标值，为了让 Q 值收敛到满足贝尔曼方程的值
$$y_t = r_t + \gamma \max_{a'} Q_\theta(s_{t+1}, a')$$
 - $r_t$ ：当前动作的即时奖励
 - $\gamma \cdot \max_{a'} Q_\theta(s_{t+1}, a')$ ：考虑未来累积奖励，折扣因子 $\gamma \in [0, 1]$ 控制权重

更新方程 
$$
Q_\theta(s_t, a_t) \leftarrow Q_\theta(s_t, a_t) + \alpha \left(y_t - Q_\theta(s_t, a_t)\right)
$$
- 通过优化神经网络，使得 $y_t$ 和 $Q_\theta(s_t, a_t)$ 的差距最小

优化目标，把上述更新公式转为一个均方误差 (MSE) 损失函数
$$
\mathcal{L}(\theta) = \mathbb{E}_{(s_t, a_t) \sim D} \left[ \left( Q_\theta(s_t, a_t) - \left( r_t + \gamma \max_{a'} Q_\text{target}(s_{t+1}, a') \right) \right)^2 \right]
$$
- D 是经验回放池 replay buffer，存储历史的 $(s_t, a_t, r_t, s_{t+1})$ ，训练时随机采样小批量数据来更新网络
- $Q_{target}$ 是目标网络，每隔一定步数根据 $Q_\theta$ 更新一次，避免训练同时修改目标和预测值，导致训练震荡

由于智能体的行为和状态是连续的，下一步的动作和上一步关系密切
如果在一次尝试中，直接用前面走过的轨迹训练，模型可能会受到之前特定动作序列的影响，无法很好地泛化
而且，这也会导致走法趋于单一化，可能达不到全局最优的走法
所以，需要引入经验放回池，把之前所有的尝试记下来

 通过最小化损失函数 $\mathcal{L}(\theta)$，就可以更新网络的参数 $\theta$
$$
\theta^\star \leftarrow \arg\min_\theta \frac{1}{2} \sum_{(s_t, a_t) \in D} \left[ Q_\theta(s_t, a_t) - \left(r + \gamma \max_{a'} Q_{target}(s_{t+1}, a')\right) \right]^2
$$





