
关键词：**GAIL**； **DDPG**； **GA-DDPG**

---
# Gripper Manipulation

一个简单的流程：可以将抓取分为多个连续的过程（如旋转夹爪，靠近物体，闭合夹爪），每个子任务分别训练一个 RL 模型，从而简化奖励函数，也便于控制机器人

## GAIL

IL 的局限性在于模型泛化能力受限于专家数据，但是如果使用 RL ，一个问题是奖励函数太难设计了，而 **GAIL (Generative Adversarial Imitation Learning)** 的想法是利用对抗网络中的 Discriminator 去生成奖励信号

>IL 常见算法：BC(DAgger/Policy Distillation/Diffusion Policy)，IRL，GAIL
>IRL 逆强化学习：从专家的轨迹中推断出奖励
>由于 GAIL 的核心目标仍是模仿专家的 demonstration 而非最大化奖励，所以仍然属于 IL 

- Generator：目标学习专家策略，与环境交互进行行动，从 Discriminator 获取奖励信号，试图骗过 Discriminator 以最大化奖励
	- Actor：要训练的智能体，负责学习策略产生动作
	- Critic（可选）：GAIL 常采用 PPO 的算法，此时需要 critic 预测 V 值用于计算梯度
- Discriminator：另一个神经网络，目标是区分哪些是专家数据，哪些是智能体的行为数据

>可见 GAIL 是一个 online 的方法，需要高频和环境交互，一般来讲能找到一个好的 reward 函数，就不怎么考虑 GAIL 了

但这也继承了对抗网络的一个缺点：一开始可能生成器表现太烂了，判别器一眼就看出来了，导致奖励很低，生成器不知道要学啥了，陷入恶性循环
所以一般需要先利用 BC 的方式训出一个表现还行的网络作为起始，或者手动添加一些 reward 项防止其太低了

## DDPG

在 RL 中，既可以去预测 V 值，也可以去预测 Q 值

>一些常见方法的分类
>On-Policy ：Policy Gradient / TRPO / PPO
>Off-Policy ：Q-learning / DQN / DDPG / SAC

使用 Q 值的网络的优点有
- 直接指导动作选择与优化，策略只需选择 Q 值最大的动作
- 天然的适合利用 buffer 中的历史数据（因为 Q 是根据 s 和 a 得出的，并没有要求 s 下 a 的分布），适合 Off-Policy 方法（而基于 V 值的方法一般是 On-Policy 的）
- 无需估计 V 值或 A 值，避免 Policy Gradient 的高方差；又因为不需要考虑分布的偏移，所以不需要 Importance Sampling ，这又避免了高方差

但传统的 Q-learning 采用 Q 值表的方法，这不适用于高维空间，因为会有很多种状态

DQN 将这个表换为神经网络预测，解决了高维状态空间不适用的问题，其目标是最大化 Q 值，学的策略是选取 Q 值最大的动作，但是只能输出离散的动作，对于高维空间的连续动作，很难直接找到最大值对应的动作

**DDPG (Deep Deterministic Policy Gradient)** 结合了 DQN 与 actor-critic 的想法，解决了不适用于高维连续动作空间的问题

>这里 Deterministic 的意思是，由于 Q 值网络学出的策略是根据当前 state ，选取 Q 值最高的 action ，所以只会输出一个 action ，而不是像 V 值网络那样输出一个 action 的分布（Gaussian Policy）

- Critic ：学习一个动作价值函数 $Q(s, a)$，使其能够准确估计在状态 s 下执行动作 a 所能获得的长期期望回报，目标是逼近最优动作价值函数 $Q^*(s, a)$
- Actor ：学习一个确定性策略 $\mu(s)$ ，使得这个策略选择的动作能够最大化Critic 评估的 Q 值

在 V 值网络中，目标值的选取有这些形式：累计奖励， TD Target， GAE
而在 Q 值网络中，一般选取 bootstrap 作为目标，去最小化对应的 loss ，这称为 bootstrap loss 
$$
\mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ \frac{1}{2} \left(Q_{\phi}(s, a) - \left(r + \gamma Q_{\phi}(s', \pi_{\theta}(s'))\right)\right)^2 \right]
$$
并利用以下公式更新 Q 函数
$$
Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]
$$
其策略的目标是最大化 Q 值的期望
$$
\max_{\theta} \mathbb{E}_{s \sim \mathcal{D}} \left(Q_{\phi}(s, \pi_{\theta}(s))\right)
$$

**GA(Goal-Auxiliary)-DDPG** 在此基础上，对 buffer 中的数据进行了增强

在许多现实任务中，环境反馈的奖励信号非常稀疏，往往只有在达到目标时才会获得一个比较大的奖励，GA-DDPG 的想法是引入一些辅助目标，比如对于一个 demonstration 轨迹，从中截取一段作为一个新的训练数据，这一小段就有一个对应的辅助目标

>数据的 sample efficiency ：
>REINFORCE(PG/On-Policy) < PPO < DDPG/SAC < DQN < Offline-RL
>在仿真环境中，一般采用 PPO ，但是对于真实世界的场景，一般采用 SAC 或者更加 sample efficient 的方式

# Non-prehensile Manipulation

## Planar Pushing

目标是把一个物体从桌面上推到指定的 pose ，一种 parameterization 如下
- observation ：观察到的物体当前的 pose ，包含位置和角度 $(x, y, \theta)$ 
- goal ：目标的 pose $(x_{goal}, y_{goal}, \theta_{goal})$
- action ：推动点的位置与推动速度 $(x,y,v_x,v_y)$

Extrinsic Dexterity ：借助外界环境实现目标，比如抓一个盒子，夹爪宽度不够直接抓，可以推到墙角把盒子掀起了，抓比较窄的那一侧

对于三维情况的 push ，动作就可以表达成一个接触点（物体表面的一个点）加上推动方向（一个单位向量）

HACMan 将接触点离散化，而推动方向仍然是保持连续的

![[EAI导论/imgs/img12/image.png]]

## Dexterous Manipulation

常见的末端执行器有以下几种

![[EAI导论/imgs/img12/image-1.png]]

对于灵巧手，需要平衡成本、灵巧度（自由度）、可靠性、大小、能提供的力度等，有两种常见的方案：一种是把电机安在手指关节处，称为舵机方案，但由于电机大小受限，能提供的力也有限；另一种是把电机安在手底下，用线去控制每根手指的运动，称为绳驱方案，但线容易疲劳，可靠性不稳定







