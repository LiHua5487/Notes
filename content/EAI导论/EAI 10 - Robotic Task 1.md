
关键词：**Online Actor-Critic Algorithm**； **Importance Sampling**； **TRPO**； **PPO**； **Model Predictive Control (MPC)**

---
# Off-Policy

Policy Gradient 有两个主要问题：噪声很大，且需要大量采样

此外，RL 还有个很严重的问题，**维度灾难 Curse of Dimensionality** ，其描述的是当数据维度/特征维度急剧增加时，各种问题会变得异常复杂
对于 RL 来说，动作维度是一个很高维的空间，在里面去探索就像大海捞针，很多行动都是无用甚至有害的，智能体需要花费很长时间才能偶然碰到一些有奖励的动作，学习进程非常缓慢，即使智能体最终获得了奖励，也很难确定是之前哪个状态下的哪个动作导致了这一结果

## Online Actor-Critic Algorithm

On-Policy 的 Actor-Critic 中，每次的参数更新的数据都只来自于当前策略，这进一步导致了每次梯度更新是比较小的，学习缓慢

Off-Policy 中去掉这个假设，让训练数据不只来自于当前策略，还能来自于历史策略，会将产生的数据储存在一个 replay buffer 里（先入先出，新的数据会替换掉老的数据）

但是不能直接把这个机制应用到 On-Policy 的 Actor-Critic 中，这会导致一些问题，需要进行一些调整

![[EAI导论/imgs/img10/image.png]]

在 Critic 的更新中，假设采样到了一个旧的数据 $(s_i,a_i,r_i,s_i')$ ，但是对于当前的策略来说，在 $s_i$ 不一定会采取 $a_i$ ，而且就连智能体所达状态的分布都会变化，如果用来自旧策略的数据更新 Critic，就是在训练它去估计旧策略的价值，这显然是不合理的，类似的问题还出现在梯度计算时

对于值函数来说，实际上是

$$
V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi(a_t \mid s_t)} \left[ Q^\pi(s_t, a_t) \right]
$$

也就是说这和你在当前状态 $s_t$ 要怎么采取动作 $a_t \sim \pi(a_t \mid s_t)$ 有关，如果采样到了旧的数据，那这个分布就是旧的分布，这不是我们期望的

不妨直接以 Q 值作为训练对象

$$Q^\pi(s, a) = \mathbb{E}_{s' \sim P} \left[ r(s, a) + \gamma \mathbb{E}_{a' \sim \pi} \left[ Q^\pi(s', a') \right] \right]$$

尽管可能采样到旧的数据 $(s_i,a_i,r_i,s_i')$ ，但 $a'$ 可以根据当前策略得到，这就解决了上述问题，目标变成了

$$y_i = r_i + \gamma \hat{Q}^\pi_\phi(s'_i, a'_i)$$

而对于梯度问题，需要保证其中的 $a_i$ （ log 中的和 A 中的）是根据当前的策略得出的，记为 $a_i^\pi$，即需要用当前策略进行采样 $a_i^\pi \sim \pi_\theta(a | s_i)$ 

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(a_i^\pi | s_i) \hat{A}^\pi(s_i, a_i^\pi)$$

实际上为了简化计算，直接把 A 换成 Q ，这虽然会让方差增大，但是可接受的

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(a_i^\pi | s_i) \hat{Q}^\pi(s_i, a_i^\pi)$$

>这种基于 Q 函数的方式可以独立地评估和选择动作，拥有极高的样本效率Sample Efficiency ，即对数据的利用率，和卓越的探索能力（产生动作时，可以使用行为策略，侧重于探索）

对于 $s$ ，这就没办法保证其按当前策略分布了，因为都已经存下来了，但也不是不能接受，因为我们存的 $s$ 往往比预期的最优策略涵盖到的范围（只走最优的）要广（好的坏的历史中都走过了），尽管不一定最优，但能够增强鲁棒性，减少 OOD 的问题产生

总结一下，**Online Actor-Critic Algorithm** 如下
1. take action $\mathbf{a} \sim \pi_\theta(\mathbf{a} | \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$, store in $\mathcal{R}$
2. sample a batch $\{\mathbf{s}_i, \mathbf{a}_i, r_i, \mathbf{s}'_i\}$ from buffer $\mathcal{R}$
3. update $\hat{Q}^\pi_\phi$ using targets $y_i = r_i + \gamma \hat{Q}^\pi_\phi(\mathbf{s}'_i, \mathbf{a}'_i)$ for each $\mathbf{s}_i, \mathbf{a}_i$
4. $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_i \nabla_\theta \log \pi_\theta(\mathbf{a}_i^\pi | \mathbf{s}_i) \hat{Q}^\pi(\mathbf{s}_i, \mathbf{a}_i^\pi)$ where $\mathbf{a}_i^\pi \sim \pi_\theta(\mathbf{a} | \mathbf{s}_i)$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

>在 Off-Policy 中，往往会引入 **行为策略** 来获取数据并存到 buffer ，这个策略可以是一个历史策略加上一定的噪声

## Importance Sampling

如果把之前的策略产生的数据储存起来然后直接用，会导致模型的学习缺少“侧重点”，就比如做题，On-Policy 是做错一道题就立马针对这个题进行学习，而 Off-Policy 就像去翻错题本和笔记，针对性较差

而且在 Off-Policy 中，训练数据的分布和当前策略的状态分布是不一样的，为了解决这个问题，引入 **重要性采样 Importance Sampling** ，从数学上看，这解决了分布不匹配的问题，而从模型学习上看，这能让模型更关注当前的问题

>下面进一步解释了这种分布不匹配导致的问题：
>另外，假设在历史数据中，更多的尝试了动作 A ，而策略更新后更倾向于奖励更高的动作 B ，但如果直接用旧数据来训练，可能会高估 A 的价值，因为 A 的奖励虽然低但频繁，B 的奖励高但罕见，如果旧数据中 B 的样本很少，可能无法正确学习 B 的价值，导致训练不稳定（用不用 B 反复横跳）

假设我们要估计 x 在分布 p(x) 下的期望，但是现在 x 服从的分布是 q(x) ，可采用以下公式解决

$$
\begin{aligned}
\mathbb{E}_{x \sim p(x)}[f(x)] &= \int p(x) f(x) dx \\
&= \int \frac{q(x)}{q(x)} p(x) f(x) dx \\
&= \int q(x) \frac{p(x)}{q(x)} f(x) dx \\
&= \mathbb{E}_{x \sim q(x)} \left[ \frac{p(x)}{q(x)} f(x) \right]
\end{aligned}
$$

这就是重要性采样的核心，对于 $J(\theta)$ 来说就是这样的

$$J(\theta) = \mathbb{E}_{\tau \sim \bar{p}(\tau)} \left[ \frac{p_\theta(\tau)}{\bar{p}(\tau)} r(\tau) \right]$$

对于 $p_\theta(\tau)$ ，其展开后得

$$p_\theta(\tau) = p(\mathbf{s}_1) \prod_{t=1}^T \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) p(\mathbf{s}_{t+1} | \mathbf{s}_t, \mathbf{a}_t)$$

这对于 $\bar{p}(\tau)$ 来说也是同理的，二者相比得

$$\frac{p_\theta(\tau)}{\bar{p}(\tau)} = \frac{p(\mathbf{s}_1) \prod_{t=1}^T \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) p(\mathbf{s}_{t+1} | \mathbf{s}_t, \mathbf{a}_t)}{p(\mathbf{s}_1) \prod_{t=1}^T \bar{\pi}(\mathbf{a}_t | \mathbf{s}_t) p(\mathbf{s}_{t+1} | \mathbf{s}_t, \mathbf{a}_t)} = \frac{\prod_{t=1}^T \pi_\theta(\mathbf{a}_t | \mathbf{s}_t)}{\prod_{t=1}^T \pi_\text{old}(\mathbf{a}_t | \mathbf{s}_t)}$$

这样以来，梯度公式就变成了

$$\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \bar{p}(\tau)} \left[ \left( \prod_{t=1}^T \frac{\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)}{\pi_\text{old}(\mathbf{a}_t | \mathbf{s}_t)} \right) \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) \right) \left( \sum_{t=1}^T r(s_t, \mathbf{a}_t) \right) \right]$$

还需考虑因果性，对于奖励来说，当前影响不了过去，对于动作，未来影响不了现在，变成这样

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \bar{p}(\tau)} \left[ \sum_{t=1}^T \left( \underbrace{\prod_{t'=1}^t \frac{\pi_\theta(\mathbf{a}_{t'}|s_{t'})}{\pi_\text{old}(\mathbf{a}_{t'}|s_{t'})}}_{\text{用于看到状态 } s_t} \cdot \underbrace{\nabla_\theta \log \pi_\theta(\mathbf{a}_t|s_t)}_{\text{当前动作的得分}} \cdot \underbrace{\left( \sum_{t'=t}^T r(s_{t'}, \mathbf{a}_{t'}) \right)}_{\text{reward to go}} \right) \right]
$$

但是这个连乘很难受，直接用 t 时刻的替代

$$\nabla_\theta J(\theta) \approx \mathbb{E}_{\tau \sim \bar{p}(\tau)} \left[ \sum_{t=1}^T \left( \frac{\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)}{\pi_\text{old}(\mathbf{a}_t | \mathbf{s}_t)} \cdot \nabla_\theta \log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) \cdot G_t \right) \right]$$

把其中 $\frac{\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)}{\pi_\text{old}(\mathbf{a}_t | \mathbf{s}_t)}$ 这个比值项定义为置信度权重 $r_t(\theta)$ ，代表新策略对于在 $s_t$ 下采取动作 $a_t$ 的倾向性（>1为偏好，<1为排斥）

$$r_t(\theta) = \frac{\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)}{\pi_\text{old}(\mathbf{a}_t | \mathbf{s}_t)}$$

但是这里有一个问题，如果 $\pi_\theta$ 和 $\pi_\text{old}$ 差距太大，那这个比值项可能变得非常小或者非常大，导致梯度方差变大，训练不稳定

# PPO 

PPO 是在 TRPO 的基础上改进的，下面先简单介绍 TRPO

为了解决上面 $r_t(\theta)$ 这个比值过于极端的问题， **TRPO (Trust Region Policy Optimization)** 的想法是使用 KL-Divergence 去约束新旧策略的差距（设置信任区域 Trust Region），以防变化太大

$$\theta_{k+1} = \operatorname{argmax}_\theta \mathcal{L}(\theta_k, \theta)\quad \text{s.t. } \bar{D}_{KL}(\theta || \theta_k) \leq \delta$$

并设置了以下的优化目标

$$\mathcal{L}(\theta_k, \theta) = \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ \frac{\pi_\theta(a | s)}{\pi_{\theta_k}(a | s)} A^{\pi_{\theta_k}}(s, a) \right]$$

TRPO 的策略更新被证明是单调收敛的，即新策略的表现不会比旧策略更差，但这也导致其可能陷入局部最优解，同时，计算 KL 散度约束使得每次更新的过程需要进行二次优化，求解代价较高

---

**PPO (Proximal Policy Optimization)** 和 TRPO 的核心思想一致，都是要方式新旧策略差异过大，但 PPO 摒弃了 TRPO 中复杂的二阶优化（强制约束差异），而是采用一阶优化（标准梯度下降）和一些 trick 来近似地实现信任域的效果，具体来讲有以下两种做法

第一种是 **Clipped Surrogate Objective (PPO-Clip)** ，相较于约束 $D_{KL}$ ，OPP 直接把 $r_t(\theta)$ 截断在 $[1-\epsilon, 1+\epsilon]$ 这个区间内，但直接这么做会有问题

![[EAI导论/imgs/img10/image-2.png]]

在优势为正时，说明这是个好动作，应当更加偏好，但如果一开始 r 比较小，此时梯度也是 0 ，不会增加偏好，这显然不符合预期；对于优势为负的情况同理，如果一开始非常偏好一个坏动作，就纠正不回来了

所以实际应用中，还要额外取 min 

$$
\begin{aligned}
\theta_{k+1} &= \operatorname{argmax}_\theta \mathbb{E}_{s, a \sim \pi_{\theta_k}} \left[ L(s, a, \theta_k, \theta) \right] \\
L(s, a, \theta_k, \theta) &= \min \left( \frac{\pi_\theta(a | s)}{\pi_{\theta_k}(a | s)} A^{\pi_{\theta_k}}(s, a), \, \text{clip} \left( \frac{\pi_\theta(a | s)}{\pi_{\theta_k}(a | s)}, 1 - \epsilon, 1 + \epsilon \right) A^{\pi_{\theta_k}}(s, a) \right)
\end{aligned}
$$

![[EAI导论/imgs/img10/image-1.png]]

- 当 $A_t$ 为正：希望增加 $r_t(\theta)$ ，但要防止加的太大，上限为 $1+\epsilon$
- 当 $A_t$ 为负：希望减小 $r_t(\theta)$ ，但要防止变得太小（接近0），下限为 $1-\epsilon$

其具体原理是：对于优势为正，当 r 比较大时，L 的梯度就是 0 了，防止 r 继续增加，优势为负时同理，防止过度减小

第二种是 **Adaptive KL Penalty Coefficient (PPO-Penalty)** ，与其把 $D_{KL}$ 设为强制约束，不如把它放到目标函数里，作为一个惩罚项

$$L^{KL PEN}(\theta) = \hat{\mathbb{E}}_t \left[ \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)} \hat{A}_t - \beta \, \text{KL}[\pi_{\theta_{\text{old}}}(\cdot \, | \, s_t), \pi_\theta(\cdot \, | \, s_t)] \right]$$

- $\beta$ 是惩罚系数，控制着对策略变化大小的厌恶程度

如果 $\beta$ 是一个固定值，我们会遇到一个难题
- 太大：惩罚过强，策略会因害怕改变而更新得非常缓慢，甚至停滞不前
- 太小：惩罚过弱，无法有效限制策略的更新幅度，可能导致训练崩溃

PPO-Penalty 的自适应 Adaptive 机制就是为了动态地调整 $\beta$，使其在训练过程中找到一个合适的值
- 首先，我们会设置一个 $D_{KL}$ 的预期值 $\delta_{\text{target}}$ ，表示我们期望每次更新后，策略的平均变化程度
- 每次更新后，计算当前的平均 KL 散度 $d = \hat{\mathbb{E}}_t [\text{KL}[\pi_{\theta_{\text{old}}}(\cdot \, | \, s_t), \pi_\theta(\cdot \, | \, s_t)]]$
- 比较 $d$ 和 $\delta_{\text{target}}$ 
	- 如果 $d>\delta_{\text{target}}\cdot1.5$ ，说明变化太大了，就增大惩罚系数 $\beta \leftarrow \beta \cdot 2$ 以更严格的惩罚大幅变化
	- 如果 $d<\delta_{\text{target}}/1.5$ ，说明变化太小了，就减小惩罚系数 $\beta \leftarrow \beta / 2$ 以允许策略有更大的变化
	- 如果在这个范围之间，说明变化大小合适，惩罚系数不变

>平均 KL 散度式子里的那个点 · 是一个占位符，类似于输入的变量，以函数为例，$f(x)$ 表示的是一个特定值，而 $f(·)$ 就表示这个函数

实际实现时，可以每次收集数据后，进行 K 次更新（K 个 epoch），这个 K 不能太大，以防分布差距太大
在一次 epoch 中，可以将 buffer 中的数据划分为许多 mini-batches （与 batch gradient decent 类似），但是需要合适的 batch 大小，过大容易陷入局部最优，过小则数据噪声太大

>PPO 可以看作介于 On-Policy 和 Off-Policy 之间（尽管分类学上归到 On-Policy），因为其采集当前策略的数据后（On-Policy 特点），会进行多次梯度更新（Off-Policy 特点），同时采用重要性采样解决其中的分布差异

---

robotics 的研究通常会采用 PPO ，而非基于 Q 值的方法（如 DDPG / SAC）
- 传统的 Q-learning 需要构建 Q 值表，即根据 $(s,a)$ 去查找 Q 值，这适合低维离散动作空间，而不适用于高维空间（尽管 DQN DDPG SAC 等后续方法进行改进，使其能应对高维连续动作）
- 对于 highly dynamic 的任务，动作稍微偏差一点，结果的差距就会有很大变化，这就导致 Q 函数的变化非常剧烈，学习可能不稳定
- 尽管 PPO 是基于 On-Policy 的采样方式，相比 Off-Policy 的采样方式效率较低，即数据利用率低，需要更多的采样，但随着并行计算的发展，这个问题几乎被解决了

# Locomotion

Navigation 解决的是路径规划问题，而 Locomotion 关注的是如何实现某种具体的动作

步态 Gait ：如何走路的模式，一般分为以下几种

![[EAI导论/imgs/img10/image-3.png]]

## Model Predictive Control (MPC)

MPC 通过使用系统的动态模型来预测其未来的行为，并实时计算出一系列最优的控制动作，但只执行第一个动作，然后在下一个时刻重复这个过程，就好比下棋时算几步然后走一步

具体来讲，MPC 定义了一个代价函数，需要求解一个 $\mathbf{u}$ ，在满足约束的前提下，使代价函数最小，其中 $\mathbf{x}(t)$ 是位置状态， $\mathbf{u}(t)$ 是控制输入

$$\int_{t_s}^{t_f} l(\mathbf{x}(t), \mathbf{u}(t), t) \, dt$$

需要满足以下约束

$$
\begin{aligneded}
&\mathbf{x}(t_s) = \mathbf{x}_s \quad &&\text{Initial Condition} \\
&\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, t) \quad &&\text{System Dynamic} \\
&\mathbf{g}(\mathbf{x}, \mathbf{u}, t) = 0 \quad &&\text{Equality Constraints} \\
&\mathbf{h}(\mathbf{x}, \mathbf{u}, t) \geq 0 \quad &&\text{Inequality Constraints}
\end{aligneded}
$$

比如下面一个例子，要维持平衡，防止上面的球掉下来

![[EAI导论/imgs/img10/image-4.png]]

欠驱动 Under-Actuated ：一个系统的控制能力（执行器数量）不足以独立、直接地控制其所有运动自由度，只能通过某种组合动作，或者借助外力来完成复杂的控制任务

对于欠驱动的系统，基本上不可能做出精确的 MPC 模型，而且对计算速度有很高的要求（因为计算过程中智能体的状态和环境也可能在实时发生变化）

基于 MPC 的种种限制，一般还是采用 RL 的方式

![[EAI导论/imgs/img10/image-5.png]]

## MDP Formulation Example

一个 Agent 与 Robot 的互动方式通常如下

![[EAI导论/imgs/img10/image-6.png]]

---

在一个马尔科夫决策过程 MDP 中，其状态的获取方式有
- 本体感知 Proprioception ：采用惯性测量单元 （Inertial Measurement Unit, IMU） 获得，具体包括
	- 加速度计 Accelerometer ，获取加速度
	- 陀螺仪 Gyroscope ，获取角速度
	- 磁力计 Magnetometer ，矫正陀螺仪的误差 
- 外感知 Exteroception （可选）：雷达点云，RGBD 相机
- 接触传感器 Contact Sensors （可选）
- Joint Encoders ：对关节位置与速度进行编码

对于表达机器人身体的 pose 来说，有两种选择
- Joint Space$[\theta_1, \dot{\theta}_1, \dots, \theta_n, \dot{\theta}_n]$ (12 joints) ：直接输入到 Joint Encoder；维度相对较低，易于训练；与 Action Space 一致
- Link Space $[(p_1, q_1, v_1, \omega_1), \dots, (p_m, q_m, v_m, \omega_m)]$ (13 links) ：描述自身各个刚体部分（link）的属性（p 位置，q 姿态，v 速度，ω 角速度），会编码空间中的位置关系

一般任务采用 Joint Space ，在跳舞任务中会采用 Link Space ，因为 Link Space 中会关注世界坐标系下的位置，而不仅仅是自身的姿态，比如跳舞时可能先往左跳再往右跳回，对于 Joint Space 来说就是重复一遍动作，但是 Link Space 中能体现其位置变化

---

对于运动，使用 PD 控制器进行控制，一般采用位置控制器
- 相比于扭矩控制 Torque Control ：位置控制器更新频率较低，但它在底层仍可高频率地输出相应的扭矩，系统响应性能更强
- 相比于速度控制：位置控制在 Sim2Real 中表现更好，而如果控制一个关节的转速，可能导致转过头了
- 神经网络的输出频率可能没有那么快，而位置控制器正好能适配这种低频

一般不采用 PID 控制器，因为
- PID 控制器中的积分项会不断累计误差值，如果机器人不处在稳定状态，这种误差积累会导致控制信号产生振荡，甚至会对系统稳定性造成不良影响
- RL 算法内在具有类似积分项的补偿能力
- RL+PD 的表现已经足够好了

---

尽管在虚拟环境中，经过域随机化 Domain Randomization ，能够适应更多的场景，但在现实世界的一个局部环境中，并不一定是最优解（比如训练走步，虚拟环境里的地面可能一会是土地，一会是冰面，那模型就会学一个更加保守的走路方式，比如频繁踏步，但如果把它部署到室内环境，这显然不是最优的）

为了解决这个问题，可以后续将模型放到真实世界里继续训练，也可以参考仿生学，把现实中动物的运动方式作为 reference 供模型学习（但鲁棒性可能没那么高）

而且由于机器人在物理层面也可能存在差异（即便是同一个型号），所以在一个机器人上 work 的策略，可能在另一个上没那么好




