
关键词：**Policy Gradient**； **Online/Offline RL**； **On-Policy/Off-Policy**； Monte Carlo Estimation； Model-Based RL； **Reinforcement Algorithm**； TD Target/Error； 软更新； Bootstrap； **(Batch/Online) Actor-Critic Algorithm**

---
# Policy Gradient

模仿学习的问题在于，学习的直接来源是专家数据，而没有自己和环境去交互，所以受到专家数据的限制

>**Offline Learning** ：训练数据是预先准备好的，模仿学习就是这种方式
>**Online Learning** ：在训练过程中实时获得数据，强化学习允许这种可能（但不是所有强化学习都是在线学习）

RL 有以下几种类别
- Online RL：通过与环境的实时交互来训练策略
	- On-Policy RL：只使用当前最新的策略产生的数据进行学习
	- Off-Policy RL：可以利用任何旧策略产生的历史数据来学习新策略
- Offline RL：不与环境交互，而是用采集到的数据替代在线交互
	- 与 BC 的区别：BC 纯粹模仿专家数据，不需要奖励函数，不属于RL；而 Offline RL 仍需要奖励信号（包含在数据集中）

![[EAI导论/imgs/img8/image.png]]

![[EAI导论/imgs/img8/image-1.png]]

---

RL 的学习目标是找到一个策略（由模型参数 $\theta$ 决定），让智能体在交互时产生的轨迹奖励期望最大化

$$
\theta^\star = \arg\max_\theta \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(s_t, a_t) \right]
$$

从决策的时间范围来看，可分为以下两种
- Finite Horizon：任务会在一个已知的固定的时间步长 T 后终止，累计奖励是这 T 步的奖励的求和；其策略与值函数依赖于时间 t 

$$
\theta^\star = \arg\max_\theta \sum_{t=1}^T \mathbb{E}_{(s_t, a_t) \sim p_\theta(s_t, a_t)} \left[ r(s_t, a_t) \right]
$$

- Infinite Horizon：任务没有明确的终止点策略与值函数和时间无关（绝大多数理论研究和算法默认都是在 infinite horizon 下进行的，因为表达更简洁）

$$
\theta^\star = \arg\max_\theta \mathbb{E}_{(s, a) \sim p_\theta(s, a)} \left[ r(s, a) \right]
$$

>无限时间步中，为了避免累计奖励无穷大，引入折扣因子 $\gamma$ 使其收敛，还能隐式地鼓励尽早完成任务，因为越往后的奖励会被折扣得越多，所以总回报最高的方式通常是尽快获得奖励，这也导致模型更偏好即时的奖励

## Reinforcement Algorithm

但实际上，这个期望是很难准确求得的，不过可以利用 **蒙特卡洛近似 Monte Carlo Estimation** 来快速求得一个估计值，其思路是，进行 N 次采样取出一些轨迹，然后求这些轨迹的累计奖励的平均值作为近似

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \sum_t r(s_t, a_t) \right] \approx \frac{1}{N} \sum_i \sum_t r(s_{i,t}, a_{i,t})
$$

- $\tau$ 为智能体的轨迹，$p_\theta(\tau)$ 为使用当前策略 $\pi_\theta(a|s)$ 得到的轨迹的概率分布
但这仍有一些困难，因为没有一个量是关于 $\theta$ 显式可导的，但我们需要求出这个期望的梯度，以便在训练时更新参数

先不用蒙特卡洛近似，根据定义，可得

$$
J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ r(\tau) \right] = \int p_\theta(\tau) r(\tau) d\tau
$$

- 对于有限时间步，则 $r(\tau) = \sum_{t=1}^T r(s_t, a_t)$

应用以下这个恒等式

$$
\nabla_\theta p_\theta(\tau) = p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} = p_\theta(\tau) \nabla_\theta \log p_\theta(\tau)
$$

可将其梯度化为

$$
\begin{aligned}
\nabla_\theta J(\theta) &= \int \nabla_\theta p_\theta(\tau) r(\tau) d\tau \\
&= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) r(\tau) d\tau \\
&= \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \nabla_\theta \log p_\theta(\tau) r(\tau) \right]
\end{aligned}
$$

对于其中的 $\log p_\theta(\tau)$ 项，由于

$$p_\theta(s_1, \mathbf{a}_1, \dots, s_T, \mathbf{a}_T) = p(s_1) \prod_{t=1}^T \pi_\theta(\mathbf{a}_t | s_t) p(s_{t+1} | s_t, \mathbf{a}_t)$$

两边取 log 得

$$
\log p_\theta(\tau) = \log p(s_1) + \sum_{t=1}^T \log \pi_\theta(\mathbf{a}_t | s_t) + \log p(s_{t+1} | s_t, \mathbf{a}_t)
$$

>**事件模型**：对环境动态的建模，即状态-动作到下一个状态和奖励的映射关系
>通常包含 $p(s_{t+1}|s_t,a_t)$ 和 $r(s_t|a_t)$ （但这并不总是能显式表达）
>**Model-Based RL**：利用环境模型来进行学习，节省了与真实环境的交互（或模拟器中的交互），能直接根据建模得到动作轨迹和奖励

对其求梯度得

$$
\nabla_\theta \left[ \log p(s_1) + \sum_{t=1}^T \log \pi_\theta(\mathbf{a}_t | s_t) + \log p(s_{t+1} | s_t, \mathbf{a}_t) \right]
$$

其中 $\log p(s_1)$ 和 $\log p(s_{t+1} | s_t, \mathbf{a}_t)$ 项与 $\theta$ 无关，梯度为 0 ，所以 $\nabla_\theta J(\theta)$ 化为

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim p_\theta(\tau)} \left[ \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_t | s_t) \right) \left( \sum_{t=1}^T r(s_t, \mathbf{a}_t) \right) \right]
$$

而后应用蒙特卡洛近似

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t} | s_{i,t}) \right) \left( \sum_{t=1}^T r(s_{i,t}, \mathbf{a}_{i,t}) \right)
$$

这样就可以利用梯度来更新参数了，由于希望期望更大，所以是梯度上升

$$\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$$

注意到，梯度式子中并没有出现奖励的梯度，而是奖励值本身，这意味着 RL 不要求奖励函数本身可导（甚至不需要知道奖励函数的具体形式，只需要能够从环境中获得每个时间步的奖励值即可），这极大地扩展了强化学习的应用范围，可以处理奖励是稀疏的、非连续的情况

以上过程可以总结成 **Reinforcement Algorithm** 
1. sample $\{\tau^i\}$ from $\pi_\theta(\mathbf{a}_t | \mathbf{s}_t)$ (run it on the robot)
2. $\nabla_\theta J(\theta) \approx \sum_i \left( \sum_t \nabla_\theta \log \pi_\theta(\mathbf{a}_t^i | \mathbf{s}_t^i) \right) \left( \sum_t r(\mathbf{s}_t^i, \mathbf{a}_t^i) \right)$
3. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

> 注：上面的计算属于 On-Policy Model-Free 的 RL，即 Policy Gradient 是针对于 On-Policy 而言的
> On-Policy 的一个问题就是采样速度低；与之相对，Off-Policy 尽管当前策略可能已经解决了一些问题，但仍可能采样到一些不好的策略，效果不如 On-Policy

其中计算 $\nabla_\theta \log \pi_\theta(a | s)$ 的部分可以通过自动微分实现

>手动/符号微分：人工/计算机推导微分公式
>数值微分：用差商 $(f(x+ε) - f(x)) / ε$ 近似
>自动微分：应用链式法则 chain rule ，把微分过程分解成一系列简单的操作（加法、乘法、指数、对数等）的组合

---

与之相比，BC 的目标是训练一个策略，使得在这个策略下，得到专家数据的概率最大，即最大似然 Maximum Likelihood，其梯度如下

$$
\nabla_\theta J_{ML}(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t} | s_{i,t}) \right)
$$

在 BC 的式子中，不同策略是平权的，没有好坏对错之分，只是单纯去模仿；而在 RL 中，加上了累计奖励这一权重，让更好的策略影响更大
这就导致 BC 可能模仿到了一些不期待的东西，比如动作中的不稳定晃动（但训练数据中的微小晃动可以让模型看到新的空间，能进行纠正）

---

假设目前的策略是一个高斯分布

$$
\pi_\theta(\mathbf{a}_t | \mathbf{s}_t) = \mathcal{N}(f_{\text{neural network}}(\mathbf{s}_t); \Sigma)
$$

对其取 log 后就变成了

$$
\log \pi_\theta(\mathbf{a}_t | \mathbf{s}_t) = -\frac{1}{2} \| f(\mathbf{s}_t) - \mathbf{a}_t \|^2_\Sigma + \text{const}
$$

那 $J(\theta)$ 的式子就类似于用累计奖励加权的 L2 loss ，这和 regression 问题有什么区别呢？在 regression 中，训练集和测试集一开始就定下来了；但 RL 中，每次更新策略后，都会运行这个策略得到一系列数据，根据这些再次更新策略，相当于每个策略都有一个自己的训练集

## Partial observability

实际观测中，不一定能获取环境中所有的 state ，比如人骑自行车时只能看到一个画面，而不知道轮子具体的转速等数据，在这种不完整的观测下，RL 学习的策略就变为根据观测做出动作，即 $\pi_{\theta}(a_t|o_t)$ 

与之前的推导相同，可以得到其梯度

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t} | \mathbf{o}_{i,t}) \right) \left( \sum_{t=1}^T r(\mathbf{s}_{i,t}, \mathbf{a}_{i,t}) \right)
$$

这里假设根据观测能做出合理的动作，但实际上观测提供的信息不一定足够（比如看不到后面但要倒车），所以实际中往往不能使用马尔可夫假设，还需要结合历史观测进行判断

## Problems with Policy Gradient

Policy Gradient 的梯度值的方差很大，核心原因是采用了蒙特卡洛近似，具体来讲，有以下问题

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t} | s_{i,t}) \right) \left( \sum_{t=1}^T r(s_{i,t}, \mathbf{a}_{i,t}) \right)
$$

- 轨迹处在一个高维空间，数量很多，但只能进行有限次采样
- 另一方面，其依赖从初始状态到终止状态的整条轨迹的累计奖励，而轨迹的变化可能是很大的，导致这个奖励项的波动很大

---

对于同一条轨迹，其在不同时间步下的权重是一样的（都是同一个累计奖励），但一个轨迹中可能有些走的对，有些是乱走

为此，可以引入**因果性 Causality**，即每个动作只能影响到之后所能获得的奖励，影响不到之前的，据此可以把一条轨迹中不同时间步 t 的权重设为从此时 t 到最后结束时 T 的奖励之和

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(\mathbf{a}_{i,t} | \mathbf{s}_{i,t}) \left( \sum_{t'=t}^T r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)
$$

- 其中 $\left( \sum_{t'=t}^T r(\mathbf{s}_{i,t'}, \mathbf{a}_{i,t'}) \right)$ 一项就是 reward to go ，记作 $\hat{Q}_{i,t}$ 或 $G_t$ 

值得注意的是，这并没有改变梯度，因为过去的奖励只依赖于之前的状态和行动，与当前无关，所以过去的累计奖励对应的梯度为 0 

---

给奖励项增减一个常数值 b ，会发现其根本不影响梯度

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau) [r(\tau) - b]$$

这是因为把常数项部分提取出来，可以得到这部分梯度为 0 

$$
\begin{aligned}
E[\nabla_\theta \log p_\theta(\tau) b] &= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) b \, d\tau \\
&= \int \nabla_\theta p_\theta(\tau) b \, d\tau \\
&= b \nabla_\theta \int p_\theta(\tau) \, d\tau \\
&= b \nabla_\theta 1 = 0
\end{aligned}
$$

这从直观上很好理解，如果所有的累计奖励都一样，那模型根本不知道啥好啥坏，自然就不知道学习啥，即没法更新梯度

>注：这和 BC 的梯度公式并不相同，本质上是因为期望的含义不同
>这里的期望实际上是$$\mathbb{E}_{\tau \sim \pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a | s) \right]$$而 BC 中的期望实际上是$$\mathbb{E}_{(s, a) \sim D} \left[ \nabla_\theta \log \pi_\theta(a | s) \right]$$其中 $\mathbb{E}_{X \sim P}(Y)$ 表示随机变量 $X$ 遵循分布 $P$ 时 $Y$ 的期望

这个 b 就称作**基线 baseline**，那 b 设置成多少合适呢，设置成平均值 $b = \frac{1}{N} \sum_{i=1}^N r(\tau)$ 虽然不是最好的，但效果还不错；更一般的，可以设为关于状态的函数 $b(s_t)$

# Actor Critic Algorithm

为了进一步解决梯度方差问题，可以采用 Actor Critic Algorithm  算法

实际上，对于一个时间步，后续可以有很多条轨迹，但现在的 reward to go 中只是选择了其中一条，理论上应该是这些轨迹的累计奖励的期望

$$
Q(s_t, a_t) = \sum_{t'=t}^T \mathbb{E}_{\pi_\theta} \left[ r(s_{t'}, a_{t'}) \mid s_t, a_t \right]
$$

将这个函数称为 **Q-function** ，代表在 $s_t$ 状态下采取 $a_t$ 动作后，按照策略 $\pi$ 进行行动得到的累计奖励的期望

另外，还可以定义**值函数 value-function** ，代表从 $s_t$ 状态下，按照策略 $\pi$ 进行行动得到的累计奖励的期望

$$
V^\pi(s_t) = \sum_{t'=t}^T \mathbb{E}_{\pi_\theta} \left[ r(s_{t'}, a_{t'}) \mid s_t \right]
$$

可得其与 Q-function 有以下关系，右式表示当行动 $a_t$ 按照策略 $\pi$ 分布时（采取不同动作都有对应的概率），采取各个动作对应的 Q 值的期望，这与值函数的含义是等价的

$$
V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi(a_t \mid s_t)} \left[ Q^\pi(s_t, a_t) \right]
$$

用值函数可以方便的表示 RL 学习的目标

$$
\mathbb{E}_{s_1 \sim p(s_1)} \left[ V^\pi(s_1) \right]
$$

这个式子表示初始状态 $s_1$ 遵循初始状态的分布 $p(s_1)$ 的情况下，初始状态的值函数的期望

采用 Q 值代替 $G_t$ 后，可以选取 $V(s_t)$ 作为 baseline ，代表 $s_t$ 下各种动作的平均好坏，而 $Q(s_t,a_t)-V(s_t)$ 就表示采取动作 $a_t$ 相较于平均水平怎么样，把这个式子定义为状态 $s_t$ 下动作 $a_t$ 的**优势函数 advantage function**

$$A(s_t,a_t)=Q(s_t,a_t)-V(s_t)$$

梯度公式就变成了

$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) A(s_{i,t}, a_{i,t})
$$

那怎么求这个优势呢，根据 Q 值和值函数的含义，可得

$$
\begin{aligned}
Q^\pi(s_t, a_t) &= r(s_t, a_t) + \sum_{t'=t+1}^T \mathbb{E}_{\pi_\theta} \left[ r(s_{t'}, a_{t'}) \mid s_t, a_t \right] \\
&= r(s_t, a_t) + \mathbb{E}_{s_{t+1} \sim p(s_{t+1})}[V^{\pi}(s_{t+1})]
\end{aligned}
$$

其中 $s_{t+1}$ 存在一个分布是因为采取相同动作后可能到达不同状态状态（比如有概率偏离），但是实际过程中，是会运行动作 $a_t$ 并得到下一状态 $s_{t+1}$ ，不妨直接取到达的这个状态的值函数代替期望值，那么

$$A^\pi(s_t,a_t)\approx r(s_t,a_t)+V^\pi(s_{t+1})-V^\pi(s_t)$$

而奖励函数一般是预先已知的，只需要获得值函数即可，这么做的合理之处在于
- 无偏估计：虽然单个样本可能不能完全代表期望，但如果我们有大量样本，样本平均值会收敛到期望值（大数定律）
- 计算效率：计算所有可能下一个状态的期望通常需要知道状态转移概率，这在复杂环境中往往未知或计算成本高

那怎么获得值函数呢，回顾公式，发现值函数也是一个期望，同样难以直接计算，需要去近似，这个过程就是**策略估值 Policy Evaluation**

$$
V^\pi(s_t) = \sum_{t'=t}^T \mathbb{E}_{\pi_\theta} \left[ r(s_{t'}, a_{t'}) \mid s_t \right]
$$

参考之前的方法，可以使用 N 次的蒙特卡洛近似

$$
V^\pi(s_t) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t'=t}^T r(s_{t'}, a_{t'})
$$

也可以直接取一条轨迹进行替代，即 N=1 的蒙特卡洛近似

$$
V^\pi(s_t) \approx \sum_{t'=t}^T r(s_{t'}, a_{t'})
$$

还可以训练一个神经网络去拟合值函数，理想的目标值 $y$ 是值函数，但是可以先进行近似，这个目标称为**蒙特卡洛目标**

$$
y_{i,t} = \sum_{t'=t}^T \mathbb{E}_{\pi_\theta} \left[ r(s_{t'}, a_{t'}) \mid s_{i,t} \right] \approx \sum_{t'=t}^T r(s_{i,t'}, a_{i,t'}) = r(s_{i,t}, a_{i,t}) + V^\pi(s_{i,t+1})
$$

对于 $V^\pi(s_{i,t+1})$ ，可以采用之前训练过程中的网络进行拟合

$$y_{i,t} \approx r(s_{i,t}, a_{i,t}) + \hat{V}^\pi_\phi(s_{i,t+1})$$

>训练过程中，每隔一定周期才保存一次模型作为目标，防止目标经常变动导致训练不稳定，尽管在目标网络的更新周期内（即目标网络保存一次，但没到下一次保存），使用旧的目标网络计算目标值可能会产生一种“往回拽”的效果，即当前网络被拉向旧的目标值
>但这种效应是可控的，并且通过精心设计的更新机制，整体训练仍然能够稳定进步，比如采用 **软更新** 的方式，在保存周期的期间，每一步都进行更新，混合进去一小部分当前网络的参数，其中 $\tau$ 为更新率，代表当前网络参数的比例$$\theta^{V'} \leftarrow \tau \theta^V + (1 - \tau) \theta^{V'}$$而且，尽管用的是旧的函数，每次更新策略后，又会产生新的训练数据，这会带来新的状态和动作，从而影响到目标值（因为其中的 $a_{i,t}$ 和 $s_{i,t+1}$ 变了），所以目标整体来说还是在前进的

则训练数据的结构如下，这个目标值被称为 **TD Target**

$$
\left\{ \left( s_{i,t}, r(s_{i,t}, a_{i,t}) + \hat{V}^\pi_\phi(s_{i,t+1}) \right) \right\}
$$

这种用一个估计值去更新另一个估计值的方法被称为**自举 bootstrap**

将 loss 设置为

$$
\mathcal{L}(\phi) = \frac{1}{2} \sum_i \left\| \hat{V}^\pi_\phi(s_i) - y_i \right\|^2
$$

这被称为 **Temporal Difference (TD) Error**

以上过程可以总结成 **Batch/Episodic Actor-Critic Algorithm**
1. sample $\{s_i, a_i\}$ from $\pi_\theta(a \mid s)$ (run it on the robot)
2. fit $\hat{V}^\pi_\phi(s)$ to sampled reward sums
3. evaluate $\hat{A}^\pi(s_i, a_i) = r(s_i, a_i) + \hat{V}^\pi_\phi(s'_i) - \hat{V}^\pi_\phi(s_i)$
4. $\nabla_\theta J(\theta) \approx \sum_i \nabla_\theta \log \pi_\theta(a_i \mid s_i) \hat{A}^\pi(s_i, a_i)$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

另外还有 **Online Actor-Critic Algorithm**
1. run N simulators to take action $\mathbf{a} \sim \pi_\theta(\mathbf{a} \mid \mathbf{s})$, get $(\mathbf{s}, \mathbf{a}, \mathbf{s}', r)$
2. update $\hat{V}^\pi_\phi$ using target $r + \gamma \hat{V}^\pi_\phi(\mathbf{s}')$
3. evaluate $\hat{A}^\pi(\mathbf{s}, \mathbf{a}) = r(\mathbf{s}, \mathbf{a}) + \gamma \hat{V}^\pi_\phi(\mathbf{s}') - \hat{V}^\pi_\phi(\mathbf{s})$
4. $\nabla_\theta J(\theta) \approx \nabla_\theta \log \pi_\theta(\mathbf{a} \mid \mathbf{s}) \hat{A}^\pi(\mathbf{s}, \mathbf{a})$
5. $\theta \leftarrow \theta + \alpha \nabla_\theta J(\theta)$

>与 Batch AC 的区别在于，Batch AC 是跑 N 个轨迹，而 Online AC 是并行 N 个模拟器，各走一个 action 
>Online 更新频率快（每一步都可能更新），数据利用率更高，能适应非平稳环境，但单步更新可能带来高方差
>Batch 更新基于更多数据（走完一整条轨迹才更新），梯度估计更稳定，方差较低，但需要存储更多数据，更新频率较低

这里 Actor 就是一个策略，负责产生具体的动作，利用 TD 误差更新策略网络，产生更好的动作；而 Critic 就是 V 或 Q ，可以计算出 A，负责评价 Actor 的动作的好坏Critic 利用 TD 误差更新价值网络，让自己未来的预测更准确

## Infinite Timestep

之前讨论的是有限时间步，对于无限时间步，需要引入折扣因子 discount factor ，其累计收益如下

$$G(\tau) = r_0 + \gamma r_1 + \gamma^2 r_2 + \cdots = \sum_{t=0}^{T} \gamma^t r_t$$

上述 TD Target 就变成了

$$y_{i,t} \approx r(s_{i,t}, a_{i,t}) + \gamma \hat{V}^\pi_\phi(s_{i,t+1})$$

则梯度公式变为

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \left( r(s_{i,t}, a_{i,t}) + \gamma \hat{V}^\pi_\phi(s_{i,t+1}) - \hat{V}^\pi_\phi(s_{i,t}) \right)$$

其中优势函数为

$$\hat{A}^\pi(s_{i,t}, a_{i,t}) = \left( r(s_{i,t}, a_{i,t}) + \gamma \hat{V}^\pi_\phi(s_{i,t+1}) - \hat{V}^\pi_\phi(s_{i,t}) \right)$$

在进行蒙特卡洛近似时，有两种选择
- Option1：一条轨迹中不同时间步 t 的权重设为从此时 t 到最后结束时 T 的奖励之和

$$\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \left( \sum_{t'=t}^T \gamma^{t'-t} r(s_{i,t'}, a_{i,t'}) \right) \tag{1}$$

- Option2：一条轨迹中不同时间步 t 的权重相同，是整条轨迹的累计奖励

$$
\begin{aligned}
\nabla_\theta J(\theta) &\approx \frac{1}{N} \sum_{i=1}^N \left( \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \right) \left( \sum_{t'=1}^T \gamma^{t'-1} r(s_{i,t'}, a_{i,t'}) \right) \tag{2} \\
&= \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \gamma^{t-1} \nabla_\theta \log \pi_\theta(a_{i,t} \mid s_{i,t}) \left( \sum_{t'=1}^T \gamma^{t'-t} r(s_{i,t'}, a_{i,t'}) \right)
\end{aligned}
$$

实际情况中一般采用的是 Option1 ，因为其考虑了因果性

## Network Architecture

在采用 Actor Critic Algorithm 网络训练时，有以下两种架构

![[EAI导论/imgs/img8/image-2.png]]

- 分离网络：Actor 和 Critic 使用独立的神经网络，简单稳定，但无特征共享
- 共享网络：Actor 和 Critic 共享部分底层网络，参数效率高，但训练可能更复杂

即使在 Online AC 中，也常常收集一个小批量数据来更新网络的，因为这有助于稳定学习过程，降低梯度估计的方差

>并行化 Parallelization：使用多个并行的 Actor（workers）同时在环境中收集经验，可以显著提高数据采集速度和多样性，进一步稳定训练
>并行又可分为同步 Synchronous 和异步 Asynchronous，同步并行存在同步点，整体速度受限于最慢的 worker；异步并行则没有同步点，会更快