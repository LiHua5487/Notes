
关键词：**PRM**； **Rejection/Gaussian/Bridge Sampling**；  **RRT**； 瞬态/稳态响应； **PID 控制器**

---
# Motion Planning

**构型空间 Configuration space ($C\text{-space} \subseteq \mathbb{R}^n$)** 包含了机器人工作状态的数学表示，如 pose 或关节角度的组合
- $C_{free} \subseteq C$ 表示能到达的地方
- $C_{obs} \subseteq C$ 表示障碍物的地方
- $C=C_{free}\cup C_{obs}$

给定 $C_{free}$ ，及其中的起点 $q_{start}$ 和终点 $q_{goal}$ ，要规划出一条路线/一系列动作
难点在于，需要躲开障碍物，需要长程视野，而且还是在高维空间进行规划

直接在高维空间用寻路算法效率太低了，可以进行采样，再尝试把邻近的两个采样点连起来，就得到一条边，只需检查这些边有没有碰撞

Pros
- 概率完备：样本越多，找到解的概率趋近于 1
- 只需知道部分 $C_{free}$ 就能给出解
- 在高维空间容易实现

Cons
- 需要在两个邻近点间规划路径，因为实际连接时需要求解一条满足动力学约束的局部轨迹
- 当 $C_{free}$ 的“连通性”差（如狭窄走廊、迷宫式环境），随机采样很难恰好落在关键通道里，导致收敛慢甚至找不到解
- 不保证最优

下面介绍两种基于采样的路径规划方法

## Probabilistic Roadmap Method (PRM)

这个算法有两个阶段
- Map construction
	- 在 $C_{free}$ 中随机取点
	- 将相邻的采样点连接
	- 将起终点与 graph 相连
- Query
	- 在这个 map 上采样 Dijkstra 等算法搜索路径

![[EAI导论/imgs/img3/image.png]]

需要在 $C_{free}$ 中尽可能均匀的采样，可以在 $C$ 内均匀采样，再把 $C_{obs}$ 中的去掉，这种采样称为**拒绝采样 rejection sampling**

难点在于连接相邻采样点与检测碰撞

但对于下面这种情况，很难保证在狭小通道内也能有采样点

![[EAI导论/imgs/img3/image-1.png]]

可以采用**高斯采样 gaussian sampling** ，把采样重点放到障碍边界附近，用来提高在狭窄通道中的连通度
- 先在 $C$ 里均匀采样一点 $q_1$
- 再以 $q_1$ 为均值（中心点），在 $\mathcal{N}(q_1,\sigma ^2)$ 采样一点 $q_2$
- 若 $q_1$ 与 $q_2$ 恰好一个落在 $C_{free}$ ，一个落在 $C_{obs}$ 时，就把落在 $C_{free}$ 的那个点加入路线图
高斯分布在 $q_1$ 附近密度高，因此大部分 $q_2$ 离 $q_1$ 不远，如果一个在 $C_{free}$ ，一个在 $C_{obs}$ ，说明两点位于障碍物的边界处

![[EAI导论/imgs/img3/image-2.png]]

但是观察结果可以发现，对于狭窄的通道，端口处点很多，但中间几乎没有点
可以采用**桥采样 bridge sampling** 进一步优化这部分
- 在 $C$ 里均匀采样一点 $q_1$
- 再以 $q_1$ 为均值（中心点），在 $\mathcal{N}(q_1,\sigma ^2)$ 采样一点 $q_2$
- 若 $q_1$ 与 $q_2$ 都在 $C_{obs}$ ，且其中点 $q_3=\frac{q_1+q_2}{2}$ 在 $C_{free}$ ，就把 $q_3$ 加入路线图
在狭窄通道里，$q_1$ 和 $q_2$ 很容易分别落在通道两侧的障碍物内，它们的中点 $q_3$ 恰好被“挤”进通道中央，从而自动填补极难采到的窄缝

![[EAI导论/imgs/img3/image-3.png]]

在实际使用时，同时采用这 3 种采样方法，基本上就能保证所有区域都有采样点覆盖

但是如果障碍物会移动，PRM 表现得就不那么好了

## Rapidly-exploring Random Trees (RRT)

起始时，将起点作为树的根节点，而后进行以下循环
- 在空间中采样一点 $q_{rand}$
- 找到树上离其最近的一点 $q_{near}$
- 从 $q_{near}$ 朝 $q_{rand}$ 前进一定步长，得到新节点 $q_{new}$ ，若整条边无碰撞，就把 $q_{new}$ 加入树中
直到 $q_{new}$ 足够接近目标或达到迭代上限时停止

![[EAI导论/imgs/img3/image-4.png]]

为了加速收敛，可以采用 greedy 的采样策略，但多数情况下表现不是很好，比如起点到终点间有个很宽墙，需要绕过去，一直 greedy 就会频繁产生碰撞，所以需要平衡 exploitation 和 exploration

难点在于，要保证最近点的查找要快速，可采用 KD Tree 等结构
此外，还要设置合适的步长 $\epsilon$ ，若过小，则采样点过多；若过大，一条边上很容易产生碰撞，新节点生成率极低

实际使用时，有一些 trick ，帮助 RRT-Connect
可以同时从 $q_{start}$ 和 $q_{goal}$ 同时扩展树，向对方靠近，而不是完全随机的采样
可以采用多个步长

一个经典的应用是 The Open Motion Planning Library (OMPL)

# Control System

在规划好路径后，需要 control system 来控制移动
一个机器人一般要有 sensor actuator 驱动器 controller 三个部分
其中 controller 的结构如下

![[EAI导论/imgs/img3/image-5.png]]

简化后长这样

![[EAI导论/imgs/img3/image-6.png]]

控制器有开环和闭环两种，还可以将两者结合
FF 的优点是速度快，FB 则可以应对外界扰动

![[EAI导论/imgs/img3/image-7.png]]

## Error Dynamics

误差动力学 Error Dynamics 研究受控系统关节误差 $\theta_e(t)$ 的变化
$$
\theta_e(t) = \theta_d(t) - \theta(t)
$$
误差响应Error response 指的是受控系统在初始条件为 $\theta_e(0) = 1$，且 $\dot{\theta}_e(0) = \ddot{\theta}_e(0) = \cdots = 0$ 下， 误差随时间的变化

典型的误差响应可以通过**瞬态响应**和**稳态响应**进行描述
过程中，可能发生超调 overshoot 现象，即误差响应第一次超过最终稳态值的最大超出量，其公式如下
$$
\text{overshoot} = \left| \frac{\theta_{e,\min} - e_{ss}}{\theta_e(0) - e_{ss}} \right| \times 100\%
$$
当误差波动在一定范围以内时（如 2%），视为进入稳态

![[EAI导论/imgs/img3/image-8.png]]

误差一般可以用线性常微分方差表示，例如以下的 First-order Error Dynamics 一阶误差动态
$$
\dot{\theta}_e(t) + \frac{1}{\tau} \theta_e(t) = 0
$$
这很容易求解，结果是一个指数级衰减，同时可以进一步求得调节时间

![[EAI导论/imgs/img3/image-9.png]]

进一步的，还有二阶误差动态，比如阻尼振动 damped oscillator

![[EAI导论/imgs/img3/image-10.png]]

其解的情况如下，分别对应阻尼过大、临界、过小的情况，当阻尼过小时，会发生超调

![[EAI导论/imgs/img3/image-11.png]]

## PID Controller

根据误差，就可以调节输出来控制运动，比如把速度作为输出，一个简单的想法是让速度与误差成比例，就得到一个基本的控制器 ，**Proportional (P) Control**
$$\dot{\theta}(t) = K_p \theta_e(t)$$
- $\dot{\theta}(t)$ 是系统的输出（在这里是速度）
- $K_p$ 是比例增益系数，是一个常数

假设目标位置 $\theta_d(t)$ 是恒定的，即 Setpoint Control ，那么 P 控制器结果如下

![[EAI导论/imgs/img3/image-12.png]]

如果目标是一个匀速运动 $\theta_d(t) = ct+a$ ，那结果就是这样
$$\theta_e(t) = \frac{c}{K_p} + \left(\theta_e(0) - \frac{c}{K_p}\right)e^{-K_p t}$$
其中有个常数项 $\frac{c}{K_p}$ ，就是稳态误差 steady-state error ，这意味着永远无法达到目标运动，一直都落后一点

需要加一个修正项来进一步消除误差，得到 **Proportional-Integral (PI) Control**
$$\dot{\theta}(t) = K_p \theta_e(t) + K_i \int_{0}^{t} \theta_e(t) \, dt$$
那么上述匀速控制的结果就是这样

![[EAI导论/imgs/img3/image-13.png]]

可见 $K_p$ 与 $K_i$ 的比例关系会影响到控制效果，如果不合适，会产生超调

为了抑制超调，还可以再加一个 derivative (D) 项，得到 PID 控制器
$$\dot{\theta}(t) = K_p \theta_e(t) + K_i \int_{0}^{t} \theta_e(t) \, dt + K_d\dot{\theta_e}(t)$$

P 控制器只能消除定点控制的稳态误差，而 PI 控制器还能消除匀速控制的稳态误差，但不是所有目标轨迹的误差都能消除

![[EAI导论/imgs/img3/image-14.png]]

除了速度，还可以将力/力矩作为输出，考虑下面一个经典情形

![[EAI导论/imgs/img3/image-15.png]]

对于定点控制问题，采用 PD 控制器，结果如下，当 $g \neq 0$ 时会有稳态误差

![[EAI导论/imgs/img3/image-16.png]]

换成 PID 控制器，结果变成一个三阶方程

![[EAI导论/imgs/img3/image-17.png]]

![[EAI导论/imgs/img3/image-18.png]]

一个经验性的调参方法，随着三个比例系数的增大，一些量的变化情况如下

![[EAI导论/imgs/img3/image-19.png]]





