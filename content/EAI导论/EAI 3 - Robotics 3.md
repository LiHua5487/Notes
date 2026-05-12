
![[EAI导论/imgs/img3/image-20.png]]

- Motion Planning ：给定起始 pose 与目标 pose ，规划出一个可行的路径
- Control ：确保机器人按照这个路径进行运动，输出一系列 actions 

>上述这个 pipeline 是静态的，如果目标位置变化，就需要循环重复这个过程；或者使用端到端方法，不直接规划完整路径，而是走一步看一步

Motion Planning 与 Navigation 的区别：这是高层指令性规划与低层可行性规划的区别，可以把它们看作导航栈中上下游的两个环节

| 特性   | Navigation中的轨迹生成 (Trajectory Generation)         | Motion Planning                                               |
| ---- | ------------------------------------------------ | ------------------------------------------------------------- |
| 核心任务 | “去哪里？”  基于高级语义目标 (如“跟踪那个人”、“去厨房”) 生成一条趋向目标的宏观路径。 | “怎么无碰撞地过去？” 给定一个起点和目标点 (或区域)，在几何/地图约束下找出一条无碰撞、通常满足动力学约束的精确路径。 |
| 输入   | 当前原始传感器数据 (图像/激光雷达) + 语义/语言指令。                   | 地图 (占据栅格地图、代价地图) + 起点和目标点的几何坐标 + 机器人外形/动力学模型。                 |
| 输出   | 一个路径点序列 (宏观意图)。                                  | 一条精确的、可执行的几何路径或动作序列，常用算法如A*、DWA、RRT等。                         |

Path 与 Trajectory 的区别：motion planning 输出的 path 只是一系列位置，trajectory 需要对 path 进行时间参数化，决定什么时间到哪个位置（一些情况，动力学上可能不允许）

# Motion Planning

## Configuration Space

- **工作空间 Workspace** ：真实世界的 3D 空间
	- 在这个空间里做 motion planning ，虽然容易规划出一个路径，但很难保证机器人各个位置都不会发生碰撞
- **构型空间 Configuration Space ($C\text{-space}$)**：空间中的一点表示机器人的关节角度的组合 joint configuration 
	- $C_{free} \subseteq C$ ：能到达的 pose 
	- $C_{obs} \subseteq C$ ：与障碍物发生碰撞的 pose 
	- $C=C_{free}\cup C_{obs}$ 
	- $C\text{-space} \subseteq \mathbb{R}^n$ ，因为每个关节能转的角度是有范围的

任务设定
- 给定 $C_{free}$ ，及其中的起点 $q_{start}$ 和终点 $q_{goal}$ ，要规划出一条路线
- 难点在于，需要躲开障碍物，需要长程视野，而且还是在高维空间进行规划

## Collision Modeling

物体的碰撞建模会影响到学习效果，比如影响到 simulator 里一个路径到底能不能走过去，一个物体到底能不能抓起来
- Visual mesh ：机器人真实的表面
- Collision mesh ：定义机器人在 simulator 里的碰撞箱，通常是一种近似处理，储存在 **URDF (Universal Robot Description Format)** 里

比如下面的图片就是用一堆球形去近似，因为球形之间的碰撞检测很容易，直接比较球心距离与半径长度就行

![[EAI导论/imgs/img3/image-21.png]]

>力控臂：在机械臂上安装力传感器，这样可以避免一些强行移动（比如卡住、怼墙等），能做出调整，具有柔顺性 compliance （这里是主动柔顺，根据传感器信息主动调整）

有时候球形近似的精度并不够，而且可能会导致 link 之间的**自碰撞 self-collision**

**Approximate Convex Decomposition (ACD)** ：用多个 convex shape 近似物体形状
- Convex-Hull ：物体的凸包，检测凸多边形之间的碰撞，比检测非凸的碰撞要容易
- 存在近似精度与检测时间的 trade-off 

## Sample-based Algorithm

直接在高维空间用传统的寻路算法效率太低了，可以进行采样，再尝试把邻近的两个采样点连起来，就得到一条边，只需检查这些边有没有碰撞

Pros
- 概率完备：样本越多，找到解的概率趋近于 1
- 只需知道部分 $C_{free}$ 就能给出解
- 在高维空间容易实现

Cons
- 需要在两个邻近点间规划路径，因为实际连接时需要求解一条满足动力学约束的局部轨迹
- 当 $C_{free}$ 的“连通性”差（如狭窄走廊、迷宫式环境），随机采样很难恰好落在关键通道里，导致收敛慢甚至找不到解
- 不保证最优

### Probabilistic Roadmap Method (PRM)

算法有两个阶段
- Map construction ：构建一个图
	- 在 $C_{free}$ 中随机取点
	- 将每个点与其周围最近的几个点连接，得到一个图
	- 将起终点连接到图里
- Query ：在图上搜索路径
	- 在这个 map 上用 Dijkstra 等算法找到路径

>需要检测连接形成的边是否发生碰撞，得遍历边上的每个点（一般是按固定距离取点），好在这是可以并行的

![[EAI导论/imgs/img3/image.png]]

需要在 $C_{free}$ 中尽可能均匀的采样，但是 $C_{free}$ 形状很不规则，为此可以使用**拒绝采样 rejection sampling**
- 先在 $C$ 内均匀采样，再把落在 $C_{obs}$ 中的去掉
- 但对于下面这种情况，很难保证在狭小通道内也能有足够的采样点

![[EAI导论/imgs/img3/image-1.png]]

**高斯采样 gaussian sampling** ：把采样重点放到障碍边界附近，用来提高在狭窄通道中的连通度
- 先在 $C$ 里均匀采样一点 $q_1$ 
- 再以 $q_1$ 为均值（中心点），在 $\mathcal{N}(q_1,\sigma ^2)$ 采样一点 $q_2$ 
- 若 $q_1$ 在 $C_{free}$ ，$q_2$ 在 $C_{obs}$ 时，就把 $q_1$ 加入路线图

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

缺点
- 有时候离得比较远的两点可以直接连线到达，而不是按照图里连接出的曲折的路径，所以无法保证最优性（可以使用 **Shortcutting** 解决）
- 起终点很近且能直接到达时，PRM 仍需要在整个空间里采样，浪费时间
- 障碍物会移动时，PRM 需要多次建图，表现下降

### Rapidly-exploring Random Trees (RRT)

构建一个树，将起点作为树根，而后进行以下循环
- 在 $C_{free}$ 中采样一点 $q_{rand}$ 
- 找到树上离其最近的一点 $q_{near}$ 
- 从 $q_{near}$ 朝 $q_{rand}$ 前进一定步长，得到新节点 $q_{new}$ ，若整条边无碰撞，就把 $q_{new}$ 加入树中
- 直到 $q_{new}$ 足够接近目标或达到迭代上限时停止

![[EAI导论/imgs/img3/image-4.png]]

为了加速收敛，可以采用 greedy 的采样策略，但多数情况下表现不是很好（比如起点到终点间有个很宽墙，需要绕过去，一直 greedy 就会频繁产生碰撞），所以需要平衡 exploitation 和 exploration 
- 在采样 $q_{rand}$ 这一步，有一定概率直接设为 $q_{goal}$ ，其余情况随机采样

难点
- 要保证最近点的查找要快速，可采用 KD Tree 等结构
- 要设置合适的步长 $\epsilon$ ，若过小，则采样点过多；若过大，一条边上很容易产生碰撞，新节点生成率极低

实际使用时，有一些 trick 
- 可以同时从 $q_{start}$ 和 $q_{goal}$ 同时扩展树，向对方靠近（**RRT-Connect**）
- 可以采用多个步长

一个经典的应用是 The Open Motion Planning Library (OMPL)

# Control System

在规划好路径后，需要 control system 来控制移动，一般有以下组成部分
- Sensor 
- Controller
- Environment / System （actuator 驱动器，physical system）

![[EAI导论/imgs/img3/image-5.png]]

简化后长这样

![[EAI导论/imgs/img3/image-6.png]]

控制器有开环 FF 和闭环 FB 两种，还可以将两者结合，FF 的优点是速度快，FB 则可以应对外界扰动

![[EAI导论/imgs/img3/image-7.png]]

## ~~Error Dynamics~~ (Wasted)

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

$$
\dot{\theta}(t) = K_p \theta_e(t)
$$

- $\dot{\theta}(t)$ 是系统的输出（在这里是速度）
- $K_p$ 是比例增益系数，是一个常数

假设目标位置 $\theta_d(t)$ 是恒定的，即 Setpoint Control ，那么 P 控制器结果如下

![[EAI导论/imgs/img3/image-12.png]]

如果目标是一个匀速运动 $\theta_d(t) = ct+a$ ，那结果就是这样

$$
\theta_e(t) = \frac{c}{K_p} + \left(\theta_e(0) - \frac{c}{K_p}\right)e^{-K_p t}
$$

其中有个常数项 $\frac{c}{K_p}$ ，就是稳态误差 steady-state error ，这意味着永远无法达到目标运动，一直都落后一点
- $K_p$ 较小：稳态误差比较大
- $K_p$ 过大：响应更快，稳态误差更小，但是达到稳态前会有严重的振动

需要加一个修正项来进一步消除误差，得到 **Proportional-Integral (PI) Control**

$$
\dot{\theta}(t) = K_p \theta_e(t) + K_i \int_{0}^{t} \theta_e(t) \, dt
$$

那么上述匀速控制的结果就是这样

![[EAI导论/imgs/img3/image-13.png]]

可见 $K_p$ 与 $K_i$ 的比例关系会影响到控制效果，如果不合适，会产生超调

为了抑制超调，还可以再加一个 derivative (D) 项，得到 **PID 控制器**

$$
\dot{\theta}(t) = K_p \theta_e(t) + K_i \int_{0}^{t} \theta_e(t) \, dt + K_d\dot{\theta_e}(t)
$$

P 控制器只能消除定点控制的稳态误差，而 PI 控制器还能消除匀速控制的稳态误差，但不是所有目标轨迹的误差都能消除

![[EAI导论/imgs/img3/image-14.png]]

---

除了速度，还可以将力/力矩作为输出，考虑下面一个经典情形

![[EAI导论/imgs/img3/image-15.png]]

对于定点控制问题，采用 PD 控制器，结果如下，当 $g \neq 0$ 时会有稳态误差

![[EAI导论/imgs/img3/image-16.png]]

换成 PID 控制器，结果变成一个三阶方程

![[EAI导论/imgs/img3/image-17.png]]

![[EAI导论/imgs/img3/image-18.png]]

一个经验性的调参方法，随着三个比例系数的增大，一些量的变化情况如下

![[EAI导论/imgs/img3/image-19.png]]

>在具身智能里，一般用的是 PD 控制器



