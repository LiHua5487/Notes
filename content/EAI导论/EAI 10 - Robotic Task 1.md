
# Locomotion

Navigation 解决的是路径规划问题，而 Locomotion 关注的是如何实现某种具体的动作

步态 Gait ：如何走路的模式，一般分为以下几种

![[EAI导论/imgs/img10/image-3.png]]

## Model Predictive Control (MPC)

MPC 通过使用系统的动态模型来预测其未来的行为，并实时计算出一系列最优的控制动作，但只执行第一个动作，然后在下一个时刻重复这个过程，就好比下棋时算几步然后走一步

具体来讲，MPC 定义了一个代价函数，需要求解一个 $\mathbf{u}$ ，在满足约束的前提下，使代价函数最小，其中 $\mathbf{x}(t)$ 是位置状态， $\mathbf{u}(t)$ 是控制输入

$$
\int_{t_s}^{t_f} l(\mathbf{x}(t), \mathbf{u}(t), t) \, dt
$$

需要满足以下约束

$$
\begin{aligned}
&\mathbf{x}(t_s) = \mathbf{x}_s \quad &&\text{Initial Condition} \\
&\dot{\mathbf{x}} = \mathbf{f}(\mathbf{x}, \mathbf{u}, t) \quad &&\text{System Dynamic} \\
&\mathbf{g}(\mathbf{x}, \mathbf{u}, t) = 0 \quad &&\text{Equality Constraints} \\
&\mathbf{h}(\mathbf{x}, \mathbf{u}, t) \geq 0 \quad &&\text{Inequality Constraints}
\end{aligned}
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




