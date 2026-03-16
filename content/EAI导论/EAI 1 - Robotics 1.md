
# Introduction

morphology 形态学
locomotion 机器人的运动能力
dark factory 黑灯工厂，无人化智能工厂

由于希望智能机器人 generalist/general-purpose robot 能做到人类能做的事，所以考虑将其形态设计为 humanoid 人形

Perception-Action Loop：感知环境，通过动作改变环境

单纯通过机器视觉难以完成动作，也不能将感知和动作关联，需要具身智能

grounding：将信息对应到现实世界（比如划出图片中某个物体的边界框就属于 language grounding）

具身智能要结合物理身体学习物理任务，而且需要泛化能力

**kinematics 运动学**：描述物体的运动（如位置、速度），不考虑怎么施力才能这么动
**dynamics 动力学**：通过 control （施加什么样的力、力矩）实现某种运动
articulated object 关节物体

# Rigid Transformation

## 6D Pose

考虑空间中一个物体，设置世界坐标系 $s$ （用 $\mathcal{F}_s$ 代表**坐标架 frame**），在物体身上设置坐标系 $b$ ，已知物体上的某点在 $b$ 下的坐标为 $\mathbf{p}^b$ ，要得到其在 $s$ 下的坐标 $\mathbf{p}^s$ 

假设坐标系 $s$ 通过某种旋转和平移与 $b$ 重合，将这两个变换在 $s$ 下分别用旋转矩阵 $\mathbf{R}^s_{s \to b}$ 和平移向量 $\mathbf{t}^s_{s \to b}$ 表示，则有如下转换关系

$$
\mathbf{p}^s = \mathbf{R}^s_{s \to b} \cdot \mathbf{p}^b + \mathbf{t}^s_{s \to b}
$$

>先把物体放到 $s$ 的坐标原点（无旋转），则该点在 $s$ 下的坐标值就等于 $\mathbf{p}^b$ ，而后把旋转和平移应用上去就行

其中 $(\mathbf{R}^s_{s \to b},\ \mathbf{t}^s_{s \to b})$ 可以看作物体在 $s$ 下的**位姿 pose**（位置 position + 朝向 orientation）
- 刚体运动包含**平移 translation** 和**旋转 rotation**，保持物体自身的距离、手性；不包含镜像、放缩等变化
- 一个 3d 物体的 pose 包含 3d 平移 + 3d 旋转，共 6 **自由度（DoF, degree of freedom）**

若知道坐标系 $b$ 的原点 $o_b$ 在 $s$ 下的坐标 $o_b^s$ ，及其坐标轴 $\mathbf{x}_b,\mathbf{y}_b,\mathbf{z}_b$ 在 $s$ 下的坐标 $\mathbf{x}_b^s,\mathbf{y}_b^s,\mathbf{z}_b^s$ ，就容易得到 $\mathbf{t}^s_{s \to b}$ 和 $\mathbf{R}^s_{s \to b}$ 

![[EAI导论/imgs/img1/image.png]]

>实际上，$\mathbf{R}^s_{s \to b}=\mathbf{R}^b_{s \to b}$ ，但 $\mathbf{t}^s_{s \to b} \neq \mathbf{t}^b_{s \to b}$

## Homogeneous Transformation

但 $x'=Rx+t$ 不是线性变换：假设

$$
\begin{aligned}
\mathbf{p}_1^s = \mathbf{R}_{s \to b}^s \mathbf{p}_1^b + \mathbf{t}_{s \to b}^s \\
\mathbf{p}_2^s = \mathbf{R}_{s \to b}^s \mathbf{p}_2^b + \mathbf{t}_{s \to b}^s
\end{aligned}
$$

易知其不遵守线性变换的定义

$$
\begin{aligned}
&\mathbf{p}_1^s + \mathbf{p}_2^s \neq \mathbf{R}_{s \to b}^s (\mathbf{p}_1^b + \mathbf{p}_2^b) + \mathbf{t}_{s \to b}^s \ ,\text{when} \ \mathbf{t}_{s \to b}^s \neq \mathbf{0} \\
&a \mathbf{p}_1^s \neq \mathbf{R}_{s \to b}^s (a \mathbf{p}_1^b) + \mathbf{t}_{s \to b}^s \ ,\text{when} \ \mathbf{t}_{s \to b}^s \neq \mathbf{0}
\end{aligned}
$$

可见平移项破坏了线性变换的性质，但是可以采用齐次坐标，这样就能表示成线性变换了：在齐次坐标系下，坐标变成

$$
\tilde{p} = \begin{bmatrix}p \\1\end{bmatrix} \in \mathbb{R}^4
$$

上述变换就可以表示为一个矩阵

$$
T_{s \to b}^s = \begin{bmatrix}R_{s \to b}^s & t_{s \to b}^s \\0 & 1\end{bmatrix} \in \mathbb{R}^{4 \times 4}
$$

那坐标变换就可以写成这样

$$
\tilde{p}^s = T_{s \to b}^s\ \tilde{p}^b
$$

这个变换还有以下性质

$$
\begin{aligned}
&\text{Composition Rule :} \quad T_{3 \to 1}^3 = T_{3 \to 2}^3 T_{2 \to 1}^2 \\
&\text{Change of Observer's Frame :} \quad T_{2 \to 1}^2 = (T_{1 \to 2}^1)^{-1}
\end{aligned}
$$

# Multi-Link Rigid-Body Geometry

## Multi-Link Rigid-Body

对于一个多链节物体，有以下组成部分
- **Link 链节**：rigid-body part connected in sequence
	- 链节 0 称为 **base / root link** ，固定在基座上，设置坐标系 $s$ 
	- 最后的链节称为 **end-effector link**，连接着**末端执行器 end effector** （如机械臂的夹子 gripper），设置坐标系 $e$ 
- **Joint 关节**：连接 link ，决定相邻链节的运动的自由度
	- fully-actuated 全驱动：能独立自由运动
	- under-actuated 欠驱动：运动受到其它部分的影响与约束

![[EAI导论/imgs/img1/image-3.png]]

这是一个 **kinematic chain** ，改变后面的关节的 pose 不影响前面的，但改变前面的会影响后面的

## Joint Types

![[EAI导论/imgs/img1/image-1.png]]

![[EAI导论/imgs/img1/image-2.png]]

## Forward Kinematics

假设有这么一个 2DoF 机械臂，第一个关节（蓝→青）是 revolute 的，第二个关节（青→绿）是 prismatic 的

![[EAI导论/imgs/img1/image-4.png]]
>坐标轴颜色 RGB 对应 xyz 轴

可得各个相邻链节的变换如下

$$
\begin{aligned}
T_{0\to1}^0 &= 
\begin{bmatrix}
\cos\theta_1 & -\sin\theta_1 & 0 & -l_2\sin\theta_1 \\
\sin\theta_1 & \cos\theta_1 & 0 & l_2\cos\theta_1 \\
0 & 0 & 1 & l_1 \\
0 & 0 & 0 & 1
\end{bmatrix} \\
T_{1\to2}^1 &= 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & l_3 \\
0 & 0 & 1 & \theta_2 \\
0 & 0 & 0 & 1
\end{bmatrix} \\
T_{2\to3}^2 &= 
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 1 & -l_4 \\
0 & 0 & 0 & 1
\end{bmatrix}
\end{aligned}
$$

那么从 base 到 end effector 的变换就是

$$
\begin{aligned}
T_{0\to3}^0 = T_{0\to1}^0 T_{1\to2}^1 T_{2\to3}^2 &=
\begin{bmatrix}
\cos\theta_1 & -\sin\theta_1 & 0 & -\sin\theta_1(l_2 + l_3) \\
\sin\theta_1 & \cos\theta_1 & 0 & \cos\theta_1(l_2 + l_3) \\
0 & 0 & 1 & l_1 - l_4 + \theta_2 \\
0 & 0 & 0 & 1
\end{bmatrix} \\
&= 
\begin{bmatrix}
R_{s\to e}^s & t_{s\to e}^s \\
0 & 1
\end{bmatrix}
\end{aligned}
$$

比如想计算夹爪的中间点（TCP, tool center position）所处的 pose ，设其在 $e$ 中坐标为 $p^e$ ，就可以直接用 $T_{0 \rightarrow 3}^0\tilde{p}^e$ 计算

---

end-effector 的 pose 有以下的描述方式
- **Joint / Configuration Space**：各个关节的 pose 的组合，又称 qpos（如果每个关节只能 1DoF 旋转，那就是旋转角度的组合）
	- 一个向量空间，每个坐标是一个 pose 的 vector
	- 空间维度对应着机械臂的 DoF
- **Cartesian / Operation Space 笛卡尔空间**：$(\mathbf{R}_{s \to e},\ \mathbf{t}_{s \to e})$ 

那这两个空间怎么转换呢
- **Forward Kinematics (FK)**：从 Joint Space 到 Cartesian Space 
	- 已知每个关节的 pose，求末端的 pose 
	- 把齐次变换矩阵乘起来就行
	- 结果是唯一确定的
- **Inverse Kinematics (IK)**：从 Cartesian Space 到 Joint  Space 
	- 已知末端要到达的 pose，求各个关节的 pose 是啥样的
	- 解不一定唯一，甚至不一定有解（可能到不了）
	- 由于末端 pose 是 6D 的，所以机械臂一般至少要 6DoF；如果机械臂 DoF > 6，往往会有无穷解；自由度太多也不好，一般是 6~7DoF
	- 通常用数值方法求解，需要满足一定条件才有解析解（比如 Pieper's Criterion）

>FK 和 IK 相关的计算函数一般在 URDG (universal robot description file) 里

末端执行器可达的范围称为 reachable space ，即 $\{p \mid \exists \theta, p=T(\theta)\}$ ，一般是不规则的形状；但实际上，当末端执行器处于一个 pose 时，不一定所有方向都能抓到，把实际上能抓到的空间称为 dexterous space 





