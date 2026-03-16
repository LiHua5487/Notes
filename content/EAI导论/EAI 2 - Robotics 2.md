
# SO(3) and SE(3)

一个旋转矩阵 $R$ 应该满足以下条件
- 保持长度：$\lVert Rp\rVert =\lVert p \rVert$ （等价于 $R$ 为正交阵）
- 保持叉乘：$(Rp)\times (Rq)=R(p\times q)$ 

>光有保持长度还不够，还可能是反射变换（除了旋转，还会进行镜像，改变手性），反射变换矩阵是行列式为 -1 的正交阵

这表明 $R$ 是一个正交阵，且行列式为 1 ，这种矩阵就构成了**特殊正交群 Special Orthogonal Group** ，$n$ 维空间下记作 $SO(n)$ 

$$
\text{SO}(n) = \{ R \in \mathbb{R}^{n \times n} \mid R R^T = I, \det(R) = 1\}
$$

- Group 群：对某种运算（比如矩阵乘法）封闭的代数结构（元素相乘后，仍在这个群里）
- Orthogonal 正交的：$R R^T = I$ 
- Special ：$\det(R) = 1$ 

---

如果把旋转和平移放到一起，就得到了**特殊欧几里得群 Special Euclidean Group** 

$$
\text{SE}(3) := \left\{ T = \begin{bmatrix}R & t \\ 0 & 1\end{bmatrix} \mid R \in \text{SO}(3), t \in \mathbb{R}^3\right\}
$$

- Euclidean 欧几里得的：这个群的变换保持了欧几里得空间的几何性质（如距离和角度）

$\text{SE(3)}$ 中的元素就可以用来表示刚体的 pose ，包括平移和旋转，共 6 自由度

# Parameterization of Rotation

一个旋转矩阵里有 9 个数字，但是实际上只有 3 个自由度（因为限制了行 / 列向量是单位向量，且彼此正交），有没有什么办法能用更少的数字表示旋转呢？

## Euler Angle

欧拉角用三个角度来描述物体的旋转，分别代表着物体绕 $x,y,z$ 轴进行的旋转，按照 $z-y-x$ 的顺序进行
- $z$ ：yaw 偏航角
- $y$ ：pitch 俯仰角
- $x$ ：roll 翻滚角

![[EAI导论/imgs/img2/image.png]]

>这里的绕轴旋转，是绕着旋转之后的轴进行的（而非世界坐标的轴），是一种 intrinsic convention ，称为内旋欧拉角

这三种旋转对应的 $\text{SO(3)}$ 中的矩阵如下

$$
\begin{aligned}
R_x(\alpha) &=
\begin{bmatrix}
1 & 0 & 0 \\
0 & \cos\alpha & -\sin\alpha \\
0 & \sin\alpha & \cos\alpha
\end{bmatrix} \\
R_y(\beta) &=
\begin{bmatrix}
\cos\beta & 0 & \sin\beta \\
0 & 1 & 0 \\
-\sin\beta & 0 & \cos\beta
\end{bmatrix} \\
R_z(\gamma) &=
\begin{bmatrix}
\cos\gamma & -\sin\gamma & 0 \\
\sin\gamma & \cos\gamma & 0 \\
0 & 0 & 1
\end{bmatrix} \\
\end{aligned}
$$

- $\beta \in [-\frac{\pi}{2},\frac{\pi}{2}],\ \alpha\ \text{and}\ \gamma \in [0,2\pi)$ 
- 可用 $R = R_x(\alpha) R_y(\beta) R_z(\gamma)$ 表示任意的 3 维旋转变换
- 这里的 $R_x,R_y,R_z$ 是绕世界坐标轴，是外旋欧拉角

但欧拉角存在一些问题
- **万向节锁 gimbal lock**：当 $\beta = \pm \pi /2$ 时，会把 $x$ 轴转到一开始 $z$ 轴的方向，失去一个自由度，导致一种旋转对应多种欧拉角表示，算法可能因多解性陷入混乱
- 一些情况下，旋转空间的连续微小变化无法通过欧拉角的连续微小变化实现，比如插值路径上如果经过万向节锁的位置，这附近的旋转变化可能非常剧烈（因为此时 $\alpha$ 和 $\gamma$ 都在变化，但是表现为同一种旋转，会相互影响）

>像万向节锁这种非常特殊、性质很烂的情况就叫**奇点 singularity**

## Axis-Angle Parameterization

根据欧拉定理，任意旋转都可以表示为绕一个固定轴的旋转 $\text{Rot}(\vec{\omega},\theta)$ 
- $\vec{\omega}$：旋转轴（单位向量）
- $\theta$：旋转角度 

定义**旋转向量** $\vec{\theta}=\vec{\omega}\theta$ 来表示上述旋转，这也被称为**指数坐标 Exponential Coordinate**，相比于旋转矩阵，存储空间更小，同时在一些计算中也更简单

### 罗德里格旋转公式

利用几何知识，可以得到对于一个向量 $x \in \mathbb{R}^3$ ，旋转后变成 

$$
\text{Rot}(\vec{\omega}, \theta)x = x + (\sin \theta)\vec{\omega} \times x + (1 - \cos \theta)\vec{\omega} \times (\vec{\omega} \times x)
$$

这里存在叉乘，但其实可以将叉乘也表示为矩阵乘法
对于一个向量 $\vec{a}$ ，定义 $[\vec{a}]$ 

$$
\vec{a} = 
\begin{bmatrix}
a_1 \\ 
a_2 \\ 
a_3
\end{bmatrix}, 
\quad [\vec{a}] := 
\begin{bmatrix}
0 & -a_3 & a_2 \\ 
a_3 & 0 & -a_1 \\ 
-a_2 & a_1 & 0
\end{bmatrix}
$$

这样叉乘就可以表示为

$$
\vec{a} \times \vec{b} = [\vec{a}]\vec{b}
$$

那上式就变为

$$
\text{Rot}(\vec{\omega}, \theta)x = \{I + [\vec{\omega}]\sin \theta + [\vec{\omega}]^2(1 - \cos \theta)\}x \tag{1}
$$

可以发现，定义的 $[\vec{a}]$ 满足 $[\vec{a}]^3=-[\vec{a}]$ ，结合 $sin$ 和 $cos$ 泰勒展开得

$$
\text{Rot}(\vec{\omega}, \theta)x = (I + \theta[\vec{\omega}] + \frac{\theta^2}{2!}[\vec{\omega}]^2 + \frac{\theta^3}{3!}[\vec{\omega}]^3 + \cdots)x \tag{2}
$$

而这正好是 $e^x$ 的泰勒展开的形式，只不过指数是矩阵，不妨采取以下定义

$$
e^{[\vec{\omega}]\theta} = I + \theta[\vec{\omega}] + \frac{\theta^2}{2!}[\vec{\omega}]^2 + \frac{\theta^3}{3!}[\vec{\omega}]^3 + \cdots
$$

那么就得到了以下结论

$$
\text{Rot}(\vec{\omega}, \theta)x = e^{[\vec{\omega}]\theta}x, \ \forall x \in \mathbb{R}^3
$$

同时，根据 $(1)(2)$ 和上述矩阵幂次的定义，有

$$
e^{[\vec{\omega}]\theta} = I + [\vec{\omega}]\sin\theta + [\vec{\omega}]^2(1 - \cos\theta)
$$

这个式子称为**罗德里格旋转公式**

### 问题

但是这种表述仍然不是一一对应的
- $(\vec{\omega}, \theta)$ 与 $(-\vec{\omega}, -\theta)$ 代表相同的旋转
- $R = I, \theta = 0$ 时 $\vec{\omega}$ 是任意的
- $(\vec{\omega}, \pi)$ 与 $(-\vec{\omega}, \pi)$ 代表相同的旋转，此时 $\operatorname{tr}(R) = -1$ 

如果要求 $\theta \in (0, \pi)$ ，那每个旋转矩阵就唯一对应一个轴角表示

$$
\theta = \arccos\frac{1}{2}[\operatorname{tr}(R) - 1],\  [\vec{\omega}] = \frac{1}{2\sin\theta}(R - R^T)
$$

对于 $\theta = 0$ 或 $\theta = \pi$ 的情况要特殊处理，尤其是 $\theta = \pi$ 处不连续

- 如果要从一个图像直接得到旋转矩阵，就不太适合用轴角表示法，因为对于转角接近 $\pi$ 的情况表现可能不太好
- 如果已知前一时刻的状态，那就只需判断旋转的变化 $\Delta R$ 就可以了，而这个变化量一般较小，适合用轴角表示法

### 旋转的距离

有了轴角表示法，就很容易衡量旋转的大小和距离
- 大小：直接用轴角表示法中的 $\theta$ 代表
- 距离：从 $R_1$ 的 pose 变成 $R_2$ 的 pose 需要的（最小）变动，根据 $\ (R_2 R_1^{-1})R_1 = R_2$ ，可以将距离定义为如下式子

$$
\operatorname{dist}(R_1, R_2) = \theta(R_2 R_1^T) = \arccos \frac{1}{2}[\operatorname{tr}(R_2 R_1^{-1}) - 1]
$$

## Quaternion

四元数是存在三个虚部的复数，形如

$$
q = w + ix + jy + kz \in \mathbb{R}^4
$$

- 其中 $i, j, k$ 是虚数单位，满足 
	- $i^2 = j^2 = k^2 = ijk = -1$ 
	- $\ i \cdot j = k，j \cdot k = i, \ k \cdot i = j$ 
	- $j\cdot i = -i\cdot j$ 
- 向量表示： $q=(w,\vec{v})$ ，其中实部为 $w$ ，虚部为向量 $\vec{v}=(x,y,z)$
- 乘积：$q_1 q_2 = \left(w_1 w_2 - \vec{v}_1^T \vec{v}_2, \, w_1 \vec{v}_2 + w_2 \vec{v}_1 + \vec{v}_1 \times \vec{v}_2 \right)$ （不满足交换律）
- 共轭：$q^* = q_0 - q_1 i - q_2 j - q_3 k = (w,-\vec{v})$ （将旋转轴的方向反向）
- 模长：$||q||^2 = qq^*=q^*q=w^2+||\vec{v}||^2$ 
- 逆：$q^{-1} = \frac{q^*}{\|q\|^2}$ （对于单位四元数，逆 $q^{-1}$ 和共轭 $q^*$ 是等价的）

>实际应用时，可能用 $(w,x,y,z)$ ，也可能用 $(x,y,z,w)$ 表示，要注意顺序

### 四元数下的向量旋转

首先将向量 $\vec{x}$ 扩充为 $x = (0, \vec{x})$ ，那旋转后就变为

$$
x' = qxq^{-1} = qxq^*
$$

对于旋转的组合，只需将四元数相乘，比如先进行旋转 $q_1$ 再进行旋转 $q_2$ 

$$(q_2(q_1 x q_1^*)q_2^*) = (q_2 q_1)x(q_1^* q_2^*)$$

但这种表示也有问题
- **双重覆盖**：$q$ 与 $-q$ 均能表示同一个旋转
- 由于需要单位化，所以会导致梯度相关的问题

### 表示方式转换

可以用单位四元数来表示 3 维旋转，因为四元数自由度是 4 ，但有长度为 1 的限制，所以单位四元数自由度是 3 ，**将单位四元数组成的空间记为 $\mathbb{S}^3$**

轴角表示和四元数的转换
- 轴角到四元数

$$
q = [\cos(\theta/2), \sin(\theta/2)\vec{\omega}]
$$

- 四元数到轴角

$$
\begin{aligned}
\theta &= 2 \arccos(w)\\
\vec{\omega} &= 
\begin{cases} 
\frac{1}{\sin(\theta/2)} \vec{v}, & \theta \neq 0 \\ 
0, & \theta = 0 
\end{cases}
\end{aligned}
$$

>可见四元数是用旋转的半角 $\theta / 2$ 来表示旋转角度的

旋转矩阵与四元数的转换
- 旋转矩阵到四元数：Rotation → Axis-angle → Quaternion
- 四元数到旋转矩阵

$$
R(q) = E(q) G(q)^T, \text{ where } E(q) = [-\vec{v}, wI + [\vec{v}]] \text{ and } G(q) = [-\vec{v}, wI - [\vec{v}]]
$$

### 四元数的角距离

3 维空间中，两个向量的夹角余弦值就是其点积，类似地，定义两个单位四元数的夹角为

$$
<p,q>=\text{arccos}(p\cdot q)
$$

>这里进行点积，是直接把 $q$ 视为一个向量 $(w,x,y,z)$ 

由于四元数用半角表示旋转角度，所以实际相差的夹角要乘 2 ，考虑到双重覆盖问题，应取 $\pm q$ 夹角更小的那一个，可得角距离定义如下

$$
\text{dist}(p,q)=2\text{arccos}(|p\cdot q|)=2\text{min}(<p,q>,<p,-q>)
$$

### 四元数的采样

需要采样一个旋转矩阵，但有以下困难
- 直接在 $\text{SO}(3)$ 上均匀采样很费劲
- 对参数化表示的均匀采样，不一定意味着在 $\text{SO(3)}$ 上是均匀的

不过可以证明，在 $\mathbb{S}^3$ 上对单位四元数均匀采样等价于在 $\text{SO(3)}$ 上均匀采样，因为因为两点之间的旋转距离与单位四元数球面上的距离（角距离）直接成正比
- 随机采样一个四维向量 $(z_1, z_2, z_3, z_4)$ ，代表一个四元数，每个 $z_i$ 是独立的标准正态分布，记作 $\mathbf{z} \sim \mathcal{N}(0, I_{4 \times 4})$ 
- 将其归一化到单位球面 $q = \frac{\mathbf{z}}{\|\mathbf{z}\|}$ 

>由于正态分布没有方向上的偏差，是一种各向同性的分布，所以在高维空间中，可以均匀分布地覆盖球体，进而保证采样的均匀性；如果 $z_i$ 是从 $[-1,1]$ 的均匀分布上的采样，相当于在一个高维立方体上均匀采样，投影到球面会变得不均匀

### 球面线性插值 SLERP

利用四元数，不仅方便采样，还方便插值，这在生成连续轨迹时很有用，但是不能直接用线性插值 LERP 
- 线性插值不能保证四元数还是模长为 1 的，需要额外归一化
- 即便进行了归一化，旋转速度不一定是匀速的
- 线性插值相当于在弦上匀速运动（线速度恒定），但我们需要在圆弧上匀速运动（角速度恒定）

需要进行**球面线性插值 Spherical linear interpolation (SLERP)** ，在球面上两点的**测地弧线 Geodesic**（最短路线，球的大圆的劣弧）之间插值，这用正弦定理很容易推导出公式

![[EAI导论/imgs/img2/image-1.png]]

最后的插值公式如下，路线是从 $q_0$ 到 $q_1$ 

$$
\begin{aligned}
\psi &= \cos^{-1}(q_0 \cdot q_1)  \\
q(t) &= \frac{q_0 \sin((1 - t)\psi) + q_1 \sin(t\psi)}{\sin \psi}
\end{aligned}
$$

具体使用时，会有一些问题
- 当夹角接近 0 时，$\sin$ 值很小，会导致精度问题，此时可以用 LERP 近似（夹角很小时，弧退化成弦）
- 当夹角接近 $\pi$ 时，最短路径有无数条（各个方向的弧），需要手动指定一个
- 由于双重覆盖问题，要保证路径最短，需要 $q_1$ 与 $q_2$ 符号相同

>有时候，我们希望 end-effector 能在两个 pose 间匀速变化，直接在 joint space 中插值只能保证每个关节是匀速变化的，而且也可能存在奇点，此时就需要 SLERP

# Summary

三种表示方式的奇点
- 欧拉角：万向节锁就是一个奇点
- 轴角表示法：$\theta = 0$ 处也是奇点
- 四元数：解决了奇点问题，但存在双重覆盖问题

在欧几里得空间中实现一个单值、无奇点、连续的参数化表示至少需要 5D，这不是说旋转本身有 5 个自由度，而是指在参数化过程中避免拓扑问题时需要额外的维度

|      | 容易求逆? | 容易组合? | SO(3) 局部变动能否用参数局部变化实现 |
| ---- | ----- | ----- | --------------------- |
| 旋转矩阵 | √     | √     | N/A                   |
| 欧拉角  | ×     | ×     | ×                     |
| 轴角表示 | √     | ×     | 多数情况下可以               |
| 四元数  | √     | √     | √                     |

那为啥用四元数，而不是直接用旋转矩阵
- 四元数表示的数字更少，能减少存储空间和计算量
- 旋转矩阵相乘后，不一定是正交阵，需要正交化，但这个过程很麻烦（比如施密特正交化），还会积累误差

应用场景
- 旋转矩阵：进行概念定义
- 欧拉角：将旋转可视化
- 轴角表示法：将旋转可视化，计算导数
- 四元数：进行运算