
关键词：**欧拉角**； **轴角表示法**； 罗德里格旋转公式； **四元数**； **球面线性插值 SLERP**； **奇点 Singularity**

---
# Parameterization of Rotation

如何参数化表示 $\text{SO(n)}$ 中的旋转矩阵呢

先从 1 维情况考虑，容易想到以下两种简单的表示方式
- 单位圆上坐标 $(x,y)$ ，其中 $x^2+y^2=1$
- 相对于极轴的旋转角度 $\theta$ ，其中 $\theta \in [0,2\pi)$

但是这两种表示都有一些缺点，比如说考虑学习过程中处理图片的实际情况，假设输入一个简单的图片，一个圆上面有个点，要输出这个点的旋转表示

第一种表示，不一定保证输出的坐标在单位圆上，所以还得加一步标准化为 $(\frac{x}{\sqrt{x^2+y^2}},\frac{y}{\sqrt{x^2+y^2}})$ ，但是这在反向传播时会增加复杂度，而且当趋近于 $(0,0)$ 时分母很小，非常不稳定，可能导致梯度爆炸

第二种表示比第一种还差，考虑分别输入在 0 附近相对偏上和偏下一点的旋转，这只是一个很小的变化，但前者的输出接近 0 ，后者的输出接近 $2\pi$ ，这个差距是非常大的
因为将角度设在 $[0,2\pi)$ 区间本身就是把连续的旋转变成不连续了（无法从 $2\pi$ 继续逆时针回到 0），两者拓扑结构不同（圆/线段）

## Euler Angle

欧拉角用三个角度来描述物体的旋转，分别代表着物体绕 x y z 轴进行的旋转
按照 z-y-x 的顺序旋转，分别称为 z - yaw 偏航角 y - pitch 俯仰角 x - roll 翻滚角

这三种旋转对应的 $\text{SO(3)}$ 中的矩阵如下
$$
\begin{align}
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
\end{align}
$$
可以设定范围 $\beta \in [-\frac{\pi}{2},\frac{\pi}{2}],\ \alpha\ \text{and}\ \gamma \in [0,2\pi)$  ，其中 $\beta$ 表示俯仰角
可用 $R = R_x(\alpha) R_y(\beta) R_z(\gamma)$ 表示任意的 3 维旋转变换

但欧拉角存在万向锁 gimbal lock 的问题，当一个角度为 ±90° 时，另两个角对应的轴的旋转平面重合，失去一个自由度，导致表示方法不唯一，在优化或控制中，算法可能因多解性陷入混乱（如路径规划中出现跳跃）

而且，一些情况下旋转空间的连续微小变化无法通过欧拉角的连续微小变化实现

## Angle-Axis Parameterization

根据欧拉定理，任意旋转都可以表示为绕一个固定轴的旋转，可得轴角表示法
- 旋转轴为单位向量 $\vec{\omega} \in \mathbb{R}^3$，满足 $\|\vec{\omega}\| = 1$
- 旋转角度 $\theta$ 
将这个旋转记作 $\text{Rot}(\vec{\omega},\theta)$ ，那怎么求其对应的旋转矩阵 $R \in \text{SO(3)}$ 

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
$$\vec{a} \times \vec{b} = [\vec{a}]\vec{b}$$
那上式就变为
$$
\text{Rot}(\vec{\omega}, \theta)x = \{I + [\vec{\omega}]\sin \theta + [\vec{\omega}]^2(1 - \cos \theta)\}x \tag{1}
$$
可以发现，定义的 $[\vec{a}]$ 满足 $[\vec{a}]^3=-[\vec{a}]$ ，结合 $sin$ 和 $cos$ 泰勒展开得
$$\text{Rot}(\vec{\omega}, \theta)x = (I + \theta[\vec{\omega}] + \frac{\theta^2}{2!}[\vec{\omega}]^2 + \frac{\theta^3}{3!}[\vec{\omega}]^3 + \cdots)x \tag{2}$$
而这正好是 $e^x$ 的泰勒展开的形式，只不过指数是矩阵，不妨采取以下定义
$$
e^{[\vec{\omega}]\theta} = I + \theta[\vec{\omega}] + \frac{\theta^2}{2!}[\vec{\omega}]^2 + \frac{\theta^3}{3!}[\vec{\omega}]^3 + \cdots
$$
那么就得到了以下结论
$$\text{Rot}(\vec{\omega}, \theta)x = e^{[\vec{\omega}]\theta}x, \ \forall x \in \mathbb{R}^3$$
同时，根据 $(1)(2)$ 和上述矩阵幂次的定义，有
$$
e^{[\vec{\omega}]\theta} = I + [\vec{\omega}]\sin\theta + [\vec{\omega}]^2(1 - \cos\theta)
$$
这个式子称为**罗德里格旋转公式**

定义**旋转向量** $\vec{\theta}=\vec{\omega}\theta$ 来表示上述旋转，这也被称为**指数坐标**，相比于旋转矩阵，存储空间更小，同时在一些计算中也更简单

---

但是这种表述仍然不是一一对应的，因为
1. $(\vec{\omega}, \theta)$ 与 $(-\vec{\omega}, -\theta)$ 代表相同的旋转
2. $R = I$, $\theta = 0$ 时 $\vec{\omega}$ 是任意的
3. $(\vec{\omega}, \pi)$ 与 $(-\vec{\omega}, \pi)$ 代表相同的旋转，此时 $\operatorname{tr}(R) = -1$

如果要求 $\theta \in (0, \pi)$ ，那每个旋转矩阵就唯一对应一个轴角表示
$$\theta = \arccos\frac{1}{2}[\operatorname{tr}(R) - 1],\  [\vec{\omega}] = \frac{1}{2\sin\theta}(R - R^T)$$
对于 $\theta = 0$ 或 $\theta = \pi$ 的情况要特殊处理，尤其是 $\theta = \pi$ 处不连续

考虑实际情况，如果要从一个图像直接得到旋转矩阵，就不太适合用轴角表示法，因为对于转角接近 $\pi$ 的情况表现可能不太好
但如果已知前一时刻的状态，那就只需判断旋转的变化 $\Delta R$ 就可以了，而这个变化量一般较小，适合用轴角表示法

---

那怎么衡量两个旋转的差距，即定义两个旋转矩阵的距离
可以定义为从 $R_1$ 的 pose 变成 $R_2$ 的 pose 需要的（最小）变动
根据 $\ (R_2 R_1^T)R_1 = R_2$ ，可以将距离定义为如下式子
$$\operatorname{dist}(R_1, R_2) = \theta(R_2 R_1^T) = \arccos \frac{1}{2}[\operatorname{tr}(R_2 R_1^T) - 1]$$

## Quaternion

1. 四元数是存在三个虚部的复数，形如
$$q = w + ix + jy + kz \in \mathbb{R}^4$$
- 其中 $i, j, k$ 是虚数单位，满足 
$$i^2 = j^2 = k^2 = ijk = -1,\ i \cdot j = k，j \cdot k = i, \ k \cdot i = j$$
- 同时 $j\cdot i = -k$ ，其余同理

- 也可表示为 $q=(w,\vec{v})$ ，其中实部为 $w$ ，虚部为向量 $\vec{v}=(x,y,z)$
- 实际应用时，可能用 $(w,x,y,z)$ ，也可能用 $(x,y,z,w)$ 表示，要注意顺序

- 对于 $q_1 = (w_1, \vec{v}_1)$ 与 $q_2 = (w_2, \vec{v}_2)$ ，其乘积如下  
$$q_1 q_2 = \left(w_1 w_2 - \vec{v}_1^T \vec{v}_2, \, w_1 \vec{v}_2 + w_2 \vec{v}_1 + \vec{v}_1 \times \vec{v}_2 \right)$$
- 注意 $\vec{v}_1 \times \vec{v}_2 \neq \vec{v}_2 \times \vec{v}_1$

2. 四元数的共轭 $q^*$ 定义为，若 $q = q_0 + q_1 i + q_2 j + q_3 k$，则其共轭为
$$q^* = q_0 - q_1 i - q_2 j - q_3 k = (w,-\vec{v})$$
- 几何意义是将旋转轴的方向反向

3. 四元数的模长 $||q||^2 = qq^*=q^*q=w^2+||\vec{v}||^2$

4. 四元数的逆 $q^{-1}$ 定义为
$$q^{-1} = \frac{q^*}{\|q\|^2}$$
- 对于单位四元数，逆 $q^{-1}$ 和共轭 $q^*$ 是等价的

---

可以用单位四元数来表示 3 维旋转，因为四元数自由度是 4 ，但有长度为 1 的限制，所以单位四元数自由度是 3 ，将单位四元数组成的空间记为 $\mathbb{S}^3$

轴角表示和四元数相互转换关系如下
Exponential coordinate → Quaternion
$$q = [\cos(\theta/2), \sin(\theta/2)\vec{\omega}]$$
Exponential coordinate ← Quaternion
$$
\begin{align}
\theta &= 2 \arccos(w)\\
\vec{\omega} &= 
\begin{cases} 
\frac{1}{\sin(\theta/2)} \vec{v}, & \theta \neq 0 \\ 
0, & \theta = 0 
\end{cases}
\end{align}
$$
注意到，四元数是用旋转的半角 $\theta / 2$ 来表示旋转角度的

旋转矩阵与四元数的转换
Rotation ← Quaternion  
$$R(q) = E(q) G(q)^T$$
- where $E(q) = [-\vec{v}, wI + [\vec{v}]]$ and $G(q) = [-\vec{v}, wI - [\vec{v}]]$  

Rotation → Quaternion  
- Rotation → Axis-angle → Quaternion

---

四元数表示下，向量的旋转如下
假设向量 $\vec{x}$ 与单位四元数 $q$ ，首先将 $\vec{x}$ 扩充为 $x = (0, \vec{x})$ ，那旋转后就变为
$$x' = qxq^{-1} = qxq^*$$
对于多次旋转的情况，只需将四元数相乘
由上，$(q_2(q_1 x q_1^*)q_2^*)$ 表示先进行旋转 $q_1$ 再进行旋转 $q_2$
而 $(q_2(q_1 x q_1^*)q_2^*) = (q_2 q_1)x(q_1^* q_2^*)$ ，所以连续旋转等价于四元数连乘

但这种表示也有问题，尽管一个单位四元数唯一对应一个旋转，但对于一个旋转，$q$ 与 $-q$ 均能表示这个旋转，如果直接假设选择一侧作为对应，会导致在交界处不连续
同时，由于需要单位化，所以会导致梯度相关的问题

---

下面定义两个单位四元数之间的角距离
3 维空间中，两个向量的夹角余弦值就是其点积，类似地，定义两个单位四元数的夹角为
$$<p,q>=\text{arccos}(p\cdot q)$$
由于四元数用半角表示旋转角度，所以实际相差的夹角要乘 2 ，同时，考虑到一个旋转对应 $q$ 和 $-q$ ，实际求角距离时应取更小的那一个，故角距离定义如下
$$\text{dist}(p,q)=2\text{arccos}(|p\cdot q|)=2\text{min}(<p,q>,<p,-q>)$$

---

有些时候，需要在 $\text{SO(3)}$ 均匀采样，但直接对矩阵进行采样很费劲，考虑对其参数化表示进行采样，但这又有一个问题，参数化表示的均匀采样不一定意味着在 $\text{SO(3)}$ 上是均匀的
不过可以证明，在 $\mathbb{S}^3$ 上对单位四元数均匀采样等价于在 $\text{SO(3)}$ 上均匀采样，因为因为两点之间的旋转距离与四元数球面上的距离直接成正比

在 $\mathbb{S}^3$ 采样时，先随机采样一个四维向量 $(z_1, z_2, z_3, z_4)$ ，代表一个四元数，每个 $z_i$ 是独立的标准正态分布，记作
$$\mathbf{z} \sim \mathcal{N}(0, I_{4 \times 4})$$
由于正态分布没有方向上的偏差，即是一种各向同性的分布，所以在高维空间中，可以均匀分布地覆盖球体，进而保证采样的均匀性

而后，将其归一化向量到单位球面，使其模长为 1
$$q = \frac{\mathbf{z}}{\|\mathbf{z}\|}$$
这样归一化后的四元数 $q$ 将位于 $\mathbb{S}^3$ 上，形成均匀分布

---

利用四元数，不仅方便采样，还方便插值，这在生成连续轨迹时很有用

![[EAI导论/imgs/img2/image.png|592x285]]

这不能直接用线性插值 LERP ，因为线性插值后不能保证四元数还是模长为 1 的，需要额外归一化，但即便进行了归一化，仍然存在旋转速度不恒定的问题

需要进行**球面线性插值 Spherical linear interpolation (SLERP)** ，在球面上两点的**测地弧线 Geodesic**（最短路线，球的优弧）之间插值，这用正弦定理很容易推导出公式

![[EAI导论/imgs/img2/image-1.png]]

最后的插值公式如下，路线是从 $q_0$ 到 $q_1$ 
$$
\begin{align}
\psi &= \cos^{-1}(q_0 \cdot q_1)  \\
q(t) &= \frac{q_0 \sin((1 - t)\psi) + q_1 \sin(t\psi)}{\sin \psi}
\end{align}
$$

具体使用时，会有一些问题
当夹角接近 0 时，$sin$ 值很小，会导致精度问题，此时可以采用 LERP 近似
当夹角接近 $\pi$ 时，最短路径有无数条（各个方向的优弧），需要手动指定一个
由于四元数的双重覆盖问题，要保证路径最短，需要 $q_1$ 与 $q_2$ 符号相同

## Problems and Summary

在参数化表示时，常常会遇到**奇点 Singularity** 的问题
数学中，奇点指的是没有定义，或者性质不好（如不可导）的点，而在旋转的参数化空间，奇点处失去自由度或导数不连续（无法光滑地描述旋转变化）
即在奇点处，表示方法失效，要么无法表示某些旋转，要么相邻旋转的微小变化导致参数变化剧烈

在欧拉角中，万向锁就是一个奇点，而对于轴角表示法，$\theta = 0$ 处也是奇点
四元数解决了奇点问题，但存在双重覆盖问题（ $q$ 和 $-q$ ）

奇点会引发梯度问题，还会导致对应关系不唯一等问题，从而使模型学习效果变差，这不是单纯通过调整 loss 能解决的

此外，这三种表示方法均存在不连续的问题

欧拉角的不连续与直接将 2 维旋转用旋转角度 $\theta \in [0,2\pi)$ 表示类似，都是改变了拓扑结构
轴角表示法在 $\theta = 0$ 和 $\theta = \pi$ 处不连续，前者对任意转轴，旋转矩阵恒为 I ，后者转轴与其反向代表的旋转相同
四元数在关于 $q$ 和 $-q$ 的处理时，强行指定保留一侧会在分界面附近出问题

![[EAI导论/imgs/img2/image-2.png]]

在欧几里得空间中实现一个单值、无奇点、连续的参数化表示至少需要 5D，这不是说旋转本身有 5 个自由度，而是指在参数化过程中避免拓扑问题时需要额外的维度

|      | 容易求逆? | 容易组合? | SO(3) 局部变动能否用参数局部变化实现 |
| ---- | ----- | ----- | --------------------- |
| 旋转矩阵 | √     | √     | N/A                   |
| 欧拉角  | ×     | ×     | ×                     |
| 轴角表示 | √     | ×     | 多数情况下可以               |
| 四元数  | √     | √     | √                     |


