---
date: 2025-04-16
tags:
  - CV
---
From 2D to 3D

将 2D 图片是一个 Regular grid ，即二维网格

人眼（双目图）对于 3D 空间的感知基于视差 disparity，是 implicit 隐式的<br>
比如说只能粗略感知远近，但不知道具体距离<br>
多视角图片、全景图对于 3D 空间的反映也是隐式的

而用3D 建模等就是 explicit 显式的

但是对于计算机视觉，可以通过多视角图片等还原 3D 形状 Structure Formation<br>
还可以用深度传感器直接感知距离，用激光雷达生成空间点云

有了显式的 3D 信息，就可以很准确的指导行动，比如机械臂、自动驾驶

总的来看，有以下数据形式

![[CV导论/imgs/img8/image.png]]

pixel $\rightarrow$ voxel 体素<br>
这样以来，对于同一物体，可以有很多的 3D representation

Q: depth 是不是 3D？<br>
A: 其实是 2.5D，涉及到 2D 与 3D 的联系<br>
2D 是 3D的投影，3D 可以从 2D 还原或生成

# Camera Model

拍照过程就是从 3D 到 2D，这可以用数学进行建模

如果直接用一个像屏接收，那物体一个点射出的光会打到很多位置，这就乱了<br>
我们想产生一一对应的效果，考虑用小孔成像的方式

![[CV导论/imgs/img8/image-1.png]]

这个孔就叫 aperture ，这个相机叫 pinhole camera

![[CV导论/imgs/img8/image-2.png]]

将孔的位置设为原点 O，考虑将空间中一点 $P$ 投影到光屏上一点 $P'$ <br>
其中 x y 为水平竖直方向，z 为沿着光轴的方向

![[CV导论/imgs/img8/image-3.png]]


$$
\begin{align}
\mathbf{P} = 
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
\rightarrow
\mathbf{P'} &= 
\begin{bmatrix}
x' \\
y'
\end{bmatrix} \\
\\
\begin{cases}
x' &= f \frac{x}{z} \\
y' &= f \frac{y}{z}
\end{cases}
\end{align}
$$

但实际上，孔的大小会影响成像效果，理论上应该无限小才是完全清晰的<br>
实际上孔如果比较大，会产生模糊 blur<br>
但要是太小了，那透过的光太少了，太暗了

可以通过加透镜 lens 解决这个矛盾<br>
把物体上一个点射出的很多条光线，经过透镜仍然可以汇聚到一个点，<br>
这就保证了亮度，而且一一对应

实际上只有在一定范围内（景深）的物体成像才是比较清晰的<br>
在范围里叫 in foucus，超出范围的叫 out of focus

![[CV导论/imgs/img8/image-4.png]]

如果不是近轴光线，会产生畸变 distortion

![[CV导论/imgs/img8/image-5.png]]


相机模型有两个重要参量
- 内参 Intrinsics ：制造时就定下来了
- 外参 Extrinsics：受拍照时相机的位置、朝向等影响

![[CV导论/imgs/img8/image-6.png]]

## Camera Intrinsics

先考虑成像的过程

![[CV导论/imgs/img8/image-7.png]]

但是，得到的 $x'$ 和 $y'$ 是以光轴在光屏上的投影点为中心的<br>
实际上，图像的原点并不是这个点，还要转换到图像坐标<br>
为此，需要加一个平移项

$$(x, y, z) \rightarrow \left(f \frac{x}{z} + c_x,\ f \frac{y}{z} + c_y\right)$$

而且，计算机里的图像是按照像素储存的，还要将坐标转换为像素位置

$$
(x, y, z) \rightarrow \left(\alpha \frac{x}{z} + c_x,\ \beta \frac{y}{z} + c_y\right) \\
$$

$$\alpha = fk,\ \beta = fl$$

其中系数 $k, l$ 单位是 pixel/m ， $f$ 单位是 m  

称这种变换为**投影变换 projective transformation**<br>
直观感受上，因为除了 $z$ ，会产生近大远小的效果<br>
这也说明这种变换不是线性的<br>
但我们希望是线性的，因为这样就能用矩阵表示

考虑从欧几里得坐标 E 变为齐次坐标 H

欧氏坐标 $\rightarrow$ 齐次坐标

$$(x, y, z) \implies \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

齐次坐标 $\rightarrow$ 欧氏坐标

$$\begin{bmatrix} x \\ y \\ z \\ w \end{bmatrix} \implies (x/w, y/w, z/w)$$

变换后的坐标添加一个占位的齐次项 1 ，再都乘 $z$ ，原始坐标也变成齐次坐标

$$
\begin{align}
P' = \left(\alpha \frac{x}{z} + c_x, \beta \frac{y}{z} + c_y\right) 
&\to P_h' = (\alpha x + c_x z, \beta y + c_y, z)\\
\\
P = (x, y, z) &\to P_h = (x, y, z, 1)
\end{align}
$$

这样这个变换就能用矩阵表示

$$
P_h' = 
\begin{bmatrix}
\alpha x + c_x z \\
\beta y + c_y z \\
z
\end{bmatrix}
=
\begin{bmatrix}
\alpha & 0 & c_x & 0 \\
0 & \beta & c_y & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$

$$
M =
\begin{bmatrix}
\alpha & 0 & c_x & 0 \\
0 & \beta & c_y & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
$$

把 M 里边的非零块提取出来，称为 K<br>
这个 K 就是相机的**内参矩阵 camera intrinsics**

$$
K =
\begin{bmatrix}
\alpha & 0 & c_x\\
0 & \beta & c_y\\
0 & 0 & 1
\end{bmatrix}
$$

$$where\ \alpha = fk,\ \beta = fl$$

则这个变换简写为

$$P' = MP = K \begin{bmatrix} I & 0 \end{bmatrix} P$$

对于 K ，可引入一些修正<br>
比如说如果 film 存在一些偏移导致 x 与 y 轴不平行，就可以这样

![[CV导论/imgs/img8/image-8.png]]

这时候内参矩阵就有 5 个自由度

## Camera Extrinsics

但实际上，上面的 $P = (x,y,z)$ 是在相机坐标系的坐标，而不是实际的世界坐标<br>
世界坐标系 $P_w = (X,Y,Z)$ 到相机坐标系的变换可以分为旋转和平移两个过程

$$R \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + T = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

为此，引入 3D 平动变换与旋转变换，并用齐次坐标表示

$$
T = 
\begin{bmatrix}
d_x \\
d_y \\
T_z
\end{bmatrix}, \quad
\mathbf{P'} \rightarrow
\begin{bmatrix}
I & T \\
0 & 1
\end{bmatrix}_{4 \times 4}
\begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$

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

$$
R = R_x(\alpha) R_y(\beta) R_z(\gamma), \quad
\mathbf{P'} \rightarrow
\begin{bmatrix}
R & 0 \\
0 & 1
\end{bmatrix}_{4 \times 4}
\begin{bmatrix}
x \\
y \\
z \\
1
\end{bmatrix}
$$

- 注：3D 旋转矩阵虽然有 9 个数，但实际上只有 3 个自由度
- 约束：每个行向量长度为 1 ，不同行向量正交
- 上面的旋转矩阵代表逆时针旋转

用齐次坐标可以表示为

$$\begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}$$

或简写为

$$\begin{bmatrix} R & T \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}$$

则整个过程可以表示为

$$
P' = K \begin{bmatrix} I & 0 \end{bmatrix} P = K \begin{bmatrix} I & 0 \end{bmatrix} 
\begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix}_{4 \times 4} P_w
$$

化简得

$$
P' = K \begin{bmatrix} R & T \end{bmatrix} P_w
$$

- $P_w$ 为世界坐标，$P'$ 为图像坐标
- K 为相机内参
- $\begin{bmatrix} R & T \end{bmatrix}$ 为相机外参

这样的相机叫**投影相机 projective / perspective camera**

## 其它类型的投影相机

### 弱投影相机

当物体到相机距离几乎不变，或者变化相比于到相机的距离很小

![[CV导论/imgs/img8/image-9.png]]

可以近似认为 z 不变，也就是没有了除 $z$ 带来的非线性影响

如果设 

$$M = K\begin{bmatrix} R & T \end{bmatrix}$$

对于正常的投影变换，M形如

$$
M = 
\begin{bmatrix} 
A & \mathbf{b} \\ 
\mathbf{v} & 1 
\end{bmatrix} = 
\begin{bmatrix} 
\mathbf{m}_1 \\ 
\mathbf{m}_2 \\ 
\mathbf{m}_3 
\end{bmatrix}
$$

- $A$ 代表旋转的影响
- $b$ 代表平移的影响
- $\mathbf{v}$  反映深度 z 的影响

齐次坐标的表示为

$$
P' = M P_w = 
\begin{bmatrix} 
\mathbf{m}_1 \\ 
\mathbf{m}_2 \\ 
\mathbf{m}_3 
\end{bmatrix} P_w = 
\begin{bmatrix} 
\mathbf{m}_1 P_w \\ 
\mathbf{m}_2 P_w \\ 
\mathbf{m}_3 P_w 
\end{bmatrix}
$$

转换为欧氏坐标

$$
P' \to \left( \frac{\mathbf{m}_1 P_w}{\mathbf{m}_3 P_w}, \frac{\mathbf{m}_2 P_w}{\mathbf{m}_3 P_w} \right)
$$

---

而对于弱投影变换，M形如

$$
M = 
\begin{bmatrix} 
A & \mathbf{b} \\ 
0 & 1 
\end{bmatrix} = 
\begin{bmatrix} 
\mathbf{m}_1 \\ 
\mathbf{m}_2 \\ 
\mathbf{m}_3 
\end{bmatrix} = 
\begin{bmatrix} 
\mathbf{m}_1 \\ 
\mathbf{m}_2 \\ 
0 \ 0 \ 0 \ 1 
\end{bmatrix}
$$

弱投影变换可以写为

$$
P' = M P_w = 
\begin{bmatrix} 
\mathbf{m}_1 \\ 
\mathbf{m}_2 \\ 
\mathbf{m}_3 
\end{bmatrix} P_w = 
\begin{bmatrix} 
\mathbf{m}_1 P_w \\ 
\mathbf{m}_2 P_w \\ 
1 
\end{bmatrix}
$$

转换为欧氏坐标

$$
P' \to \left( \mathbf{m}_1 P_w, \mathbf{m}_2 P_w \right)
$$

### 正交投影

当入射光全是平行光时，称为正交投影<br>
这可以保证投影大小与原物体大小相同

![[CV导论/imgs/img8/image-11.png]]

# Camera Calibration 相机标定

实际应用时，往往需要从一个 2D 图片恢复出 3D 场景<br>
相机内参把 3D 投影到 2D ，如果不知道内参，那就没法恢复到 3D<br>
如果不知道外参，那这就不是实际的位置<br>
如果想从 2D 还原出 3D ，就需要知道内参和外参

内参与外参一共 5 + (3 + 3) = 11 个未知量<br>
现在要从 2D 图片中得到方程，进而求解出这些未知量

引入**校准架 calibration rig**<br>
校准架的点、线、面在三维空间中的位置和分布是预先已知的<br>

![[CV导论/imgs/img8/image-12.png]]

对于图片中的点 $p_1, \cdots, p_n$ ，我们指定其在校准架中的对应点 $P_1, \cdots, P_n$ <br>
这个过程就相当于手动标定图像中每个点在实际世界的坐标<br>
这些点可以用角点检测等获得，一般是比较有特征的点

需要注意的是，$P_i$ 不能全都在一个平面内<br>
图像中的每一个像素经过光线传播，可以看作对应实际世界空间中的一个锥形<br>
K 的 $\alpha$ 和 $\beta$ 控制的是不同像素对应的光线在世界空间里的夹角<br>
再根据图片大小 H W ，可以得到水平、竖直方向上光线的最大夹角<br>
这个范围叫做**视场角 filed-of-view FOV**

![[CV导论/imgs/img8/image-20.png]]

如果取的点都在一个平面内，无法判断深度对于投影效果的影响<br>
那就无法准确判断物体的深度 depth 与大小 size

人眼可以近似为投影相机<br>
单目相机由于要满足近轴光线的近似条件，视场角比人眼小很多

对于 11 个未知量，需要 11 个方程，那就至少要 6 组对应关系<br>
但为了更鲁棒，一般会用更多组点<br>
此时不同方程解可能不完全一样，需要用 RANSAC 等方法求出近似解

假设 $p_i$ 的坐标为

$$
\mathbf{p}_i = 
\begin{bmatrix} 
u_i \\ 
v_i 
\end{bmatrix} = 
\begin{bmatrix} 
\frac{\mathbf{m}_1 P_i}{\mathbf{m}_3 P_i} \\ 
\frac{\mathbf{m}_2 P_i}{\mathbf{m}_3 P_i} 
\end{bmatrix}
$$

得到方程组

$$\begin{cases} -u_1 (\mathbf{m}_3 P_1) + \mathbf{m}_1 P_1 = 0 \\-v_1 (\mathbf{m}_3 P_1) + \mathbf{m}_2 P_1 = 0 \\\vdots \\-u_n (\mathbf{m}_3 P_n) + \mathbf{m}_1 P_n = 0 \\-v_n (\mathbf{m}_3 P_n) + \mathbf{m}_2 P_n = 0\end{cases}$$

化为矩阵形式

$$\mathbf{P} \mathbf{m} = 0$$

其中

$$\mathbf{P} \overset{\text{def}}{=} \begin{pmatrix}P_1^T & 0^T & -u_1 P_1^T \\0^T & P_1^T & -v_1 P_1^T \\\vdots & \vdots & \vdots \\P_n^T & 0^T & -u_n P_n^T \\0^T & P_n^T & -v_n P_n^T\end{pmatrix}_{2n \times 12}$$

$$\mathbf{m} \overset{\text{def}}{=} \begin{pmatrix}\mathbf{m}_1^T \\\mathbf{m}_2^T \\\mathbf{m}_3^T\end{pmatrix}_{12 \times 1}$$

为了防止 $m = 0$ ，指定归一化形式 $||m|| = 1$<br>
这样以来，这个问题就化为了和线性回归类似的形式<br>
采用 SVD，取 V 的最后一列即可解得 $m$ ，而后<br>

![[CV导论/imgs/img8/image-10.png]]

需要将 $\hat M$ 乘一个系数变回没有归一化的形式

![[CV导论/imgs/img8/image-13.png]]

其中

$$
\mathbf{K} =
\begin{pmatrix}
\alpha & -\alpha \cot \theta & c_x \\
0 & \frac{\beta}{\sin \theta} & c_y \\
0 & 0 & 1
\end{pmatrix}
$$

可以解得相机内参与外参

之后，需要验证标定的准确性<br>
考虑利用计算得到的内参与外参，将空间中的点投影到图像<br>
这个过程称为**重投影 reprojection**<br>
比较其结果与实际上的图像中的点的误差（一般认为 1 个像素以内就很好了）

假设现在有一个棋盘，取如图的绿色校准架，标定棋盘网格的点的世界坐标

![[CV导论/imgs/img8/image-14.png]]

对于一组不同的图片，假设只考虑棋盘，不考虑背景，那么有两种解释

![[CV导论/imgs/img8/image-15.png]]

- 相机不动，棋盘动

![[CV导论/imgs/img8/image-16.png]]

- 相机动，棋盘不动

![[CV导论/imgs/img8/image-17.png]]

将误差可视化，得到下面这个图（每个颜色代表同一组标定点）

![[CV导论/imgs/img8/image-18.png]]

- 红色的点：误差在 1 个像素以内，效果很好
- 黑色点：大部分效果很好，少部分在 x 方向偏差较大
	- 校准板位置“极端”或因拍摄失误数据不准，导致点往一边偏得离谱
- 蓝色点： 普遍存在一些误差，少部分在 y 方向偏差较大
	- 相机没对准或点的检测出错，导致点在竖直方向上集体发生漂移
- 粉色点：误差沿着一条斜线分布，且较为明显 
	- 成像发生畸变，或因为校准板的倾斜视角导致多个点散着“跑远”

# Depth Image

深度图大小为 H × W × 1

![[CV导论/imgs/img8/image-19.png]]

- 1 通道，每个像素位置记录深度信息

深度有两种
- **ray depth**：光线长度，常用于光线追踪，模拟真实光照
- **z depth**：光线沿 z 轴的长度，常用于**深度缓冲 Depth Buffering**，处理物体遮挡关系 

深度图记录的是 z depth

Q：为什么深度图是 2.5D<br>
A：对于图像上的一个点

$$(u, v) = \left(\alpha \frac{x}{z} + c_x, \beta \frac{y}{z} + c_y\right)$$

已知 $z$ ，还需要知道相机内参才能还原为 3D

$$x = z(u - c_x)/\alpha, \quad y = z(v - c_y)/\beta$$

这个过程称为depth back projection 深度图反投影

Q：为啥不是无限远<br>
A： z 方向设置 near plane 和 far plane ，过近或者过远都不检测，因为过远或过近都测不准

