---
date: 2025-04-16
tags:
  - CV
---

# From 2D to 3D

二维图片是一个规则网格 regular grid，每个格子记录颜色或亮度，但不直接告诉我们这个位置离相机多远，三维视觉则进一步关心物体在空间中的位置、形状和结构

双眼看到的同一物体位置不同，称为**视差 disparity**，这是感知远近的重要线索之一，但普通图片并不直接列出物体的三维坐标，这里的三维信息是**隐式 implicit**的

如果直接记录空间坐标、表面或体积，比如建立三维模型，就得到**显式 explicit**的三维表示

计算机视觉可以根据多视角图片中的对应关系估计三维结构，比如后面会涉及的 Structure from Motion，也可以结合已知条件，从图像推断形状
还可以用深度传感器直接感知距离，用激光雷达生成空间点云

有了显式的 3D 信息，就可以很准确的指导行动，比如机械臂、自动驾驶

总的来看，有以下数据形式

![[CV导论/imgs/img8/image.png]]

pixel $\rightarrow$ voxel 体素
这样一来，同一个物体可以用多种 **3D representation（三维表示）**描述，比如体素、网格和点云，下面会分别介绍

Q：深度图是不是完整的 3D？
深度图通常称为 2.5D，因为它给出了当前视角可见表面的深度，但一般没有记录物体背面和被遮挡的结构，结合相机内参，可以把这些可见点还原到三维空间

# Camera Model

拍照过程就是从 3D 到 2D，这可以用数学进行建模

如果直接用一个光屏接收光线，物体一个点发出的光会落在屏上很多位置，不同点的光混在一起，无法形成清晰图像
我们想产生一一对应的效果，考虑用小孔成像的方式

![[CV导论/imgs/img8/image-1.png|645x285]]

这个孔称为**光圈 aperture**，对应的理想模型称为**针孔相机 pinhole camera**，真实成像平面在孔后方，会得到倒像；推导时常把一个虚拟成像平面放在孔前方，方便使用正号表示坐标，下面采用后一种约定

![[CV导论/imgs/img8/image-2.png|651x405]]

将孔的位置设为原点 O，考虑将空间中一点 $P$ 投影到光屏上一点 $P'$
其中 x y 为水平竖直方向，z 为沿着光轴的方向

![[CV导论/imgs/img8/image-3.png|675x259]]

$$
\begin{aligned}
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
\end{aligned}
$$

小孔较大时，同一个物点会在屏上形成一小片模糊区域；缩小孔能减轻这种几何模糊，但透过的光也变少，图像变暗，真实光学中孔过小还会受衍射影响，因此不能无限缩小

可以通过加透镜 lens 解决这个矛盾
在合适的物距、像距下，透镜可以将同一个物点发出的多条光线汇聚到对应像点，同时接收更多光线，兼顾成像亮度和清晰度

实际上只有在一定范围内（景深）的物体成像才是比较清晰的
在景深范围内称为 in focus，对焦清晰，范围外则是 out of focus，产生离焦模糊

![[CV导论/imgs/img8/image-4.png|700x273]]

真实透镜还可能产生**畸变 distortion**，例如原本的直线在图像中变弯，理想针孔模型不包含这些效应，实际标定时可以额外估计畸变参数

![[CV导论/imgs/img8/image-5.png]]

相机模型有两个重要参量

- **内参 intrinsics**：描述相机内部的成像关系，比如焦距、主点位置，固定成像设置下通常视为不变，变焦、对焦或图像缩放等可能改变有效内参
- **外参 extrinsics**：描述相机相对于世界坐标系的位置和朝向

![[CV导论/imgs/img8/image-6.png|698x455]]

## Camera Intrinsics

先考虑成像的过程

![[CV导论/imgs/img8/image-7.png]]

但是，得到的 $x'$ 和 $y'$ 是以光轴在光屏上的投影点为中心的
实际上，图像的原点并不是这个点，还要转换到图像坐标
为此，需要加一个平移项

$$
(x, y, z) \rightarrow \left(f \frac{x}{z} + c_x,\ f \frac{y}{z} + c_y\right)
$$

而且，计算机里的图像是按照像素储存的，还要将坐标转换为像素位置

$$
(x, y, z) \rightarrow \left(\alpha \frac{x}{z} + c_x,\ \beta \frac{y}{z} + c_y\right)
$$
$$
\alpha = fk,\ \beta = fl
$$

其中系数 $k, l$ 单位是 pixel/m ，表示图像传感器在两个方向上每米有多少个像素， $f$ 单位是 m

这就是**透视投影 perspective projection**，直观上，因为除以了深度 $z$，会产生近大远小的效果
这也说明这种变换不是线性的
直接除以 $z$ 不能写成普通的线性变换，但引入齐次坐标后，可以先用矩阵计算，再通过一次除法还原像素坐标

考虑从欧几里得坐标 E 变为齐次坐标 H
欧氏坐标 $\rightarrow$ 齐次坐标

$$
(x, y, z) \implies \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

齐次坐标 $\rightarrow$ 欧氏坐标（最后一个分量 $w\ne0$ 时）

$$
\begin{bmatrix} x \\ y \\ z \\ w \end{bmatrix} \implies (x/w, y/w, z/w)
$$

齐次坐标允许所有分量同时乘同一个非零系数，而表示同一个点，因此可以给图像坐标补上 1，再整体乘 $z$，把分母消掉；原始三维点也补上一个齐次分量

$$
\begin{aligned}
P' = \left(\alpha \frac{x}{z} + c_x, \beta \frac{y}{z} + c_y\right) 
&\to P_h' = (\alpha x + c_x z, \beta y + c_y z, z)\\
\\
P = (x, y, z) &\to P_h = (x, y, z, 1)
\end{aligned}
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

把 M 里边的非零块提取出来，称为 K
这个 K 就是相机的**内参矩阵 camera intrinsics**

$$
K =
\begin{bmatrix}
\alpha & 0 & c_x\\
0 & \beta & c_y\\
0 & 0 & 1
\end{bmatrix}
$$
$$
where\ \alpha = fk,\ \beta = fl
$$

接下来的矩阵公式中，$P$ 是三维点的齐次坐标，$P'$ 是投影后尚未除以第三个分量的图像齐次向量，实际像素坐标还要除回去，简写为

$$
P' = MP = K \begin{bmatrix} I & 0 \end{bmatrix} P
$$

>可见一般来讲计算时先得到的是 $\alpha$ 和 $\beta$ ，即以像素为单位的焦距 $f_X$ 和 $f_Y$

对于 K ，可引入一些修正
比如图像坐标的两个轴不完全正交时，可以加入倾斜项 skew，描述两轴夹角偏离直角的影响

![[CV导论/imgs/img8/image-8.png|555x373]]

这时候内参矩阵就有 5 个自由度

## Camera Extrinsics

前面的 $(x,y,z)$ 是**相机坐标系**中的坐标，以相机中心为原点；而我们还可以在场景中选一个固定的**世界坐标系**，用 $(X,Y,Z)$ 记录物体位置

同一个点在两个坐标系中只是描述方式不同，世界坐标变成相机坐标，可以分为旋转和平移两个过程

$$
R \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} + T = \begin{bmatrix} x \\ y \\ z \end{bmatrix}
$$

为此，引入 3D 平动变换与旋转变换，并用齐次坐标表示

$$
T = 
\begin{bmatrix}
d_x \\
d_y \\
d_z
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
- 上面的矩阵采用右手坐标系中的标准旋转约定，正方向按右手定则判断

用齐次坐标可以把旋转、平移合并为一次矩阵乘法

$$
\begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \\ 1 \end{bmatrix}
$$

或简写为

$$
\begin{bmatrix} R & T \end{bmatrix} \begin{bmatrix} X \\ Y \\ Z \\ 1 \end{bmatrix} = \begin{bmatrix} x \\ y \\ z \end{bmatrix}
$$

则整个过程可以表示为

$$
P' = K \begin{bmatrix} I & 0 \end{bmatrix} P = K \begin{bmatrix} I & 0 \end{bmatrix} 
\begin{bmatrix} R & T \\ 0 & 1 \end{bmatrix}_{4 \times 4} P_w
$$

化简得

$$
P' = K \begin{bmatrix} R & T \end{bmatrix} P_w
$$

- $P_w$ 为世界点的齐次坐标，$P'$ 为图像齐次向量，转成实际像素坐标仍需除以第三个分量
- K 为相机内参
- $\begin{bmatrix} R & T \end{bmatrix}$ 为相机外参

这样的相机叫**投影相机 projective / perspective camera**

## 其它类型的投影相机

### 弱投影相机

当物体到相机距离几乎不变，或者变化相比于到相机的距离很小
![[CV导论/imgs/img8/image-9.png]]
可以近似认为 z 不变，也就是没有了除 $z$ 带来的非线性影响

如果设

$$
M = K\begin{bmatrix} R & T \end{bmatrix}
$$

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

- $A$ 是前两行、前三列的块，综合了内参和旋转的影响
- $\mathbf b$ 与平移等因素有关
- 最后一行决定还原像素坐标时的分母，反映相机坐标深度对投影的影响，这里假设可以将右下角归一化为 1

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
在物体的深度变化相比平均深度很小时，可以把分母近似为常数，将弱透视投影写成以下仿射形式

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

**正交投影 orthographic projection**可以理解为沿同一个方向，把空间点用平行投影线投到成像平面，物体不会因为离相机更远就变小，但仍会有比例缩放、朝向和遮挡的影响，不能认为投影总与物体实际大小相同

![[CV导论/imgs/img8/image-11.png|634x302]]

# Camera Calibration 相机标定

**相机标定 camera calibration**就是估计相机的内参、外参等参数，把图像位置与空间位置的关系确定下来

单张普通图片把不同深度压在同一条视线上，即使知道内外参，也不能凭一个像素唯一恢复三维点，还需要深度、多视图对应关系或其他几何约束，标定是建立这些联系的重要基础

内参与外参一共 5 + (3 + 3) = 11 个未知量
现在要从 2D 图片中得到方程，进而求解出这些未知量

引入**校准架 calibration rig**
校准架的点、线、面在三维空间中的位置和分布是预先已知的
![[CV导论/imgs/img8/image-12.png]]

对于图片中的点 $p_1, \cdots, p_n$ ，我们指定其在校准架中的对应点 $P_1, \cdots, P_n$
这个过程就相当于手动标定图像中每个点在实际世界的坐标
这些点可以用角点检测等获得，一般是比较有特征的点

对于下面直接求一个一般 $3\times4$ 投影矩阵的线性方法，控制点不能全在同一平面；这不是所有标定方法的共同限制，多张不同姿态的平面棋盘图也可以用于其他标定方法
理想针孔模型中，一个像素中心对应从相机中心出发的一条视线，像素具有有限面积时，可以理解为对应一小束光线
K 的 $\alpha$ 和 $\beta$ 控制的是不同像素对应的光线在世界空间里的夹角
再根据图片大小 H W ，可以得到水平、竖直方向上光线的最大夹角
这个范围叫做**视场角 field of view (FOV)**

![[CV导论/imgs/img8/image-20.png]]

如果取的点都在一个平面内，无法判断深度对于投影效果的影响
那就无法准确判断物体的深度 depth 与大小 size

人眼也可以用类似的透视成像模型粗略理解，不过实际视场角还与镜头设计有关，单目相机并不必然比人眼窄，广角、鱼眼相机需要相应的成像和畸变模型

对于 11 个未知量，需要 11 个方程，那就至少要 6 组对应关系
但为了更鲁棒，一般会用更多组点
有噪声时，这些方程一般不能同时精确满足，可以用后面的 SVD 最小二乘求近似解；若还存在错误对应点，可以结合 RANSAC 排除离群点

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

$$
\begin{cases} -u_1 (\mathbf{m}_3 P_1) + \mathbf{m}_1 P_1 = 0 \\-v_1 (\mathbf{m}_3 P_1) + \mathbf{m}_2 P_1 = 0 \\\vdots \\-u_n (\mathbf{m}_3 P_n) + \mathbf{m}_1 P_n = 0 \\-v_n (\mathbf{m}_3 P_n) + \mathbf{m}_2 P_n = 0\end{cases}
$$

化为矩阵形式

$$
\mathbf{P} \mathbf{m} = 0
$$

其中

$$
\mathbf{P} \overset{\text{def}}{=} \begin{pmatrix}P_1^T & 0^T & -u_1 P_1^T \\0^T & P_1^T & -v_1 P_1^T \\\vdots & \vdots & \vdots \\P_n^T & 0^T & -u_n P_n^T \\0^T & P_n^T & -v_n P_n^T\end{pmatrix}_{2n \times 12}
$$
$$
\mathbf{m} \overset{\text{def}}{=} \begin{pmatrix}\mathbf{m}_1^T \\\mathbf{m}_2^T \\\mathbf{m}_3^T\end{pmatrix}_{12 \times 1}
$$

为了防止 $m = 0$ ，指定归一化形式 $||m|| = 1$
这样一来，问题就变成了前面讲过的齐次最小二乘形式：在单位长度约束下，让 $\|\mathbf P\mathbf m\|_2$ 尽量小
采用 SVD，取 V 的最后一列即可解得 $m$ ，而后
![[CV导论/imgs/img8/image-10.png|547x370]]
SVD 得到的投影矩阵只确定到一个整体比例，需要选择尺度，再将其分解为内参和外参，使内参符合标准形式、旋转矩阵满足正交约束
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

之后，需要验证标定的准确性
考虑利用计算得到的内参与外参，将空间中的点投影到图像
这个过程称为**重投影 reprojection**
比较重投影位置与实际观测位置的距离，称为重投影误差，误差越小通常越好，能否接受还要结合图像分辨率、标注精度和任务要求

假设现在有一个棋盘，取如图的绿色校准架，标定棋盘网格的点的世界坐标

![[CV导论/imgs/img8/image-14.png|478x385]]

对于一组不同的图片，假设只考虑棋盘，不考虑背景，那么有两种解释

![[CV导论/imgs/img8/image-15.png|487x330]]

- 相机不动，棋盘动
![[CV导论/imgs/img8/image-16.png|424x335]]

- 相机动，棋盘不动

![[CV导论/imgs/img8/image-17.png|419x319]]

将误差可视化，得到下面这个图（每个颜色代表同一组标定点）

![[CV导论/imgs/img8/image-18.png|495x421]]

- 红色点：误差集中在原点附近
- 黑色点：大部分误差较小，少部分在水平方向偏差较大
- 蓝色点：部分点在竖直方向偏差较大
- 粉色点：误差沿斜方向分布，具有一定方向性

误差分布可以帮助发现问题，但仅凭散点形状不能唯一判断原因，还需要检查角点对应、畸变模型和标定数据

# Depth Image

深度图大小为 H × W × 1

![[CV导论/imgs/img8/image-19.png]]

- 1 通道，每个像素位置记录深度信息

深度有两种

- **ray depth**：光线长度
	- 常用于光线追踪，模拟真实光照
- **z depth**：光线沿 z 轴的长度
	- 常用于**深度缓冲 Depth Buffering**，处理物体遮挡关系

很多视觉任务中的深度图约定记录 z-depth，具体也要看传感器或数据集定义，如果传感器输出的是 ray-depth，需要结合像素视线方向转换

对于图像上的一个点，已知深度和内参后，可以通过**深度反投影 depth back-projection**恢复其三维位置

$$
(u, v) = \left(\alpha \frac{x}{z} + c_x, \beta \frac{y}{z} + c_y\right)
$$

如果记录的是 $z$ 深度，根据投影公式反解 $x,y$，就得到相机坐标系中的三维点

$$
x = z(u - c_x)/\alpha, \quad y = z(v - c_y)/\beta
$$

这样恢复的点只覆盖当前视角看得到的表面，因此仍称为 2.5D；如果要转成世界坐标，还需要应用外参的逆变换

深度传感器通常有有效测距范围，过近或过远可能无法可靠测量，图形渲染中也常设置 near plane 和 far plane 限制处理的深度范围，两者需要结合具体场景区分
