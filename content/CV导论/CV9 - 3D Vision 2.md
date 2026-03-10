---
date: 2025-04-23
tags:
  - CV
---
# Depth Sensor

Stereo Sensors 双目传感器

找到两个图像上的两个点 p 和 p' ，处在一条线上，叫做极线 Epipolar Line<br>
可以通过极线矫正保证这样的点存在

![[CV导论/imgs/img9/image.png]]

![[CV导论/imgs/img9/image-1.png]]

记这两个点位置为 u 和 u' ，定义**视差 disparity** 

$$disparity=u-u'=\frac{Bf}{z}$$

- B 是双目距离，f 是焦距
- $u - u'$ 不是世界空间中的距离，而是投到一个图里，在图像坐标系的距离

双目传感器可以获得双目图，一个左图，一个右图<br>
对于一个图的一个点，根据周围特征，可以找到另一个图里的对应点<br>
通过 $u - u'$ 计算视察，进而计算 z ，得出深度图

但是一些点，可能左图里有，右图里没有<br>
那画出来的深度图，这些点就没有值，得用一些算法补上<br>
都能看到的部分叫 co-visible<br>
不只是最左侧或者最右侧看不到，中间一些物体的一部分侧面可能也不能都看到

对于一些纹理不明显，或者有重复的部分，很难判断对应点关系<br>
比如周围背景都一样，那很难计算准确的视差，得到的深度图也不准

如果看到的东西是经过镜面反射的<br>
镜子上同一点，左右眼看到的可能甚至是不同的东西<br>
这一点局部特征就对不上，那对应点判断也会出问题

类似的，如果是透过透明物体看东西，光线经过折射，也会出问题

在漫反射表面 diffuse surface / Lambertian surface ，纹理明显且独特<br>
而且点要 co-visible ，表现才比较理想

specular 反射 transparent 折射

对于纹理不明显的情况，可以用红外线 IR 打上一个纹理，称为**结构光**<br>
这种方法属于**主动成像 Active Stereo**

![[CV导论/imgs/img9/image-2.png]]

但是如果有外来的红外干扰，比如阳光下，打上的红外光就不明显了<br>
此外，对于黑色情况，吸光多，也不太好

![[CV导论/imgs/img9/image-3.png]]

对于汽车模型，表面反光很强，会有 missing point<br>
对于玻璃瓶，存在折射，表面深度塌陷

双目传感器得到的是 z-depth<br>
对于激光传感器，用光线传播时间 Time-of-Flight 计算得到的深度是 ray-depth<br>
光线射出后存在一部分原路返回的光，所以能这么弄<br>
但是几乎全反射的情况，也没辙

# 3D Representations

## Voxel

![[CV导论/imgs/img9/image-4.png]]

- H×W×D
- 可以进行 index
- 是 geometry representation ，但并不是 surface representation
- 内存与计算开销大

mesh 和 point cloud 可以解决 voxel 的问题<br>
但并不是 regular grid，有时候不能实现对于一个点的周围的快速访问

## Mesh

![[CV导论/imgs/img9/image-5.png]]

- 分段线性 piece-wise Linear 的近似，由一堆三角形/四边形平面构成
- 是 geometry representation ，也是 surface representation

把 mesh 视为 graph
每个三角形的顶点 vertex，边缘 edge

顶点的集合 $V = \{v_1, v_2, \dots, v_n\} \subseteq \mathbb{R}^3$
边缘的集合 $E = \{e_1, e_2, \dots, e_k\} \subseteq V \times V$
面的集合 $F = \{f_1, f_2, \dots, f_m\} \subseteq V \times V \times V$

储存信息
- Geometry
	- 3D coordinates 坐标 
- Topology 
- Attributes 每个面的属性
	- Normal 单位法向量 , color, texture coordinates 
	- Per vertex, face, edge

储存格式

1. STL format 

Triangle List
![[CV导论/imgs/img9/image-6.png]]

Stored information 
- Face: 3 positions

但是没有表面连接的信息

2. OBJ

![[CV导论/imgs/img9/image-7.png]]

Stored information
- Vertex: position 
- Face: vertex indices

约定按照右手定则的顺序存放 vertex ，这样法向量指向表面外侧<br>
法向量如果是反的，直接当表面会有问题，还会出现渲染错误

**Geodesic Distance 测地距离**<br>
在立体表面，从一个点到另一个点的最短距离<br>
将表面展开为平面，画出来不一定是直线<br>
比如对于世界地图，高纬度有失真，图片上区域不能代表实际大小

![[CV导论/imgs/img9/image-8.png]]

对于 mesh ，不一定非得沿着 edge 走<br>
一种近似算法：Fast Marching method

mesh 经常用于可视化，以及生成 ground-truth

## Point Cloud

![[CV导论/imgs/img9/image-9.png]]

- N×3 
	- 不是张量，而是代表集合，因为顺序不重要
- irregular
- a light-weight geometric representation 
	- compact 紧凑 to store 
	- easy to understand and generally easy to build algorithms
- 不是 surface representation

只保留 mesh 的 vertex ，不是正经的点云 

**点云 = 表面 + 采样**

**Uniform Sampling 均匀采样**
1. 计算每个三角面片的面积 <br>
	设三角面片的三个顶点分别为 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$，则该三角面片的面积为

$$A_i = \frac{1}{2} \| (\mathbf{v}_2 - \mathbf{v}_1) \times (\mathbf{v}_3 - \mathbf{v}_1) \|$$

3. 计算每个面片的采样权重<br>
	根据每个三角面片的面积，将其面积占比作为权重

$$P_i = \frac{A_i}{\sum_{j=1}^N A_j}$$

4. 采样三角面片 <br>
	根据概率分布 $P_i$，独立同分布地采样三角面片<br>
	每次采样都在所有面片中进行随机选择<br>
	更大的三角面片上采样的点数在统计意义上会更多

5. 在每个采样的三角面片上进行均匀采样 <br>
	对于所选择的三角面片 $\mathbf{f}_i$，假设其顶点为 $\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3$<br>
	可以补一个对称的三角形变成平行四边形<br>
	在这个平行四边形均匀采样，如果落在补充区域，就对称回原来的三角形

$$\mathbf{p} = (1 - \sqrt{r_1}) \mathbf{v}_1 + \sqrt{r_1}(1 - r_2) \mathbf{v}_2 + \sqrt{r_1} r_2 \mathbf{v}_3$$

- $\mathbf{p}$ 是采样得到的点
- $r_1, r_2 \in [0, 1]$ 是两个独立的随机数，用于均匀采样三角面片

面内的均匀采样：任取一个小的面元，内部采样点占总数的比例趋近于这个面元的面积占整个区域的比例

但整体上看，点云的密度直观上可能不是均匀的

**Farthest Point Sampling (FPS)** <br>
对于采样出来的点，想让两两之间的距离和最大<br>
但这是 NP-hard 的，不过可以近似做到

1. 先用均匀采样采取过量的点，保证没洞
2. 贪心选点
- 先随机取一个点
- 从剩下的点里选出一个点，考虑这个点到所有已选的点的距离的最小值，让这个值最大，逐渐循环，直到选出目标数量个点

![[CV导论/imgs/img9/image-10.png]]

直观上看，FPS 更为均匀，覆盖更全，而均匀采样反而在直观上不均匀<br>
训练时，基于 FPS 采取的点云表现会比均匀采样好一点

对于二维图像，对应像素做差即可比较，但是对于点云怎么比？

**Chamfer distance**
对于点云 A 上的一个点，在另一个点云 B 里找到距离最近的点<br>
如果反过来，两种找法得到的距离不相等

$$
d_{CD}(S_1, S_2) = \sum_{x \in S_1} \min_{y \in S_2} \|x - y\|_2 + \sum_{y \in S_2} \min_{x \in S_1} \|x - y\|_2
$$

这种方法有些点可能会漏掉，并不是两两匹配的

**Earth Mover's distance**
对于点云 A 里的一个点，指定点云 B 里的一个点，形成一堆一一对应的点对，把这些点对的两点距离和加起来，求最小值

$$
d_{EMD}(S_1,S_2)= \min _ {\phi:S_1\rightarrow S_2} \sum _ {x\in S_1} \left \| x- \phi (x) \right \| _2
$$

对于两个点云疏密程度互补的情况，CD 比较小，但 EMD 就比较大，可见 EMD 很依赖点云的取样，CD 则不敏感

# 3D Deep Learning

对于 voxel ，只需要把 2d 情况加个维度就行

对于点云，不能简单地把 N×3 展成 3N 再过 FC，因为点云中的点顺序改变，还是同一个点云，理论上结果应该不变，但 3N 向量中顺序改变会影响 FC 输出

也不能简单地指定排序，比如在点云去掉一个点，从另一个地方加一个点，那展平后的向量中的值就全都窜位了，也不行

想要找到对于排列顺序不变 permutation invariant 的 function，也就是对称函数 Symmetric Function，比如取最大值 max 和直接求和

## PointNet

![[CV导论/imgs/img9/image-11.png]]

每一个点经过 MLP 得到一个 C 维的特征向量，对于所有点的同一个特征维度，取最大值，得到一个 C 维向量，再过 MLP 

![[CV导论/imgs/img9/image-12.png]]

- 输入点云 (n×3)
	每行表示一个点的三维坐标 $(x,y,z)$，总共有 n 个点
- Input Transform
	- 即使输入的点云有不同的旋转或平移，这个模块能够通过 T-Net 学习一个 3×3 的变换矩阵，将点云对齐到共享的参考空间
	- T-Net 的结构
	    - 本质上是一个 PointNet 子网络，输入是 n×3 的点坐标
		- 通过一系列共享的 MLP 提取特征（类似后续流程）
	    - 最终学习一个 3×3 矩阵，与原始输入点云相乘，实现对齐
- Local Embedding 局部特征提取 
	每个特征向量经过一个共享的 MLP ，提取高维特征
- Feature Transform
	类似于 Input Transform ，也是一个 T-Net，但它作用于局部特征 
	在特征空间维度上的对齐，学习了一个 64×64 的矩阵
- Max Pool
	所有 n 个特征向量的同一特征维度上取最大值，得到一个等长向量，代表整个点云的全局特征
- 输出
	将这个向量作为 MLP 的输入，实现分类等任务

对于 segmentation ，由于经过 max pool 压成了一个向量，想要还原回去，还需要前面 local embedding 得到的特征向量，补充局部特征

对于不同点数的输入，都能运行，而且表现降低程度较小，因为取 max 时很多信息都没用，只关注一些关键点，更加鲁棒<br>
经过学习，这些关键点往往是代表框架的点

![[CV导论/imgs/img9/image-13.png]]

PointNet 擅长提取全局特征，但局部特征表现不如 3D CNN <br>
因为 MLP 时每一个点的感受野只是这个点本身，然后取 max 就直接给融了，并没有 3D CNN 那种逐步提取特征

而且，如果将输入的点云平移，表现会差很多，因为是直接对点云坐标处理的

同时，PointNet 中的一些部分被证明是不必要的

## PointNet++

基本的想法是，对于多个局部分别应用 PointNet ，逐层循环

这样就有了 CNN 那样的逐步特征提取，也有了 Local translation invariance

对于一个局部，取一个球形范围，半径称为 **ball query** <br>
每个球包含 k 个点，ball query 和 k 均为超参

![[CV导论/imgs/img9/image-14.png]]

如果球里超过 k 个点，就随机选 k 个<br>
如果不够 k 个点，就将一些点的数据进行复制，这样不会影响到 max pool

对于这个局部，将所有点到球心的相对坐标作为输入，经 PointNet 处理

![[CV导论/imgs/img9/image-15.png]]

先用 FPS 选出一些点作为球心，再经过上述处理，得到了一些高维特征点<br>
其中 X Y 代表欧氏空间的坐标，F 是高维特征向量，把它们拼在一起

![[CV导论/imgs/img9/image-16.png]]

重复多次，得到一些特征点，输入到PointNet进行后续分类处理

对于 segmentation，通过 skip link 把前面的信息传给后面

![[CV导论/imgs/img9/image-17.png]]

对于前面得到的特征图，点的数量有点少，要做 3D 插值

- 对于已有的点，保留其 $C_2$ 维的特征
- 通过 skip link 获得了其它的点的坐标，用反距离加权平均计算特征
- 通过 skip link 获得前面的 $C_1$ 维特征，拼到一起

其中反距离加权平均指的是<br>
给定一个需要插值的点 $P$，和一群已知点（有坐标和特征值的点）集合 $\{P_1, P_2, \dots, P_k\}$，用这些点对 $P$ 进行插值，计算 $P$ 的预测特征 $F(P)$

$$
F(P) = \frac{\sum_{i=1}^k w_i F(P_i)}{\sum_{i=1}^k w_i}
$$

- $F(P_i)$ 表示已知点 $P_i$ 的特征值（例如点云中的特征向量）。
- $w_i$ 表示权重，基于距离的倒数计算的权重
  
$$
w_i = \frac{1}{d(P, P_i)^\alpha}
$$

- $d(P, P_i)$: 插值点 $P$ 和已知点 $P_i$ 之间的距离（通常用欧氏距离）
- $\alpha$: 控制距离对权重影响的参数，通常是正数（$\alpha = 2$ 是常见的设置）
