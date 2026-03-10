
关键词：**Iterative Closest Point (ICP)**； **Normalized Object Coordinate Space (NOCS)**

---
# Instance-Level 6D Pose Estimation

![[EAI导论/imgs/img5/image.png]]

在 instance-level 中，平移矩阵可以先判断 2D 中心，再借助深度信息得到；而旋转矩阵一般采用 regression 方法得到
旋转任务对于旋转的要求一般没有平动那么严格，因为即便有轻微的旋转误差，东西也能抓起来

![[EAI导论/imgs/img5/image-1.png]]

可以构建出一个 **PoseCNN** ，用于预测图片中各个物体的 pose 

![[EAI导论/imgs/img5/image-2.png]]

但是这个预测可能不太准确，可以利用 ICP 算法进一步优化

## Iterative Closest Point (ICP)

ICP 旨在通过逐步的调整一个点云 P ，使得两个点云 Q 与 P 尽可能对齐
其中 Q 是通过传感器得到的物体现实状态的点云（一般需要多机位）；P 是将 CAD 模型根据预测的 pose 进行调整而得到的点云
$$
\begin{align}
\hat{\mathbf{R}}, \hat{\mathbf{T}} = \arg\min_{\mathbf{R} \in SO(3), \mathbf{T} \in \mathbb{R}^{1 \times 3}} \| \mathbf{Q} - (\mathbf{R}\mathbf{P} + \mathbf{T}) \|_F^2 \\
where\ \mathbf{R} \in SO(3), \mathbf{T} \in \mathbb{R}^3, \mathbf{Q} \in \mathbb{R}^{3 \times n}, \mathbf{P} \in \mathbb{R}^{3 \times n}
\end{align}
$$

1. 求解时，先让两个点云变到中心
$$
\begin{align}
\tilde{\mathbf{P}} = \mathbf{P} - \bar{\mathbf{P}}\\
\tilde{\mathbf{Q}} = \mathbf{Q} - \bar{\mathbf{Q}}
\end{align}
$$
2. 这两个点云目前还没有对应关系，可以利用 Chamfer Distance 的想法，即对于每个 P 中的点，找到 Q 中最近的点，得到一系列对应点
$$
\tilde{\mathbf{p}}_{i,\text{corr}} = \arg\min_{\tilde{\mathbf{q}}_j \in \tilde{\mathbf{Q}}} \| \tilde{\mathbf{q}}_j - \tilde{\mathbf{p}}_i \|_2^2
$$
3. 问题就变成了 constrained OPP 的形式，可求解出 R 
$$
\begin{align}
&\text{Question:}\quad \hat{\mathbf{R}} = \arg\min_{\mathbf{R} \in SO(3)} \| \tilde{\mathbf{P}}_{\text{corr}} - \hat{\mathbf{R}} \tilde{\mathbf{P}} \|_F^2 \\
&\text{Sol:}\quad \mathbf{U}, \mathbf{D}, \mathbf{V}^T = \text{SVD}(\tilde{\mathbf{P}}_{\text{corr}} \tilde{\mathbf{P}}^T) \quad \hat{\mathbf{R}} = \mathbf{U} \operatorname{diag}\{1, 1, \det(\mathbf{U}\mathbf{V}^T)\} \mathbf{V}^T \\
\end{align}
$$
4. 通过做差得到 T 
$$
\hat{\mathbf{T}} = \bar{\mathbf{Q}} - \hat{\mathbf{R}} \bar{\mathbf{P}}
$$
5. 随后，更新 P ，并重复上述过程
$$
\mathbf{P}_{\text{new}} = \hat{\mathbf{R}} \mathbf{P} + \hat{\mathbf{T}}
$$

当 R 和 T 小于阈值，或 loss（可以设为 P 到 Q 的 Chamfer Distance）变化小于阈值，或达到迭代次数时停止
将迭代过程中的 R 和 T 累加到模型预测的结果上，就得到了优化后的 pose

优点
- 简单，无需点云分割或特征提取
- 当对应关系找的好时，有不错的准确性和收敛性
缺点
- 寻找最近对应点的计算成本高（可用降采样密集点云、小样本采样加速）
- 仅考虑点对点距离，没有充分利用点云结构的信息
- 高度依赖于找对应关系的准确程度

---

但是 instance-level 的局限在于，仅仅针对特定的物体，不适用于新的物体，而且需要物体的 CAD 模型

# Category-Level 6D Pose Estimation

对于未知物体，如果要获取其 pose ，得先定义其标准状态下是啥样，可以按照类别进行划分，一种定义是 **Normalized Object Coordinate Space (NOCS)**
- rotation normalization ：将物体的旋转方式统一到一个固定的方向
- translation normalization ：给物体搞一个正方体的边界框，二者中心对齐，把这个框的中心放在坐标原点，各边与坐标轴平行
- scale normalization ：将边界框的边长放缩成 1 

据此，Category-level 的 pose 就是从 NOCS 到 Camera Space 的变换

![[EAI导论/imgs/img5/image-4.png]]

![[EAI导论/imgs/img5/image-5.png]]

一方面，将 RGB 图片输入到神经网络，预测物体上的每个像素在 NOCS 中的坐标，并变成点云；另一方面，将深度图反投影成物体当前状态的点云，把这俩个点云进行 Pose fitting ，得到 6D Pose + 3D Size

![[EAI导论/imgs/img5/image-3.png]]

两个点云应近似满足以下关系
$$
P_{\text{Cam}} = s \cdot R \cdot P_{\text{NOCS}} + T
$$
这个式子可以用 Umeyama 算法进行求解，得到 s R T ，其中 s 代表对角线长度，所以结果总共只是 7D 的，还要利用 NOCS 中各方向的比例进一步分解成 3 个维度上的放缩比例，进而又能得到物体在世界场景下的 bounding box

---

由于手动标注数据很难，可以采用虚拟生成的训练数据

![[EAI导论/imgs/img5/image-6.png]]

Mixed-reality data generation pipeline ：背景是真实的，前景物体是虚拟的
Sim2Real Gap ：虚拟数据和真实情况有一定差距，用虚拟数据训练，在真实世界的表现不一定好
Domain Randomization ：既然虚拟数据不一定是真实的，那就可以生成尽可能多的训练数据，让真实情况成为其子集

对于上面这种数据，由于真实背景和虚拟物体差距很大，导致边界很清晰，但现实中边界不一定这么清晰，所以真实场景中 segmentation 这一步的表现可能很差，这可以采用联合训练结合真实数据去解决

---

即便有了 category-level pose ，也不知道怎么去抓取，不过如果是对于零部件，比如把手按钮，再标注如何去抓取使用，那就能在未知物体上去进行操作

抓取不适合看成 regression 问题，因为一般情况下不只有一个解，更类似于条件生成问题，最终给出一个 distribution 并从中 sample 
早期抓取被看作 detection 问题，检测出若干抓取方式，使用 NMS 保留最优的，这个过程将抓取方式离散成有限解



