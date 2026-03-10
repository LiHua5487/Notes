
关键词：**6D/9D 表示法**； **Orthogonal Procrustes Problem (OPP)**

---
# Grasping

![[EAI导论/imgs/img4/image.png]]

- Grasp Synthesis 是一个高维的搜索优化问题
- Grasp Pose 包含位置、旋转、关节信息
	- 4-DoF：3D 位置 + 1D 沿重力方向的旋转 “top-down grasping” 
	- 6-DoF：3D 位置 + 3D 旋转

对于一个二指夹爪，其自由度就是 6D 抓取位置 + 1D 夹爪开合程度，共 7D

抓取可以分为 open-loop 和 closed-loop
- Open-loop Grasping：抓取前进行视觉观测，抓取过程中按照规划进行
- Closed-loop Grasping：实时进行观测，抓取过程中能进行调整

开环的抓取就是一个模块化系统，由视觉、运动规划、控制等模块组成；而闭环的就是端到端的

对于开环抓取，如果物体的形状等信息已知，标注了根据位姿的抓取方式，可以直接根据物体位姿进行抓取，比如看到一个倒了的杯子就直接找把手去抓，这需要位姿预测；对于未知物体，可以直接根据其现在的状态预测抓取位姿

# 6D Pose Estimation

![[EAI导论/imgs/img4/image-1.png]]

预测物体的位姿，即预测在相机坐标系下的 R 和 T ，满足

$$
X_{cam} = R \cdot X_{obj} + T
$$

对于已知物体，根据一个RGB图片，若相机内参已知，且各个角度看起来不一样，就能一一对应到位姿 ；根据点云，也能一一对应（对于对称的物体，多种位姿本身就是是等价的）

## Rotation Regression

得到平移信息很容易，那如何根据图片学习预测旋转呢

3D 旋转只有 3 个自由度，但是旋转矩阵有 9 个数字，这使得旋转很难学，考虑其它的旋转表示法，比如欧拉角、轴角、四元数，但是它们又不连续，不能与旋转矩阵一一对应

旋转矩阵本身就是连续的，但是有冗余，把最后一列抹掉，就是**6D表示法**
由于旋转矩阵是正交阵且行列式为 1 ，可以用施密特正交化变成 3×3 的矩阵

$$
f_{GS} \left( 
\begin{bmatrix}
\vert & \vert \\ 
a_1 & a_2 \\ 
\vert & \vert
\end{bmatrix} 
\right) 
= 
\begin{bmatrix}
\vert & \vert & \vert \\ 
b_1 & b_2 & b_3 \\ 
\vert & \vert & \vert
\end{bmatrix}
\quad
b_i = 
\begin{cases} 
Norm(a_1) & \text{if } i = 1 \\ 
Norm( a_2 - (b_1 \cdot a_2) b_1 ) & \text{if } i = 2 \\ 
b_1 \times b_2 & \text{if } i = 3
\end{cases}
$$

采用 6D 表示法后，loss就可以采用L2 loss，即每个元素与ground truth的旋转矩阵 $[c_1,c_2,c_3]$ 中对应位置做差后的平方和

$$
\mathcal{L}_{6D} = \lVert \{{a}_1,{a}_2\} - \{c_1,c_2\} \rVert^2 = \lVert {a}_1 - c_1 \rVert^2 + \lVert {a}_2 - c_2 \rVert^2
$$

但是L2 loss对于每个元素都是平权的，两个向量都被强制要求匹配ground truth，而从 6D 表示用施密特正交化变成旋转矩阵时，是从第一列逐渐变化得到的，相当于第二列的重要性会被削弱，比如第二列加上一个与第一列平行的向量，并不影响结果

---

**9D表示法**解决了上述loss权重的问题，每一个 3×3 矩阵都与一个旋转矩阵一一对应，可以通过 SVD 进行转化，将 $\mathbb{R}^3$ 映射到 $\text{SO}(3)$ 

任意一个 3×3 的矩阵 $\mathbf{M}$ 可以通过 SVD 分解为

$$
\mathbf{M} = \mathbf{U} \mathbf{S} \mathbf{V}^T
$$

- $\mathbf{U}$ 和 $\mathbf{V}$ 是正交矩阵
- $\mathbf{S}$ 是对角矩阵，包含奇异值（非负）

而后将矩阵 $\mathbf{M}$ 转换为满足旋转矩阵性质的正交矩阵 $\hat{\mathbf{R}}$

$$
\hat{\mathbf{R}} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T \quad \mathbf{\Sigma} = 
\begin{bmatrix}
1 & 0 & 0 \\ 
0 & 1 & 0 \\ 
0 & 0 & \det(\mathbf{U} \mathbf{V}^T)
\end{bmatrix}
$$

- $\mathbf{\Sigma}$ 是一个修正后的对角矩阵，用于确保行列式为 1

其loss就可以是直接将预测结果与旋转矩阵计算L2 loss

如果要预测全空间内的旋转，高维表示法会更优；但如果要预测两个时刻间的旋转的变化，轴角、四元数可能更好，因为这个变化一般是微小的，几乎不包含奇点，而且比高维表示法容易学习

## Rotation Fitting

另一种预测物体的位姿的方法，可以先获取观测到的物体上的每个点与物体的 CAD 模型（预先已知）的对应关系，进而求解出 R 与 T

需要先用一个训好的模型对图片进行处理，得到mask，预测图片中的点在不在物体上，而后通过另一个模型预测这些点对应到模型上的坐标
另一方面，可以利用相机内参计算出图片中物体上的点在相机坐标系下的坐标
得到一系列 3D-3D 对应的坐标后，求解 R 与 T

先求解 R ，假设只有旋转没有平移，那求解过程就可以看作通过旋转模型，使其与观测到的姿态尽可能相近，即以下问题，这种问题被称为 **Orthogonal Procrustes Problem**

$$
\hat{\mathbf{A}} = \arg\min_{\mathbf{A} \in \mathbb{R}^{3 \times 3}} \| \mathbf{M}^T - \mathbf{A} \mathbf{N}^T \|_F^2 \quad \text{subject to } \mathbf{A}^T \mathbf{A} = \mathbf{I} \quad \mathbf{M},\mathbf{N} \in \mathbb{R}^{n \times 3}
$$

- 其中 $\|\mathbf{X}\|_F = \sqrt{\text{trace}(\mathbf{X}^\top \mathbf{X})} = \sqrt{\sum_{i,j} x_{ij}^2}$
- $\mathbf{M}$ 和 $\mathbf{N}$ 代表 n 组点组成的矩阵，这些点是相对于各自的几何中心的坐标

对 $\mathbf{M}^T \mathbf{N}$ 进行 SVD ，得到 $\mathbf{M}^T \mathbf{N} = \mathbf{U} \mathbf{D} \mathbf{V}^T$ ，那么

$$
\hat{\mathbf{A}} = \mathbf{U} \mathbf{V}^T
$$

这个 $\hat{\mathbf{A}}$ 行列式不一定是 1 ，还需要进行处理，由于额外要求行列式为 1 ，所以称为 **constrained OPP**

$$
\hat{\mathbf{A}} = \mathbf{U} 
\begin{bmatrix}
1 & 0 & 0 \\
0 & 1 & 0 \\
0 & 0 & \det(\mathbf{U} \mathbf{V}^T)
\end{bmatrix}
\mathbf{V}^T
$$

对于一般情况，求解 R 和 T ，即求解以下问题

$$
\hat{\mathbf{R}}, \hat{\mathbf{T}} = \arg\min_{\mathbf{R} \in SO(3), \mathbf{T} \in \mathbb{R}^{1 \times 3}} \|\mathbf{M}^T - \mathbf{R} \mathbf{N}^T - \mathbf{T}\|_F^2
$$

1. 先转化成相对于几何中心的坐标 $\tilde{\mathbf{M}} = \mathbf{M} - \bar{\mathbf{M}}, \tilde{\mathbf{N}} = \mathbf{N} - \bar{\mathbf{N}}$ 
2. 利用上述方法求解 $\tilde{\mathbf{M}}^T \approx \hat{\mathbf{R}} \tilde{\mathbf{N}}^T$ 得到 $\hat{\mathbf{R}}$
3. $\hat{\mathbf{T}} = \bar{\mathbf{M}}^T - \hat{\mathbf{R}} \bar{\mathbf{N}}^T$

但是 SVD 对于 outliers 很敏感，比如如果有某些点预测误差较大，那整体结果就会差很多，可以使用 RANSAC

上述模型训练，生成ground truth时，可以将模型变换后进行渲染，并得到各个点的对应坐标，从而得到一系列点对

>Instance-Level 的位姿预测一般采用 regression 方法，但仅仅针对特定物体，不适用于新的未知物体，而且需要 CAD 模型
>Category-Level 的一般采用 coordinate 的方法，适用于一个类别的物体，已知深度信息则可以预测 6D Pose + 3D Size






