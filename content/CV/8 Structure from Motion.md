
# SFM

回顾一下
- 相机标定：已知图像上点的 2D 坐标及其对应的 3D 坐标，求解出相机参数
- 三角测量：已知相机参数、两个图像上对应点的 2D 坐标，求解出这个点的 3D 坐标
- SFM：已知一个点在很多照片中的 2D 坐标，求解出相机参数、3D坐标

![[CV/img/img8/image.png]]

## Ambiguity

但是 SFM 的解不一定是唯一的，比如对于下图中的 Necker cube （立方体的二维线框图），它没有任何深度信息，可以有多种解读方式（顶部朝前 / 底部朝前）

![[CV/img/img8/image-1.png]]

从数学的角度来看，由于我们只知道一堆 2D 坐标，没有更多的信息，如果对 $X$ 应用一个变换 $Q_{4\times 4}$ ，并把相机参数对应的投影矩阵 $P$ 调整为 $PQ^{-1}$ ，仍然可以得到相同的 2D 坐标，这种情况称为 **Projective Ambiguity** 

$$x \simeq PX = (P Q^{-1})(Q X)$$

如果直接求解，还原出的 3D 场景可能很离谱

![[CV/img/img8/image-2.png]]

考虑给 $Q$ 添加一些约束，比如平行性约束（平行线变换后仍然平行），那 $Q$ 就应该长这样，这种情况称为 **Affine Ambiguity** 

$$Q_A = \begin{bmatrix} A & t \\ 0^T & 1 \end{bmatrix}$$

- $A$ 为一个 3×3 的满秩矩阵，对应一个仿射变换
- $t$ 为一个平移变换

![[CV/img/img8/image-3.png]]

再严格一点，添加正交约束（相互垂直的线变换后仍然垂直），那 $Q$ 就长这样，这种情况称为 **Similarity Ambiguity** ，这对应使用投影相机拍摄下的约束

$$Q_S = \begin{bmatrix} sR & t \\ 0^T & 1 \end{bmatrix}$$

- $R$ 是一个旋转矩阵，$s$ 是缩放因子，$t$ 是平移变换

![[CV/img/img8/image-4.png]]

## Affine SFM

先从 Affine Ambiguity 的情况考虑，这对应着使用弱投影相机拍摄下的约束

弱投影相机：假设场景中的物体离相机较远，深度变化较小，此时投影被近似为缩放和平移操作，即一个仿射变换

![[CV/img/img8/image-5.png]]

从数学的角度去看，在透视投影中，投影点 $(x,y)$ 通常遵循如下关系

$$x = f \frac{X}{Z}, \quad y = f \frac{Y}{Z}$$

然而在弱投影下，我们把深度近似为一个常量 $Z \approx Z_\text{avg}$ ​，则投影关系变为

$$x \approx f \frac{X}{Z_\text{avg}}, \quad y \approx f \frac{Y}{Z_\text{avg}}$$

这等价于对点 $(X,Y,Z)$ 进行仿射变换，如果把 $\frac{f}{Z_\text{avg}}$ 视为一个常数，那这个变换正好与正交投影类似（直接把物体垂直投影到成像平面上，丢弃了深度 $z$ ）

$$\begin{pmatrix}
x \\
y \\
1
\end{pmatrix}
=
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{pmatrix}
x \\
y \\
z \\
1
\end{pmatrix}
$$

那么相机的投影矩阵 $P$ 可以简化为

$$P = \begin{bmatrix}
K_{2D} & t_{2D} \\
0 & 1
\end{bmatrix}
\begin{bmatrix}
1 & 0 & 0 & 0 \\
0 & 1 & 0 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
R_{3D} & t_{3D} \\
0 & 1
\end{bmatrix} \quad \Rightarrow \quad P = \begin{bmatrix}
a_{11} & a_{12} & a_{13} & t_{1} \\
a_{21} & a_{22} & a_{23} & t_{2} \\
0 & 0 & 0 & 1
\end{bmatrix}$$

则坐标变换在笛卡尔坐标下就可以表示为

$$\begin{pmatrix}
x \\
y
\end{pmatrix}
=
\begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23}
\end{bmatrix}
\begin{pmatrix}
X \\
Y \\
Z
\end{pmatrix}
+
\begin{pmatrix}
t_{1} \\
t_{2}
\end{pmatrix}
= AX + t \quad $$

其中 $t$ 就对应着世界坐标系原点在图像中的坐标，于是SFM 的问题就变成了

![[CV/img/img8/image-6.png]]

假设有 $m$ 个图片，知道 $n$ 个点在这些图片上的坐标分别是多少，那总共有 $2mn$ 个已知量 $(x_{ij})$ ，需要求 $8m + 3n$ 个未知量 $(A_{i}, t_{i}, \text{ and } X_{j})$ ，为了能求解这个方程组，需要 $2mn \geq 8m + 3n - 12$ （对于 2 个相机，就至少需要 4 个点）

>这里 -12 是因为考虑到 $Q$ 引入的不确定性，而 Affine Ambiguity 中 $Q$ 为仿射变换，有 12 个自由度

为了简化方程，把一个图像上的 $n$ 个点中心化，消去平移向量

$$
\begin{align}
\hat{x}_{ij} & = x_{ij} - \frac{1}{n} \sum_{k=1}^n x_{ik} \\
& = A_i X_j + t_i - \frac{1}{n} \sum_{k=1}^n \left( A_i X_k + t_i \right) \\
& = A_i \left( X_j - \frac{1}{n} \sum_{k=1}^n X_k \right) \\
& = A_i \hat{X}_j
\end{align}
$$

同时把世界坐标原点设在这 $n$ 个点的 3D 坐标的中心，那么 $\hat{X}_j=X_j$ ，于是方程变为

$$\hat x_{ij}=A_iX_j$$

把所有的方程合并成矩阵形式

$$
\underbrace{\begin{bmatrix}
\hat{x}_{11} & \hat{x}_{12} & \cdots & \hat{x}_{1n} \\
\hat{x}_{21} & \hat{x}_{22} & \cdots & \hat{x}_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
\hat{x}_{m1} & \hat{x}_{m2} & \cdots & \hat{x}_{mn}
\end{bmatrix}}_{D_{2m\times n}} = 
\underbrace{\begin{bmatrix}
A_1 \\
A_2 \\
\vdots \\
A_m
\end{bmatrix}}_{M_{2m\times 3}}
\underbrace{\begin{bmatrix}
X_1 & \cdots & X_n
\end{bmatrix}}_{S_{3\times n}}
$$

注意这里 $\hat x_{ij}=A_iX_j$ 是 2D 的，所以 $D$ 一共有 $2m$ 行，$M$ 代表着一系列相机参数，$S$ 就是还原出的 3D 坐标 

考虑 $D$ 的秩，根据矩阵乘积的秩的规律，有

$$\text{rank}(D) \leq \min(\text{rank}(M), \text{rank}(S))$$

而由于 $M$ 和 $S$ 的形状，可得这俩矩阵的秩不超过 3 ，则 $D$ 的秩不超过 3 ，于是我们将 $D$ 进行 SVD 分解，并取出前 3 行/列

![[CV/img/img8/image-7.png]]

那么 $M$ 和 $S$ 就是

$$M = U_3 \Sigma^{\frac{1}{2}}_3, \quad S = \Sigma^{\frac{1}{2}}_3 V^T_3$$

但是由于 Affine Ambiguity ，这里得到的解是仿射解，我们希望得到的是欧几里得解（比如拍摄了一个正方体，直接得到的仿射解是一个平行六面体，而希望得到的欧几里得解是一个立方体）

所以我们给 $M$ 和 $S$ 添加一个 $Q$ ，使得添加 $Q$ 后的 $M'$ 和 $S'$ 是欧几里得解

$$
D = 
\begin{bmatrix}
A_1 Q \\
A_2 Q \\
\vdots \\
A_m Q
\end{bmatrix}
\begin{bmatrix}
Q^{-1} X_1 & Q^{-1} X_2 & \cdots & Q^{-1} X_n
\end{bmatrix}
$$

>这里的 $Q$ 不是 Affine Ambiguity 中的 $Q_A$ ，而是对应着 $Q_A$ 中的 $A$ ，需要矫正 $A$ 带来的仿射影响

这要求 $A_iQ$ 是一个正交矩阵，即

$$(A_iQ)(A_iQ)^T=A_i(QQ^T)A_i^T=I_{2\times 2}$$

直接用这个方程求解 $Q$ 比较困难，因为其关于 $Q$ 是非线性的，可以定义 $N=QQ^T$ ，求解 $N$ ，这就是线性的了，可以用最小二乘法求解，需要把这个方程转换为 $Hx=b$ 的形式

由于 $N$ 是对称阵，可以假设

$$
N = 
\begin{bmatrix}
n_{11} & n_{12} & n_{13} \\
n_{12} & n_{22} & n_{23} \\
n_{13} & n_{23} & n_{33}
\end{bmatrix}
$$

而后把 $A_iNA_i^T=I$ 展开，再把展开后的方程写为 $Hx=b$ 的形式求解

得到 $N$ 后，需要进一步解出 $Q$ ，这可以使用 Cholesky decomposition （将一个正定对称阵分解为下三角矩阵右乘其转置 $A=LL^T$ ）

最后，只需把 $M$ 和 $S$ 用 $Q$ 更新一下

$$M\leftarrow MQ,\quad S\leftarrow Q^{-1}S$$

## Incremental SFM

但是上述推导是假设空间中一个点在每个图片中都有投影，实际上由于遮挡等原因，不一定每个图片都有这个点，$D$ 矩阵类似于这样

```
.    P1 P2 P3 P4 P5
C1 [ ✓  ✓  -  ✓  ✓ ]
C2 [ ✓  ✓  ✓  ✓  - ]
C3 [ ✓  ✓  ✓  ✓  - ]
C4 [ ✓  -  ✓  ✓  ✓ ]

P:point  C:camera
✓:observed  -:occluded
```

一个简单的想法是，找到其中所有全是 `✓` 的子块，分别进行求解，再把它们综合一下，但这个过程是 NP-complete 的，即随着问题规模增大，计算时间会爆炸性增长

Incremental bilinear refinement 的想法是，先找到一个全是 `✓` 的子块，比如对于上面那个矩阵，可以找到

```
.    P1 P2 P4
C1 [ ✓  ✓  ✓ ]
C2 [ ✓  ✓  ✓ ]
C3 [ ✓  ✓  ✓ ]
```

用之前的方法可以求解出相机参数 $C_1$ $C_2$ 和 3D 坐标 $P_1$ $P_2$ $P_4$ ，然后我们考虑添加一个点，用 triangulation 计算其 3D 坐标

考虑添加 $P_5$ ，但是在已知的相机中，只有 $C_1$ 观测到了，而三角测量要求至少 2 个图像上的 2D 坐标才能求解，所以跳过

考虑添加 $P_3$ ，我们知道它在 $C_2$ 和 $C_3$ 中的 2D 坐标，又解得了它们的相机参数，就可以求解出其 3D 坐标

```
.    P1 P2 P3 P4
C1 [ ✓  ✓  -  ✓ ]
C2 [ ✓  ✓  ✓  ✓ ]
C3 [ ✓  ✓  ✓  ✓ ]
```

而后考虑添加一个相机，并用 calibration 计算其相机参数，比如添加 $C_4$ ，已知其观测到的 $P_1$ $P_3$ $P_4$ 的 2D 坐标，又求解出了它们的 3D 坐标，就能得到其相机参数（根据场景不同，可能需要已知更多的点才能求解，这里只是个例子）

```
.    P1 P2 P3 P4
C1 [ ✓  ✓  -  ✓ ]
C2 [ ✓  ✓  ✓  ✓ ]
C3 [ ✓  ✓  ✓  ✓ ]
C4 [ ✓  -  ✓  ✓ ]
```

而后再添加一个点，由于求出了 $C_4$ 的参数，这下可以添加 $P_5$ 了，之后交替的添加相机与点

## Projective SFM

问题的描述与 Affine SFM 差不多，只不过这里要求 $2mn ≥ 11m + 3n −15$ ，对于 2 个相机，至少需要 7 个点

>$11m$ 和 $-15$ 是因为齐次坐标下 $P_{3\times 4}$ 和 $Q_{4\times 4}$ 有 11 / 15 个自由度

以 2 个相机的情况为例，过程如下
- 利用八点法求解基本矩阵 $F$ （实际中 SFM 的流程一般采用预测的内参，结合五点法计算本质矩阵 $E$ ，进而求解出外参）
- 将第一个相机的参数设为 $[I\ 0]$ ，那么就可以得到第二个相机的参数
	- 如果采用 $F$ ，那第二个相机的参数就是 $[A\ t]$ ，其中 $A=-[e_{\times}]F$ ，$e$ 为极点（$F^Te=0$）
	- 如果采用 $E$ ，那就能从中计算出第二个相机的外参 $[R\ t]$
- 得到了相机参数，就可以通过三角测量计算出 3D 坐标
- 利用 bundle adjustment 对结果进行全局优化
- 选择一个放缩单位/尺度（比如对于 $E$ 设置 $\lVert t \rVert=1$ 或依据已知基准） 
- 通过 self-calibration 将投影重建升级为度量重建

>采用 $F$ 时，相机参数 $P_1 = [I\ 0]$ 和 $P_2 = [A\ t]$ 是射影相机矩阵（直接对相机的投影矩阵进行参数化），而不是度量相机矩阵 $K[R\ t]$ 

一个常见的 SFM pipeline 如下
- 特征提取：使用 SIFT/ORB 提取特征点
- 特征匹配：根据特征描述子，使用  ANN/KNN + ratio test
- 几何验证：使用 RANSAC + 基本矩阵选取 inlier 最多的情况，剔除 outlier
- 初始化重建：利用上面的 Projective SFM 方法，初始化内参，五点法计算 $E$ 进而得到外参，再三角测量得到一个初始的三维重建
- 逐步往里加入新的图像并优化调整


