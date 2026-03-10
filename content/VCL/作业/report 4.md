
# 1 Inverse Kinematics

## Sub1 FK

前向动力学过程从根节点开始，根据一个关节的父关节的全局 pose，和这个关节的局部 pose ，计算出这个关节的全局 pose 

## Sub2 CCD

从倒数第二个关节开始（因为最后一个关节没有效果），从后往前调整每个关节的旋转，由图可知，需要调整的角度是：从这个关节到末端执行器的朝向，到这个关节到目标位置的朝向，二者之间所需的旋转

![[VCL/作业/img/img4/image.png]]

## Sub3 FABR

FABR (Forward And Backward Reaching) 分为两个过程进行调整：forward 和 backward 过程，二者交替的进行

以下图为例，在 forward 过程中，从最后一个关节 $P_4$ 开始，直接把它移动到目标位置，然后连接目标点和 $P_3$ ，在这条线上取距离 $d_3$ 得到 $P_3'$ ，以此类推，其中 $d_3$ 是 $P_3P_4$ 这段链节的长度

![[VCL/作业/img/img4/image-1.png]]

但是上述过程中，可能导致根节点调整后偏离原先的位置，所以在 backward 过程中，从从根节点 $P_1'$ 开始，把它平移到原先的位置，然后在与 $P_2'$ 的连线上取距离 $d_1$ 得到 $P_2''$ 以此类推

![[VCL/作业/img/img4/image-2.png]]

> 此处参考 [FABRIK算法详解](https://blog.csdn.net/zhaishengfu/article/details/88195246) ，但是 lab 代码中的 forward 和 backward 的称呼貌似是反过来的，这从原论文 [FABRIK: A fast, iterative solver for the Inverse Kinematics problem](https://www.sciencedirect.com/science/article/pii/S1524070311000178#s0020) 的 Algorithm 1 中可以确定

## Sub4 狭挤霸画

我搞了个类似于 DNA 双链的东西，效果如下

![[VCL/作业/img/img4/image-3.png]]

## Sub4.1 直观上均匀的采样

考虑到曲线参数化的方式，如果对参数 $t$ 均匀采样，实际上呈现的效果可能并不均匀，比如曲线在某些部分弯曲得更厉害或变化更快，那采样点就会聚集在这些区域

一种方式是更换曲线的参数化方式，但这需要具体情况具体分析；还有一种方式是根据弧长等间隔的采样，但这又需要计算弧长，对于一些复杂曲线很费劲

考虑借鉴 Farthest Point Sampling 的思路
1. 先用均匀采样采取过量的点，保证覆盖率够大（我设为 50000 个）
2. 贪心选点
	- 先随机取一个点
	- 从剩下的点里选出一个点，考虑这个点到所有已选的点的距离的最小值，让这个值最大，逐渐循环，直到选出目标数量个点（5000 个）

但是这样直接选出来的点的顺序是随机的，不一定是按曲线上的顺序排布的，绘制时会乱窜，还得根据原先的参数化顺序重新排列一下，最终效果如下（经过打印，这的确只有 5000 个点，数量和原来一样）

![[VCL/作业/img/img4/image-4.png]]

## Q1

如果超出可达距离，IK 算法会使末端停在尽可能接近目标的位置，但仍会有距离，这时关节可能会旋转到极限角度，导致不自然的姿势

## Q2

CCD IK 通常需要更多迭代次数，因为它从末端关节开始，逐个关节调整旋转，每次只优化一个关节，效率低收敛慢；FABR IK 通常需要较少迭代次数，它同时从根关节和末端关节双向处理，每次迭代更全局地优化整个链，收敛更快

## Q3

可以考虑添加平滑约束，比如添加阻尼因子，或者设置一个差异值选择前后最接近的解

# 2 Mass-Spring System

对于一个质点系统，记第 $k$ 个时间步的位置如下（速度和力同理）

$$\mathbf{x}^k = \begin{pmatrix}
\mathbf{x}_1(t_k) \\
\mathbf{x}_2(t_k) \\
\vdots \\
\mathbf{x}_n(t_k)
\end{pmatrix}\in \mathbb{R}^{3n}$$
其质量矩阵形如
$$M = \begin{bmatrix}

m_1 & 0 & 0 & 0 & 0 & 0 \\
0 & m_1 & 0 & 0 & 0 & 0 \\
0 & 0 & m_1 & 0 & 0 & 0 \\
0 & 0 & 0 & m_2 & 0 & 0 \\
0 & 0 & 0 & 0 & m_2 & 0 \\
0 & 0 & 0 & 0 & 0 & m_2
\end{bmatrix}\in \mathbb{R}^{3n\times 3n}$$

隐式欧拉方程的形式如下，其中 $h$ 为时间步长，要求解出 $x^{k+1}$ 和 $v^{k+1}$

$$
\begin{aligned}
&x^{k+1} = x^k + h v^{k+1}\\
&v^{k+1} = v^k + h M^{-1} f(x^{k+1})
\end{aligned}
$$

对于弹簧系统，要求 $x^{k+1}$ ，通过一番推导可得，等价于求解以下目标

$$x^{k+1} = \arg \min_{x} \frac{1}{2h^2} \| x - y^k \|^2_M + E(x)$$

- $y^k = x^k + h v^k + h^2 M^{-1} f_{ext}$ ，$f_{ext}$ 为外力，可以认为是已知量
- $E(x)$ 为系统的总弹性势能

---

令 $g(x)=\frac{1}{2h^2} \| x - y^k \|^2_M + E(x)$ ，利用二阶泰勒展开可以转化为以下迭代方程

$$x_{i+1} = x_i - H^{-1}_g(x_i) \nabla g(x_i)$$

其中 $H_g(x_i)$ 是 $g(x)$ 关于 $x_i$ 的 Hessian 矩阵

$$H_g(x_i) = \begin{bmatrix}
\frac{\partial^2 g(x_i)}{\partial x^2} & \frac{\partial^2 g(x_i)}{\partial x \partial y} & \frac{\partial^2 g(x_i)}{\partial x \partial z} \\
\frac{\partial^2 g(x_i)}{\partial x \partial y} & \frac{\partial^2 g(x_i)}{\partial y^2} & \frac{\partial^2 g(x_i)}{\partial y \partial z} \\
\frac{\partial^2 g(x_i)}{\partial x \partial z} & \frac{\partial^2 g(x_i)}{\partial y \partial z} & \frac{\partial^2 g(x_i)}{\partial z^2}
\end{bmatrix}$$
对于弹簧系统，我们认为 $g(x)$ 性质足够好，只需要进行一步牛顿迭代即可，即求解以下方程（可以把 $x^{k+1} - x^k$ 视为一个整体，这就变成 $Ax=b$ 的形式）
$$H_g(x^k)(x^{k+1} - x^k) = -\nabla g(x^k)$$

下面需要计算 $\nabla g(x^k)$ 和 $H_g(x^k)$ ，由 $g(x)$ 定义得

$$
\begin{aligned}
&\nabla g(x^k) = \frac{1}{h^2} M (x^k - y^k) + \nabla E(x^k)\\
&H_g(x^k) = \frac{1}{h^2} M + H(x^k)
\end{aligned}
$$

其中 $H(x^k)$ 是系统弹性势能 $E(x)$ 的 Hessian 矩阵，我们先看 $\nabla E(x^k)$ 

---

设 $E_{ij}$ 表示质点 $i$ 和 $j$ 之间的弹簧的弹性势能，则其关于 $x_i$ 的负梯度就是质点 $i$ 受到这根弹簧的弹力

$$\nabla_i E_{ij} = f_{ij}$$

把与质点 $i$ 相连的所有弹簧的梯度求和，就可以得到 $E(x)$ 关于 $x_i$ 的梯度

$$\nabla_i E(x^k) = \sum_j \nabla_i E_{ij}(x^k)$$

把所有质点的这个梯度放到一起，就得到了 $E(x)$ 关于 $x^k$ 的梯度

$$\nabla E(x^k) = \begin{pmatrix}
\nabla_1 E(x^k) \\
\vdots \\
\nabla_n E(x^k)
\end{pmatrix}$$

---

再看 $H(x^k)$ ，先考虑一个弹簧 $(i,j)$ 的势能的 Hessian 矩阵 $H_{ij}(x^k)$ 
$$

\begin{aligned}
\frac{\partial^2 E_{ij}(x^k)}{\partial x_i^2} &= k_{ij} \frac{(x_i - x_j)(x_i - x_j)^{\top}}{\|x_i - x_j\|^2} \\
&+ k_{ij} \left(1 - \frac{l_{ij}}{\|x_i - x_j\|}\right) \left(I - \frac{(x_i - x_j)(x_i - x_j)^{\top}}{\|x_i - x_j\|^2}\right)
\end{aligned}

$$
记其为 $H_e\in \mathbb{R}^{3\times 3}$ ，则可得
$$

\frac{\partial^2 E_{ij}(x^k)}{\partial x_i \partial x_j} = -H_e,\quad
\frac{\partial^2 E_{ij}(x^k)}{\partial x_j^2} = H_e

$$
那么 $H_{ij}(x^k)$ 就是下面这样，其中省略号部分全是 0 ，而 $H_e$ 和 $-H_e$ 块位于行/列的第 $i$ 和 $j$ 个 3×3 子块处，这代表弹簧 $(i,j)$ 对于整个系统的贡献
$$H_{ij}(x^k) = \begin{bmatrix}

\vdots & \vdots & \vdots & \vdots & \vdots \\
\cdots & H_e & \cdots & -H_e & \cdots \\
\vdots & \vdots & \vdots & \vdots & \vdots \\
\cdots & -H_e & \cdots & H_e & \cdots \\
\vdots & \vdots & \vdots & \vdots & \vdots 
\end{bmatrix}$$
那么整个系统的 Hessian 矩阵就是

$$H(x^k) = \sum_{(i,j)} H_{ij}(x^k)$$

最终可得效果如下

![[VCL/作业/img/img4/image-5.png]]




