---
date: 2025-03-05
tags:
  - CV
---

# Line Fitting

边缘检测可以得到一组像素点，但有时我们想进一步知道这些点对应哪条直线，这就需要**直线拟合 line fitting**

## Least Squares Method

先假设直线为 $y=mx+b$，对于点 $(x_i,y_i)$，直线预测的高度为 $mx_i+b$，与实际高度的差称为**残差 residual**，最小二乘法就是选择 $m,b$，使残差的平方和最小

$$
E(m,b)=\sum_{i=1}^n(mx_i+b-y_i)^2
$$

![](CV导论/imgs/img2/image-4.png)

把不同参数对应的误差画出来，就是误差曲面 energy landscape，我们想找到它的最低处，这个目标是凸的，局部最小值也是全局最小值；数据不足等情况下，最优参数可能不唯一

![](CV导论/imgs/img2/image-5.png)

这种方式有两个问题

1. $y=mx+b$ 无法表示竖直的直线
2. 残差会被平方，少数偏离很远的点就可能主导结果，对**离群点 outlier** 很敏感

### Implicit Line and SVD

对于第一个问题，可以改用隐式方程 $ax+by-d=0$，竖直直线也能表示，比如 $x=2$

![](CV导论/imgs/img2/image-6.png)

此时可以把所有点的约束写成矩阵形式

$$
\begin{aligned}
A&=\begin{bmatrix}x_1&y_1&-1\\\vdots&\vdots&\vdots\\x_n&y_n&-1\end{bmatrix},\quad
h=\begin{bmatrix}a\\b\\d\end{bmatrix} \\
E(h)&=\|Ah\|_2^2
\end{aligned}
$$

$Ah$ 的每个元素就是一个点代入直线方程后的值，$\|\cdot\|_2^2$ 表示把这些值平方后求和，但如果直接最小化，会得到 $h=0$，这根本不是一条直线，所以需要给参数加上约束

一种常见的齐次代数拟合是要求 $\|h\|_2=1$，再最小化 $\|Ah\|_2$，这可以用**奇异值分解 SVD** 求解

$$
A_{n\times3}=U_{n\times n}\Sigma_{n\times3}V_{3\times3}^T
$$

这里 $U,V$ 可以理解为改变坐标方向，$\Sigma$ 的对角元素是各个方向上的缩放量，即奇异值，在单位长度约束下，沿缩放最小的方向取 $h$，就能让 $\|Ah\|_2$ 最小，所以取 $V$ 中最小奇异值对应的列，奇异值通常从大到小排列，也就是最后一列

```python
U, S, Vt = np.linalg.svd(A)
h = Vt[-1]  # Vt 的最后一行，就是 V 的最后一列
```

也可以从特征值理解：$\|Ah\|_2^2=h^TA^TAh$，由于 $A^TA$ 是对称半正定矩阵，最优方向就是其最小特征值对应的特征向量，SVD 可以直接得到这个方向

>这里最小化的是代数误差，如果想最小化点到直线的垂直距离，约束应为 $a^2+b^2=1$，可以先将点减去均值，再对中心化后的 $n\times2$ 坐标矩阵做 SVD，取最小奇异值对应的方向作为法向量 $(a,b)$，最后由 $d=a\bar x+b\bar y$ 得到直线

对于一般的 $AX=B$，则可以用伪逆求最小二乘解

$$
X=A^+B=V\Sigma^+U^TB
$$

$\Sigma^+$ 将非零奇异值取倒数，并调整矩阵形状，实际一般直接调用现成的库

```python
X = np.linalg.lstsq(A, B, rcond=None)[0]
# 也可以使用伪逆
X = np.linalg.pinv(A) @ B
```

### Outliers

最小二乘对小幅噪声通常还能应对，但离群点会把结果拉偏

![](CV导论/imgs/img2/image.png)

如果将残差平方改成残差绝对值，即 L1 损失，大误差的影响会减弱，不过仍然需要处理离群点，另一种思路是先找出哪些点属于同一条直线，再只用这些点拟合

## RANSAC

**随机抽样一致性 RANSAC**：反复从数据中抽取少量点来提出假设，再看有多少数据支持这个假设，符合当前模型的点称为**内点 inlier**，偏离较远的点称为离群点 outlier

![](CV导论/imgs/img2/image-7.png)

以直线拟合为例

1. 随机选 2 个不同的点，确定一条候选直线
2. 计算其余点到直线的距离，小于阈值的视为内点
3. 重复取样，保留内点最多的候选；内点数相同时，可以再比较拟合误差
4. 用最好候选的所有内点重新拟合直线，也可以重新判断内点并进一步优化

![](CV导论/imgs/img2/image-1.png)

![](CV导论/imgs/img2/image-2.png)

主要的超参数是取样次数和内点距离阈值，阈值可以结合噪声水平选择；内点最多的候选也不保证就是真实模型，需要结合数据情况判断，多个候选的计算可以并行进行

### Number of Hypotheses

假设内点比例为 $w$，一次抽取 $n$ 个点，在近似独立的条件下，一组点全部是内点的概率为 $w^n$，取样 $K$ 次却没有一次成功的概率为 $(1-w^n)^K$

如果希望至少成功一次的概率达到 $p$，就需要

$$
\begin{aligned}
1-(1-w^n)^K&\geq p \\
K&\geq\frac{\log(1-p)}{\log(1-w^n)}
\end{aligned}
$$

因此取右侧向上取整即可，这里假设 $0<w<1$、$0<p<1$，每次取的点数应尽量少，但必须足够确定模型，比如直线至少需要 2 个点；内点越少，通常需要取样越多次

RANSAC 适用于很多模型拟合问题，实现简单，也能估算取样失败率，但离群点比例太高时，计算成本会迅速增加

## Hough Transform

**霍夫变换 Hough transform**采用投票的思路：让每个点为所有可能经过它的直线投票，得票最多的直线就是较好的候选

以 $y=mx+b$ 为例，对于点 $(x_i,y_i)$，有 $b=y_i-mx_i$，如果把 $m,b$ 当作参数空间中的坐标，这就是一条直线，参数空间中两条线的交点，代表一条同时经过原图中两个点的直线

![[CV导论/imgs/img2/image-21.png]]

把参数空间划成网格，统计每个网格得到多少票，即可找到很多点共同支持的参数，即使存在较多离群点，真实直线仍可能形成明显的峰值

![[CV导论/imgs/img2/image-22.png]]

实际为了表示竖直线，常用 $x\cos\theta+y\sin\theta=\rho$，每个原图中的点在 $(\theta,\rho)$ 空间中对应一条曲线，同样通过找投票峰值确定直线

>传统视觉系统常把边缘检测、拟合、后处理等模块接起来，前一个模块的错误可能影响后续结果，因此每个模块都需要一定鲁棒性；端到端 end-to-end 方法则尝试把多个环节一起学习优化

# Corner Detection

## Keypoint and Corner

匹配两张图片时，需要找出能够稳定对应的位置，称为**关键点 keypoint**，我们希望这些点

- 显著：与周围容易区分
- 可重复检测：光照、视角或尺度适当变化后，仍能找到同一位置
- 定位准确：能够确定具体的点，而不是一整段模糊区域
- 数量充足：方便进行匹配和后续估计

角点 corner 就是一类常见关键点，一个平坦区域无论怎么挪动都差不多，一条边缘沿着边的方向移动也变化不大，而角点附近沿不同方向移动，局部图案通常都会明显变化，可以据此构建 **Harris Corner Detector**

>角点具有成为关键点的条件，但普通 Harris 检测并不自动具备尺度不变性，对各种视角、光照变化的适应能力也有限

## Window and Intensity Change

先取一个滑动窗口 window，观察当前位置附近的一块区域，实际可以并行计算各个位置

![](CV导论/imgs/img2/image-3.png)

![|209x209](CV导论/imgs/img2/image-10.png)

将窗口中心固定在 $(x_0,y_0)$，把其中的图案平移 $(u,v)$，比较移动前后的灰度差异

![](CV导论/imgs/img2/image-8.png)

![](CV导论/imgs/img2/image-9.png)

用 $w(x-x_0,y-y_0)$ 表示窗口内各位置的权重，定义灰度差平方和

$$
E_{(x_0,y_0)}(u,v)=\sum_{x,y}w(x-x_0,y-y_0)\big(I(x+u,y+v)-I(x,y)\big)^2
$$

窗口可以是矩形内权重相同、外面为 0，也可以用高斯函数让靠近中心的像素更重要，窗口函数对称时，加权求和也可以写成与窗口核的卷积

![](CV导论/imgs/img2/image-11.png)

## First-Order Approximation

如果位移很小，可以用一阶泰勒展开近似：某方向上的变化量，约等于这个方向的导数乘以移动距离

$$
\begin{aligned}
I(x+u,y+v)-I(x,y)&\approx I_xu+I_yv \\
E(u,v)&\approx\sum_{x,y}w(I_xu+I_yv)^2 \\
&=\begin{bmatrix}u&v\end{bmatrix}M\begin{bmatrix}u\\v\end{bmatrix} \\
M&=\sum_{x,y}w\begin{bmatrix}I_x^2&I_xI_y\\I_xI_y&I_y^2\end{bmatrix}
\end{aligned}
$$

$M$ 综合了窗口内各个像素的梯度，是一个 $2\times2$ 的对称半正定矩阵，可以换到两个互相垂直的主方向上理解它，沿这两个方向的变化强弱分别由非负特征值 $\lambda_1,\lambda_2$ 表示

$$
\begin{aligned}
M&=R^T\begin{bmatrix}\lambda_1&0\\0&\lambda_2\end{bmatrix}R \\
E(u,v)&\approx\lambda_1u_R^2+\lambda_2v_R^2,\quad
\begin{bmatrix}u_R\\v_R\end{bmatrix}=R\begin{bmatrix}u\\v\end{bmatrix}
\end{aligned}
$$

这里 $R$ 只是把位移换到主方向坐标系，两个特征值越大，沿对应方向移动时图案变化越明显

![|372x170](CV导论/imgs/img2/image-12.png)

- 两个特征值都小：平坦区域
- 一个大、一个小：边缘，沿一个方向移动变化小
- 两个都大：角点，沿各个方向移动都有明显变化

![](CV导论/imgs/img2/image-13.png)

## Corner Response Function

直接计算每个位置的两个特征值比较麻烦，可以用一个**角点响应函数**来判断

$$
\begin{aligned}
R_H&=\det(M)-\alpha\operatorname{tr}(M)^2 \\
&=\lambda_1\lambda_2-\alpha(\lambda_1+\lambda_2)^2
\end{aligned}
$$

$\det(M)$ 是行列式，$\operatorname{tr}(M)$ 是对角线元素之和，$\alpha$ 是一个较小的正系数，两个特征值都大时，响应通常较大；一大一小时通常为负；都小时接近 0

![|430x292](CV导论/imgs/img2/image-14.png)

设置阈值 $t$，保留 $R_H>t$ 的位置，也可以定义 $\theta=R_H-t$，再判断 $\theta>0$，窗口函数的选择也会影响结果

![](CV导论/imgs/img2/image-15.png)

## Summary

对于一张输入图像

![|203x154](CV导论/imgs/img2/image-16.png)

1. 计算水平、竖直方向的梯度 $I_x,I_y$

![](CV导论/imgs/img2/image-17.png)

2. 逐像素计算 $I_x^2,I_y^2,I_xI_y$

![](CV导论/imgs/img2/image-18.png)

3. 用矩形窗口或高斯核 $g$ 对这些结果做局部加权求和，组成 $M$

![](CV导论/imgs/img2/image-19.png)

4. 计算响应，可以直接写成

$$
\theta=g(I_x^2)g(I_y^2)-g(I_xI_y)^2-\alpha\big(g(I_x^2)+g(I_y^2)\big)^2-t
$$

5. 阈值筛选，再进行局部非极大值抑制，让同一个角点附近只保留响应最强的位置

![|243x190](CV导论/imgs/img2/image-20.png)
