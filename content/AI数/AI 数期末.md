
# 重要考点

- 概率期中后
	- 协方差矩阵、多元正态分布、独立分布相加
	- 依概率收敛（压轴证明）
- 统计
	- 最大似然估计、矩估计、无偏性、UMVUE
	- 两点假设检验、$t$ 分布与置信区间
- 图论
	- 握手定理、扩大路径法
	- 树（定义）、生成树
	- 图的矩阵表示（关联、邻接矩阵与相邻矩阵、拉普拉斯矩阵）

# n 维随机向量

## 条件分布

**条件分布函数**：设 $X = (X_1, \cdots, X_m)$ 和 $Y = (Y_1, \cdots, Y_n)$ 分别是 $m$ 维和 $n$ 维随机向量，给定 $y = (y_1, \cdots, y_n)$，若 $P(Y = y) > 0$，则 $x_1, \cdots, x_m$ 的函数

$$P(X_1 \leq x_1, \cdots, X_m \leq x_m \mid Y = y)$$

称为在 $Y = y$ 条件下 $X$ 的条件分布函数，记为 $F_{X \mid Y}(x_1, \cdots, x_m \mid y)$

**定理**：若 $X = (X_1, \cdots, X_m)$ 和 $Y = (Y_1, \cdots, Y_n)$ 有联合密度 $p(x_1, \cdots, x_m, y_1, \cdots, y_n)$，则

$$F_{X \mid Y}(x_1, \cdots, x_m \mid y_1, \cdots, y_n) = \int_{-\infty}^{x_1} \cdots \int_{-\infty}^{x_m} p_{X\mid Y}(u_1,\cdots,u_m) du_1 \cdots du_m$$

其中

$$p_{X\mid Y}(u_1,\cdots,u_m)=\frac{p(u_1, \cdots, u_m, y_1, \cdots, y_n)}{p_Y(y_1, \cdots, y_n)}$$

这里 $p_Y(y_1, \cdots, y_n)$ 是 $Y = (Y_1, \cdots, Y_n)$ 的联合密度，也可以视为 $p(u_1, \cdots, u_m, y_1, \cdots, y_n)$ 的一个边缘分布

>**n 维随机向量的边缘分布公式**：对于 $n$ 维随机向量 $(X_1, X_2, \dots, X_n)$ ，要得到其中任意 $k$ 个分量的边缘概率密度函数，需对剩余的 $n-k$ 个变量进行积分

## 卡方分布

**伽马分布的独立可加性**：若 $X$ 与 $Y$ 独立，$X \sim \Gamma(r, \lambda), Y \sim \Gamma(s, \lambda)$，则  

$$Z = X + Y \sim \Gamma(r + s, \lambda)$$

**$\chi^2$ 分布**：假设 $X_1, \cdots, X_n \overset{i.i.d.}{\sim} N(0, 1)$ ，则  

$$Y_n := X_1^2 + \cdots + X_n^2 \sim \Gamma\left(\frac{n}{2}, \frac{1}{2}\right)$$

称 $Y_n$ 服从自由度为 $n$ 的卡方分布，记为 $Y_n\sim \chi^2(n)$ ，其概率密度为  

$$p_n(x) = \frac{1}{2^{n/2}\Gamma\left(\frac{n}{2}\right)} x^{n/2 - 1} e^{-x/2}, \quad x > 0$$

其中

$$\Gamma(\alpha) = \int_0^{+\infty} x^{\alpha - 1} e^{-x} dx$$

## 独立性

**相互独立**：若对任意 $a_i < b_i, i = 1, \cdots, n$ 都有

$$P(a_1 < X_1 < b_1, \cdots, a_n < X_n < b_n) = P(a_1 < X_1 < b_1) \cdots P(a_n < X_n < b_n)$$

则称 $n$ 个随机变量 $X_1, \cdots, X_n$ 相互独立

**定理**：设 $X_1, \cdots, X_n$ 都是随机变量，分别有概率密度 $p_1(x_1), \cdots, p_n(x_n)$，则 $X_1, \cdots, X_n$ 相互独立的充分必要条件是 $n$ 元函数  

$$p_{X_1, \cdots, X_n}(x_1, \cdots, x_n) = p_1(x_1) \cdots p_n(x_n)$$

为 $n$ 维随机向量 $(X_1, \cdots, X_n)$ 的联合密度

**两两独立**：若对任意 $i \neq j$，都有 $X_i$ 与 $X_j$ 相互独立，则称 $X_1, \cdots, X_n$ 两两独立（**相互独立强于两两独立**）

## 数学期望与协方差矩阵⭐

**n 维随机向量的数学期望**：设 $\xi = (X_1, \dots, X_n)$ 是 $n$ 维随机向量，若  

$$(E(X_1), \dots, E(X_n))$$

存在，则称其为 $\xi$ 的数学期望，记为 $E(\xi)$

**n 维随机向量的数学特征**：设 $\xi = (X_1, \dots, X_n)$ 是 $n$ 维随机向量，且满足 $\mathrm{var}(X_i) < \infty \ (i = 1, \dots, n)$，记  

$$\sigma_{ij} = \mathrm{cov}(X_i, X_j), \quad \rho_{ij} = \frac{\sigma_{ij}}{\sqrt{\sigma_{ii}\sigma_{jj}}}$$

称 $\Sigma = (\sigma_{ij})_{n \times n}$ 为 $\xi$ 的协方差矩阵，$R = (\rho_{ij})_{n \times n}$ 为 $\xi$ 的相关系数矩阵

**n 维随机向量的线性变换的均值和协方差**：设 $\xi = (X_1, \cdots, X_n)$ 的协方差矩阵为 $\Sigma$，$A_{m\times n}$ 的秩为 $m$ ，$\lvert A \rvert \neq 0$，$Y=AX$，则

$$\begin{aligned}
(E(Y))^\top &= A(E(\xi))^\top \\
\text{cov}(Y, Y) &= A \Sigma A^\top
\end{aligned}$$

---

期望的性质与计算
- $E(a(X)) = aE(X), \quad E(X + Y) = E(X) + E(Y)$ 
- 若随机变量 $X, Y$ 独立，则 $E(XY) = E(X)E(Y)$ 

方差的性质与计算
- $\text{var}(X) = E(X^2) - (E(X))^2$ 
- $\text{var}(aX + b) = a^2 \text{var}(X)$ 
- 若随机变量 $X,Y$ 独立，则 $\text{var}(X+Y)=\text{var}(X)+\text{var}(Y)$ 

协方差的性质与计算
- $\text{cov}(X,Y)=E(XY)-E(X)E(Y)$ 
- $\text{cov}(X + Y, U + V) = \text{cov}(X,U) + \text{cov}(X,V) + \text{cov}(Y,U) + \text{cov}(Y,V)$ 
- $\text{cov}(aX,bY)=ab\text{cov}(X,Y)$ 
- $\text{cov}(X + c,Y + d) = \text{cov}(X,Y)$ 
- $\text{cov}(X,X)=\text{var}(X)$ 
- $\text{cov}(X,Y)=\text{cov}(Y,X)$ 

## n 维随机向量的函数

**n 维随机向量的函数**：设 $Y = f(X_1, \dots, X_n)$ 的分布函数是 $F(y)$ ，令  
$$A(y) = \{(x_1, \dots, x_n) : f(x_1, \dots, x_n) \leq y\}$$

其中 $y$ 是任意实数，则

$$F_Y(y) = P(f(\xi) \leq y) = \int \cdots \int_{A(y)} p(x_1, \dots, x_n) dx_1 \cdots dx_n$$  

**n 维随机向量的函数的期望（均值公式）**：设 $Y = f(X_1, \dots, X_n)$ ，$n$ 维随机向量 $(X_1, \dots, X_n)$ 有联合密度 $p(x_1, \dots, x_n)$ ，则  

$$E(Y) = \int \cdots \int f(x_1, \dots, x_n) p(x_1, \dots, x_n) dx_1 \cdots dx_n$$

**n 维随机向量的联合密度变换**：设 $\xi = (X_1, \cdots, X_n)$ 为连续型随机向量，且 $\mathbb{R}^n$ 中的区域 $A$ 满足 $P(\xi \in A) = 1$，函数 $f_1(x_1, \cdots, x_n), \cdots, f_n(x_1, \cdots, x_n)$ 满足下列条件
1. 对任何实数 $u_1, \cdots, u_n$，方程组 $f_k(x_1, \cdots, x_n) = u_k, \quad (k = 1, \cdots, n)$ 在 $A$ 中至少多有一个解 $x_i = x_i(u_1, \cdots, u_n), \quad i = 1, \cdots, n$ 
2. 对一切 $k = 1, \cdots, n$，$f_k$ 在 $A$ 中有连续偏导数
3. 雅可比行列式

$$J = \frac{\partial(y_1, \cdots, y_n)}{\partial(x_1, \cdots, x_n)} \neq 0.$$

设 $Y_k = f_k(X_1, \cdots, X_n) \ (k = 1, \cdots, n)$，$G = \{(u_1, \cdots, u_n) : \text{方程组 } f_k(x_1, \cdots, x_n) = u_k, \ (k = 1, \cdots, n) \text{ 在 } A \text{ 中有解}\}$，则 $\eta = (Y_1, \cdots, Y_n)$ 是连续型，且联合密度  

$$p_\eta(y_1, \cdots, y_n) = p_\xi(x_1, \cdots, x_n) \left| J^{-1} \right|, \ (y_1, \cdots, y_n) \in G.$$

## n 维正态分布⭐

**n 维正态分布**：假设 $n$ 维随机向量 $\xi$ 有如下的联合密度，则称 $\xi$ 服从 $n$ 维正态分布，记为 $\xi \sim N(\mu, \Sigma)$   

$$p(x) = \frac{1}{\sqrt{(2\pi)^n \lvert \Sigma \rvert}} \exp \left\{ -\frac{1}{2} (x - \mu)^\top \Sigma^{-1} (x - \mu) \right\}$$

且 $E(\xi)=\mu,\ \text{cov}(\xi,\xi)=\Sigma$ 

>可见**要求 n 元正态分布的联合密度，只需求其均值与协方差矩阵**

**定理**：设 $(X_1, \cdots, X_n)^\top \sim N(\mu, \Sigma)$，$A_{m\times n}$ 的秩为 $m$ ，$\lvert A \rvert \neq 0$，$Y=AX$，则  

$$(Y_1, \cdots, Y_m)^\top \sim N(A\mu, A\Sigma A^\top)$$

推论
- 若 $\xi$ 服从 $n$ 元正态分布 $N(\mu, \Sigma)$，则存在一个正交变换 $U$，使得 $\eta = U\xi$ 是一个具有独立正态分布分量的随机向量，它的数学期望为 $U\mu$，方差分量是 $\Sigma$ 的特征值  
- 正交变换下，多维标准正态变量保持其独立性，同方差性不变  
- 若 $\xi \sim N(\mu, \Sigma)$，其中 $\Sigma$ 是 $n$ 阶正定阵，则 $(\xi - \mu)^\top \Sigma^{-1} (\xi - \mu) \sim \chi_n^2$

**定理**：设 $(X_1, \cdots, X_m, X_{m+1}, \cdots, X_n)^\top \sim N(\mu, \Sigma)(1 \leq m < n)$，且  

$$  
\mu = \begin{pmatrix}  
\mu^{(1)} \\  
\mu^{(2)}  
\end{pmatrix}, \quad  
\Sigma = \begin{pmatrix}  
\Sigma^{(1)} & \mathbf{O} \\  
\mathbf{O} & \Sigma^{(2)}  
\end{pmatrix}
$$

其中 $\mu^{(1)}$ 是 $m$ 维列向量，$\mu^{(2)}$ 是 $n-m$ 维列向量，$\Sigma^{(1)}$ 是 $m$ 阶矩阵，$\Sigma^{(2)}$ 是 $n-m$ 阶矩阵，则

$$\begin{aligned} 
X^{(1)} &= (X_1, \cdots, X_m)^\top \sim N(\mu^{(1)}, \Sigma^{(1)})\\
X^{(2)} &= (X_{m+1}, \cdots, X_n)^\top \sim N(\mu^{(2)}, \Sigma^{(2)})
\end{aligned}$$
推论：多元正态分布 $(X_1,\cdots,X_n)$ 两两独立的充要条件是两两不相关

**定理**：$(X_1, \cdots, X_m, \cdots, X_n)^\top \sim N(\mu, \Sigma)(1 \leq m < n)$，则  
$$(X_1, \cdots, X_m) \sim N(\mu^{(1)}, \Sigma_{11})$$

其中

$$\Sigma =  
\begin{bmatrix}  
\Sigma_{11} & \Sigma_{12} \\  
\Sigma_{21} & \Sigma_{22}  
\end{bmatrix}, \quad  
\mu =  
\begin{bmatrix}  
\mu^{(1)} \\  
\mu^{(2)}  
\end{bmatrix}$$

# 概率极限定理

## 随机序列的收敛性⭐

**依分布收敛（弱收敛）**：随机变量序列 $X_n$ 的分布函数 $F_n(x)$，如果对于目标随机变量 $X$ 的分布函数 $F(x)$ 的所有连续点，都满足下式，记为 $X_n \overset{w}{\to} X$ 
$$\lim_{n \to \infty} F_n(x) = F(x)$$

- 含义：不关心随机变量具体取值，只关心它们的概率分布是否相似

**依概率收敛**：对于任意小的正数 $\varepsilon > 0$，满足下式，记为 $X_n \overset{P}{\to} X$ 

$$\lim_{n \to \infty} P(|X_n - X| \geq \varepsilon) = 0$$

- 含义：关心随机变量 $X_n$ 的取值偏离目标值 $X$ 的可能性是否越来越小

**几乎必然收敛（概率为 1 的收敛）**：满足下式，记为 $X_n \overset{a.s.}{\to} X$ 

$$P(\lim_{n \to \infty} X_n = X) = 1$$

- 含义：在“几乎所有”可能的观测序列中，$X_n$ 都像数列一样收敛到 $X$ 

**强弱关系**：几乎必然收敛 > 依概率收敛 > 依分布收敛

## 大数律

**大数律的基本含义**：样本和 / 均值趋近于数学期望

**切比雪夫大数律**：设 $X_1, X_2, \dots$ 相互独立（不要求遵循相同的分布），且存在 $M$ 使得 $\forall i,\ \text{var}(X_i) \leq M$ ，设 $S_n = X_1 + \cdots + X_n$ ，则

$$
\frac{1}{n}(S_n - E[S_n]) \overset{P}{\to} 0 \quad (n \to \infty)
$$

- 含义：只要数据足够多，且波动性有限，那么它们的平均值就会非常稳定地接近理论上的期望值

**切比雪夫不等式**：对于方差有限的随机变量 $X$ ，它给出了 $X$ 偏离其期望值 $\mu$ 的概率上界

$$
P(|X - \mu| \geq \varepsilon) \leq \frac{\text{Var}(X)}{\varepsilon^2}, \quad \forall \varepsilon > 0
$$

- 含义：方差越小，$X$ 越集中在 $\mu$ 附近，反之则可能偏离更远
- 左式正好符合依概率收敛定义中的式子，所以**证明依概率收敛时，可以利用切比雪夫不等式确定上界**

## 中心极限定理

**中心极限定理的基本含义**：大数律刻画了样本和 / 均值的收敛性，而中心极限定理精确刻画了其收敛到的分布形式

**中心极限定理**：设 $X_1, X_2, \dots$ 为随机变量序列，若 $E(X_n), \text{var}(X_n), n = 1, 2, \dots$ 都存在，且 $\text{var}(X_n)$ 不全为 0，令 $S_n = \sum_{i=1}^n X_i$ 且

$$
S_n^* = \frac{S_n - E(S_n)}{\sqrt{\text{var}(S_n)}} \overset{w}{\to} N(0, 1)
$$

则称 $X_1, X_2, \dots$ 服从中心极限定理（Central Limit Theorem, CLT）

**中心极限定理的一个充分条件**：设 $X_1, X_2, \dots$ 独立同分布，$E(X_1)$ 存在，且 $0 < \text{var}(X_1) < \infty$ ，那么

$$S_n^* \overset{w}{\to} Z \sim N(0, 1)$$

- 含义：无论原始数据来自什么分布（只要不太极端），当我们大量、独立地采样并求和 / 求平均时，这个和 / 平均值的分布形态会越来越接近正态分布
- 中心极限定理的一个用途是**计算独立随机变量的和在某个范围内的概率**，常会用到 $\Phi(x)$ 

>标准正态函数 $\Phi(x)$ ：随机变量 $Z$ 服从标准正态分布 $Z\sim N(0,1)$ ，定义函数 $\Phi(x) = P(Z \le x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt$

# 估计

## 最大似然估计（ML 估计）⭐

**似然函数**：设 $(X_1, \dots, X_n)$ 为独立重复观察得到的样本，其中 $X_i (i = 1, \dots, n)$ 为离散型随机变量（连续情况同理），样本分布列具有下列一般性质

$$
P_\theta((X_1, \dots, X_n) = (x_1, \dots, x_n)) = \prod_{i=1}^n P_\theta(X_i = x_i) \quad (\theta \in \Theta),
$$

此时 $\theta$ 为参数，对于固定的样本值 $(x_1, \dots, x_n)$，右式就是关于 $\theta$ 的函数，称为似然函数

$$
L(\theta) = \prod_{i=1}^n P_\theta(X_i = x_i)
$$

**最大似然估计（ML估计）**：设 $\theta \in \Theta$ 为统计模型 $(X_1, \dots, X_n) \sim P_\theta$ 的参数，又设 $x_1, \dots, x_n$ 为总体的样本值，若存在 $\hat{\theta}(x_1, \dots, x_n)$，使得

$$
L(\hat{\theta}(x_1, \dots, x_n)) = \max_{\theta \in \Theta} L(\theta)
$$

其中 $L(\cdot)$ 为似然函数，则称 $\hat{\theta}(x_1, \dots, x_n)$ 为 $\theta$ 的最大似然估计

**参数的函数的 ML 估计**：若 $\hat{\theta}$ 为参数 $\theta$ 的 ML 估计，则 $g(\theta)$ 的 ML 估计定义为 $g(\hat{\theta})$ 

## 矩估计⭐

**矩估计**：设总体分布包含 $k$ 个未知参数 $\theta = (\theta_1, \theta_2, \dots, \theta_k)$，称 $\alpha_j = E(X^j)$ 为其 $j$ 阶总体矩，从总体中抽取样本 $X_1, X_2, \dots, X_n$，称 $a_j = \frac{1}{n} \sum_{i=1}^n X_i^j$ 为其 $j$ 阶样本矩，矩估计法通过解方程组

$$
E(X^j) = \frac{1}{n} \sum_{i=1}^n X_i^j, \quad j = 1, 2, \dots, k
$$

得到参数 $\theta$ 的估计值 $\hat{\theta}_1, \hat{\theta}_2, \dots, \hat{\theta}_k$ ，这些估计量称为矩估计量

## 估计的无偏性与 UMVU⭐

**无偏估计**：设 $X_1, \dots, X_n \overset{i.i.d.}{\sim} F(x, \theta)$ 是一个统计模型，$g(\theta)$ 为待估量，若统计量 $T = T(X_1, \dots, X_n)$ 满足

$$
E_\theta[T(X_1, \dots, X_n)] = g(\theta), \ \forall \theta \in \Theta
$$

则称 $T$ 为 $g(\theta)$ 的无偏估计

**方差的无偏估计**：若总体方差 $\sigma^2$ 存在，则其无偏估计如下（这个估计还具有相合性，所以也是方差的相合估计）

$$
S^2 = \frac{1}{n-1} \sum_{i=1}^n (X_i - \overline{X})^2.
$$

>一般来讲，为了得到一个统计量的无偏估计，可以先利用ML估计或矩估计得到一个初步的估计，再对其取期望（或尝试求 $E(\bar X^m)$ 等），根据得到的等式，把选定的统计量放到一边，另一边的期望内部的就是对应的无偏估计

**均方误差（MSE）**：设 $X_1, \dots, X_n \overset{i.i.d.}{\sim} F_\theta(x)(\theta \in \Theta)$ 为统计模型，$g(\theta)$ 为待估量，$g(\theta)$ 的估计量 $T(X_1, \dots, X_n)$ 的均方误差定义为

$$
R(\theta, T) = E_\theta[T(X_1, \dots, X_n) - g(\theta)]^2
$$

>均方误差的另一种计算方式为 $\text{MSE}=\text{Bias}^2(T)+\text{Var}(T)$ ，其中 $\text{Bias}(T)=E[T]-g(\theta)$ ，本质上是利用 $\text{Var}(T)=E[T^2]-E^2[T]$ 把 MSE 的定义式转换成 $E^2[T] + \text{Var}(T)$ 的形式

**一致最小方差无偏估计（UMVUE）**：设 $X_1, \dots, X_n \overset{i.i.d.}{\sim} F_\theta(x)(\theta \in \Theta)$ 为统计模型，$g(\theta)$ 为待估量，$g(\theta)$ 的一个估计量为 $T(X_1, \dots, X_n)$ ，若其满足    
(1) $T$ 是 $g(\theta)$ 的无偏估计；  
(2) 对于 $g(\theta)$ 的任意无偏估计 $\tilde{T} = \tilde{T}(X_1, \dots, X_n)$，都有  

$$
\text{var}_\theta(T) \leq \text{var}_\theta(\tilde{T}), \ \forall \theta \in \Theta.
$$

则称 $T$ 为 $g(\theta)$ 的（一致）最小方差无偏估计（Uniformly Minimum Variance Unbiased, UMVU）；记 $F_\theta(t) = P_\theta(T \leq t)$  为统计量 $T$ 的分布

---

**充分统计量**：假设统计量 $T = T(X_1, \dots, X_n)$ 满足对任意统计量 $\tilde{T} = \tilde{T}(X_1, \dots, X_n)$，都有在 $T = t$ 的条件下，$\tilde{T}$ 的分布与参数 $\theta$ 无关，则称 $T$ 为充分统计量
- 含义：你确定了要用一个概率模型去估计数据背后的分布，但是里面有一堆参数不知道，你只需从样本中得到这些统计量，就能用 ML 估计等方法算出模型参数，即**得到这个统计量对于求解模型参数是充分的**
- 注意：充分统计量不一定只有一个统计量，**也可能是一堆统计量的集合**
- **求解方法**：因子分解定理

**因子分解定理**：设 $X_1, \dots, X_n \overset{i.i.d.}{\sim} p(x, \theta)$，其中 $p(x, \theta)$ 为分布密度或分布列，若 $T = T(X_1, \dots, X_n)$ 满足

$$
\prod_{i=1}^n p(x_i, \theta) = q(T,\theta)\cdot h(x_1, \dots, x_n)
$$

则 $T$ 是充分统计量

---

**完全充分统计量**：设 $X_1, \dots, X_n \overset{i.i.d.}{\sim} p(x, \theta)$，又设 $T = T(X_1, \dots, X_n)$ 为充分统计量，若对任意 $\phi$，有  

$$
E_\theta(\phi(T)) = 0, \forall \theta \in \Theta \implies P_\theta(\phi(T) = 0) = 1, \forall \theta \in \Theta
$$

则称 $T$ 为完全充分统计量
- 含义：**这里“完全性”是指这些统计量完全足够求解出模型参数，没有冗余**，就好比方程数 = 未知量个数的线性方程组，每个方程都用得上，如果方程数量比未知量多，就说明有些方程可以通过其它方程组合出来，这就冗余了
- **求解方法**：指数族分布

**指数族分布**：若密度或分布列 $p(x, \theta)$ 能进行如下分解

$$
p(x, \theta) = S(\theta) h(x) \exp\left\{\sum_{k=1}^m C_k(\theta) T_k(x)\right\} \quad (\theta \in \Theta)
$$

则称 $p(x, \theta), \theta \in \Theta$ 为指数族分布
- 不要求 $p(x,\theta)$ 表达式本身显含 $\exp$ 项，但要能变形为上述形式

**引理**：若总体 $X$ 是指数族，则设 $(X_1, \dots, X_n)$ 为 $X$ 的一个样本，将 $(X_1, \dots, X_n)$ 看作随机向量，则其联合分布也是指数族分布

**定理**：总体 $X$ 具有指数族分布，$\Theta \subset \mathbb{R}^m$ 且含内点，$(C_1, \dots, C_m)$ 是在 $\Theta$ 上一对一、连续的函数，诸 $C_i$ 之间 ($T_i$ 之间) 无线性关系，则  

$$
\left( \sum_{i=1}^n T_1(X_i), \dots, \sum_{i=1}^n T_k(X_i) \right)
$$

是完全充分统计量

---

**Lehmann-Scheff 定理**：设 $T = T(X_1, \dots, X_n)$ 为完全充分统计量，若 $T$ 的函数 $\phi(T)$ 是 $g(\theta)$ 的无偏估计，即

$$
E_\theta(\phi(T)) = g(\theta), \forall \theta \in \Theta
$$

则 $\phi(T)$ 是 $g(\theta)$ 的唯一 UMVU 估计

**UMVUE 求法**
- 先证明是不是指数族分布，再根据其性质得到完全充分统计量
- 根据完全充分统计量，得到无偏估计（对统计量取期望并移项，或者从 $g(\theta)$ 出发，利用 $\text{Var}(T)=E[T^2]-E^2[T]$ 等进行变形）
- 所得的无偏估计就是 UMVUE

## 估计的相合性

**相合估计（弱相合）**：设 $T_n = T_n(X_1, \dots, X_n)$ 满足 $\forall \varepsilon > 0$ ，都有

$$
\lim_{n \to \infty} P_\theta(|T_n - g(\theta)| \geq \varepsilon) = 0, \quad \forall \theta \in \Theta.
$$

则称 $T_n$ 为 $g(\theta)$ 的相合估计，或估计 $T_n$ 具有相合性
- 含义：**样本量越大，估计的就越准**，即随着样本量 $n$ 增大，估计量 $\hat{\theta}_n$​ 会依概率（弱相合） / 几乎必然（强相合）收敛到真实参数 $\theta$ 
- 与无偏性的区别：无偏性是对固定样本量 $n$ ，估计量 $\hat{\theta}_n$​ 的期望等于真实参数 $\theta$ ，即多次取样并估计的平均结果是正确的，但是单次估计可能误差很大

**相合性的基本定理**：设 $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} F_\theta(x)(\theta \in \Theta)$ ，$E_\theta(X_i)$ 存在且有限，则

$$
\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i \overset{P}{\to} E_\theta(X_1) \quad (n \to \infty).
$$

即在简单随机抽样的情况下，样本均值是总体均值的相合估计

**矩估计的相合性**：设 $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} F_\theta(x)(\theta \in \Theta)$ ，则 $\theta$ 的函数 $g(\theta)$ 的矩估计具有相合性

## 置信区间⭐

**置信区间与置信限**：设 $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} F(x, \theta)$ 是一个统计模型，$g(\theta)$ 为实值函数，假设 $\underline T = \underline T(X_1, \dots, X_n)$ 与 $\overline{T} = \overline{T}(X_1, \dots, X_n)$ 为统计量，$\alpha \in (0, 1)$ 
1. 若 $\underline T < \overline{T}$ 且满足下式，则称 $[\underline T, \overline{T}]$ 为 $g(\theta)$ 的置信度为 $1-\alpha$ 的置信区间

$$
P_\theta(\underline T \leq g(\theta) \leq \overline{T}) \geq 1-\alpha, \quad \forall \theta \in \Theta
$$

2. 若满足下式，则称 $T$ 为 $g(\theta)$ 的置信度为 $1-\alpha$ 的置信下限

$$
P_\theta(T \leq g(\theta)) \geq 1-\alpha, \quad \forall \theta \in \Theta
$$

3. 若满足下式，则称 $T$ 为 $g(\theta)$ 的置信度为 $1-\alpha$ 的置信上限

$$
P_\theta(g(\theta) \leq T) \geq 1-\alpha, \quad \forall \theta \in \Theta
$$

- 含义：在给定样本的条件下，模型参数的函数 $g(\theta)$ 的取值会存在一个概率分布，置信区间就是图像面积为 $1-\alpha$ 的区间
- 意义：**衡量统计方法的可信度**，认为这种统计方法得到的结果有 $1-\alpha$ 的把握是可信的

**枢轴量**：设 $X_1,\dots,X_n \overset{\text{i.i.d.}}{\sim} F(x,\theta)$ 是一个统计模型，$g(\theta)$ 是待估量，若

$$
h = h(X_1,\dots,X_n;g(\theta))
$$

的分布与 $\theta$ 无关，则称 $h$ 为枢轴量

**求解置信区间或置信限的方法**
- 找枢轴量 $h = h(X, g(\theta))$ 及其分布 $F$ 
- 根据 $F$ 确定一个区间 $[a, b]$ ，使得 $P(a \leq h \leq b) = 1 - \alpha$ 
- 将 $a \leq h \leq b$ 化为 $\underline T \leq g(\theta) \leq \overline{T}$ ，就得到了 $g(\theta)$ 的区间

---

**z 分数**：设 $Z\sim N(0,1)$ ，则 $z_\alpha$ 定义上侧 $\alpha$ 分位数，即

$$P(Z>z_\alpha)=\alpha$$

- 可见 z 分数与标准正态函数 $\Phi(x)$ 的关系为 $\Phi(z_\alpha)=1-\alpha$ 
- 要求 $1-\alpha$ 的置信区间，就对应着 $z_{\alpha / 2}$ ，因为 $P(Z>z_{\alpha / 2})=\alpha / 2$ ，则 $P(-z_{\alpha / 2}\leq Z\leq z_{\alpha / 2})=1-\alpha$  
- 一些地方是去查找 $z_{1-\alpha / 2}$ ，这里 $z_\alpha$ 代表下侧 $\alpha$ 分位数，即 $P(Z<z_\alpha)=\alpha$ 

**t 分布与 t 分数**：设 $\xi \sim N(0,1),\ \eta \sim \chi^2(n)$ ，且 $\xi$ 与 $\eta$ 独立，则 $T = \frac{\xi}{\sqrt{\eta/n}}$ 服从的分布就是自由度 $df=n$ 的 t 分布，记为 $t(n)$ ，即

$$T = \frac{\xi}{\sqrt{\eta/n}}\sim t(n)$$

设 $T\sim t(n)$ ，则 $t_{\alpha,n}$ （或写为 $t_\alpha(n)$ ）定义为

$$P(T>t_{\alpha,n})=\alpha$$

**定理**：假设总体 $X \sim N(\mu, \sigma^2)$，样本量为 $n$ ，则
1. $\bar{X} \sim N\left(\mu, \frac{1}{n}\sigma^2\right)$ 
2. $\frac{1}{\sigma^2}\sum_{i=1}^n (X_i - \bar{X})^2 \sim \chi^2(n-1)$ 
3. $\bar{X}$ 与 $\sum_{i=1}^n (X_i - \bar{X})^2$ 相互独立

**正态分布的 $1-\alpha$ 置信区间的求法**
- 求 $\mu$ 的置信区间
	- 已知方差 $\sigma^2$ ：可得枢轴量 $\frac{\bar X-\mu}{\sigma/\sqrt n}\sim N(0,1)$ ，根据 $z_{\alpha / 2}$ 计算出置信区间为 $\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$ 
	- 未知方差：用方差的无偏估计 $S^2$ 来代替 $\sigma^2$ ，此时枢轴量为 $\frac{\bar X-\mu}{S/\sqrt n}\sim t(n-1)$ ，根据 $t_{\alpha / 2,n-1}$ 计算置信区间
- 求 $\sigma^2$ 的置信区间
	- 计算方差的无偏估计 $S^2$ ，枢轴量为 $\frac{(n-1)S^2}{\sigma^2}\sim \chi^2(n-1)$ ，根据 $\chi^2_{\alpha / 2,n-1}$ 计算置信区间 

---

**近似置信区间**：设 $X_1, \dots, X_n \overset{\text{i.i.d.}}{\sim} F(x,\theta)$ 是一个统计模型, $g(\theta)$ 是待估量, $T(X_1, \dots, X_n)$ 是 $g(\theta)$ 的渐近正态估计
1. 若 $\sigma^2$ 已知, 则 $g(\theta)$ 的置信度为 $1-\alpha$ 的近似置信区间如下

$$
\left[
T - \frac{\sigma}{\sqrt{n}} z_{1-\alpha/2}, T + \frac{\sigma}{\sqrt{n}} z_{1-\alpha/2}
\right]
$$

2. 若 $\sigma^2$ 未知, 则 $g(\theta)$ 的置信度为 $1-\alpha$ 的近似置信区间如下，其中 $\hat{\sigma}_n$ 为 $\sigma$ 的无偏估计

$$
\left[
T - \frac{\hat{\sigma}_n}{\sqrt{n}} t_{n-1, 1-\alpha/2}, T + \frac{\hat{\sigma}_n}{\sqrt{n}} t_{n-1, 1-\alpha/2}
\right]
$$

- 基础是大数律与中心极限定理，比如样本均值 $\bar X$ 近似服从正态分布，就是对总体均值 $\mu$ 的渐进正态估计

# 假设检验

## 假设检验与 UMP 否定域

**假设检验**：设 $X \sim F_\theta(\theta \in \Theta)$ 为总体模型，所谓假设检验问题是两个关于总体真值的互相对立判断 ($\theta \in \Theta_0, \theta \in \Theta_1$) 的鉴定问题，其中 $\Theta_0$ 是 $\Theta$ 的一个真子集， $\Theta_1 = \Theta \setminus \Theta_0$ 为 $\Theta_0$ 的余集
- 判断 $\theta \in \Theta_0$ 称为**零假设（原假设）**，记为 $0$ 
- 判断 $\theta \in \Theta_1$ 称为**对立假设 （备择假设）**，记为 $1$ 
- 通常用 $H_0 : \theta \in \Theta_0 \leftrightarrow H_1 : \theta \in \Theta_1$ 或 $(\Theta_0, \Theta_1)$ 表示假设检验问题

**否定域**：假设检验要求回答是否接受零假设 $\theta \in \Theta_0$ 成立，该回答依赖于样本观测值 $x = (x_1, \dots, x_n)$ ，当 $x$ 的取值处于某个指定的区间 $W$ 时，就拒绝零假设，即认为它不成立，这个 $W$ 就称为否定域

**两类错误**
- **第一类错误（$\alpha$ 错误）**：零假设为真，但我们拒绝了它，犯这类错误的概率就是显著性水平 $\alpha$ 
- **第二类错误（$\beta$ 错误）**：零假设为假，但我们接受了它，犯这类错误的概率就是显著性水平 $\beta$ 
- **控制第一类错误是首要目标**

**否定域的显著性水平**：设 $(\Theta_1, \Theta_2)$ ，称 $\beta_W(\theta) := P_\theta(X \in W)$ 为 $W$ 的功效函数，若

$$
P_\theta(X \in W) \leq \alpha, \quad \forall \theta \in \Theta_0
$$

则称 $W$ 为检验问题 $(\Theta_0, \Theta_1)$ 的一个（显著性）水平为 $\alpha$ 的否定域
- 含义：**发生第一类错误的概率为** $\alpha$ （对于处在零假设 $\Theta_0$ 的情况，我们希望更小概率去拒绝它）

**UMP 否定域**：若 $W$ 是检验问题 $(\Theta_0, \Theta_1)$ 的水平为 $\alpha$ 的否定域，并且对任意水平为 $\alpha$ 的否定域 $\widetilde{W}$ 都有

$$
P_\theta(X \in W) \geq P_\theta(X \in \widetilde{W}), \quad \forall \theta \in \Theta_1
$$

则称 $W$ 为检验问题 $(\Theta_0, \Theta_1)$ 的水平为 $\alpha$ 的一致最大功效（UMP）否定域
- 含义：**在发生第一类错误概率相同的情况下，发生第二类错误的概率最小的那个否定域**（对于处在备择假设 $\Theta1$ 的情况，我们希望更大概率去拒绝它）

## N-P 引理和似然比检验⭐

**似然比**：对于简单假设检验问题 $\Theta = \{\theta_0, \theta_1\}$ ，即 $H_0: \theta = \theta_0 \leftrightarrow H_1: \theta = \theta_1$ ，
似然函数为 $L(x, \theta) = \prod_{i=1}^n p(x_i, \theta)$ （离散型同理），则似然比 $\lambda$ 定义为

$$\lambda(x)=\frac{L(x,\theta_1)}{L(x,\theta_0)}$$

用似然比构建的否定域称为**似然比否定域**，其中 $\lambda$ 为一个常数

$$
W_{\lambda} = \{x : \lambda(x)>\lambda\}
$$

用似然比否定域进行假设检验的方式就叫做**似然比检验**

**N-P 引理**：若 $\lambda_0$ 使得

$$
P_{\theta_0}(X \in W_{\lambda_0}) = \alpha
$$

则 $W_{\lambda_0}$ 是水平为 $\alpha$ 的 UMP 否定域

**检验统计量**：作为假设检验的解，通常经过化简以后，否定域化简后，具有 $\{x : T(x) > c\}$ 或 $\{x : T(x) < c_1\} \cup \{x : T(x) > c_2\}$ 的形式，它是通过统计量 $T(X)$ 构造得到的，则 $T(X)$ 就称为检验统计量

**求假设检验问题 $H_0: \theta = \theta_0 \leftrightarrow H_1: \theta = \theta_1$ 的水平为 $\alpha$ 的 UMP 否定域**
- 计算似然函数，并根据似然比构建否定域
- 将似然比否定域从 $\{x : \lambda(x)>\lambda\}$ 的形式变为 $\{x : T(x) > c\}$ 等类似的形式
- 求解在 $H_0$ 的条件下， $P_{\theta_0}(T(x)>c)=\alpha$ ，解得 $c$ 
- 根据 N-P 引理，所得的否定域 $\{x : T(x) > c\}$ 即为水平为 $\alpha$ 的 UMP 否定域

## 广义似然比检验（GLR 检验）⭐

**广义似然比**：对于假设检验问题 $H_0: \theta \in \Theta_0 \leftrightarrow H_1: \theta \in \Theta_1$ ，似然函数为 $L(x, \theta) = \prod_{i=1}^n p(x_i, \theta)$ ，令 $\hat{\theta}$ 为 $\theta$ 的 ML 估计，即 $\hat{\theta}$ 满足条件

$$
L(x, \hat{\theta}) = \sup_{\theta \in \Theta} L(x, \theta).
$$

令 $\hat{\theta}_0$ 为在总体模型 $X \sim p(x, \theta)$ $(\theta \in \Theta_0)$ 的假设之下，参数 $\theta$ 的 ML 估计，即 $\hat{\theta}_0$ 满足条件

$$
L(x, \hat{\theta}_0) = \sup_{\theta \in \Theta_0} L(x, \theta).
$$

则广义似然比定义为

$$\lambda(x) = \frac{L(x, \hat{\theta})}{L(x, \hat{\theta}_0)}$$

- **广义似然比检验得到的否定域不一定是 UMP 否定域**
- **当满足“单调似然比”与“单边假设”的条件时，得到的是 UMP 否定域**
	- 单调似然比：似然比是关于一个统计量 $T(X)$ 的单调函数
	- 单边假设：问题具有 $H_0:\theta \leq \theta_0 \leftrightarrow H_1:\theta > \theta_0$ 等类似的形式
	- 当这两个条件同时满足时，GLR 会自然导出一个基于充分统计量 $T(X)$ 的检验，并且该检验的否定域具有 $\{T(X) > c\}$ 等类似的形式，可以被证明这是 UMP 的

# 图论

## 图的基本概念

**图论基本定理（握手定理）**
1. 设 $G = \langle V, E \rangle$ 是无向图，$V = \{v_1, v_2, \ldots, v_n\}$，$|E| = m$，则各顶点度数之和为

$$
d(v_1) + d(v_2) + \cdots + d(v_n) = 2m
$$

2. 设 $D = \langle V, E \rangle$ 是有向图，$V = \{v_1, v_2, \ldots, v_n\}$，$|E| = m$，则各顶点出度（+）和与入度（-）和为

$$
d^+(v_1) + d^+(v_2) + \cdots + d^+(v_n) = d^-(v_1) + d^-(v_2) + \cdots + d^-(v_n) = m
$$

- 推论：任何图中，奇数度顶点的个数为偶数

度数列与可图化：设 $G = \langle V, E \rangle$, $V = \{v_1, v_2, \ldots, v_n\}$ ，称 $d = (d(v_1), d(v_2), \ldots, d(v_n))$ 为 $G$ 的度数列，设非负整数列 $d = (d_1, d_2, \ldots, d_n)$ ，若存在图 $G$ ，使得 $G$ 的度数列是 $d$ ，则称 $d$ 为可图化的；特别地，若该 $G$ 是简单图，则称 $d$ 为可简单图化的

**可图化的充要条件**：非负整数列 $d = (d_1, d_2, \ldots, d_n)$ 是可图化的，当且仅当 $d_1 + d_2 + \cdots + d_n = 0 \pmod{2}$ 

图同构：设图 $G_1 = \langle V_1, E_1 \rangle,\ G_2 = \langle V_2, E_2 \rangle$ ，若存在双射 $f: V_1 \to V_2$ 满足

$$
\forall u \in V_1, v \in V_1, (u, v) \in E_1 \iff (f(u), f(v)) \in E_2
$$

且 $\langle u, v \rangle$ 与 $\langle f(u), f(v) \rangle$ 重数相同，则称 $G_1$ 与 $G_2$ 同构，记作 $G_1 \cong G_2$ 

一些类别
- $k$ 正则图：每个顶点读书均为 $k$ 的简单无向图
- $r$ 部图：顶点集合 $V$ 分成 $r$ 个互不相交的子集 $V_i$ ，使得 $G$ 中任何一条边的两个端点都不在同一个 $V_i$ 中
- 完全 $r$ 部图：$V_i$ 中任一个顶点均与 $V_j (i \neq j)$ 所有顶点相邻

周长与围长
- 周长：图中最长圈的长度，记为 $c(G)$ 
- 围长：图中最短圈的长度，记为 $g(G)$ 

定理：在 $n$ 阶图（有 $n$ 个顶点）中，若两个顶点间存在通路，则其长度至多为 $n-1$ ；若存在回路，则其长度至多为 $n$ 

**极大路径**
- 在无向简单图中，路径的两个端点不与路径以外的顶点相邻
- 在有向图中，路径起点的前驱，终点的后继，都在路径内部

**扩大路径法**：任何一条非极大路径，其至少有一个端点与路径本身以外的顶点相邻，则路径还可以扩大，直到变成极大路径为止，所以在分析一些问题时，**可以取出这个图的一个极大路径进行分析**

距离与直径
- 距离：无向图 $G$ 中两个顶点 $u$ 和 $v$ ，称二者间长度最短的路径为短程线，其长度就是二者间的距离，记为 $d_G(u,v)$ （若不连通，则距离为无穷大）
- 直径：图中顶点间的最大距离，记为 $d(G)$ 

二部图判别定理：$G$ 是二部图 $\iff G$ 中无奇圈（长度为奇数的圈）

定理：若 $n$ 阶无向图是连通图，则其边数 $m\geq n-1$ 

## 树

**树的等价定义**：设 $G=(V,E)$ 是 $n$ 阶 $m$ 边无向图，则以下命题等价
1. $G$ 是树（连通无回路）
2. $G$ 中任何 $2$ 顶点之间有唯一路径
3. $G$ 无圈，且边数 $m=n-1$
4. $G$ 连通，且边数 $m=n-1$
5. $G$ 极小连通：连通，且所有边是桥
6. $G$ 极大无回：无圈，且增加任何新边会得到唯一的圈

**生成树与余树**
- 设 $G=(V,E)$ 是 $n$ 阶 $m$ 边无向图，若存在一棵树 $T$，满足 $T \subseteq G \land V(T) = V(G)$，则称 $T$ 为 $G$ 的生成树
- 图上所有边 $e$ ，满足 $e \in E(T)$ 的称为生成树的树枝，共有 $n-1$ 条；否则称为弦，共有 $m-n+1$ 条
- $G$ 中由弦导出的子图被称为生成树 $T$ 的余树 $\bar T = G[E(G) - E(T)]$ 

**定理**：无向图 $G$ 联通等价于 $G$ 有生成树

**定理**：$n$ 阶无向连通图，记 $G-e$ 表示删除边 $e$ ，记 $G\setminus e$ 表示沿着 $e$ 收缩（合并两个端点），$\tau(G)$ 表示 $G$ 的生成树的个数，则对 $G$ 的任意非环边 $e$ ，有

$$\tau(G) = \tau(G-e) + \tau(G\setminus e)$$

- $\tau(G-e)$ 是不含 $e$ 的生成树的个数，$\tau(G\setminus e)$ 是含 $e$ 的生成树的个数

## 图的矩阵表示

**关联矩阵**：设图 $G=\langle V,E \rangle$ ，$V = \{v_1, v_2, \ldots, v_n\}$，$E = \{e_1, e_2, \ldots, e_m\}$，把 $v_i$ 对应到每一行，$e_j$ 对应到每一列，则**有向图的关联矩阵**定义为

$$
M(G) = [m_{ij}]_{n \times m}, \quad m_{ij} =
\begin{cases}
1, & v_i \text{ 是 } e_j \text{ 的起点}, \\
0, & v_i \text{ 与 } e_j \text{ 不关联}, \\
-1, & v_i \text{ 是 } e_j \text{ 的终点}.
\end{cases}
$$

**无向图的关联矩阵**定义为

$$
M(G) = [m_{ij}]_{n \times m}, \quad m_{ij} =
\begin{cases}
1, & v_i \text{ 与 } e_j \text{ 关联}, \\
0, & v_i \text{ 与 } e_j \text{ 不关联}.
\end{cases}
$$

- 若无向图 $G$ 有 $k$ 个连通分支，则 $M(G)$ 为伪对角阵（分块对角阵，由若干对角块构成）

**无向图的基本关联矩阵**：设 $G = \langle V, E \rangle$ 是无环无向图，$V = \{v_1, v_2, ..., v_n\}$，$E = \{e_1, e_2, ..., e_m\}$，取一个顶点为参考点，称 $G$ 的基本关联矩阵是从 $M(G)$ 删除参考点对应的行形成的矩阵，记作 $M_f(G)$ 

**定理**：在模 2 加法意义下，$n$ 阶无环无向连通图 $G$ 的关联矩阵的秩 $r(M(G)) = n - 1$ 
- 在模 2 加法下，图中一个环 / 圈所对应的若干列向量相加会得到 0 向量，因为环上每个顶点的关联次数在模 2 下都是偶数，这正好对应了“这些列线性相关”这一代数性质
- 推论：若 $G$ 的连通分支数量为 $p$ ，则在模 2 加法意义下，$r(M(G))=r(M_f(G))=n-p$ 

**用基本关联矩阵判断生成树**：设 $M_f(G)$ 是 $n$ 阶连通图 $G$ 的一个基本关联矩阵，$M_f'$ 是 $M_f(G)$ 中任意 $n-1$ 列组成的方阵，则在模 2 加法意义下，$M_f'$ 各列所对应的边集 $\{e_{i_1}, e_{i_2}, ..., e_{i_{n-1}}\}$ 的导出子图 $G[\{e_{i_1}, e_{i_2}, ..., e_{i_{n-1}}\}]$ 是 $G$ 的生成树，当且仅当 $M_f'$ 的行列式 $\lvert M_f' \rvert \neq 0$ 

---

**邻接矩阵**：$A(G)=[a_{ij}]$ ，有向图的 $a_{i,j}$ 表示从 $v_i$ 到 $v_j$ 的路径数，无向图的 $a_{ij}$ 表示 $v_i$ 与 $v_j$ 是否相邻（0 为不相邻，1 为相邻）

**定理**：$A = [a_{ij}]_{n \times n}, \ (r \geq 2), \ A^r = \left[a_{ij}^{(r)}\right]_{n \times n}$，则
- $a_{ij}^{(r)}$ ：从 $v_i$ 到 $v_j$ 长度为 $r$ 的通路总数
- $\sum_{i=1}^n \sum_{j=1}^n a_{ij}^{(r)}$ ：长度为 $r$ 的通路总数
- $\sum_{i=1}^n a_{ii}^{(r)}$ ：长度为 $r$ 的回路总数
- 对于无向图，$a^{(2)}_{ii}=d(v_i)$ 
- 推论：$B_r = A + A^2 + \ldots + A^r = \left[ b_{ij}^{(r)} \right]_{n \times n}$，则 $b_{ij}^{(r)}$ 表示从 $v_i$ 到 $v_j$ 长度小于等于 $r$ 的通路总数

**有向图的可达矩阵 / 无向图的连通矩阵**：$P(G)=[p_{ij}]$ ，$p_{i,j}$ 表示从 $v_i$ 到 $v_j$ 是否可达 / 连通（0 为不可达，1 为可达）

---

**无向图的度数矩阵**：每个顶点的度数 $d(v_i)$ 组成的对角阵

**拉普拉斯矩阵**：设 $G = \langle V, E \rangle$ 是无向简单图，则其拉普拉斯矩阵为 $L=D-A$ ，其中 $D$ 为度数矩阵，$A$ 为邻接矩阵，称 $x^TLx$ 为拉普拉斯二次型
- $L_{ii}$ 为顶点 $v_i$ 的度数
- $L_{ij}(i\neq j)$ 表示 $v_i$ 与 $v_j$ 间是否有边（有边为 -1）
- 行和为 0 
- $x^TLx=\sum_{(a,b)\in E}(x_a-x_b)^2$ 

**拉普拉斯矩阵的性质**
- 半正定性：所有特征值均为非负数
- 零特征值的数量等于连通分支的个数

# 补充

例 3.9.6 ：$p(x)dx=dF(x)$ 
例 3.9.17 ：$\max \{ X,Y\}=\frac{1}{2}(X+Y+\lvert X-Y\rvert)$ 
习题 3.18 ：求 $P(\min \{x_1,\cdots,x_n \}\leq y)$ 转换为  $P(\min \{x_1,\cdots,x_n \}\leq y) = 1-P(\min \{x_1,\cdots,x_n \}> y)=1-P(x_1>y,\cdots,x_n>y)$ 




