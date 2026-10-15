# Multi-Party Computation 

问题设定：Alice 和 Bob 一起计算 $f(x,y)$ 的值，其中 $x,y \in \{0,1\}^n,f(x,y) \in \{0,1\}$ ，但是 Alice 只知道 $x$ ，Bob 只知道 $y$ ，二者之间至少需要传递多少信息，才能让双方都知道结果

记 $D(f)$ 为确定性通信复杂度，即在最坏输入情况下，为了正确计算 $f(x,y)$ 所需通信 bit 数的最小值

一个平凡上界是 Alice 直接把 $x$ 发给 Bob，Bob 算出 $f(x,y)$ 后再把结果告诉 Alice，因此 $D(f)\leq n+1$ ，如果只要求 Bob 知道结果，则 $D(f)\leq n$

---

把函数 $f$ 写成一个矩阵 $M_f$ ，其中行对应 Alice 的输入 $x$ ，列对应 Bob 的输入 $y$ ，矩阵元素为

$$
M_f[x,y]=f(x,y)
$$

由于 $x,y\in\{0,1\}^n$ ，因此 $M_f$ 是一个 $2^n\times 2^n$ 的 $0/1$ 矩阵

通信协议的作用可以理解为：通过 Alice 和 Bob 的对话，把当前输入 $(x,y)$ 所在的位置逐渐缩小到一个能确定函数值的区域，这个区域中的所有位置函数值都相同，又显然是一个矩形，被称为**单色矩形**

定义 $\chi(f)$ 为 $M_f$ 的**最小单色矩形划分数**，如果 $D(f)=c$ ，则存在一个 $c$ bit 的协议，从而 $M_f$ 可以被划分成至多 $2^c$ 个单色矩形，因此

$$
\chi(f)\leq 2^{D(f)}
$$

两边取对数得到

$$
D(f)\geq \log_2\chi(f)
$$

直观上，$c$ bit 的通信最多只能区分 $2^c$ 种通信记录，而 $\chi(f)$ 表示至少需要区分多少个单色区域，所以通信量至少要达到 $\log\chi(f)$

反方向也有一个经典上界

$$
D(f)\leq O((\log\chi(f))^2)
$$

因此

$$
\log\chi(f)\leq D(f)\leq O((\log\chi(f))^2)
$$

也就是说，$\chi(f)$ 在一定程度上刻画了确定性通信复杂度

## Rank Lower Bound

直接计算 $\chi(f)$ 通常很难，$\chi(f)$ 的计算是 NP-Hard 的，因此常用更容易计算的 $\operatorname{rank}(M_f)$ 给出下界

将 $M_f$ 看作实数矩阵，有

$$
\chi(f)\geq \operatorname{rank}(M_f)
$$

原因是：若 $M_f$ 被划分成若干单色矩形，则只考虑值为 $1$ 的矩形。每个值为 $1$ 的单色矩形都可以写成一个 rank 为 $1$ 的矩阵

$$
R_i=\mathbf{1}_{A_i}\mathbf{1}_{B_i}^T
$$

于是

$$
M_f=R_1+R_2+\cdots+R_k
$$

利用

$$
r(A+B)\leq r(A)+r(B)
$$

可得

$$
\operatorname{rank}(M_f)\leq \sum_i \operatorname{rank}(R_i)\leq k\leq \chi(f)
$$

所以

$$
D(f)\geq \log\chi(f)\geq \log\operatorname{rank}(M_f)
$$

即

$$
D(f)\geq \log_2\operatorname{rank}(M_f)
$$

## Equality Function

考虑

$$
f_{Eq}(x,y)=1\iff x=y
$$

它的通信矩阵中，对角线为 $1$ ，其他位置为 $0$

一个 $1$-单色矩形最多只能包含一个对角线上的 $1$ 。因为如果它同时包含 $(x,x)$ 和 $(z,z)$ ，其中 $x\neq z$ ，那么由矩形性质，它也必须包含 $(x,z)$ 和 $(z,x)$ ，但这两个位置的函数值为 $0$ ，矛盾

对角线上共有 $2^n$ 个 $1$ ，因此至少需要 $2^n$ 个 $1$-单色矩形，所以

$$
\chi(f_{Eq})\geq 2^n
$$

进而

$$
D(f_{Eq})\geq n
$$

也就是说，在确定性 worst case 正确的要求下，判断两个 $n$ bit 字符串是否相等需要 $\Omega(n)$ bit 通信

## Inner Product Function

例：

$$
f_{\text{innerproduct}}(x,y)=x\odot y=\oplus_{i=1}^n x_iy_i
$$

其中 $\oplus$ 表示模 $2$ 下的加法。这个函数表示模 $2$ 下的内积，即先找出 $x,y$ 哪些位置上均为 $1$ ，再看这些位置的数量是奇数还是偶数

令

$$
f'(x,y)=(-1)^{f(x,y)}
$$

则

$$
M_{f'}[x,y]=(-1)^{x\odot y}
$$

考虑 $M_{f'}$ 中任意两行 $x,z$ 的内积：

$$
\sum_{y\in\{0,1\}^n}(-1)^{x\odot y}(-1)^{z\odot y}
=
\sum_{y\in\{0,1\}^n}(-1)^{(x\oplus z)\odot y}
$$

如果 $x=z$ ，则 $x\oplus z=0$ ，上式为 $2^n$

如果 $x\neq z$ ，则 $x\oplus z\neq 0$ 。当 $y$ 遍历所有 $\{0,1\}^n$ 时，$(x\oplus z)\odot y$ 有一半为 $0$ ，一半为 $1$ ，所以上式为 $0$

因此 $M_{f'}$ 任意两行正交，故

$$
\operatorname{rank}(M_{f'})=2^n
$$

从而 inner product 函数具有线性通信复杂度下界

$$
D(f_{\text{innerproduct}})\geq n
$$

# Randomized Communication

从 worst case 保证正确改为 high prob correct 后，有些函数的通信复杂度会显著降低。例如 Equality 函数在确定性情况下需要 $\Omega(n)$ bit ，但若允许随机化并只要求高概率正确，则只需 $O(\log n)$ bit

做法是使用 fingerprinting 。把 $x=(x_1,\cdots,x_n)$ 看成多项式

$$
p_x(t)=x_1+x_2t+\cdots+x_nt^{n-1}
$$

Bob 同理构造

$$
p_y(t)=y_1+y_2t+\cdots+y_nt^{n-1}
$$

在有限域 $\mathbb{F}_q$ 中随机选取 $r$ ，Alice 发送 $p_x(r)$ ，Bob 比较它和 $p_y(r)$ 是否相等

如果 $x=y$ ，则 $p_x=p_y$ ，一定判断正确；如果 $x\neq y$ ，则 $p_x-p_y$ 是一个次数至多为 $n-1$ 的非零多项式，由代数基本定理，它最多有 $n-1$ 个根，因此误判概率至多为

$$
\frac{n-1}{q}
$$

取 $q$ 为 $n$ 的多项式大小时，错误率可以很小，通信只需 $O(\log q)=O(\log n)$ bit

# Fisher Info and Cramer-Rao Ineq

问题：现有样本 $X=(X_1,\cdots,X_n)$ ，要估计概率分布函数 $f(x;\theta)$ 的参数 $\theta$ ，想让无偏估计的方差尽可能小

定义 Score Function 为

$$
S(x;\theta)=\frac{\partial}{\partial \theta}\ln f(x;\theta)
$$

它表示当参数 $\theta$ 发生微小变化时，样本 $x$ 的 log-likelihood 变化得有多快

定理：

$$
E[S(x;\theta)]=0
$$

证明：

$$
E[S(x;\theta)]
=
\int f(x;\theta)\frac{\partial}{\partial\theta}\ln f(x;\theta)\,dx
$$

由于

$$
\frac{\partial}{\partial\theta}\ln f(x;\theta)
=
\frac{1}{f(x;\theta)}\frac{\partial}{\partial\theta}f(x;\theta)
$$

所以

$$
E[S(x;\theta)]
=
\int \frac{\partial}{\partial\theta}f(x;\theta)\,dx
=
\frac{\partial}{\partial\theta}\int f(x;\theta)\,dx
=
\frac{\partial}{\partial\theta}1
=
0
$$

因此 score 的均值为 $0$ 

## Fisher Information

Fisher Information 定义为

$$
I(\theta)=E[S^2(x;\theta)]
$$

由于 $E[S(x;\theta)]=0$ ，所以也有

$$
I(\theta)=\operatorname{Var}(S(x;\theta))
$$

直观上，Fisher Information 衡量样本对参数 $\theta$ 的敏感程度。如果 $I(\theta)$ 很大，说明参数稍微变化，分布就会发生明显变化，因此样本中包含较多关于 $\theta$ 的信息；如果 $I(\theta)$ 很小，说明参数变化不容易被样本反映出来，因此估计会更困难

对 $n$ 个 i.i.d. 样本，有

$$
I_n(\theta)=nI(\theta)
$$

因为 log-likelihood 可以相加：

$$
\ln f(x_1,\cdots,x_n;\theta)
=
\sum_{i=1}^n \ln f(x_i;\theta)
$$

对应的 score 也相加，而独立样本的信息量相加

## Equivalent Definition

Fisher Info 还有一个等价定义：

$$
I(\theta)
=
-E\left[\frac{\partial^2}{\partial\theta^2}\ln f(x;\theta)\right]
$$

证明思路是从

$$
E[S(x;\theta)]=0
$$

出发，对 $\theta$ 再求导

$$
0
=
\frac{\partial}{\partial\theta}E[S(x;\theta)]
=
\frac{\partial}{\partial\theta}\int f(x;\theta)S(x;\theta)\,dx
$$

展开得到

$$
0
=
\int \frac{\partial f(x;\theta)}{\partial\theta}S(x;\theta)\,dx
+
\int f(x;\theta)\frac{\partial S(x;\theta)}{\partial\theta}\,dx
$$

又因为

$$
\frac{\partial f(x;\theta)}{\partial\theta}
=
f(x;\theta)S(x;\theta)
$$

所以

$$
0
=
E[S^2(x;\theta)]
+
E\left[\frac{\partial S(x;\theta)}{\partial\theta}\right]
$$

即

$$
I(\theta)
=
-E\left[\frac{\partial^2}{\partial\theta^2}\ln f(x;\theta)\right]
$$

## Cramer-Rao Inequality

Cramer-Rao Ineq：对于任意无偏估计 $\hat{\theta}$ ，有

$$
\operatorname{Var}(\hat{\theta})
\geq
\frac{1}{I(\theta)}
$$

若有 $n$ 个 i.i.d. 样本，则

$$
\operatorname{Var}(\hat{\theta})
\geq
\frac{1}{nI(\theta)}
$$

证明思路：由于 $\hat{\theta}$ 无偏，有

$$
E[\hat{\theta}]=\theta
$$

两边对 $\theta$ 求导：

$$
\frac{\partial}{\partial\theta}E[\hat{\theta}]=1
$$

写成积分形式：

$$
\frac{\partial}{\partial\theta}\int \hat{\theta}(x)f(x;\theta)\,dx=1
$$

交换求导和积分：

$$
\int \hat{\theta}(x)\frac{\partial}{\partial\theta}f(x;\theta)\,dx=1
$$

利用

$$
\frac{\partial}{\partial\theta}f(x;\theta)
=
f(x;\theta)S(x;\theta)
$$

得到

$$
E[\hat{\theta}S]=1
$$

又因为 $E[S]=0$ ，所以

$$
E[(\hat{\theta}-\theta)S]=1
$$

由 Cauchy-Schwarz 不等式，

$$
(E[(\hat{\theta}-\theta)S])^2
\leq
E[(\hat{\theta}-\theta)^2]E[S^2]
$$

因此

$$
1
\leq
\operatorname{Var}(\hat{\theta})I(\theta)
$$

即

$$
\operatorname{Var}(\hat{\theta})
\geq
\frac{1}{I(\theta)}
$$
