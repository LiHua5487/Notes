
# 随机变量与概率分布

## 离散型随机变量

### 泊松分布

**泊松分布**：设随机变量 $X$ 的所有可能取值是全体非负整数，且

$$P(X = k) = \frac{\lambda^k}{k!} e^{-\lambda}, \quad k = 0, 1, 2,\ldots$$

则称 $X$ 从参数为 $\lambda$ 的泊松分布，记为 $X \sim \mathcal{P}(\lambda)$，其中参数 $\lambda > 0$ 

**性质**：设随机变量 $X \sim \mathcal{P}(\lambda)$，记 $p_k = P(X = k)$，则有如下结论

$$p_0 < p_1 < \cdots < p_{\lfloor \lambda \rfloor} > p_{\lfloor \lambda \rfloor + 1} > \cdots$$

### 几何分布⭐

**几何分布**：若随机变量 $X$ 的所有可能取值是全体正整数，且概率分布满足

$$P(X = k) = (1 - p)^{k - 1} p, \quad k = 1, 2, \ldots,$$

则称 $X$ 服从几何分布，记为 $X \sim G(p)$，其中参数 $0 < p < 1$

**定理**：设 $X$ 是只取正整数的离散型随机变量，若 $X$ 的分布具有无记忆性，则 $X$ 的分布一定为几何分布

无记忆性：表示随机变量 $X$ 未来的行为与过去无关，即满足

$$P(X>n+m|X>m) = P(X>n)$$

## 连续型随机变量

### 指数分布⭐

**指数分布**：如果随机变量 $X$ 的分布密度为

$$p(x) = \begin{cases}
\lambda e^{-\lambda x}, & x \ge 0 \\
0, & x < 0
\end{cases}$$
其中参数 $\lambda > 0$，则称 $X$ 服从参数为 $\lambda$ 的指数分布，记为 $X \sim \text{Exp}(\lambda)$ ，易知其具有无记忆性

### 正态分布⭐

**正态分布**：如果随机变量 $X$ 的分布密度为
$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left\{-\frac{(x - \mu)^2}{2\sigma^2}\right\}$$

其中参数 $\mu \in \mathbb{R}, \sigma > 0$，则称 $X$ 服从参数为 $\mu, \sigma$ 的正态分布，记为 $X \sim N(\mu, \sigma^2)$ 

**标准正态分布函数**：如果随机变量 $Z$ 服从标准正态分布 $Z\sim N(0,1)$ ，定义以下函数

$$\Phi(x) = P(Z \le x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-\frac{t^2}{2}} dt$$

将 $\Phi(x)$ 称为标准正态函数

**定理**：设 $X \sim N(\mu, \sigma^2)$ ，则

$$P(a < X < b) = \int_{a}^{b} \frac{1}{\sigma} \phi\left(\frac{x - \mu}{\sigma}\right) dx = \Phi\left(\frac{b - \mu}{\sigma}\right) - \Phi\left(\frac{a - \mu}{\sigma}\right)$$

**推论**：设 $X \sim N(\mu, \sigma^2)$ ，则对于一切正数 $k$ ，有

$$P(\mu - k\sigma < X < \mu + k\sigma) = \Phi(k) - \Phi(-k) = 2\Phi(k) - 1$$

### 伽马分布

伽马函数 $\Gamma : (0, +\infty) \to \mathbb{R}$ 定义为

$$\Gamma(\alpha) = \int_{0}^{+\infty} y^{\alpha - 1} e^{-y} dy$$

- $\Gamma(\alpha + 1) = \alpha \Gamma(\alpha)$
- $\Gamma(1) = 1, \ \Gamma\left(\frac{1}{2}\right) = \sqrt{\pi}$

**伽马分布**：如果随机变量 $X$ 的分布密度为

$$p(x) = 
\begin{cases}
\frac{\beta^{\alpha}}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, & x > 0 \\
0, & x \le 0
\end{cases}$$
其中参数 $\alpha, \beta > 0$ ，则称随机变量 $X$ 服从伽马分布，记为 $X \sim \Gamma(\alpha, \beta)$ 

## 随机变量的函数

### 概率密度变换⭐

**复合函数的概率分布**：随机变量 $X$ 有分布密度 $p(x)$ ，且在区间 $(a, b)$ 上满足 $P(a < X < b) = 1$ ；随机变量 $Y = f(X)$ ，其中 $f(x)$ 是 $(a, b)$ 上可导的连续函数，$g(y)$ 是 $f(x)$ 的反函数，且 $g'(y)$ 存在，则 $Y$ 的分布密度为
$$q(y) = 

\begin{cases}
p(g(y))|g'(y)|, & y \in (\alpha, \beta) \\
0, & \text{其他情况}
\end{cases}$$
其中 $(\alpha, \beta)$ 是反函数 $g(y)$ 的存在区间，即

$$\alpha = \min\{A, B\}, \ \beta = \max\{A, B\}, \ A \triangleq \lim_{x \to a^+} f(x), \ B \triangleq \lim_{x \to b^-} f(x),$$

### 期望与方差

**常见分布的期望**
- 泊松分布 $X \sim \mathcal{P}(\lambda)$ ：$E(X)=\lambda$ 
- 几何分布 $X \sim G(p)$ ：$E(X)=\frac{1}{p}$
- 指数分布 $X \sim \text{Exp}(\lambda)$ ：$E(X)=\frac{1}{\lambda}$ 
- 伽马分布 $X \sim \Gamma(\alpha, \beta)$ ：$E(X)=\frac{\alpha}{\beta}$ 

**期望的计算与性质**
（1）离散型 $E(f(X))=\sum_k f(x_k)p_k$ ；连续型 $E(f(X)) = \int_{-\infty}^{\infty} f(x)p(x)dx$
（2）线性性：$E(a(X)) = aE(X), \quad E(X + Y) = E(X) + E(Y)$
（3）单调性：若 $X \ge Y$ ，则 $E(X) \ge E(Y)$ 
（4）若随机变量 $X, Y$ 独立，则 $E(XY) = E(X)E(Y)$
（5）$E|X| \ge |E(X)|$

**方差的计算与性质**
（1）$\text{var}(X) = E(X^2) - (E(X))^2$
（2）$\text{var}(aX + b) = a^2 \text{var}(X)$
（3）若随机变量 $X,Y$ 独立，则 $\text{var}(X + Y) = \text{var}(X) + \text{var}(Y)$ 

指标函数：定义一个指标函数 $1_A$，它的值在事件 $A$ 上为 1，在事件的补集上为 0 

$$1_A = 
\begin{cases}
1, & \text{当事件A发生时} \\
0, & \text{其它情况}
\end{cases}$$
其期望 $E(1_A)=P(A)$ 

## 随机变量的其它数学特征

**分位数**：若 $X$ 是随机变量， $0 < p < 1$ ，且
$$P(X < a) \le p \le P(X \le a)$$

则称 $a$ 为 $X$ 的一个 $p$ 分位数；当 $p = 0.5$ 时, 也称 $a$ 为一个中位数

# 随机向量

## 二维随机向量的分布

### 联合分布与边缘分布⭐

**离散型随机变量的联合分布**：如果 $\xi=(X,Y)$ 只能取至多可列个不同的值 $a_1,a_2,\cdots$ （有限或无穷可列），则称 $\xi$ 为离散型二维随机变量，且

$$P(\xi = (x,y)) = P(X = x,Y = y)$$

可以用以下形式的概率分布表来表示

$$P(X_1,X_2) = \begin{array}{c|cc}
& X_2 = 0 & X_2 = 1 \\
\hline
X_1 = 0 & P(0,0) & P(0,1) \\
X_1 = 1 & P(1,0) & P(1,1) \\
\end{array}$$

**连续型随机变量的联合分布**：设 $\xi = (X,Y)$ 为二维随机量，若存在密度函数 $p(x,y)$ 使得
$$P(\xi \in D) = \iint_D p(x,y)dxdy$$

对任意开矩形 $D$ 成立，则称 $\xi$ 为连续型随机量，称 $p(x,y)$ 为 $\xi$ 的联合密度，也称概率密度

**连续型二维随机变量的均匀分布**：设 $G$ 是平面上面积为 $a(0 < a < +\infty)$ 的区域，若

$$p(x,y) = \begin{cases}
\frac{1}{a}, & (x,y) \in G \\
0, & \text{其他}
\end{cases}$$
则称二维随机量 $\xi = (X,Y)$ 服从 $G$ 上的均匀分布，可得 $\xi$ 落在 $G$ 的一个子区域的概率等于其面积占比，即 $P(\xi \in A) = \frac{S(A)}{a}$ 

**边缘分布**：对于二维随机量 $\xi = (X,Y)$ ，$X$ 的概率分布称为 $\xi$ 关于 $X$ 的边缘分布，$Y$ 的概率分布称为 $\xi$ 关于 $Y$ 的边缘分布；对于连续型随机变量，边缘分布密度为
$$p_X(x) \triangleq \int_{-\infty}^{+\infty} p(x,y)dy, \quad p_Y(y) \triangleq \int_{-\infty}^{+\infty} p(x,y)dx$$

**随机向量的分布函数**：设 $\xi = (X,Y)$ 是二维随机变量，则称

$$F(x,y) = P(X \leq x, Y \leq y) \quad (x,y \in \mathbb{R})$$

为 $\xi$ 的分布函数，也称为 $(X,Y)$ 的联合分布函数

### 二维正态分布

**二维正态分布**：若 $\xi = (X,Y)$ 的联合密度 $p(x,y)$ 满足如下表示

$$p(x,y) = \frac{1}{2\pi\sigma_1\sigma_2\sqrt{1 - \rho^2}} \exp \left\{ -\frac{u^2 + v^2 - 2\rho uv}{2(1 - \rho^2)} \right\}$$

则称 $\xi$ 服从二维正态分布，其中

$$u = \frac{x - \mu_1}{\sigma_1}, \quad v = \frac{y - \mu_2}{\sigma_2}$$

共有 5 个参数 $\mu_1, \mu_2 \in \mathbb{R}, \sigma_1, \sigma_2 > 0, \rho \in (-1,1)$ ，其中 $\rho$ 为相关系数；可得 $X$ 和 $Y$ 的边缘分布为 $X \sim N(\mu_1, \sigma_1^2), Y \sim N(\mu_2, \sigma_2^2)$

## 条件分布⭐

**条件分布密度**：设二维随机变量 $(X,Y)$ 有联合密度 $p(x,y)$ ，若边缘分布 $p_Y(y) > 0$ ，则称

$$p_{X|Y}(x|y) = \frac{p(x,y)}{p_Y(y)}$$

为在 $Y = y$ 条件下 $X$ 的条件分布密度

**随机向量的全概率公式**：设 $X$ 和 $Y$ 是连续型随机变量，则

$$\begin{aligned}p_X(x) = \int_{-\infty}^{+\infty} p(x,y)dy = \int_{-\infty}^{+\infty} p_Y(y)p(x|y)dy \\
p_Y(y) = \int_{-\infty}^{+\infty} p(x,y)dx = \int_{-\infty}^{+\infty} p_X(x)p(y|x)dx\end{aligned}$$

**随机向量的贝叶斯公式**：设 $X$ 和 $Y$ 是连续型随机变量，则
$$p(y|x) = \frac{p_Y(y)p(x|y)}{\int_{-\infty}^{+\infty} p_Y(y)p(x|y)dy}\\

\quad p(x|y) = \frac{p_X(x)p(y|x)}{\int_{-\infty}^{+\infty} p_X(x)p(y|x)dx}$$

## 随机变量的独立性

**二维随机向量的独立性**：对于二维随机向量 $\xi=(X,Y)$ ，$X$ 与 $Y$ 相互独立的充要条件为

$$p(x,y) = p_X(x)p_Y(y)$$

或

$$F(x,y)=F_X(x)F_Y(y)$$

## 二维随机向量的函数⭐

**随机变量的二元函数的概率分布**：假设二维随机量 $(X,Y)$ 有联合密度 $p(x,y)$ ，随机变量 $Z = f(X,Y)$ ，对于任何实数 $z$ ， 令 $A = \{(x,y) : f(x,y) \leq z\}$ ， 则 $Z$ 的分布函数为

$$P(Z \leq z) = P(Z \in A) = \iint_A p(x,y)dxdy.$$

**定理**：设二维随机量 $(X,Y)$ 有联合密度 $p(x,y)$ ，随机变量 $Z = X + Y$ ，则 $Z$ 的分布密度为

$$p_Z(z) = \int_{-\infty}^{+\infty} p(x, y)dx = \int_{-\infty}^{+\infty} p(x, z - x)dx$$

## 二维随机向量的概率密度变换⭐

**定理**： $\xi = (X,Y)$ 为连续型，有密度 $p(x,y)$ ，区域 $A$ 满足 $P((X,Y) \in A) = 1$ ，假设 

$$\eta = (U,V), \quad 其中 \quad U = f(X,Y), \quad V = g(X,Y)$$

如果满足
（1） $P(\xi \in A) = 1 \quad \text{且} \quad (f,g) : A \to G \text{ 是一一的}$
（2） $f,g \in C^1(A), \quad \text{且} \quad \frac{\partial(u,v)}{\partial(x,y)} \neq 0, \quad \forall (x,y) \in A$
那么，$\eta$ 是连续型，且

$$p_{U,V}(u,v) = p(x(u,v),y(u,v)) \frac{1}{\lvert J\rvert}, \quad (u,v) \in G$$

其中

$$J=\left| \frac{\partial(u,v)}{\partial(x,y)} \right| = \left| \begin{array}{cc}
\frac{\partial u}{\partial x} & \frac{\partial u}{\partial y} \\
\frac{\partial v}{\partial x} & \frac{\partial v}{\partial y}
\end{array} \right|$$

**定理**：假设二维随机量 $(X,Y)$ 有联合密度 $p(x,y)$ ，随机变量 $Z = f(X,Y)$ ，则
$$E[Z] = E[f(X,Y)] = \int_{x} \int_{y} f(x,y)p(x,y) dx dy$$

## 二维随机向量的数字特征⭐

**协方差**：设随机变量 $X,Y$ 的期望和方差存在，则称

$$\text{cov}(X,Y)=E(X - E(X))(Y - E(Y))=E(XY)-E(X)E(Y)$$

为 $X$ 与 $Y$ 的协方差，记为 $\text{cov}(X,Y)$ 或 $\sigma_{XY}$ ；若 $\text{cov}(X,Y) = 0$ ，则称 $X$ 与 $Y$ 不相关；协方差具有以下性质

$$(\text{cov}(X,Y))^2 \leq \text{var}(X)\text{var}(Y)$$

**性质**
（1）$\text{cov}(X + Y, U + V) = \text{cov}(X,U) + \text{cov}(X,V) + \text{cov}(Y,U) + \text{cov}(Y,V)$
（2）$\text{cov}(aX,bY)=ab\text{cov}(X,Y)$
（3）$\text{cov}(X + c,Y + d) = \text{cov}(X,Y)$
（4）$\text{cov}(X,X)=\text{var}(X)$
（5）$\text{cov}(X,Y)=\text{cov}(Y,X)$
（6）协方差为 0 不代表 $X$ 和 $Y$ 不相关，只代表二者不线性相关

**相关系数**：设随机变量 $X$ 和 $Y$ 的方差存在且有限，则称

$$\rho=\frac{\text{cov}(X,Y)}{\sqrt{\text{var}(X)}\sqrt{\text{var}(Y)}}$$

为 $X$ 与 $Y$ 的相关系数，记为 $\rho_{XY}$ ，简记为 $\rho$ ；$|\rho| \leq 1$ ， 二者独立则 $\rho=0$ ，但 $\rho = 0$ 仅代表二者不是线性相关，$|\rho| = 1$ 说明二者符合线性关系

## 条件期望⭐

**条件期望**：设 $X$ 和 $Y$ 是两个随机变量
（1）离散型

$$E(X|Y = y)=\sum_i x_i P(X = x_i|Y = y)$$

（2）连续型

$$E(X|Y = y)=\int_{-\infty}^{+\infty} x p_{X|Y}(x|y)dx$$

**定理**：设二维随机变量 $(X,Y)$ 有联合密度 $p(x,y)$，则

$$E(X) = \int_{\{y: p_Y(y) > 0\}} E(X|Y = y) p_Y(y) dy.$$
