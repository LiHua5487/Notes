
# Channel Capacity

我们想知道：在一个有噪声的通信信道里，信息最多能以多快的速度可靠传递（这里的“可靠”意思是：只要编码方式足够好，错误率可以被压得非常低）

**带噪信道 noisy channel**：设输入输出是随机变量 $X,Y$ ，则可以用 $P(Y\mid X)$ 去建模

在这种建模下，发送方传递给接收方的信息量可以用互信息 $I(X;Y)$ 表示，即“观察到输出 $Y$ 后，我们对输入 $X$ 额外了解了多少”
- 如果信道可靠，看到 $Y$ 基本就知道 $X$，则 $I(X;Y)$ 较大
- 如果信道噪声很大，$Y$ 和 $X$ 几乎没关系，则 $I(X;Y)$ 较小

注意 $I(X;Y)$ 不仅取决于 $P(Y\mid X)$ ，还取决于输入的分布 $P(X)$ （给定 $P(X)$ 和 $P(Y\mid X)$ 可以算出 $P(X,Y)$ 和 $P(Y)$ ，就能算出互信息）
- 比如输入端总是发送 0 ，那无论输出端接收到什么，也了解不到新的信息，此时互信息很低
- 但如果 $P(X)$ 是均匀的，互信息就比较大

**信道容量 Channel Capacity**：对给定的信道（与输入 $X$ 使用的字符表，如 $\{0,1\}$ ），信道最大能传递的信息量是多少

$$
C = \max_{P(X)} I(X;Y)
$$

---

例：有一个信道，输入是 $A,B,\cdots,Z$ ，传递过程中 $A$ 有 0.5 概率变为$A/B$ ，$B$ 有 0.5 概率变为 $B/C$ ，……，$Z$ 有 0.5 概率变为 $Z/A$ ，求其信道容量

由 $I(X;Y) = H(Y) - H(Y\mid X)$ ，只需求 $H(Y)$ 和 $H(Y\mid X)$ 

对于 $H(Y\mid X)$ ，其含义是“已知 $X$ 的情况下，$Y$ 的不确定性”，那就是两种情况各 0.5 概率，故 $H(Y\mid X) = 1$ 

而 $H(Y) \leq \log_2 26$ ，此时对应 $Y$ 是均匀分布，故 $I(X;Y) \leq \log_2 13$ 

# Asymptotic Equi-partition Property (AEP)

大数律：对于 $X_1, \cdots, X_n \overset{i.i.d.}{\sim} P(X)$ ，有

$$
\lim_{n\to \infty} P(\lvert \frac{1}{n} \sum g(x_i) - E[g(x_i)] \rvert \geq \epsilon) \to 0
$$

取 $g(x) = \log \frac{1}{p(x)}$ ，可得 $P(X_1,\cdots,X_n) \to 2^{-nH(X)}$ ，据此定义**典型集 Typical Set**

$$
A_\epsilon^{(n)} = \{(x_1,\cdots,x_n) \mid 2^{-n(H(x)+\epsilon)} \leq p(x_1,\cdots,x_n) \leq 2^{-n(H(x)-\epsilon)}\}
$$

其中满足这种条件的序列 $(x_1,\cdots,x_n)$ 就是**典型序列**，表示“在期望情况”附近的序列





# Channel Coding Theory

那怎么辨别得到的信息中，哪些是噪声，哪些真正来源于 $X$ 



