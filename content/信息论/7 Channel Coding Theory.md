
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
\lim_{n\to \infty} P(\lvert \frac{1}{n} \sum g(x_i) - E[g(x)] \rvert \geq \epsilon) \to 0
$$

取 $g(x) = \log \frac{1}{p(x)}$ ，可得 $P(X_1,\cdots,X_n) \to 2^{-nH(X)}$ ，据此定义**典型集 Typical Set**

$$
A_\epsilon^{(n)} = \{(x_1,\cdots,x_n) \mid 2^{-n(H(x)+\epsilon)} \leq p(x_1,\cdots,x_n) \leq 2^{-n(H(x)-\epsilon)}\}
$$

其中满足这种条件的序列 $(x_1,\cdots,x_n)$ 就是**典型序列**，表示“在期望情况”附近的序列，即满足 $p(x_1,\cdots,x_n) \in 2^{-n(H(X)\pm\epsilon)}$ 

>如果 $P(X)$ 为伯努利分布，期望为 $p$ ，当 $p \neq \frac{1}{2}$ 时，典型集的数量占比很小

典型集的性质
- 典型集的概率接近 1 ：$P(X^n \in A_\epsilon^{(n)}) \to 1$ ，即随机序列几乎一定落在典型集，所以只需考察典型序列
- 每个典型序列出现概率差不多：$P(x^n) \approx 2^{-nH(X)}$ ，可见典型集的大小约为 $\lvert A_\epsilon^{(n)} \rvert \approx 2^{nH(X)}$ 

# Joint Typical Set & Seq

大数律：对于 $(X_1,Y_1), \cdots, (X_n,Y_n) \overset{i.i.d.}{\sim} P(X,Y)$ ，有

$$
\lim_{n\to \infty} P(\lvert \frac{1}{n} \sum g(x_i,y_i) - E[g(x,y)] \rvert \geq \epsilon) \to 0
$$

定义**联合典型序列** $(x_1,y_1,\cdots,x_n,y_n)$ 为满足以下条件的序列
- $p(x_1,y_1,\cdots,x_n,y_n) \in 2^{-n(H(X,Y)\pm\epsilon)}$ 
- $p(x_1,\cdots,x_n) \in 2^{-n(H(X)\pm\epsilon)}$ 
- $p(y_1,\cdots,y_n) \in 2^{-n(H(Y)\pm\epsilon)}$ 

**联合典型集**为联合典型序列组成的集合，有以下性质
- $P((X,Y)^n \in A_\epsilon^{(n)}) \to 1$ 
- $P((x,y)^n) \approx 2^{-nH(X,Y)}, \lvert A_\epsilon^{(n)} \rvert \approx 2^{nH(X,Y)}$ 

联合典型序列有两种随机生成方式
- 直接根据联合分布抽取：$(X_i,Y_i) \sim P(X,Y)$ 
- 先抽 $X$ ，再按条件概率抽 $Y$ ：$X_i \sim P(X), Y_i \sim P(Y\mid X)$ 

**定理**：如果独立的随机抽取 $(x_1,\cdots,x_n)\in A_\epsilon^{(n)}(x)$ 与 $(y_1,\cdots,y_n)\in A_\epsilon^{(n)}(y)$ ，则二者恰好组成联合典型序列的概率为 $\frac{2^{nH(X,Y)}}{2^{nH(X)}\cdot 2^{nH(Y)}} = 2^{-nI(X;Y)}$ 

# Channel Coding Theory

那怎么辨别得到的信息中，哪些是噪声，哪些真正来源于 $X$ 

**码率 rate**：每次平均传递的信息量

$$
R = \lim_{n\to \infty} \frac{\log M}{n}
$$

- $M$ ：消息数量（假设消息均匀分布，则信息量为 $\log M$）
- $n$ ：码字长度，亦即信道使用次数

**信道编码定理 Noisy Channel Coding Thm**：设信道容量为 $C$ ，码率为 $R$ 
- 若 $R < C$ ，则存在一种编码方式，使得解码错误率能无限接近 0 
- 若 $R > C$ ，则存在 $\delta > 0$ ，使得解码错误率总是不小于 $\delta$ （无法接近 0 ）

---

证明（$R < C$ 的情况）：消息总数为 $M = 2^{nR}$ ，且服从均匀分布

编码过程：设计一个码本 codebook ，用一个矩阵 $C = (C_i(j))_{M\times n}$ 表示，将每个消息 $m_i$ 对应到长度为 $n$ 的编码 $c_i = (C_i(1),\cdots,C_i(n))$ 

不去构造一个具体的码本，而是考察随机的码本，看平均情况下的错误率，即设 $C_i(j) \overset{i.i.d}{\sim} P(X)$ ，其中 $P(X)$ 是使得 $C = I(X;Y)$ 的那个分布

解码过程：设收到 $(y_1,\cdots,y_n)$ ，从所有的 $c_i$ 中找到一个编码 $(x_1,\cdots,x_n)$ ，使得 $x^n$ 与 $y^n$ 组成联合典型序列，且满足
- $x^n$ 和 $y^n$ 均为典型序列（若收到的 $y^n$ 不是典型序列，则视为解码失败）
- $x^n$ 是唯一的（若 $y^n$ 与不只一个编码构成联合典型序列，也视为失败）

下证为什么这种解码的错误率趋于 0 
- $y^n$ 是根据 $P(Y\mid X)$ 与 $P(X)$ 抽取的，其不是典型序列的概率趋于 0 
- 发送的编码 $x^n$ 与收到的 $y^n$ 是按 $P(X,Y)$ 抽取的，二者不是联合典型序列的概率也趋于 0 
- 对任意 $c_i$ ，与 $y^n$ 组成联合典型序列概率为 $2^{-nI(X;Y)}=2^{-nC}$ ，则 $x^n$ 不是唯一的概率 $\leq (M-1)\cdot 2^{-nC}=2^{-n(C-R)}\to 0$ 

---

证明（$R > C$ 的情况）：会用到 Fano 不等式


