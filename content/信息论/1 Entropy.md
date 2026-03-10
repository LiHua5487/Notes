
**Random Source**：将信源视为一个随机变量 $X$ ，消息 $x_i$ 出现概率为 $p_i$ 
**Source Coding**：将每个 $x_i$ 都映射成 01 串 $c_i$ （**codewords**）

我们希望 $c_i$ 尽可能短，记码长 $l_i=\lvert c_i\rvert \ \text{(bit)}$ ，一个简单的目标就是让平均码长 $\bar l=\sum p_il_i$ 最小（**minimum code length**）

# Uniquely Decodable Coding

这里有一个隐含的前提，就是需要**无损编码**，希望解码出的信息和发送过来的一模一样，比如想要传输 $A,B,C$ 三个字母，每个字母用一串 01 进行编码，一个简单的想法是

$$
A=0,B=1,C=01
$$

但是**信息的传输可能有无穷多次 infinite times**，这种方式会产生歧义（比如 01 可能是 $C$ ，也可能是 $AB$），这就是有损的，可以调整成下面的方式

$$
A=0,B=10,C=11
$$

>但是现实中很多时候是有损编码，比如传输图像 / 视频，由于人眼的灵敏度有限，一些细小的差别分辨不出来，所以损一点无所谓，还能压缩空间

无损编码需要满足一些条件
- Prefix-free Code：任意 $c_i$ 不是其它 $c_j$ 的前缀（充分条件）
- Kraft Inequality：$\sum 2^{-l_i}\leq 1$ （必要条件）

## Prefix-free Code

Prefix-free Code 不是必要条件，比如前面的例子

$$
A=0,B=01,C=11
$$

假设收到 01，如果把一开始的 0 解释为 A，那剩下的 1 就没法解释了，所以只能解释为 B 

那为啥一般研究的都是 Prefix-free Code 呢，因为有这个定理

$$
\begin{aligned}
\text{Claim:} \\
&\text{For every uniquely decodable code } C=(c_1,\cdots,c_n), \\
&\text{ there exists a prefix-code } C'=(c_1',\cdots,c'_n) \text{ such that } \lvert c_i \rvert=\lvert c'_i \rvert
\end{aligned}
$$

## Kraft Inequality

$$
\begin{aligned}
\text{Lemma:} \\
&\text{Let } C=(c_1,\cdots,c_n),l_i=\lvert c_i\rvert \text{ be a prefix-free code}, \text{ then} \sum 2^{-l_i}\leq 1
\end{aligned}
$$

证明：可以用二叉树进行表示，那 $c_1,\cdots,c_n$ 就是叶子节点，当二叉树为满时，求和正好是 1 ，不满就 <1 

上面二叉树为满的情况对应 **optimal code** ，即平均码长最小，因为如果不满，总可以把无兄弟的那个叶节点对应的编码，换为其父节点对应的编码

# Entropy

有了上面的结论，要求最小平均码长，就可以变为下面的优化问题

$$
\min_{l_1,\cdots,l_n} \sum p_il_i,\quad \text{where}\sum 2^{-l_i}= 1,l_i\geq 0
$$

>这里我们暂时不要求 $l_i$ 时整数，这样能得到更深刻的结论，但是这个解不一定能取到（即只能得到下界，不一定是下确界）

设 $q_i=2^{-l_i}$ ，则

$$
\sum p_il_i=-\sum p_i\log q_i,\quad \sum q_i=1
$$

根据下面的引理

$$
\text{Let}\sum p_i=\sum q_i=1,\quad \text{then}\sum p_i\log q_i \leq \sum p_i\log p_i
$$

可得

$$
l_i=\log \frac{1}{p_i},\quad \min \bar l=\sum p_i\log \frac{1}{p_i} \text{(bit)}
$$

这个最小平均码长就定义为**熵 Entropy**：设 $X$ 是一个随机变量，服从分布 $p=(p_1,\cdots,p_n)$ ，则 $X$ 的熵为

$$
H(X) \coloneqq \sum p_i\log\frac{1}{p_i} \text{(bit)}
$$

由于 $l_i$ 实际上只能取整数，所以 $\lvert l_i-\log\frac{1}{p_i} \rvert < 1$ ，则

$$
\bar l - H(X) < 1
$$

熵的含义
- 最小平均码长
- 信源的信息量的大小（用多长的编码就能把这些信息表示完了）
- 对于每个信息 $x_i$ 的信息量，可以定义为 $\log\frac{1}{p_i}$ 
- 随机变量 $X$ 的不确定性 uncertainty （当 $p_i=1/n$ 时，取得最大值 $\log n$）






