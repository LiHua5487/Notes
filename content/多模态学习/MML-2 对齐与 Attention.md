
# Multimodal Alignment

前面的 Fusion 主要关心如何将不同模态的信息组合起来，Alignment 则更加细致，关注不同模态内部元素之间的对应关系，而不仅是把两种模态整体关联起来

**Discrete Alignment（离散对齐）**：不同模态都可以拆成离散元素，模型直接寻找元素之间的联系，比如一个单词对应一个图像中的对象

**Continuous Alignment（连续对齐）**：当模态是连续信号，例如音频和视频，除了对应关系，还要考虑时间分割、速度变化与同步，比如说话声音和口型需要在时间上对齐

**Contextualized Representation（上下文化表示）**：不一定显式地告诉模型谁对应谁，而是让每个元素在编码时结合其他元素的信息，使它的表示包含上下文，它既可以支持模态内部关系建模，也可以进一步用于跨模态关系建模

# Attention

**Image Captioning 任务**：给定一个图像，输出一句描述它的文本

早期方法使用 **CNN+RNN**：CNN 提取图像特征，得到一个特征向量 $c$ ，RNN 根据这个向量逐词生成句子，但是 RNN 每次生成下一个词用的都是同一个 $c$ ，这就成为了信息瓶颈

![[多模态学习/imgs/img2/image.png]]

**RNN+Attention**：每生成一个词，重新看一次图片，在每时间步生成一个新的上下文向量 $c_t$ 

假设 CNN 提取了 $N=HW$ 个区域特征

$$
Z=\{z_1,z_2,\ldots,z_N\},\quad z_i\in\mathbb R^D
$$

在第 $t$ 个时间步，RNN 已经具有上一时刻的隐藏状态 $h_{t-1}$，其中包含之前生成的文本信息，可以利用 $h_{t-1}$ 判断当前应该关注哪些图像区域

整个 Attention 计算可以分为三步

第一步：计算匹配分数（Alignment Score）

$$
e_{t,i}=f_{\mathrm{att}}(h_{t-1},z_i)
$$

其中 $f_{\mathrm{att}}$ 是一个 MLP，输出一个标量，表示根据当前已经生成的文字信息，第 $i$ 个图像区域有多值得关注

第二步：用 Softmax 得到注意力权重

$$
\alpha_{t,i} =\frac{\exp(e_{t,i})} {\sum_{j=1}^N\exp(e_{t,j})}
$$

第三步：对区域特征加权求和，得到 $c_t$ 

$$
c_t=\sum_{i=1}^N\alpha_{t,i}z_i
$$

![[多模态学习/imgs/img2/image-1.png]]

整个计算过程都是可微的，只需要监督最终生成的文字，就能利用反向传播更新计算注意力权重的网络

## General Attention

我们可以不再把 $h$ 限定为 RNN 隐藏状态，也不把 $z_i$ 限定为图像特征，而是抽象成查询向量 $q$ 和输入向量 $x$ ，分别表示现在想找什么、有哪些信息可以查找，同时把计算 Attention Score 时的打分函数从 MLP 换成点积

$$
e_i=\frac{q^\top x_i}{\sqrt D},\quad \alpha_i=\operatorname{softmax}(e)_i,\quad c=\sum_i\alpha_ix_i
$$

为什么要除以 $\sqrt D$ ？如果不缩放，维度越大，点积往往越大，经过 Softmax 就容易产生极端分布，这会让注意力过于集中，也可能导致 Softmax 梯度变小，而除以 $\sqrt D$，能让打分的数值尺度更稳定

为什么除的数值是 $\sqrt D$ ？假设 $q,x_i$ 的每个分量近似独立，均值为 0、方差为 1，那么 $q^\top x_i=\sum_{j=1}^Dq_jx_{i,j}$ 这一求和包含 $D$ 项，其方差大约是 $D$，标准差约为 $\sqrt D$ 

---

前面的 Attention 使用同一个 $x_i$ 完成两件事：一是与 Query 比较，决定相关性；二是按照权重贡献给最后的输出

但这两件事其实不完全一样，比如你在图书馆找一本书
- Query：你想查找什么书？
- Key：书的索引信息，供你匹配
- Value：书里的实际内容，供你读取

书的索引（用于匹配）和书的内容（用于生成结果）不需要采用完全相同的表示，所以可以把 $x$ 进一步拆分成

$$
k_i=x_iW_K,\quad v_i=x_iW_V
$$

而且查询也不一定只有一次，而是可以有很多 $q$ 

$$
Q=\begin{bmatrix}q_1\\q_2\\\vdots\\q_M\end{bmatrix}
$$

那么 Attention 的完整公式变为

$$
\operatorname{Attention}(Q,K,V) = \operatorname{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)V
$$

设 $x\in \mathbb{R}^{N},\ q\in \mathbb{R}^{d_k}$ ，则

$$
Q\in \mathbb{R}^{M\times d_k},\quad K\in \mathbb{R}^{N\times d_k},\quad V\in \mathbb{R}^{N\times d_v}\\
\operatorname{Attention}(Q,K,V) \in \mathbb{R}^{M\times d_v}
$$

相当于每一个 Query 都会从 Value 中提取一个 $d_v$ 维向量

## Self-Attention

前面的 Query 可以来自外部，比如 RNN 的隐藏状态，而 Self-Attention 的 Query 来自于 $x$ 自身，通过三个不同的可学习矩阵得到

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

![[多模态学习/imgs/img2/image-2.png]]

## Positional Encoding

对于 Query 来自外部的 Attention，是**置换不变**的，即调换 $x$ 中的元素的位置，输出仍然不变；而 Self-Attention 是**置换等变**的，输入顺序改变后，输出结果也会改变，比如输入从 $(x_1,x_2,x_3)$ 变为 $(x_2,x_1,x_3)$ ，则输出变为 $(y_2,y_1,y_3)$ 

但是每一个输出向量 $y_i$ 中并没有包含位置信息，比如“人咬狗”和“狗咬人”，对于“咬”这个词，Self-Attention 得到了完全相同的表示，尽管这两句话中动作的主客体已经交换了

为了把顺序信息加进去，最直接的方式是给每个位置一个不同的向量 $p_t$，让模型同时看到内容和位置

一个经典方法是正弦位置编码：给序列中的每个位置 $pos$ ，生成一个 $d$ 维的位置编码向量，这个向量中的第 $2i$ 与 $2i+1$ 个元素交替使用正余弦的方式计算

$$
\begin{aligned}
PE(pos,2i) &= \sin\left(\frac{pos}{10000^{2i/d}}\right) \\
PE(pos,2i+1) &= \cos\left(\frac{pos}{10000^{2i/d}}\right)
\end{aligned}
$$

为什么要使用 Sin/Cos？除了让不同位置有不同编码，它还有很好的数学性质
- 两个位置编码 $PE_{t}$ 与 $PE_{t+\Delta t}$ 的点积仅取决于偏移量 $\Delta t$ 
- 位置编码的点积是无向的，即 $PE_{t}\cdot PE_{t+\Delta t} = PE_{t}\cdot PE_{t-\Delta t}$ 

## Masked Self-Attention

普通 Self-Attention 允许一个位置关注所有位置，但在生成文本时，预测一个词只能根据之前的词来进行，不能提前看到后面的内容，解决方式非常简单：在 Softmax 之前，屏蔽未来位置

$$
A=\operatorname{softmax}\left( \frac{QK^\top}{\sqrt{d_k}}+M \right)
$$

其中 Mask 矩阵

$$
M_{ij}= \begin{cases} 0,&j\le i\\ -\infty,&j>i \end{cases}
$$

因为 $e^{-\infty}=0$ ，所以未来位置在 Softmax 之后得到的权重就是 0

## Multi-Head Attention

使用多个注意力头，为不同头设置不同的 Q、K、V 投影，关注不同方面

$$
\mathrm{head}_i = \operatorname{Attention} (QW_Q^{(i)},KW_K^{(i)},VW_V^{(i)})\\
\operatorname{MultiHead}(Q,K,V) = \operatorname{Concat}(\mathrm{head}_1,\ldots,\mathrm{head}_H)W_O
$$

![[多模态学习/imgs/img2/image-3.png]]

# Transformer

Transformer 使用 Encoder–Decoder 架构

![[多模态学习/imgs/img2/image-4.png]]

![[多模态学习/imgs/img2/image-5.png]]








































