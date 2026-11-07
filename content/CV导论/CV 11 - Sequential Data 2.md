---
date: 2025-05-14
tags:
  - CV
---

# RNN

可以堆叠多个 RNN ，有多次矩阵乘法，增强表达能力

![[CV导论/imgs/img11/image-5.png|315x359]]

也可以用两个 RNN，一个从前向后、一个从后向前，把两者的状态组合起来，称为**双向 RNN**，这样每个位置都能利用前后文，适合整段输入已经给定的理解或标注任务；逐词自回归生成时，未来的词尚不存在，就不能直接使用未来信息

## Truncation

序列很长时，完整展开并保存所有中间状态会占用大量内存，因此可以采用**截断 truncation**，按较短片段处理，并限制一次反向传播跨越的时间步数

![[CV导论/imgs/img11/image.png|226x267]]

其中的 $h_0$ 如何获取？

一种方式是把各片段视为独立序列，每段都从 $h_0=0$ 开始，处理简单，但片段之间的信息被丢掉，模型无法利用跨片段的上下文

![[CV导论/imgs/img11/image-1.png|614x346]]

另一种方式是把上一片段末尾的隐藏状态传给下一片段，但在边界处切断梯度，称为 **truncated BPTT**，这样前向信息能跨越片段，反向传播只在当前窗口内进行

实现时可以对传来的状态执行 detach，保留数值但不保留之前的计算图，然后处理下一片段，不必每次更新后重新计算整个历史序列，代价是无法通过当前损失直接训练窗口之前的依赖

## Beam Search

在输出时，有以下策略

**贪心解码 greedy decoding**：每次取 softmax 后概率最高的 token

- 对于特定的开头输入，只能产生固定的输出

**随机采样 sampling**：根据预测的概率分布抽取 token，不只考虑概率最高的那个

- 可以有多样的输出
- 但是有小概率可能选一个不合理的，偏差越来越大

输出每个词时，考虑前面 T 个词，其概率为

$$
\begin{aligned}
P(y|x) &= P(y_1|x) P(y_2|y_1, x) P(y_3|y_1, y_2, x) \cdots P(y_T|y_1, \dots, y_{T-1}, x) \\
&= \prod_{t=1}^T P(y_t | y_1, \dots, y_{t-1}, x)
\end{aligned}
$$

考虑所有的组合，使这个概率最大化，但这开销很大

采用 **beam search（束搜索）**，每一步只保留累计序列概率最高的 $k$ 个候选，$k$ 称为 beam size

每个候选都可以扩展下一个词，再从所有扩展后的候选序列中取前 $k$ 个，也可以先为每个候选保留自己的前 $k$ 个扩展，再统一筛选，比较的是整段候选序列的累计概率，实际通常累加对数概率以避免数值下溢
虽然不一定保证最优，但效率更高

## Embedding Layer

有时候，输入的词可能有很多种，如果直接用 one-hot vector ，维数会很大
可以先转化为一个向量，即词嵌入 embedding

![[CV导论/imgs/img11/image-2.png]]

embedding 可以随机初始化后随任务训练，也可以使用预训练向量，它把离散词表中的编号映射为可学习的连续特征

## Gradient Flow

![[CV导论/imgs/img11/image-3.png]]

![[CV导论/imgs/img11/image-4.png]]

隐藏状态会通过许多时间步影响后面的损失，可以先看从 $h_1$ 传到最后损失 $L_T$ 的这一条路径，记 $J_t=\partial h_t/\partial h_{t-1}$，则

$$
\begin{aligned}
\frac{\partial L_T}{\partial h_1}&=\frac{\partial L_T}{\partial h_T}J_TJ_{T-1}\cdots J_2 \\
J_t&=\operatorname{diag}(1-h_t^2)W_{hh}
\end{aligned}
$$

这里使用 tanh 激活，$\operatorname{diag}$ 把各分量导数组成对角矩阵，矩阵乘积需要保留时间顺序，因为不同矩阵一般不能交换；参数共享时，参数梯度还需要把它在不同时间步产生的贡献相加

tanh 的导数在 $[0,1]$ 内，多个导数与权重矩阵连乘，可能让梯度逐步缩小，也可能因权重的放大作用而爆炸，关键在于整个连乘的效果
这进一步限制了长程学习能力，使得实际上考虑的信息小于 sequence length
最终导致生成的内容前言不搭后语

如果去掉 tanh ，连乘仍然可能导致梯度消失/爆炸，解决不了问题
对于梯度爆炸，可以采用**梯度裁剪 gradient clipping**，比如梯度范数超过阈值时整体缩小，避免一次更新过大
对于梯度消失，不能简单的直接缩放，要调整结构

## Summary

RNN 优点

- 能处理任意长度的输入序列
- 单向 RNN 的状态理论上可以利用此前所有输入，但实际受梯度传播和容量限制
- 模型大小不会随着输入长度增加而增大
- 不同时间步共享参数，用同一套更新规则处理输入

缺点

- 需要不断循环，速度慢
- 长程关系的学习能力差

# LSTM (Long Short Term Memory)

普通 RNN 的 $h_t$ 理论上能携带很久以前的信息，但实际难以长期保留，**长短期记忆网络 LSTM**额外引入 **cell state** $c_t$，通过门控控制信息的保留、写入与输出
$h_t$ 与 $c_t$ 等长，对于 cell state ，可以读写与清空

![[CV导论/imgs/img11/image-6.png]]

对于前面的 hidden state $h_{t-1}$ 与当前输入 $x_t$ ，先计算得到 4 个值

![[CV导论/imgs/img11/image-7.png|379x243]]

这里 $i,f,o$ 是门 gate，通常通过 sigmoid 得到 $0\sim1$ 的数，分别控制写入、保留和输出的程度；$g$ 是候选写入内容，通常通过 tanh 得到，与门承担的作用不同

- $i$：input gate，控制候选内容写入多少
- $f$：forget gate，控制旧 cell state 保留多少
- $o$：output gate，控制 cell state 如何影响隐藏输出
- $g$：candidate，准备写入的新内容

$$
\begin{aligned}
c_t&=f_t\odot c_{t-1}+i_t\odot g_t \\
h_t&=o_t\odot\tanh(c_t)
\end{aligned}
$$

$\odot$ 表示逐元素相乘，cell state 通过加法累积更新，当遗忘门接近 1 时，能提供较直接的信息和梯度通路，改善长期依赖的学习

这个单元的计算图如下，最上面 $c_t$ 的传播就类似于 skip link
极端情况 $f = 1,\ i = 0$ ，那 $c_t$ 永远不变

![[CV导论/imgs/img11/image-8.png|430x271]]

# Sequence to Sequence

给一段序列，输出一个序列，比如翻译问题
与之前的文本生成不同（可以只给一个开始的词）

可以使用 RNN 实现，引入两个特殊 token `[start]` `[stop]`

![[CV导论/imgs/img11/image-9.png]]

通过 **encoder（编码器）**将输入序列编码成上下文表示，再利用 **decoder（解码器）**逐步生成输出，这里的“编码”是提取特征，并不是加密
但 encoder 压缩到一个特征向量，可能导致信息丢失

补充：
Fusion 信息聚合

- 直接加
- 逐元素乘 point/element-wise multiply
- concat
ablation study 消融实验：在其他条件尽量一致时，去掉或更换某个部分，观察它对结果的影响
multi-modal 多模态

## Attention

在 segmentation 中，bottleneck 的问题可以通过 skip  link 解决
但对于 Seq2Seq ，显然不能直接把句子中的一段词直接传给后面
需要整句处理后传给后面，生成每个词都能看到整句信息

![[CV导论/imgs/img11/image-10.png]]
生成第 $t$ 个词时，利用 decoder 当前可用的状态作为 query，与 encoder 的各个隐藏状态 $h_i$ 计算相关性分数，再对输入位置做 softmax，得到注意力权重

$$
\begin{aligned}
e_{t,i}&=\operatorname{score}(s_{t-1},h_i) \\
\alpha_{t,i}&=\frac{\exp(e_{t,i})}{\sum_j\exp(e_{t,j})} \\
c_t&=\sum_i\alpha_{t,i}h_i
\end{aligned}
$$

权重较大的位置对上下文向量贡献更多，attention 的参数跟随最终任务损失一起训练，通常不需要额外标注“这个输出词该看哪个输入词”，但翻译等任务本身仍然可以使用监督数据

生成下一个词时，decoder 状态已经改变，因此可以重新分配注意力，得到另一个上下文向量

![[CV导论/imgs/img11/image-11.png]]

由于每个上下文向量都不同，可以避免 bottleneck
同时每个上下文向量关注了原句的不同侧重的信息
