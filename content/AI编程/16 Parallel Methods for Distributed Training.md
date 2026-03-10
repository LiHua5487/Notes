
# Model Parallelism

在训练时，还需要存储前向传播的各层的中间结果，因为计算关于参数梯度时也会用到；如果采用混合精度训练，还需要存储 FP32 版本的参数；此外，对于 Adam 等优化器，还可能需要储存动量 / 一二阶矩等额外信息

如果只采用数据并行，一个 GPU 的存储开销太大了，所以考虑把模型分到不同设备上，一个简单的想法是把模型中的每几层放到一个设备，但是这样的话后面的设备就需要等待前面设备的计算，反传时也得等待

![[AI编程/imgs/img16/image.png]]

## Pipeline Parallelism

与其先算一个 batch 再传给后面，不如每计算一点数据（micro batch）就传给后面，而且拆分得越细，设备利用率越高

![[AI编程/imgs/img16/image-1.png]]

此外，每个设备只需要存储对应的那几层的参数和梯度，不需要存储整个模型的参数和梯度

## Tensor Parallelism

实际上，还可以把每一层进一步拆分，比如说一个线性层计算矩阵乘法时，完全可以把矩阵拆成一堆行块 / 列块，因为每次计算输出中的一个元素时，只使用了一部分权重

![[AI编程/imgs/img16/image-2.png]]

比如一个 MLP ，第一层是 $Y=GeLU(XA)$ ，第二层是 $Z=Dropout(YB)$ ，把每一层都分成两块

![[AI编程/imgs/img16/image-3.png]]

如果在第一层的时候把 A 竖着分成了两份，那第二层就要把 B 横着分成两份，这样第二层计算时，每个设备只根据 $Y_i$ 就能进行计算，中间不需要再彼此通信，只需要最后再合起来就行

![[AI编程/imgs/img16/image-4.png]]

对于多头注意力层，可以天然的按每个注意力头去分割，不过需要注意一下 softmax 时需要用到其它地方的元素，有些时候不能简单的把它分开

![[AI编程/imgs/img16/image-5.png]]

![[AI编程/imgs/img16/image-6.png]]

## New Perspective

把前向和反向过程表示成计算图

![[AI编程/imgs/img16/image-7.png]]

前面的两种并行方式可以表示为

![[AI编程/imgs/img16/image-8.png]]

它们的计算过程如下

![[AI编程/imgs/img16/image-9.png]]

- Inter-op parallelism 数据传输次数更少，但是设备空闲时间更多
- Intra-op parallelism 数据传输次数更多（需要先复制输入，再汇总输出），但设备空闲时间更少

---

我们考虑其中一个计算节点的划分方式，对于矩阵加法，爱咋分咋分，因为不同维度之间是独立的

![[AI编程/imgs/img16/image-10.png]]

但是对于矩阵乘法，不能随便划分
- 如果要并行化循环 $i$ ：A 按行划分，把 B copy 给每个设备，最后每个设备持有 C 中的一行
- 如果要并行化循环 $k$ ：A 按列划分，B 按行划分，但是这样计算完后，每个设备得到的结果都不是 C 中的某一块，需要通过 all-reduce 操作还原 C ，最后每个设备都持有完整的 C 

![[AI编程/imgs/img16/image-11.png]]

---

考虑多个计算节点的划分方式，如果一个数据在两个计算节点中的划分方式不同，就需要重新进行划分 re-partition 以适配下一个计算节点的划分方式

![[AI编程/imgs/img16/image-12.png]]

下图总结了两个不同的划分方式之间，需要的 re-partition 操作

![[AI编程/imgs/img16/image-13.png]]

经过了这么多铺垫，这个新视角想说这样一个事：要想找到一种效率最高的划分方式，可以定义下面这样一个代价函数，并进行图优化

$$\text{NodeCost}(\text{computation}+\text{communication})+\text{EdgeCost}(\text{re-partition})$$

# Sequence Parallelism

对于 LLM，一个数据（一个文字序列）可能就很长，先回顾一下 Transformer 中的 Self-Attention 层

![[AI编程/imgs/img16/image-16.png]]

---

Solution 1: Re-partition data in Attention layers

![[AI编程/imgs/img16/image-17.png]]

先把序列分成若干份，每个设备持有其中一部分，然后根据持有的序列和 $W_{Q,K,V}$ 计算各自的 $Q_i,K_i,V_i$ ，但是此时每个设备的 $Q,K,V$ 都不是完整的，如果不相互通信，那每个设备只能计算出下图所示的这些部分，没法往下算

![[AI编程/imgs/img16/image-18.png]]

所以需要用 All-Gather 操作，每个设备收集所有其他设备的 $K$ 和 $V$ ，从而得到完整的 $K$ 和 $V$ 矩阵，这样每个设备就能根据各自所持的序列，计算出对应的输出

![[AI编程/imgs/img16/image-20.png]]

缺点
- All-Gather 通信数据量较大
- 每个设备需要存储完整的 $K$ 和 $V$ ，内存开销大

---

Solution 2: Ring Attention

转着圈的把当前持有的 $K_j$ 和 $V_j$ 传递给下一个设备，这样每个设备就能逐渐补充上自己那列中缺少的部分

![[AI编程/imgs/img16/image-21.png]]

尽管 softmax 需要对整个一列进行，但实际上只用转一圈（而不用先转一圈传递 $K$ ，等 softmax 完了再转圈传 $V$ ），具体原理参考下面的推导

简化起见，设 $Q=(q_1,q_2,q_3)^T,K=(k_1,k_2,k_3)^T,V=(v_1,v_2,v_3)^T$ ，要求 $Y=\text{softmax}(QK^T)V$ ，考虑第 $i$ 个设备，即对于 $q_i$ ，可得

$$q_iK^T=(q_ik_1,q_ik_2,q_ik_3)$$

那么 softmax 结果为

$$\text{softmax}(q_iK^T)=(\frac{e^{q_ik_1}}{\sum_{j=1}^3e^{q_ik_j}},\frac{e^{q_ik_2}}{\sum_{j=1}^3e^{q_ik_j}},\frac{e^{q_ik_3}}{\sum_{j=1}^3e^{q_ik_j}})$$

再乘上 $V$ 可得

$$y_i=\frac{\sum_{j=1}^3e^{q_ik_j}\cdot v_j}{\sum_{j=1}^3e^{q_ik_j}}$$

也就是说，我们只需要维护两个累积量，分别累积 $e^{q_ik_j}\cdot v_j$ 和 $e^{q_ik_j}$ 就行

# Bottlenecks of Distributed Training

分布式训练最大的瓶颈就在于不同设备间的通信

![[AI编程/imgs/img16/image-14.png]]

而且随着尺度增大，通信的延迟也会成倍的增大

![[AI编程/imgs/img16/image-15.png]]




