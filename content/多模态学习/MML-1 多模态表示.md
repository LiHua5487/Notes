
# 三种表示方式

不同模态的信息要怎么进行表示，让模型既能理解各自的信息，又能理解不同模态之间的关联？

**Fusion（融合）**：把多个模态融合成一个表示

$$
z=f(x_A,x_B)
$$

**Coordination（协调）**：不同模态分别编码，并学习它们的对应关系

$$
z_A=f_A(x_A),\quad z_B=f_B(x_B)
$$

**Fission（拆分）**：把不同模态的信息进一步拆开，避免不同信息混在一起

$$
(x_A,x_B)\rightarrow (z_A,z_{\text{shared}},z_B)
$$

# Fusion

## 基本的融合方法

实际输入通常不是直接可以相加的向量，因此各模态输入一般先通过各自的 编码器得到特征 $a=f_A(x_A)$ 和 $b=f_B(x_B)$ ，根据融合的时期有以下划分

![[多模态学习/imgs/img1/image-5.png]]

- **早期融合**：有机会学习特征之间的交互
- **晚期融合**：更容易独立训练各模态，但可能丢失细粒度交互信息

**拼接**：$z=[a;b]$，保留两边信息，特征的交互则交给后续网络学习

**加法**：$z=W_Aa+W_Bb$，先映射到相同维度再相加

**逐元素乘法**：$z=(W_Aa)\odot(W_Bb)$，通过对应维度乘积形成交互

**双线性融合**：$z_k=a^\mathsf TW_kb=\sum_i\sum_jW_{kij}a_ib_j$，每个输出都可以利用任意跨模态维度组合，表达力更强，但权重规模可达 $d_{\mathrm{out}}d_Ad_B$

## Tensor Fusion 与低秩融合

**Tensor Fusion**：变成齐次形式再做外积

![[多模态学习/imgs/img1/image-7.png]]

$$
\bar a=\begin{bmatrix}a\\1\end{bmatrix},\qquad
\bar b=\begin{bmatrix}b\\1\end{bmatrix},\qquad
z=\bar a\bar b^\mathsf T=
\begin{bmatrix}
ab^\mathsf T&a\\
b^\mathsf T&1
\end{bmatrix} \\
h_k=\sum_{i,j}W^{(k)}_{ij}z_{ij}=\bar a^\mathsf T W^{(k)} \bar b
$$

- 外积 $z$ 中含有单模态项 $a$ 和 $b^\mathsf T$ ，以及交互项 $ab^\mathsf T$ ，不同模态各自的信息和之间的交互同时保留
- 维度问题：$m$ 个模态完整外积有 $\prod_m(d_m+1)$ 个元素，后续还要过权重矩阵 $W$，参数量会变得非常大

**Low-rank Fusion**：不直接学习巨大的权重张量，而是把它分解为多个较小因子的组合

![[多模态学习/imgs/img1/image-6.png]]

对权重矩阵进行低秩分解

$$
W_{ij}\approx\sum_{r=1}^{R}u_{ri}v_{rj}
$$

则输出 $h$ 变为

$$
\begin{aligned}
h&=\sum_{i,j}W_{ij}\bar a_i\bar b_j\\

h&\approx\sum_{r=1}^{R}
\left(\sum_i u_{ri}\bar a_i\right)
\left(\sum_j v_{rj}\bar b_j\right)\\
&=\sum_{r=1}^{R}(u_r^\mathsf T\bar a)(v_r^\mathsf T\bar b)
\end{aligned}
$$

三模态同理

$$
h\approx\sum_{r=1}^{R}
(u_r^\mathsf T\bar a)
(v_r^\mathsf T\bar b)
(w_r^\mathsf T\bar c)
$$

**参数规模比较**

$$
\begin{aligned}
\mathrm{Full}&:\quad O\left(d_{\mathrm{out}}\prod_m(d_m+1)\right)\\
\mathrm{LowRank}&:\quad O\left(Rd_{\mathrm{out}}\sum_m(d_m+1)\right)
\end{aligned}
$$

## 门控与动态融合

**Gated Fusion**：权重随当前输入变化，不同情况下可以关注不同模态的信息

$$
z=g_A(a,b)\odot a+g_B(a,b)\odot b
$$

其中 $g_A,g_B$ 是模型学出的权重，可以是整个模态的一个数，也可以是每个特征维度对应的权重

**Dynamic Fusion**：不仅调整模态权重，还根据输入选择交互方式

$$
z=\sum_{k=1}^{K}\alpha_k(a,b)F_k(a,b)
$$

其中 $F_k$ 是融合方法，可以是单模态分支、加法、双线性等

# Coordination

## 基本的相似度函数

Fusion 是把不同模态的编码融合后喂给后续模型，而 Coordination 在分别编码后，用成对样本与协调目标 $g(z_A,z_B)$ 来训练学习跨模态关系

**余弦相似度**：向量夹角的余弦值

$$
g(z_A,z_B) = \frac{z_A\cdot z_B}{\lVert z_A\rVert \lVert z_B\rVert}
$$

**Pearson 相关系数**：先对向量中心化（减去均值），再计算余弦相似度

$$
r(a,b)=\text{corr}(a,b)=\frac{\text{Cov}(a,b)}{\text{Var}(a)\text{Var}(b)} =
\frac{\sum_i(a_i-\bar a)(b_i-\bar b)}
{\sqrt{\sum_i(a_i-\bar a)^2}\sqrt{\sum_i(b_i-\bar b)^2}}
$$

- 只代表线性相关程度，$r=0$ 不代表独立，非线性关系可能很强
- 如果两个模态之间的关系非常复杂，线性相似度可能不足以描述

**核相似度**：原空间中难以描述的关系，在一个更高维的特征空间中可能更容易表达；解决线性相似度不足以描述相关性的问题

$$
K(a,b)=\langle\phi(a),\phi(b)\rangle=\phi(a)^\mathsf T\phi(b)\\
K_{\mathrm{RBF}}(a,b)=\exp\left(-\frac{\|a-b\|^2}{2\sigma^2}\right)
$$

- 通过特征映射 $\phi$ 表达非线性关系
- RBF 是一个常用的核函数，距离越近，RBF 相似度越高

## CCA

**CCA（Canonical Correlation Analysis，典型相关分析）**：分别寻找线性投影 $u,v$，使配对样本 $X,Y$ 投影后尽可能相关，即

$$
(u^*,v^*)=\arg\max_{u,v}\text{corr}(u^\mathsf TX,v^\mathsf T Y)
$$

设中心化随机向量 $X,Y$，协方差为 $\Sigma_{XX}=\text{Cov}(X,X),\ \Sigma_{YY}=\text{Cov}(Y,Y)$，跨模态协方差 $\Sigma_{XY}=\text{Cov}(X,Y)$ 

**单方向 CCA**

$$
\max_{u,v}
\frac{u^\mathsf T\Sigma_{XY}v}
{\sqrt{u^\mathsf T\Sigma_{XX}u}\sqrt{v^\mathsf T\Sigma_{YY}v}}
$$

- 分子是两个投影结果的协方差，分母对它们的尺度进行标准化，避免单纯靠放大投影向量提高目标值
- 等价于 $\max_{u,v}u^\mathsf T\Sigma_{XY}v,\quad u^\mathsf T\Sigma_{XX}u=v^\mathsf T\Sigma_{YY}v=1$ 

**多方向 CCA**：寻找多组投影方向 $U=[u_1,\ldots,u_k],\ V=[v_1,\ldots,v_k]$ 

$$
\begin{aligned}
\max_{U,V}\quad&
\operatorname{tr}(U^\mathsf T\Sigma_{XY}V)\\
\text{s.t.}\quad&
U^\mathsf T\Sigma_{XX}U=I,\quad
V^\mathsf T\Sigma_{YY}V=I
\end{aligned}
$$

**DCCAE**：先进行非线性编码，使中间表示相关，再用 Decoder 重建各自的输入；解决 CCA 的线性投影有局限的问题

![[多模态学习/imgs/img1/image-8.png]]

**Multi-view Intact Space**：假设存在较完整潜在表示 $h$，各视图是其部分投影 $z_m\approx W_mh$，通过多个不完整视图恢复潜在结构

**Gated Coordination**：两个模态互相影响，但仍分别输出表示

$$
z_A=g_A(a,b)\odot a,\qquad
z_B=g_B(a,b)\odot b
$$

## Contrastive Learning

**Contrastive Learning（对比学习）**：让匹配的模态信息相互靠近，不匹配的相互远离
- Positive Pair（正样本对）：匹配的图片和文字，应该靠近
- Negative Pair（负样本对）：不匹配的图片和文字，应该远离

一个简单的对比损失：：正样本要比负样本更相似，而且最好至少高出 $\alpha$ （与 Hinge Loss 类似）

$$
\mathcal{L}=\max\left(0,\alpha+\operatorname{sim}(z_A,z_B^-)-\operatorname{sim}(z_A,z_B^+)\right)
$$

- $z_B^+$ ：与 A 匹配的正样本
- $z_B^-$ ：不匹配的负样本
- $\alpha$ ：希望正负样本至少相差多少间隔
- $sim(\cdot,\cdot)$ ：相似度函数 

## CLIP

**CLIP（Contrastive Language–Image Pre-training）**：通过对比学习进行图文对齐

它有两个 Encoder ，分别编码图像和文本

$$
v_i=f_V(\text{image}_i),\quad t_i=f_T(\text{text}_i)
$$

训练时，每个 batch 包含 $N$ 对匹配的图像和文本，模型计算每张图像与所有图文对中的文本的相似度，构建相似度矩阵，其中对角线上的元素对应着匹配的图文

![[多模态学习/imgs/img1/image-9.png]]

一个常用的 loss 是 InfoNCE：对于第 $i$ 个图像，在所有文本里，正确文本的相似度占多大优势

$$
\mathcal{L}=-\frac{1}{N}\sum_{i=1}^{N}\log\frac{sim(z_A^i,z_B^i)}{\sum_{j=1}^{N}sim(z_A^i,z_B^j)}
$$

CLIP 使用双向训练，不仅用图像找文本，还会反过来用文本找图片，使用两个方向损失的平均值

$$
\mathcal L_{\text{CLIP}} =\frac{\mathcal L_{I\to T}+ \mathcal L_{T\to I}}{2}
$$

原始 CLIP 使用约 4 亿对图文进行预训练，它的一个重要结果是：图像和文字编码器学到可以跨模态比较的表示，但这两个表示不必完全相同

**零样本分类**：把分类标签构造成一句话，使用 CLIP 看图像与各个标签文本的匹配程度，不需要针对每个分类任务从头训练

![[多模态学习/imgs/img1/image-10.png]]

# Fission

前面的方法要么把信息融合，要么让它们对齐，但还有一种情况：两个模态既有共同的信息，又各自包含另一方没有的信息，如果强行把所有信息挤进一个完全共享的表示，就可能丢掉一个模态特有的信息

**Modality-level Fission（模态级拆分）**：区分各模态专有的信息与共同的信息

$$
x_A,x_B\longrightarrow (z_A,z_{\text{shared}},z_B)
$$

**Fine-grained Fission（细粒度拆分）**：进一步把信息拆分为更细致的语义因素或特征组，而不只按模态划分

## Modality-level Fission

为了判断拆分结果的好坏，一个简单的方法是，把拆出的表示送入预测器，对预测标签 $y$ 进行监督，但仅靠预测正确，并不能保证信息没有丢失，也不能保证没有信息的重复

![[多模态学习/imgs/img1/image-11.png]]

- $\mathcal L_1$：Discriminative Loss，让相关表示能够完成预测任务
- $\mathcal L_2$：Generative Loss，让表示能够重建原始模态，避免信息丢失
- $\mathcal L_3$：No-overlap Constraint，共享信息与专有信息尽量不要重复

# 自监督学习

前面讲了各种表示学习方法，但它们往往需要大量匹配好的数据，甚至需要人工标注，自监督学习试图解决这个问题：从原始数据自身构造训练任务，让模型在完成这些任务时，学到可以迁移到其他任务的特征

## SimCLR

假设有一张猫的图片，我们对它做旋转、裁剪、颜色变化，虽然图片在像素上发生了很大变化，但它们仍然表示同一只猫

SimCLR 的训练过程是：
- 对一个 batch 中的 $N$ 张图片，每张随机增强两次，得到 $2N$ 个视图
-  通过同一个 Encoder 提取表示。
- 同一张原图产生的两个视图作为正样本对，其他图像的视图作为负样本
- 使用对比损失，让正样本更接近，负样本更远

这和 CLIP 的思想非常相似，区别在于：CLIP 的正样本是匹配的图片和文字，SimCLR 的正样本是同一张图片的两个增强视图

## DINO

SimCLR 依赖正负样本对，那么能不能不用显式负样本，也学习好的表示？**DINO（Distillation with No Labels，无标签蒸馏）** 就是一种方法，它使用教师和学生两个网络，将同一图像的不同增强版本分别送给它们，让学生网络学习教师网络的输出分布

设 $p_T$ 和 $p_S$ 是 Teacher 和 Student 的输出概率分布，一种简化的损失形式是

$$
\mathcal L=-\sum_k p_T(k)\log p_S(k)
$$

Student 使用该 loss 进行反向传播更新参数，而 Teacher 则使用指数移动平均（EMA）更新（初始参数和 Student 相同）

$$
\theta_T\leftarrow \lambda\theta_T+(1-\lambda)\theta_S
$$

既然 Teacher 一开始和 Student 一样，那怎么保证 Student 能学到更多的东西？这是因为 DINO 给两个网络的图像是不同的，但是仍然要求 Student 做出和 Teacher 相近的预测，这样 Student 就不能只依赖某个视图特有的细节，而需要学习不同视图之间共通的信息

为了让 Teacher 给 Student 提供有用的学习目标，DINO 对 Teacher 的输出做了两种处理
- Centering（中心化）：防止 Teacher 总是偏向某几个固定的输出维度
- Sharpening（锐化）：让 Teacher 的预测更明确，而不是每个维度都差不多

## MAE 与其它预训练任务

在构建任务方面，除了判断两个视图是不是同一个对象，还可以让模型恢复被遮挡的图片，这就是 **MAE（Masked Autoencoder）**，训练过程为
- 把图像切成多个 Patch ，并随机遮住其中大部分 Patch 
- Encoder 只处理可见的 Patch，Decoder 尝试重建完整图片
- 只对被遮住的 Patch 计算重建损失，记 $M$ 为被遮挡的位置集合

$$
L_{\mathrm{MAE}}=\frac1{|M|}\sum_{p\in M}\|\hat x_p-x_p\|^2
$$

为什么遮住图片也能学到表示？因为模型想把一只猫缺失的部分补出来，就必须理解周围图像的结构、纹理，以及猫通常长什么样

遮挡太少任务可能太简单了，模型学不到真东西，遮挡太多又缺乏线索，可能完全学不会；最佳遮挡比例依赖模型、数据与下游评价方式

| 方法 | 自动得到的监督 | 要学习什么 |
| --- | --- | --- |
| 旋转预测 | 已施加的 $0/90/180/270^\circ$ 变换 | 物体方向与结构 |
| Inpainting | 遮挡前的原始像素 | 根据上下文补连续缺失区域 |
| Colorization | 原有颜色通道 | 根据亮度与语义预测颜色 |
| 跨视图预测 | 同一样本的其它通道 / 模态 | 学习 RGB、深度、亮度、颜色等之间的关系 |

这些任务称为 **Pretext Tasks**，就是先设计一个能自动生成的、让模型学到东西的任务，完成预训练以后，再把 Encoder 用于真正的下游任务，比如分类、检测或分割

# 与信息论的联系

## 信息论与多模态

可以使用信息论中的熵 $H(X)$ 来表示一个模态包含的信息量
- 联合熵：两个模态总共有多少信息
- 条件熵：已知一个模态的信息，另一个模态还有多少信息无法确定
- 互信息：两个模态共享多少信息

$$
\begin{aligned}
H(X,Y)&=H(X)+H(Y\mid X)\\
I(X;Y)&=H(X)+H(Y)-H(X,Y)\\
&=H(X)-H(X\mid Y)\\
&=D_{\mathrm{KL}}(p_{XY}\|p_Xp_Y)
\end{aligned}
$$

**互信息最大化的假设**：两个模态共有的信息，对下游任务最重要

这个假设在一些任务中很合理，比如识别图文共同描述的对象，但它并不是永远成立的，单个模态的独有信息也可能很重要，这正是前面 Fission 要解决的问题之一

## 信息论与自监督学习

自监督学习会从原始数据构造出不同的视图，学习一个表示 $z$ ，希望学到它们共同的、有用的信息，而不是死记每个视角的细节

![[多模态学习/imgs/img1/image-12.png]]

- 最大化互信息：希望 $z$ 尽可能多的包含各个视图中的信息
- 最小化条件熵：如果我只看到了一个视图，能不能确定完整的 $z$，即希望 $z$ 中尽可能不要包含每个视图独有的信息

InfoNCE 可以作为互信息的一个下界

$$
I(f(x);f(x^+))\geq \log N-\mathcal L_{\text{InfoNCE}}
$$

可见增加负样本数量 $N$，降低对比损失，都有助于提高互信息的下界

