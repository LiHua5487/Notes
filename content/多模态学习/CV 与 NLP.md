
# 深度学习基础

## MLP 与激活函数

单层全连接网络

$$
y=Wx+b
$$

如果连续堆叠多层，但没有激活函数，仍然是一个线性变换，所以需要非线性激活函数

| 函数           | 公式                   | 主要特点                   |
| ------------ | -------------------- | ---------------------- |
| Sigmoid      | $\frac{1}{1+e^{-x}}$ | 输出在 (0,1)，两端饱和，容易梯度消失  |
| Tanh         | $\tanh(x)$           | 输出在 (-1,1)，零中心，但仍有饱和问题 |
| ReLU         | $\max(0,x)$          | 简单、计算快，负半轴梯度为 0        |
| Leaky ReLU   | $\max(\alpha x,x)$   | 负半轴保留小梯度               |
| GELU         | $x\Phi(x)$           | 平滑地控制输入通过的程度           |
| Swish / SiLU | $x\sigma(x)$         | 平滑、允许小幅负输出             |

**Non-zero centered problem（非零中心问题）**：如果激活函数的输出始终为正，那下一层的各个权重在一次反向传播中会同时增大或同时减小（梯度同号），如果最优参数希望 $w_0$ 增大、$w_1$ 减小，那优化过程就可能出现“之字形”震荡，收敛变慢

![[多模态学习/imgs/img0/image.png]]

## Normalization

归一化的共同目的，是控制中间特征的数值尺度，让训练更加稳定。不同方法最重要的区别是：沿哪些维度计算均值和方差

标准归一化公式：

$$
\hat{x}_i=\frac{x_i-\mu}{\sqrt{\sigma^2+\epsilon}}, \qquad y_i=\gamma\hat{x}_i+\beta
$$

- $\mu,\sigma^2$ 来自指定的统计维度，计算得到均值和方差
- $\gamma,\beta$ 是学习到的缩放和偏移参数，避免归一化后只能使用固定的尺度和中心，通过 moving average 进行更新

以 CNN 特征 $X\in\mathbb R^{N\times C\times H\times W}$ 为例

![[多模态学习/imgs/img0/image-1.png]]

| 方法           | 统计范围             | 特点           |
| ------------ | ---------------- | ------------ |
| BatchNorm    | 每个通道的 $N,H,W$    | 依赖 batch 统计量 |
| LayerNorm    | 每个样本的 $C,H,W$    | 不依赖其他样本      |
| InstanceNorm | 每个样本、每个通道的 $H,W$ | 常用于风格迁移      |
| GroupNorm    | 每个样本的一组通道及空间维度   | 适合较小 batch   |

对于语言模型的 $N\times L\times D$ 张量，常见 LayerNorm 是在每个 token 的 $D$ 个特征维度上计算，而不是把整句话的所有 token 混在一起

**RMSNorm（Root Mean Square Normalization）**：进一步简化了 LN，不减均值，直接用均方根调整尺度，计算更简单

$$
\operatorname{RMSNorm}(x) = \gamma\odot \frac{x}{\sqrt{\frac1d\sum_{i=1}^d x_i^2+\epsilon}}
$$

**AdaLN（Adaptive Layer Normalization）**：普通 LN 的 $\gamma,\beta$ 是模型参数，不会随当前输入条件改变，AdaLN 则让它们由条件 $c$ 决定

$$
[\gamma(c),\beta(c)]=\operatorname{MLP}(c)\\
y=\gamma(c)\odot\operatorname{LN}(x)+\beta(c)
$$

例如扩散模型的条件 $c$ 可以是时间步 $t$、文本嵌入或类别标签，这样同一网络层在不同条件下可以采用不同的特征缩放和偏移

**AdaLN-Zero**：在 AdaLN 的基础上引入残差分支的门控 $\alpha$ ，并对相关参数做零初始化

$$
x'=x+\alpha(c)\odot F\big(\gamma(c)\odot\operatorname{LN}(x)+\beta(c)\big)
$$

初始时 $\alpha(c)\approx0$，所以 $x'\approx x$，残差分支从接近恒等映射开始训练，有利于稳定优化

## 损失函数、梯度下降、正则化

| 任务  | 损失函数                      | 公式                                                              |
| --- | ------------------------- | --------------------------------------------------------------- |
| 回归  | MSE（Mean Squared Error）   | $\mathcal L_{\mathrm{MSE}}=\frac1N\sum_{i=1}^N(y_i-\hat y_i)^2$ |
| 二分类 | BCE（Binary Cross-Entropy） | $\mathcal L_{\mathrm{BCE}}=-[y\log p+(1-y)\log(1-p)]$           |
| 多分类 | CE（Cross-Entropy）         | $\mathcal L_{\mathrm{CE}}=-\sum_{c=1}^C y_c\log p_c$            |

多分类中的 $p_c$ 使用 softmax 计算 $p_c=\frac{e^{z_c}}{\sum_j e^{z_j}}$ ，其中 $z_c$ 是分类器直接输出的 logit

**梯度下降**基本更新

$$
\theta_{t+1}=\theta_t-\eta\nabla_\theta\mathcal L
$$

SGD 使用 mini-batch 估计梯度，Adam 等优化器进一步利用梯度的一阶、二阶统计量调整更新

**正则化**解决的是模型过拟合问题，常用形式

$$
\mathcal L_{\mathrm{total}} =\mathcal L_{\mathrm{task}}+\lambda R(\theta)
$$

- L1：$R(\theta)=\|\theta\|_1$，倾向于产生稀疏权重
- L2：$R(\theta)=\|\theta\|_2^2$，倾向于抑制过大的权重

**Dropout**：训练时随机置零部分激活值，减少对某些特征的过度依赖

**训练流程**：检查初始 loss → 尝试过拟合极小数据集 → 找到能让 loss 明显下降的学习率 → 粗调、细调超参数 → 检查训练/验证曲线

判断曲线时，重点看训练集与验证集的差距
- 训练表现很好但验证表现明显差，通常说明过拟合，考虑使用正则化与数据增强
- 两者表现都不好，则优先检查模型容量、训练是否充分以及优化设置

# CV

## CNN

CNN 利用两个假设
- 局部连接：一个位置的特征主要取决于附近区域，不必连接整幅图像
- 权重共享：同一个卷积核在不同位置重复使用，而不是每个位置训练一套参数

设输入为 $C_{\mathrm{in}}\times H\times W$，卷积核大小为 $K\times K$，输出通道数为 $C_{\mathrm{out}}$，步长为 $S$，Padding 为 $P$ ，则**输出空间尺寸**

$$
H_{\mathrm{out}}= \left\lfloor\frac{H+2P-K}{S}\right\rfloor+1
$$

宽度同理，输出通道数则直接等于卷积核的数量 $C_{\mathrm{out}}$ 

**参数量**

$$
N_{\mathrm{param}} =(K^2C_{\mathrm{in}}+1)C_{\mathrm{out}}
$$

一个输出通道对应的卷积核大小是 $C_{\mathrm{in}}\times K\times K$ ，总共需要 $C_{\mathrm{out}}$ 个这样的卷积核，这里的 +1 来自于每个输出通道的 bias

**乘加运算次数（MACs）**

$$
\mathrm{MACs} =H_{\mathrm{out}}W_{\mathrm{out}} C_{\mathrm{out}}\cdot K^2C_{\mathrm{in}}
$$

若把乘法和加法分别计作一次浮点运算，FLOPs 约为 MACs 的两倍

**$1\times1$ 卷积**：不聚合周围像素，但会混合通道，主要用于通道变换和降维，尤其是在昂贵的 $3\times3$、$5\times5$ 卷积前压缩通道

$$
y_{h,w}=Wx_{h,w}+b,\quad W\in\mathbb R^{C_{\mathrm{out}}\times C_{\mathrm{in}}}
$$

## Pooling

Max Pooling 取局部最大值，Average Pooling 取局部平均值，两者都可以降低空间分辨率，增加对小幅位置变化的鲁棒性

设输入为 $C_{\mathrm{in}}\times H\times W$，池化窗口大小为 $F\times F$，步长为 $S$，则

$$
H_{\mathrm{out}}= \left\lfloor\frac{H-F}{S}\right\rfloor+1,\quad C_{\text{out}} = C_{\text{in}}
$$

可学习参数量为 0 

## 经典 CNN 架构

| 模型        | 核心设计                                                                                       | 作用                                         |
| --------- | ------------------------------------------------------------------------------------------ | ------------------------------------------ |
| AlexNet   | 5 个卷积层、3 个全连接层、ReLU、Max Pooling                                                            | 证明深度 CNN 在大规模图像分类上的有效性                     |
| VGG       | 统一使用 3×3 卷积，增加网络深度                                                                         | 通过堆叠小卷积，在扩大感受野的同时，有着更少的参数量                 |
| GoogLeNet | 前期积极下采样（Aggressive Downsampling）、Inception Module 多分支卷积、1×1 卷积、Global Average Pooling（GAP） | 前期积极下采样降低分辨率、1×1 卷积降低通道数，GAP ，进一步减少参数量和计算量 |
| ResNet    | Residual Link 残差连接                                                                         | 解决深层网络难以优化的问题                              |
| SENet     | Squeeze-Excitation branch                                                                  | 不同的特征通道携带不同信息，让网络学习哪些通道应该被增强或抑制            |

## 轻量化 CNN

普通卷积的每一个输出通道，都需要综合全部输入通道、以及每个通道邻近的 $K\times K$ 个位置，实际上同时做了两件事
- 空间特征提取： 整合同一通道附近的像素
- 通道特征融合： 将不同输入通道的信息组合成新的输出通道

**Depthwise Separable Convolution（深度可分离卷积）**：把这两件事拆开做
- **Depthwise Conv**：每个通道分别卷积
- **Pointwise Conv**：在每个位置混合不同通道，实际上就是 1×1 卷积

![[多模态学习/imgs/img0/image-2.png]]

参数量（不计 bias） 

$$
\frac{N_{\mathrm{DS}}}{N_{\mathrm{standard}}} = \frac{K^2C_{\mathrm{in}}C_{\mathrm{out}}}{K^2C_{\mathrm{in}}+C_{\mathrm{in}}C_{\mathrm{out}}} = \frac{1}{C_{\mathrm{out}}}+\frac{1}{K^2}
$$

其他轻量化思路：减少通道、去除大型 FC 层、知识蒸馏、模型压缩和量化

## 大卷积核

**RepLKNet**：小卷积核虽然参数量少，但是主要关注局部特征，而大卷积核能在单层里直接聚合更大范围的信息，这对于分割等需要较大上下文范围的下游任务尤其有帮助

设计思路
- 利用 Depthwise Convolution，使大卷积核不再必然带来难以承受的计算成本
- 大卷积核需要 Shortcut 辅助优化
- 采用结构重参数化，训练时在大卷积核旁边增加一个小卷积核分支，小卷积核提供更容易学习的局部特征，有助于优化大卷积核，推理时可以直接将大小核合并计算
- 大卷积核带来的收益在 Segmentation 等下游任务上可能比 ImageNet 分类更明显

## 动态卷积

传统 CNN 的特征提取主要是自底向上的，比如先关注边缘、纹理，再关注局部结构，最后看高级语义，但它缺少显式的自顶向下语义指导，例如经典的鸭兔错觉图，单看局部线条容易产生歧义，而整体上下文可以帮助判断应该关注哪些特征

**OverLoCK**：先理解全局，再指导局部

![[多模态学习/imgs/img0/image-3.png]]

- Base-Net： 提取低层、中层图像特征
- Overview-Net： 提取粗粒度的高级语义信息，形成上下文先验
- Focus-Net： 利用上下文先验，进一步提取准确、细致的视觉特征

**Context-Mixing Dynamic Kernels**：普通卷积对于所有空间位置使用相同的核，动态卷积则根据当前位置与全局上下文之间的关系，调整卷积核参数

# 目标检测与图像分割

目标分类 Classification 只需要预测一张图中有什么，目标检测 Detection 需要进一步输出每个目标的位置和类别，一个检测结果通常表示为

$$
(c,x,y,w,h)
$$

其中 $c$ 是类别，$(x,y)$ 表示框的位置，$w,h$ 表示宽高

直接让 CNN 回归所有目标的坐标会遇到问题：不同图片中目标数量不同，输出长度不固定

一种早期思路是滑动窗口：从图像裁出大量不同位置和尺度的小块，再逐个分类，但这需要对高度重叠的区域反复运行 CNN，计算量极大

因此出现了 **Region Proposal** 方法：先找出可能包含目标的候选区域，称为 **RoI（Region of Interest）**，再交给 CNN 判断

**Selective Search** 就是一种典型的 Region Proposal 方法：从初始区域出发，按颜色、纹理、大小等相似性逐步合并，收集区域框作为候选

## R-CNN 系列

![[多模态学习/imgs/img0/image-4.png]]

| 方法           | 主要变化                                    | 解决的问题                            |
| ------------ | --------------------------------------- | -------------------------------- |
| R-CNN        | 用 Region Proposal 得到候选区域，每个区域用 CNN 提取特征 | 不用穷举原图的所有窗口                      |
| Fast R-CNN   | 对原图用 CNN 提取特征，在特征图上计算候选区域的特征            | 减少重复卷积，不需要多阶段训练                  |
| Faster R-CNN | 把 Region Proposal 换成 RPN 神经网络           | 解决 Region Proposal 的速度瓶颈，成为端到端方法 |

### R-CNN

**R-CNN（Region-based Convolutional Neural Network）**
- 先用 Region Proposal 方法从原图得到候选区域
- 将每个区域裁剪、缩放到 CNN 所需的固定输入尺寸
- 每个区域分别提取特征
- 最后使用两个独立的输出头并进行监督：SVM 判断目标类别，Bounding Box Regressor 对框的位置进行修正

解决的问题：不需要遍历原图中所有位置、所有大小的窗口，只需对一部分区域进行 CNN 检测

存在的缺点
- 每个候选区域都需要独立运行 CNN，大量候选框彼此重叠，却没有共享卷积计算
- 训练分多个阶段：先微调 CNN，再训练 SVM，还要训练框回归器，整个流程复杂，时间和存储成本都很高

### Fast R-CNN

**Fast R-CNN**：在计算候选区域的特征时，不是每个区域分别提取，而是先提取整图特征，再根据这个特征图计算候选区域的特征

Fast R-CNN 仍然会用 Region Proposal 从原图得到候选框，但是提取候选框特征时，是在特征图上进行的，需要把原图的候选框对应到特征图

假设原图大小是 $800\times800$，经过 CNN 下采样 32 倍，得到一张 $25\times25$ 的特征图，则原图坐标 $(x,y)=(665,560)$ 就对应到 $(x',y')=(665/32,560/32)$ 

不同候选框的大小不一样，但后面的 FC 层需要固定长度的输入，每个候选框的特征长度应相同，为此有两种提取办法：RoI Pooling 和 RoI Align 

**RoI Pooling**：将候选框坐标取整，把特征图中对应的候选区域划分成固定数量的网格（不一定能均匀划分），每个网格做 Max Pooling，得到固定大小的特征

![[多模态学习/imgs/img0/image-5.png]]

但是这种方法强行把候选框坐标取整，可能会造成特征错位

**RoI Align**：保留小数坐标，并在特征图中对应的候选区域采样一定数量的点，使用双线性插值计算采样点的特征值

![[多模态学习/imgs/img0/image-6.png]]

Fast R-CNN 还改进了训练方式，不再单独训练 SVM 和框回归器，而是用一个多任务损失联合训练

$$
L=L_{\mathrm{cls}}+\lambda L_{\mathrm{box}}
$$

分类使用交叉熵损失，框回归使用 Smooth L1 损失

### Faster R-CNN

Fast R-CNN 已经解决了重复运行 CNN 的问题，但仍然依赖 Region Proposal 产生候选框，这一步不参与端到端学习，也成为了新的速度瓶颈

Faster R-CNN 的核心改进，就是引入 **RPN（Region Proposal Network）**，使用神经网络提取候选框

它把目标检测划分为两个阶段
- 第一阶段：CNN 对整张图片提取一次特征，RPN 在特征图上预测哪些位置可能有目标，以及对应的候选框位置
- 第二阶段：使用这些候选框提取 RoI 特征，再做具体目标类别的分类与进一步的位置修正

RPN 的过程

![[多模态学习/imgs/img0/image-7.png]]

- 在特征图的每个位置，预设 $K$ 个不同大小、长宽比的候选框，称为 **Anchor** （其大小通常是指在原图中的大小，只是把 Anchor 的中心放在特征图各位置所对应的原图坐标上）
- 对每个 Anchor 预测两种东西
	- **Objectness Score**： 这个框包含目标的可能性有多大（二分类，是否包含目标）
	- **BBox Regression**： 应该如何调整这个框的中心位置和宽高（4 个偏移量）

RPN 产生候选框后，通常会通过排序及 NMS 去除大量重叠框，再选出一部分高质量候选框交给第二阶段

整个 Faster R-CNN 联合优化四项损失

$$
L=L_{\mathrm{RPN,cls}}+ L_{\mathrm{RPN,box}}+ L_{\mathrm{det,cls}}+ L_{\mathrm{det,box}}
$$

前两项训练候选框生成，后两项训练最终检测结果

## 评价指标与损失函数

**IoU（Intersection over Union）**：衡量预测框 $A$ 与真实框 $B$ 的重叠程度

$$
\operatorname{IoU}(A,B) =\frac{|A\cap B|}{|A\cup B|},\quad L_{\mathrm{IoU}}=1-\mathrm{IoU}
$$

**GIoU（Generalized IoU）**：如果两个预测框都没有和真实框重叠，但是一个离得更近，怎么区分？

![[多模态学习/imgs/img0/image-8.png]]

$$
\mathrm{GIoU}=\mathrm{IoU}- \frac{|C|-|A\cup B|}{|C|}
$$

-  $C$ ：包含两个框的最小外接矩形

**DIoU（Distance IoU）**：比较候选框中心距离，直接要求两个框的中心点靠近，改善了 GIoU 对某些位置关系区分不足的问题

![[多模态学习/imgs/img0/image-9.png]]

$$
\mathrm{DIoU}=\mathrm{IoU}- \frac{\rho^2(b,b^{gt})}{c^2}
$$

- $b,b^{gt}$ 是两个框的中心点，$\rho$ 是欧氏距离
- $c$ 是最小外接框的对角线长度

**CIoU（Complete IoU）**：在 DIoU 的基础上，对长宽比做出要求，比如真实框是扁的，那候选框最好也是扁的

---

**Focal Loss**：在交叉熵前加入一个调节因子，降低容易样本的影响，让训练更加关注困难样本

$$
L_{\mathrm{FL}}=-\alpha_t(1-p_t)^\gamma\log p_t
$$

- 当 $p_t$ 接近 1 时，说明样本很容易分类，$(1-p_t)^\gamma$ 接近 0，它的损失被大幅减小
- 当 $p_t$ 很小时，说明模型预测得不好，这个调节因子接近 1，损失得以保留

## YOLO

前面的 Faster R-CNN 属于 Two-Stage Detector，先预测候选框，再做具体检测

**YOLO（You Only Look Once）** 则属于 One-Stage Detector：不再分阶段独立运行候选区域检测阶段，而是一次前向传播直接输出目标类别与边界框

![[多模态学习/imgs/img0/image-10.png]]

将图像划分成 $S\times S$ 网格，每个网格预测 $B$ 个边界框和 $C$ 个类别的概率，每个边界框有 5 个预测量

$$
(x,y,w,h,\mathrm{confidence})
$$

其中 confidence 同时反映目标存在概率和框定位质量

## 语义分割

语义分割：给每个像素预测类别，需要输出每个位置对 $C$ 个类别的预测分数

**FCN（Fully Convolutional Network）**：直接使用卷积产生整张图的预测，不需要对每个像素单独运行一次网络

问题是如果始终在原始分辨率计算，成本很高；如果下采样，空间细节又会丢失，不知道具体的位置，因此采用 Encoder-Decoder 结构：先用 Encoder 压缩成低分辨率特征，再用 Decoder 上采样，还原回高分辨率预测

上采样的几种方式
- 最近邻插值：把一个位置的数值复制到附近多个位置
- Max Unpooling：在 Max Pooling 时记录最大值所在的位置，上采样时将特征放回对应位置，其余位置补零
- Transpose Convolution：通过可学习的卷积核将低分辨率特征映射到更大的空间，重叠区域的贡献相加

![[多模态学习/imgs/img0/image-11.png]]

![[多模态学习/imgs/img0/image-12.png]]

**U-Net**：使用 Skip Connection 补回下采样丢失的细节

![[多模态学习/imgs/img0/image-13.png]]

# NLP

## 词表示

**One-hot**：为词表中的每个词分配一个独立维度

$$
e_i=[0,\ldots,1,\ldots,0]
$$

问题是维度高，而且不同词之间的距离都相同，无法直接表达语义相似性

**Bag-of-Words（BoW，词袋）**：把一个句子表示为词频向量

$$
x_i=\operatorname{Count}(w_i,\text{sentence})
$$

它适合简单文本分类，但不保留词序

**Distributional Hypothesis（分布假设）**：经常出现在相似上下文里的词，其语义往往也比较相似

衡量向量相似性时，常用**余弦相似度**

$$
\operatorname{sim}(u,v) =\frac{u^\top v}{\|u\|\|v\|}
$$

它主要关注向量方向，而不是绝对长度，因此能减轻词频造成的尺度差异

**Word2Vec**：用预测任务学习词向量，有两种经典的训练方式

|                      | 输入      | 预测目标    |
| -------------------- | ------- | ------- |
| CBOW（Continuous BoW） | 周围的上下文词 | 中间的目标词  |
| Skip-gram            | 中间的目标词  | 周围的上下文词 |

## 语言模型

语言模型希望给一段文本分配概率

$$
P(w_1,w_2,\ldots,w_T)
$$

通过概率链式法则

$$
P(w_1,\ldots,w_T) =\prod_{t=1}^{T}P(w_t\mid w_1,\ldots,w_{t-1})
$$

因此，语言建模可以转化为一个序列预测任务：根据之前的词预测下一个词

### N-garm

**N-gram**：如果完整考虑所有历史词，条件概率很难估计，所以使用有限长度的上下文近似，只看前 $n-1$ 个词

$$
P(w_t\mid w_{1:t-1}) \approx P(w_t\mid w_{t-n+1:t-1})
$$

概率可以通过频率估计：

$$
P(w_t\mid w_{t-n+1:t-1}) = \frac{\operatorname{Count}(w_{t-n+1:t})} {\operatorname{Count}(w_{t-n+1:t-1})}
$$

问题很明显：N 太小无法捕捉长距离依赖，N 太大又会导致大量词语组合在训练数据中从未出现

### RNN

**RNN（Recurrent Neural Network，循环神经网络）**：通过一个不断更新的隐藏状态 $h_t$ 来表示已经处理过的序列

$$
h_t=\tanh(W_{hh}h_{t-1}+W_{xh}x_t+b_h),\quad y_t=W_{hy}h_t+b_y
$$

根据输入输出形式，RNN 可以实现 Many-to-One（例如句子分类）、One-to-Many（例如根据输入生成序列）、Many-to-Many（例如逐词标注或序列转换）

训练自回归语言模型时，常使用 **Teacher Forcing**：把真实的前文提供给模型，让模型预测下一个词，但推理时没有所谓的真实词，错误会逐渐累积

可通过 **Scheduled Sampling** ，在训练时逐渐增加使用模型自身预测结果的比例，减轻这种不一致

RNN 通过 **BPTT（Backpropagation Through Time，时间反向传播）** 训练：将网络沿时间展开后，损失需要向前面所有时间步反向传播梯度，容易导致梯度消失或爆炸，有以下解决办法
- **Gradient Clipping**：超出范围直接截断，主要用于控制梯度爆炸
- **Truncated BPTT**：只沿有限长度的时间片段反向传播，但隐藏状态仍然可以继续向后传递

### LSTM

**LSTM（Long Short-Term Memory，长短期记忆网络）**：在 RNN 的隐藏状态的基础上，引入了表示长期记忆的状态 $c_t$ ，通过门控决定保留、写入和输出哪些信息

![[多模态学习/imgs/img0/image-14.png]]

遗忘门（Forget Gate）：决定过去的记忆中保留哪些信息

$$
f_t=\sigma(W_fz_t+b_f)
$$


输入门（Input Gate）与候选记忆：决定写入多少新信息、新信息的内容

$$
i_t=\sigma(W_iz_t+b_i),\quad \tilde c_t=\tanh(W_cz_t+b_c)
$$

记忆更新

$$
c_t=f_t\odot c_{t-1} +i_t\odot\tilde c_t
$$

输出门（Output Gate）：决定当前记忆中有多少信息通过隐藏状态向外输出

$$
o_t=\sigma(W_oz_t+b_o),\quad h_t=o_t\odot\tanh(c_t)
$$

为什么 LSTM 更不容易梯度消失？根据记忆更新公式，梯度只需要经过逐元素乘法

$$
\left.\frac{\partial c_t}{\partial c_{t-1}}\right|_{\text{direct}} =\operatorname{diag}(f_t)
$$

当遗忘门 $f_t$ 接近 1 时，历史信息与梯度都可以较好地沿这条路径传递，而不需要在每一步反复经过完整的 $W_{hh}$ 和 Tanh ；当然，如果遗忘门长期接近 0，梯度仍可能衰减

![[多模态学习/imgs/img0/image-15.png]]

### GNN

RNN 假设输入有明确的先后顺序，但很多数据不是线性序列，例如社交关系网络、知识图谱和分子结构

**GNN（Graph Neural Network）** 的基本思路是：一个节点的表示不仅取决于自身特征，还取决于相邻节点的信息

**邻接矩阵 $A$**：表示节点的连接关系

**特征矩阵 $X$**：表示每个节点的其属性，比如一个节点代表一个用户，每个用户分别具有两个特征，每天发帖次数和关注人数，则

$$
X= \begin{bmatrix} 5&100\\ 50&20\\ 3&200 \end{bmatrix}
$$

每一行对应一个用户，每一列对应一种属性

**节点标签 $Y$**：表示我们想预测什么，比如预测每个节点的用户属于真人还是机器人

$$
Y=[0,1,0]
$$

普通 MLP 预测时只考虑自身的信息，但可能出现这样的情况：A 的发帖次数正常，个人资料也正常，可是它连接的大量账号都有明显的异常行为，因此我们希望

$$
z_A=f(x_A,\text{A 的邻居信息})
$$

也就是说，节点的表示需要结合其在图中的位置，以及与它相连的节点的特征，这就涉及到 GNN 的 **Message Passing** 机制：先聚合邻居特征，再更新当前节点的特征

比如 A 与 B、C、D 相邻，最简单的方式就是取平均收集邻居特征，然后结合自己的特征

$$
m_A=\frac{x_B+x_C+x_D}{3},\quad h_A=\sigma(Wm_A+Bx_A)
$$

这只是一层 GNN ，还可以堆叠多层 GNN 收集更远的节点的信息

![[多模态学习/imgs/img0/image-16.png]]
















