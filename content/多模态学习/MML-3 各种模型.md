
# ViT

Transformer 也可以用来处理图像，但是如果直接把每个像素作为一个 token 输入，开销太大了，所以 ViT 把图像切成一系列 $16\times16$ 的 **Patch** ，每个 Patch 展平成向量，再通过可学习的线性映射转换成 $D$ 维特征

光有 Patch 特征还不够，Transformer 本身不知道各个 Patch 在图片里的位置，因此还要加入**位置编码**

比如对于图像分类任务，ViT 的过程如下

![[多模态学习/imgs/img3/image.png]]

- Patch Projection：每个 Patch 映射到一个 $D$ 维向量
- Position Embedding：为每个 Patch 进行位置编码
- `[CLS]` Token：额外插入一个可学习的向量，用来汇总整张图片的信息
- Transformer Encoder：所有 Token 通过 Self-Attention 交换信息
- Classification Head：取最后的 `[CLS]` 输出，映射到类别分数

>其中的 `[CLS]` Token 可以看作模型参数的一部分，可以随着反向传播进行更新

## Inductive Bias

但是 ViT 的表现不总是比基于 CNN 的方法好，比如在 ImageNet 分类任务上：训练数据较少时，ResNet 表现更好；但是当训练数据增多时，ViT 发生了反超

为什么数据规模会产生这么大的影响？关键概念是 **Inductive Bias（归纳偏置）**，意思是模型在看到数据之前，结构中已经预先包含的一些假设

比如 CNN 天生包含两种重要假设
- 局部性：附近的像素通常比远处的更相关，所以用卷积核处理局部区域
- 平移等变性：同一个卷积核在不同位置共享，因此可以识别出现在不同位置的相似特征

这些假设对图像非常有效，相当于提前告诉模型一些图像的规律，但 ViT 主要依靠 Self-Attention 自己学习空间关系，没有像 CNN 那样强的局部结构约束，因此数据少的时候，CNN 的先验能帮助泛化；数据足够多时，ViT 的灵活性就更可能发挥出来

## Improving ViT

### 正则化与数据增强

| 方法               | 核心思路                                 |
| ---------------- | ------------------------------------ |
| Weight Decay     | 惩罚过大的模型权重，防止模型过度依赖某些参数               |
| Dropout          | 训练时随机将部分神经元输出置零（在 FNN 层）             |
| Stochastic Depth | 训练时随机跳过部分残差分支，让模型学会在不同深度下工作          |
| MixUp            | 将两张图片和对应标签按比例混合，让模型在不同样本之间也能给出较平滑的预测 |
| RandAugment      | 随机选择旋转、颜色调整等图像增强操作                   |

### Knowledge Distillation

假设我们已经训练好了一个准确率很高的大模型，现在想训练一个更小的模型，正常训练时，我们给学生图片和真实标签（位置依次代表猫、老虎、车）

$$
y=[1,0,0]
$$

但真实标签只能告诉学生「答案是猫」，无法告诉学生模型对其他类别的判断，一个训练好的教师模型可能输出

$$
P_T=[0.8,0.15,0.05]
$$

教师模型的输出更平滑，也提供了更多的信息：图片最像猫，也有些像老虎，但几乎不可能是车

知识蒸馏的想法，就是让学生不仅学习最终答案，还学习教师对各类别的判断

![[多模态学习/imgs/img3/image-1.png]]

通常先训练好教师并固定其参数，再让学生拟合教师输出的概率分布，为了比较两个分布，可以使用 KL 散度

$$
D_{\mathrm{KL}}(P_T\|P_S) = \sum_i P_T(i)\log\frac{P_T(i)}{P_S(i)}
$$

有时候学生还要同时学习真实标签，因此损失可以组合为

$$
\mathcal L = \alpha\mathcal L_{\mathrm{CE}} + (1-\alpha)\mathcal L_{\mathrm{KD}}
$$

在计算 $D_{KL}$ 时，还可以引入一个参数：温度系数 $T$

$$
P_i^{(T)} = \frac{\exp(z_i/T)} {\sum_j\exp(z_j/T)}
$$

- 这里 $z_i$ 是分类的 Logit ，当 $T$ 较大时，各类别之间的概率差距被缩小，输出分布更平滑，这样那些原本概率很小的类别也能提供一定的训练信号
- 标准软蒸馏中，KL 项通常乘上 $T^2$ 来调整梯度尺度

**DeiT（Data-efficient Image Transformers）**：在 ViT 的输入中额外添加一个可学习向量 Distillation Token ，用来引入教师模型的监督

![[多模态学习/imgs/img3/image-2.png]]

# BERT

**BERT（Bidirectional Encoder Representations from Transformers，基于 Transformer 的双向编码表示）**：Google 在 2018 年提出的预训练语言编码模型，主要用于自然语言理解，让模型理解一个词在具体上下文中的含义，并将这种理解能力迁移到不同的 NLP 任务中

BERT 的输入由三部分组成

![[多模态学习/imgs/img3/image-3.png]]

$$
h_i=E_{\mathrm{token}}(x_i)+E_{\mathrm{segment}}(i)+E_{\mathrm{position}}(i)
$$

- **Token Embedding**：这个词是什么，比如 dog、cute
- **Segment Embedding**：这个词属于哪句话，因为 BERT 可以同时输入两个句子，用 $E_A$ 和 $E_B$ 区分
- **Position Embedding**：BERT 采用可学习的位置向量，而不是原始 Transformer 使用的正弦余弦位置编码

还有两个特殊 Token
- `[CLS]`：放在开头，主要作为句子级任务的汇总表示
- `[SEP]`：用于分隔两个句子，或标记句子结束

## MLM

BERT 的关键创新是它的预训练任务：传统自回归语言模型通过前面的词预测下一个词，而 BERT 使用 **Masked Language Modeling（MLM）**，同时结合前后文预测当前 token 

MLM 会随机遮住一句话中的一些词（记为集合 $M$），并把整句话交给模型，预测被遮住的词

$$
\mathcal L_{\mathrm{MLM}} = -\sum_{i\in M}\log P_\theta(x_i\mid\tilde x)
$$

- $M$：选中用于预测的位置
- $x_i$：被选中位置的原始词
- $\tilde x$：经过随机遮盖后的输入序列

前向传播时，BERT 会为每个位置输出一个特征向量，再通过 MLM 预测头得到该位置在整个词表上的概率分布，计算 Loss 时，只取集合 $M$ 中的位置，与对应的原始词计算交叉熵

BERT 随机选择 15% 的 Token 作为预测目标，在这些被选中的 Token 中：80% 替换成 `[MASK]` ，10% 替换成随机 Token ，10% 保持原样

为什么不把这 15% 全部替换成 `[MASK]`？因为真实使用 BERT 时，输入句子通常没有 `[MASK]` ，如果预训练时只见过这种人工符号，就会产生预训练与下游任务之间的分布差异

## NSP

BERT 还有第二个预训练任务，**Next Sentence Prediction（NSP）**：在一段语料中抽取两个句子 A 和 B ，让模型判断 B 是否紧接着 A 

NSP 希望模型学习句子级关系，预测依据通常是 `[CLS]` 位置的最终表示

因此，BERT 通过两个预训练目标获得不同层次的信息：

$$
\mathcal L_{\mathrm{BERT}} = \mathcal L_{\mathrm{MLM}} + \mathcal L_{\mathrm{NSP}}
$$

MLM 偏向词与上下文的关系，NSP 则引入句子之间的关系监督

## 应用

BERT 学习了对于语言的通用的理解能力，所以对于具体的下游任务，只需要给 BERT 接上不同的预测头就能完成

|任务|使用哪部分输出|预测什么|
|---|---|---|
|情感分类、句子分类|`[CLS]`|整句话的类别|
|句子对关系判断|`[CLS]`|两句话的关系|
|命名实体识别（NER）|每个 Token|每个词的实体类别|
|抽取式问答（SQuAD）|每个 Token|答案的起始与结束位置|

# GPT

既然 Transformer 能够通过预测 Token 学习语言，为什么不直接把预测下一个 Token 当作统一任务，让模型自己学习如何回答问题、翻译和完成其他任务？这就是 GPT 的思路
- **GPT-1**：先预训练，再针对任务微调
- **GPT-2**：能不能只用文字告诉模型任务是什么？
- **GPT-3**：能不能只通过给几个例子，让模型直接学会新任务？

| |BERT|GPT|
|---|---|---|
|主体结构|Transformer Encoder|Decoder-only Transformer|
|注意力方式|双向 Self-Attention|Causal Self-Attention|
|预训练目标|恢复被遮盖的 Token|预测下一个 Token|
|典型用途|理解、分类、信息抽取|文本生成、续写、问答|

GPT 的 Decoder 把原始 Transformer Decoder 的 Cross-Attention 部分，替换为 Masked Self-Attention

## GPT-1

先在“预测下一个词”的任务上预训练，然后再根据具体任务，添加特殊 token 和分类头进行微调

![[多模态学习/imgs/img3/image-4.png]]

![[多模态学习/imgs/img3/image-5.png]]

## GPT-2

既然 GPT 已经能理解一段文本并继续生成，那能不能把任务要求也写进输入，让模型直接按要求输出？这背后有一个非常重要的统一思想：把不同任务都转换成文本续写问题

翻译、摘要、问答看起来是不同任务，但只要都能表示成文本输入与文本输出，就可以通过同一个语言模型来处理，比如要做翻译，不必专门添加翻译模块，而是输入

```
Translate English to French:
cat => chat
dog =>
```

让语言模型继续生成

```
chien
```

架构方面，GPT-2 延续了 Decoder-only Transformer，同时做了一系列训练与规模上的改进
- 将 LayerNorm 移到子模块的输入侧，即 Pre-LN 风格
- 在最终输出处增加 LayerNorm，并调整残差层的初始化缩放
- 词表扩大到 50,257，使用更大规模的 WebText 数据（约 800 万文档、40GB）
- 上下文长度从 512 增加到 1024，最大模型约 15 亿参数，Batch Size 扩大到 512

当 GPT-2 从 1.17 亿扩展到约 15 亿参数时，阅读理解、翻译、摘要、问答等任务的表现普遍提升，这开始体现 **Scaling Law**：通过堆数据就能让模型的表现得到提升

## GPT-3

PT-3 在基本架构上与 GPT-2 相似，但将最大模型扩大到 175 Billion（1750亿），并引入 **In-Context Learning（上下文学习）**：只修改输入上下文（比如提供几个任务示例），不修改模型参数，让模型完成不同的任务

GPT 在大规模预训练中已经学习了许多语言和任务模式，例子相当于告诉模型「当前应该遵循哪种规律」，模型就能根据上下文来预测合适的输出

## GPT-4 与 RLHF

GPT-4 引入了 **RLHF（Reinforcement Learning from Human Feedback，基于人类反馈的强化学习）**

模型学会了生成在训练语料中看起来合理的文本，但这不一定等同于用户希望得到的回答

RLHF 的基本思路是：先收集人类对不同回答的偏好，再训练一个偏好或奖励模型，最后利用这个模型提供的信号优化语言模型，其主要作用是对齐人类偏好，但是如果训练不合理，还可能降低部分原有能力

# LLaMA

从 GPT-3 开始，大模型已经展现出非常强的能力，但有两个现实问题：训练成本极高，模型太大，推理和部署也非常昂贵

**LLaMA（Large Language Model Meta AI）** 则追求在模型架构、训练数据和推理效率之间寻找更好的平衡

## LLaMA 1

LLaMA 的架构仍然是 Decoder-only Transformer，但在三个关键位置做了修改

**第一：RMSNorm 代替 LayerNorm**

普通 LayerNorm 会先减去均值，再除以标准差

$$
\operatorname{LN}(x) = \frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}\odot\gamma+\beta
$$

RMSNorm 不再显式减去均值

$$
\operatorname{RMSNorm}(x) = \frac{x}{\sqrt{\frac1d\sum_{i=1}^d x_i^2+\epsilon}} \odot\gamma
$$

它主要根据向量的均方根来调整尺度，计算更简单

**第二：使用 SiLU 激活函数**

$$
\operatorname{SiLU}(x)=x\cdot\sigma(x)
$$ 

其中 $\sigma(x)$ 是 Sigmoid 函数；SiLU 和 ReLU 不同，允许负数区域保留平滑、非零的响应

**第三：RoPE（Rotary Position Embedding）**

传统的位置编码一般通过相加引入位置信息；RoPE 则将位置编码直接融入 Query 和 Key，可以把它直观地理解为：根据 Token 的位置，把 Query 和 Key 向量旋转不同的角度

## LLaMA 2

模型逐个生成 Token 时，为了避免每生成一个 Token 就重新计算所有历史位置的 Key 和 Value，通常会把它们保存到 **KV Cache** 中，随着序列长度增加，缓存越来越大

对于在 Multi-Head Attention（MHA）部分，每个 Head 都有自己对应的 $Q,K,V$ ，Head 数量越多，需要保存的 Key 和 Value 也越多，内存开销大

LLaMA 2 使用了 **GQA（Grouped-Query Attention）** ，把 Query Head 分成若干组，同一组内共享 Key 和 Value 

![[多模态学习/imgs/img3/image-6.png]]

## LLaMA 3

LLaMA 3 继续沿着大规模训练和高效推理的方向发展

|         | LLaMA 2    | LLaMA 3     |
| ------- | ---------- | ----------- |
| 预训练数据规模 | 约 2T Token | 约 15T Token |
| 上下文长度   | 4096       | 8192        |
| 词表大小    | 32K        | 128K        |
| GQA     | 主要用于较大模型   | 8B 模型也采用    |
| 数据组成    | 以英语为主      | 增加代码和多语言数据  |

词表扩大为什么有用？因为 Tokenizer 不一定把一个完整的汉字、词语或代码片段编码成一个 Token，一个 Token 可能是多个字母/词语的组合，大词表的 Token 种类更丰富，使得更多的字符串可以用较少 Token 表达，减少序列长度，从而改善推理效率

# LoRA

假设已经有一个预训练好的大模型，现在想让它适应一个新任务，最直接的方法是 Full Fine-tuning，即更新模型的所有参数

但大语言模型参数量可能有几十亿甚至上千亿，全参数微调不仅需要保存模型权重，还要保存梯度和优化器状态，训练成本很高

LoRA（Low-Rank Adaptation）的思路是：模型原本已经具备大量知识，为了适应一个新任务，可能不需要对权重矩阵做任意复杂的修改，只需要学习一个低秩的更新量

假设某个预训练线性层的权重为

$$
W_0\in\mathbb R^{d\times k}
$$

原本的前向传播是

$$
h=W_0x
$$

全参数微调相当于训练一个权重变化量 $\Delta W$ 

$$
h=(W_0+\Delta W)x
$$

这里 $\Delta W$ 和 $W_0$ 一样大，所以需要训练 $d\cdot k$ 个参数

LoRA 假设更新量能够表示为两个小矩阵的乘积

$$
\Delta W=BA
$$

其中 $B\in\mathbb R^{d\times r}, \quad A\in\mathbb R^{r\times k}$ ，并且 $r\ll\min(d,k)$ 

于是前向传播变为

$$
h=W_0x+BAx
$$

这里 $W_0$ 完全冻结，只训练 $A$ 和 $B$ ，其中 $A$ 随机初始化，$B$ 初始化为零

为什么不都初始化成零？因为如果两者都为零，乘积 $BA$ 对两个矩阵的梯度也都会为零，导致无法学习

为什么不都随机初始化？因为我们希望刚开始微调时，模型与预训练模型完全一致

为什么叫低秩？因为根据矩阵乘积的秩不等式

$$
\operatorname{rank}(BA) \leq \min(\operatorname{rank}(B),\operatorname{rank}(A)) \leq r
$$

也就是说，虽然 $BA$ 的形状仍然是 $d\times k$，但它的秩最多只有 $r$ ，而且相比 $\Delta W$ 大大减少了参数量

此外，LoRA 的权重更新可以在推理前合并进原始权重 $W_{\mathrm{new}}=W_0+BA$ ，这样合并后仍然只需要一次矩阵乘法 $h=W_{\mathrm{new}}x$ ，而不必额外计算一个分支













