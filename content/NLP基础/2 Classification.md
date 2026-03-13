
# Probabilistic Model

分类任务就是要根据输入 $x$ ，判断最可能是哪个标签 $y$ ，可以表示成这样

$$
\text{Find function } g: y=g(x), \text{where } g(x)=\arg\max_y p(y\mid x)
$$

两种模型
- 判别式模型：直接学习 $p(y\mid x)$ 
- 生成式模型：学习 $p(x,y)$ ，或 $p(x\mid y)$ 和 $p(y)$ ，再用贝叶斯得到 $p(y\mid x)$ 
	- 前面的贝叶斯模型就属于生成式模型

对于生成式模型，由贝叶斯公式，可得

$$
p(y|x) = \frac{p(x,y)}{p(x)} = \frac{p(x|y)p(y)}{p(x)}
$$

由于我们只需要找到概率最大的一个，而分母 $p(x)$ 都一样，所以可以不管，这样目标就变成了

$$
g(x) = \arg\max_y p(x,y) = \arg\max_y p(x\mid y) p(y)
$$

## Bayesian Modeling

两个核心关注点
 - 数据生成分布 $p(x|y)$：已知标签 $y$，观察到的 $x$ 是如何生成的
- 模型先验 $p(y)$：模型对 $y$ 的初始假设，也就是关于 $y$ 的先验知识

目标是学习 $p(y\mid x)$ ，由贝叶斯公式，即

$$
p(y|x) = \frac{p(x|y)p(y)}{p(x)} \propto p(x|y)p(y)
$$

但通常来讲，有以下困难
- $p(x\mid y)$ 可能会很复杂，难以通过显式公式精准建模
- $p(y\mid x)$ 可能很难进行解析求解

所以需要做出一些假设，比如图书分类任务中
- $Y$ ：书的类别
	- $Y \sim \text{Multinominal}(\gamma)$ （多项分布） 
	- 参数 $\gamma$ 是一个概率向量，比如 $\gamma=[0.2,0.5,0.3]$ ，表示各个类别出现的概率分别是多少
- $X$ ：书中出现的单词
- $P(X\mid Y)$ ：某个类别的书中，可能出现什么单词
	- $X \mid Y \sim p(X \mid \theta_Y)$ 
	- 参数 $\theta_Y$ 也是一个概率向量，表示在类别 $Y$ 的书中，各种词可能出现的概率
- $P(Y\mid X)$ ：给定书中出现的单词，判断这个书啥类别

根据训练集估计参数 $\gamma$ 和 $\theta_Y$ ，可以使用 MLE 

$$
\hat \gamma_k = \frac{\text{类别 k 出现的次数}}{\text{总样本数}}, \quad
\hat \theta_{k,w} = \frac{\text{类别 k 中单词 w 出现的次数}}{\text{类别 k 的总词数}}
$$

但这有一些问题
- 零概率问题：如果一个单词从没在训练集出现过，此时 $P(X\mid Y)=0$，很可能发生误判（对于一个没出现过的类别也同理）
	- 比如训练集的“科幻书”中从没出现过“爱情”，现在来了一本科幻言情小说，模型会认为这本书是“科幻”的概率是 0 
- 小样本下的过拟合：当某个类别的样本很少时，对应的参数 $\hat \theta_k$ 噪声极大，泛化能力差
	- 比如“侦探书”中只收录了福尔摩斯，那“伦敦”这种词出现的概率就可能会被高估，但这显然无法代表其它的侦探小说

---

为了解决上述问题，我们引入一些先验假设
- $\gamma \sim \text{Dirichlet}(\beta)$ （狄利克雷分布）
- $\theta_1,\cdots,\theta_n \overset{i.i.d.}{\sim} p(\theta)$ （$p(\theta)$ 一般也是狄利克雷分布，即 $\theta_k \sim \text{Dirichlet}(\alpha)$）

**狄利克雷分布**相当于“概率分布的分布”，参数 $\alpha=(\alpha_1,\cdots,\alpha_n)$ 控制各种概率向量 $x=(x_1,\cdots,x_n),\sum x_i=1$ 出现的概率

$$
f(\mathbf{x}; \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{i=1}^{K} x_i^{\alpha_i - 1}, \quad
\text{where }B(\boldsymbol{\alpha}) = \frac{\prod_{i=1}^{K} \Gamma(\alpha_i)}{\Gamma\left(\sum_{i=1}^{K} \alpha_i\right)}
$$

- $\alpha_k$ 可以理解为给 $k$ 添加的伪计数，$\alpha_k$ 越大，表示先验上觉着它更可能出现
- 所有 $\alpha_i$​ 值较小时偏向稀疏分布，会极端的偏向某个成分（即更可能出现某个 $x_k$ 特别大的情况）；较大时偏向均匀分布

这样相当于给 $\gamma$ 和 $\theta$ 添加了平滑项

$$
\hat{\gamma}_k = \frac{\text{类别}k\text{出现次数} + \beta_k}{N + \sum_j \beta_j}, \quad
\hat{\theta}_{k,w} = \frac{\text{类别}k\text{中单词}w\text{出现次数} + \alpha_w}{\text{类别}k\text{总词数} + \sum_v \alpha_v}
$$

这样以来，原本为 0 的地方，由于加上了伪计数，不再是 0 了；对于小样本，它的参数估计会更强烈的向先验靠拢，缓解过拟合

## LDA (Latent Dirichlet Allocation)

综合上面的概率分布假设，就得到了 **LDA 潜在狄利克雷分配**
- $Y \sim \text{Multinominal}(\gamma),\quad \gamma \sim \text{Dirichlet}(\beta)$ 
- $X \mid Y \sim p(X \mid \theta_Y),\quad \theta_1,\cdots,\theta_n \overset{i.i.d.}{\sim} \text{Dirichlet}(\alpha)$ 

LDA 是无监督的，训练时不需要给标签 $y$ ，它是一种主题模型，假设每个文档由多个主题混合而成，并假设文档是这样搞出来的
- 从 $\text{Dirichlet}(\alpha)$ 采样各个主题的词分布 $\theta_k$ 
- 生成一系列文档
	- 决定一个文档的总词数 $N$ 
	- 从 $\text{Dirichlet}(\beta)$ 采样主题分布 $\gamma$ 
	- 循环生成 $N$ 个词
		- 根据 $\gamma$ 采样一个主题 $z$ 
		- 根据 $\theta_z$ 采样一个词 $w$ 

- 训练时，根据训练集（即观测到的文档），通过算法（如吉布斯采样）推断一个文档中每个词的主题，和这个文档的主题分布，最后估计出 $\alpha$ 和 $\beta$ 
- 预测时，根据给定文档中的词，通过算法（如吉布斯采样）迭代估计每个词的主题，预测这个文档的主题分布

# Feature Representation

## Features in NLP

在 NLP 中，特征是用来描述数据实例的规则或属性，它是关于输入 $x$ 和输出 $y$ 的函数 $f_i(x,y)$ ，通常是一个指示函数，输出 0 或 1，表示某个条件是否成立

>为啥特征还是关于输出值 $y$ 的函数？实际上，这种特征建模主要是在传统的基于特征的机器学习方法中使用，模型会枚举所有可能的 $y$ 候选，为每个候选计算特征向量，然后使用训练好的权重打分，选择最高分的 $y$ 

比如对目标词 bank 进行 WSD 时，可以定义这样一个特征：前一个词是 transfer ，且目标词义为 FINANCIAL ，那就可以这么表示

$$
f_i =
\begin{cases}
1 & \text{if } w_{-1} = \text{transfer and } y = \text{FINANCIAL}, \\
0 & \text{otherwise}
\end{cases}
$$

可以定义一堆特征，组成一个特征向量 $f(x,y)=[f_1(x,y),\cdots, f_m(x,y)]$ 

>可见这种方式极大的依赖先验，需要预先定义很多特征规则

## Bag-of-Words

词袋模型是一种简单的特征建模，其基本思想是
- 把每个词作为一个特征，只关注每个单词出没出现、出现多少次
- 忽略词语顺序，每个单词彼此独立

需要预先定义一个词典，比如 $[w_1,w_2,w_3]$ ，而后统计一段文本中每个单词 $w_i$ 是否出现 / 出现次数，得到特征向量（比如 $[1,0,1]$ / $[2,0,4]$ ）

但是这么做有个问题，一些辅助性的词（比如 “的” “是”）会有很大权重，影响判断，为此可以使用 **tf-idf 加权**
- **TF**：词在当前文本的频率（局部重要性）
- **IDF**：词的“稀缺性”权重（全局重要性）
    - “的” “是”这种几乎每个文档都有的词，IDF值很低
    - “神经网络”这种只在少数文档出现的专业词，IDF值很高

>回顾一下怎么算 tf-idf [[AI 3 - NLP#信息检索指标 tf-idf]] 

文本中可能有一些低权重特征（出现次数少 / 不重要的词），通常贡献很小，但会让特征变得稀疏，常见的应对方法有
- **stop-word list 停用词表**：直接过滤掉常见的无实义的词
- **stemming 词根提取**：把派生词合并到词根（比如 running ran 等变形，都算作 run）

# Feature Based Discriminative Model

前面提到过，传统的基于特征的机器学习方法中，模型会枚举所有可能的 $y$ 候选，计算特征 $f_i(x,y)$ ，然后使用训练好的权重 $\lambda_{f_i}$ 打分，为了学习 $\lambda_{f_i}$ ，通常有以下方法
- 感知器算法（单层感知器 Perceptron，是线性模型）
- 基于边际的 margin-based 模型：如 SVM
- 指数模型：如对数线性模型、最大熵模型、逻辑回归模型

## Log-linear Model

对于一个样本 $(x,y)$ ，我们给它打分

$$
p(y\mid x) = \frac{\exp \sum_i \lambda_{f_i(x,y)} f_i(x,y)}{\sum_{y'} \exp \sum_i \lambda_{f_i(x,y')} f_i(x,y')} = \frac{\exp(\lambda \cdot f(x,y))}{\sum_{y'} \exp(\lambda \cdot f(x,y'))}
$$

其中 $\lambda=(\lambda_{f_1},\cdots,\lambda_{f_n}),\quad f(x,y)=(f_1,\cdots,f_n)$ ，取对数得

$$
\log p(y\mid x; \lambda) = \lambda \cdot f(x,y) - \log \sum_{y'} \exp(\lambda \cdot f(x,y'))
$$

那咋根据训练集估计参数 $\lambda$ 呢？假设训练集是 $\{(x_1,y_1),\cdots,(x_n,y_n)\}$ ，使用 MLE ，$\lambda$ 的似然函数就是在给定参数的情况下，观测到训练集的概率

$$
L(\lambda) = \prod_k p(y_k | x_k; \lambda) = \prod_k \frac{\exp(\lambda \cdot f(x_k, y_k))}{\sum_{y'} \exp(\lambda \cdot f(x_k, y'))}
$$

取对数得

$$
LL(\lambda) = \sum_k \log p(y_k | x_k; \lambda)
= \sum_k \lambda \cdot f(x_k, y_k) - \sum_k \log \sum_{y'} \exp(\lambda \cdot f(x_k, y'))
$$

我们希望最大化 $L(\lambda)$ ，即最大化 $LL(\lambda)$ ，对 $\lambda_{f_i}$ 求偏导得

$$
\begin{aligned}
\frac{\partial LL(\lambda)}{\partial \lambda_{f_i(\cdot)}} &= \sum_k f_i(x_k, y_k) - \sum_k \frac{\sum_{y'} f_i(x_k, y') \exp(\lambda \cdot f(x_k, y'))}{\sum_{z'} \exp(\lambda \cdot f(x_k, z'))} \\
&= \sum_k f_i(x_k, y_k) - \sum_k \sum_{y'} f_i(x_k, y') \frac{\exp(\lambda \cdot f(x_k, y'))}{\sum_{z'} \exp(\lambda \cdot f(x_k, z'))} \\
&= \underbrace{\sum_k f_i(x_k, y_k)}_{\text{Empirical Counts}} - \underbrace{\sum_k \sum_{y'} f_i(x_k, y') p(y' | x_k; \lambda)}_{\text{Expected Counts}}
\end{aligned}
$$

- 经验计数 Empirical Counts：特征 $f_i$ 在训练集出现的总次数
- 期望计数 Expected Counts：特征 $f_i$ 在当前模型参数下，估计的出现次数
- 希望经验计数 = 期望计数 

后面就是梯度上升（也可以取负然后梯度下降）、正则化了



