# Word Sense Disambiguation (WSD)

一个词在不同上下文中可能对应不同含义（**义项 sense**），**词义消歧 WSD** 就是要判断出这个词到底啥意思（选一个义项）
- 输入：目标词 $w$ 、上下文 $C$ 、**义项集合 sense inventory** $S$ （目标词可选哪些义项）
- 输出：在当前上下文中最合适的义项标签

上下文线索通常包括
- 邻近词和局部搭配
- 目标词及邻词词性（POS）
- 句法结构信息
- 其他模态信息

常见资源
- WordNet （English）
- HowNet （Chinese）
- 现代汉语语义词典（Chinese）
- BabelNet （Multilingual）
- SenseEval / SemEval 数据集（含标注样本）

WordNet 的组织方式是 synset（同义词集） + gloss（解释） + example（例句），这也是很多基于词典的方法（如 Lesk）的基础

![[NLP基础/img/img1/image.png]]

BabelNet 把不同语言中的同义词联系在一起

![[NLP基础/img/img1/image-1.png]]

注意 WordNet 是手动构建，而 BabelNet 是自动构建，相对没那么精确

# Machine Learning Methods for WSD

- 三种学习方式：自监督、半监督、无监督
- 适用场景的区别：有没有可用的义项标签、有没有足够的标注数据

## Supervised Learning

监督学习把 WSD 当成标准分类任务，前提是候选义项集合已经定义好，而且训练集中每条样本都有人标了正确义项

它的核心逻辑是“从已标注上下文学决策边界”
- 第一步准备数据：每个样本是 `目标词 + 上下文 + 义项标签`
- 第二步做表示：把上下文变成模型可用的特征或向量
- 第三步训练模型：学习“什么上下文对应什么义项”
- 第四步预测新样本：在候选义项里选概率最高的标签

这种方法通常效果最稳，但成本也最高，标注量不够时容易过拟合，遇到训练集没见过的新义项也容易失效

常见上下文特征
- neighboring words
- local collocations / n-grams
- POS tags
- 句法特征
- 分布式语义特征

常见模型
- Naive Bayes
- Logistic Regression
- SVM / MaxEnt
- Neural Network Models

## Semi-Supervised Learning

半监督学习用于“有一点标注，但远远不够”这种最常见场景

基本做法是先用小规模人工标注训练一个初始模型，再让模型去大规模无标注语料里找高置信样本，把这些伪标签回流继续训练

常见自举流程
- 小规模人工标注作为 seeds
- 训练初始分类器
- 在无标注语料上打分，选高置信预测
- 加入训练集重训，重复多轮

它的好处是能显著减少人工标注成本，难点是错误标签会被反复放大

## Unsupervised Learning

无监督学习不依赖人工标签，通常把 WSD 当作“同一词不同用法的聚类问题”

直觉是“相似上下文表达相似词义”
- 先把每次词出现的上下文编码成向量
- 再把这些用法聚成若干簇
- 每个簇对应一个潜在义项

这条路线适合低资源语言、新词新义、或者义项集合还不稳定的任务

主要难点不是聚类本身，而是解释和对齐
- 一个聚类簇不一定和词典里的单一义项一一对应（但是也有可能会发现新的词义）
- 落地时往往还要把簇映射回现有的 sense inventory

# Naive Bayes Model

把 WSD 视作一个多分类任务，判断一句话中，目标词是啥意思
- 给定
	- 目标词 $w$ 
	- 候选义项集合 $S=\{s_1,s_2,...,s_n\}$ 
	- 训练集 $D_w=\{(C_i,y_i)\}_{i=1}^N$ ，其中 $y_i \in S$ 
	- 特征词表 $V=\{v_1,v_2,...,v_m\}$ 
- 输入：一条未见样本的上下文 $C$ （一句话）
- 输出：该样本对应的义项标签 $s^*$ 

>其中特征 $v$ 是从上下文 $C$ 中提取出来的、用于帮助判断目标词意思的其它相关联的词（比如周围的词、修饰词等），也可能是其它一些东西（比如一个长度为 n 的词组，称为 **n-gram**）；一种简单的做法是，一句话由很多单词组成，把每个单词视为一个特征

目标可以表示为在给定上下文 $C$ 的条件下，目标词是哪个意思的概率最高，即

$$
s^* = \arg\max_{s_k} P(s_k\mid C)
$$

由贝叶斯公式

$$
P(s_k\mid C)=\frac{P(C\mid s_k)P(s_k)}{P(C)}
$$

因为 $P(C)$ 对所有候选义项相同，所以不用管分母

$$
s^* = \arg\max_{s_k} P(C\mid s_k)P(s_k)
$$

其中 $P(s_k)$ 可以根据频率进行估计（虽然并不一定准确），那 $P(C\mid s_k)$ （已知目标词的意思，能看到这句话的概率）咋整呢？我们认为一句话 $C$ 可以由若干特征 $v$ 去描述，可以假设**各特征的出现与否相互独立**（这也不一定准确），这就好办了

$$
P(C\mid s_k)=P(\{v_x \mid v_x \in C\}\mid s_k)=\prod_{v_x\in C} P(v_x\mid s_k)
$$

于是目标就变为

$$
s^* = \arg\max_{s_k} \left(\prod_{v_x\in C} P(v_x\mid s_k)\right) P(s_k)
$$

---

$$
P(s_k)=\frac{\text{Count}(s_k)}{\text{Count}(w)}
$$

- 含义：目标词 $w$ 取 $s_k$ 义项的概率
- $\text{Count}(w)$ ：训练集中 $w$ 总共出现次数
- $\text{Count}(s_k)$ ：$w$ 取 $s_k$ 义项的次数

$$
P(v_x\mid s_k)=\frac{\text{Count}(v_x,s_k)}{\sum_{v\in V}\text{Count}(v,s_k)}
$$

- 含义：在目标词义项为 $s_k$ 的情况下，在上下文看到特征 $v_x$ 的概率
- $\text{Count}(v_x,s_k)$ ：在所有义项为 $s_k$ 的句子里，特征 $v_x$ 的出现次数
- $\sum_{v\in V}\text{Count}(v,s_k)$ ：在所有义项为 $s_k$ 的句子里，所有特征总共出现次数
- 实际应用中常常会引入平滑项，给每个特征加一个基础的出现次数 $\alpha$ ，此时分子 $+\alpha$ ，分母 $+\alpha \lvert V\rvert$ （$\lvert V \rvert$ 为特征词表大小）

比如下面的例子，目标词 $w=\text{can}$ ，取每个单词作为一个特征词
- 训练集
	- `I have a can of coke.` （$s_2$）
	- `I can not do this.` （$s_1$）
	- `I would like to can a salmon.` （$s_3$）
- 义项集合
	- $s_1$ ：“能”
	- $s_2$ ：“易拉罐”
	- $s_3$ ：“开罐子”

则 $P(I\mid s_2)=1/6$ 
- 分子 1 表示在所有义项为 $s_2$ 的句子中，"I" 出现的次数
- 分母 6 表示在所有义项为 $s_2$ 的句子中，单词的总数

>这里 $P(I\mid s_2)$ 并不是采用"在所有 $s_2$ 的句子中，含 I 的句子占比"来求解（伯努利贝叶斯），而是看单词的出现频率占比（多项式贝叶斯），这对应着两种朴素贝叶斯模型

>可见多项式贝叶斯更关注词频的影响，但实际上，在 WSD 中伯努利贝叶斯会更适合一些，因为在判断词义时，其它词的词频通常不是核心依据，选择多项式贝叶斯可能有一些缺点
>- 如果目标词某种意思常常伴随更复杂的语句（词很多），那句子中的每个单词的概率就很小，乘起来也很小，会导致基本上不可能选择到这个义项
>- 如果某种词义常常伴随另一个词出现（类似于固定搭配），多项式贝叶斯也无法直接反映

## Evaluation

使用经典的 precision 和 recall 来衡量
- precision：在所有预测的句子里，预测 $s_k$ 对了的有多少
- recall：在所有 $s_k$ 的句子里，预测出来的有多少

![[NLP基础/img/img1/image-3.png]]

此外还有 F-measure 指标，是精确率和召回率的加权调和平均

$$F_{\beta} = \frac{1 + \beta^2}{\frac{1}{Pre} + \frac{\beta^2}{Rec}} = \frac{(\beta^2 + 1) Pre \times Rec}{\beta^2 \times Pre + Rec}
$$

常常使用 F1，即取 $\beta=1$ 


$$
F_{1} = \frac{2}{\frac{1}{Pre} + \frac{1}{Rec}} = \frac{2 \times Pre \times Rec}{Pre + Rec}
$$


- 宏平均 Marco F1：对每个义项算 F1 ，再取平均
- 微平均 Micro F1：把所有义项的 TP/FP/FN 各自加起来，算总体的 precision 和 recall ，再拿这俩算 F1 

两种 setup 
- Lexical-Sample：只做少量目标词的消歧，一般一句话只有一个目标词
- All-Words：把文本里所有实词都进行消歧

