
# What is Language Model

语言模型，简单说就是给一句话打分，看它在某种语言里“自然不自然”，本质上是在回答：一个词串作为该语言表达的可能性有多大
- “the dog barks” 看起来正常
- “dog the dog bark a laugh smile” 看起来就不像话

这有啥用？比如在语音识别里，一个发音可能有若干候选词，就需要结合上下文信息判断选哪个词，这句话才是人话

$$
words^* = \arg \max_{words} \text{Faithfulness}(signals, words) + \text{Fluency}(words)
$$

- Faithfulness ：这句话和听到的声音信号 $signal$ 像不像
- Fluency ：这句话本身通不通顺（是不是人话）

# N-gram Language Models

既然语言模型是判断一句话的概率，那咋算呢？显然不能以句子为单位进行统计，可以把句子看作词序列
- 句子由一堆词组成，还可以引入 `STOP` 这种标志符号
- 需要考虑词的顺序

设一个句子中包含单词 $w_1,\cdots,w_n$ （其中 $w_n$ 为 `STOP` 符号），考虑一个句子的生成过程，一开始有一个词 $w_1$ ，根据 $w_1$ 选择出 $w_2$ ，再根据 $w_1,w_2$ 选择出 $w_3$ ，可得一句话的概率为一堆条件概率连乘

$$
\begin{aligned}
&P(X_1 = w_1, X_2 = w_2, X_3 = w_3, ..., X_n = STOP) \\
&= P(X_1 = w_1) \prod_{i=2}^{n} P(X_i = w_i \mid X_1 = w_1, ..., X_{i-1} = w_{i-1})
\end{aligned}
$$

但是如果每一个词都依赖之前的所有词，这个条件概率太难算了，需要做一些假设，比如每个词只依赖前一个词，这就是**一阶马尔可夫假设**，得到的模型就是 **Bigram（二元）语言模型**（前后两个词相关联）

$$
\begin{aligned}
&P(X_1 = w_1, X_2 = w_2, X_3 = w_3, ..., X_n = STOP) \\
&= P(X_1 = w_1) \prod_{i=2}^{n} P(X_i = w_i \mid X_1 = w_1, ..., X_{i-1} = w_{i-1}) \\
&\approx P(X_1 = w_1) \prod_{i=2}^{n} P(X_i = w_i \mid X_{i-1} = w_{i-1})
\end{aligned}
$$

>极端一点，直接假设每个词和之前没有任何联系，即都是独立的，不考虑词的顺序，得到的就是 **Unigram （一元）模型**，实际上就是 BoW 

如果假设每个词依赖前面两个词，即**二阶马尔科夫假设**，得到的模型就是 **Trigram（三元）模型**

$$
p(w_1, w_2, ..., w_n) = p(w_1) p(w_2 \mid w_1) \prod_{i=3}^{n} p(w_i \mid w_{i-2}, w_{i-1})
$$
比如给一句话 `the cat laughs. STOP` ，使用三元模型，其概率就是

$$
\begin{aligned}
p(\text{the cat laughs . STOP}) &= p(\text{the}) \\
&\times\ p(\text{cat}\mid \text{the}) \\
&\times\ p(\text{laughs}\mid \text{the},\ \text{cat}) \\
&\times\ p(\text{.}\mid \text{cat},\ \text{laughs}) \\
&\times\ p(\text{STOP}\mid \text{laughs},\ \text{.})
\end{aligned}
$$

推广下去，假设每个词与前面 $n-1$ 个词相关联，得到的就是 **n-gram 模型**

## Parameter Estimation

要得到这些概率，可以从训练集中进行统计

$$
\begin{aligned}
&\text{Bigram} \quad p(w_i \mid w_{i-1}) = \frac{\text{count}(w_{i-1}, w_i)}{\text{count}(w_{i-1})} \\
&\text{Trigram} \quad p(w_i \mid w_{i-2}, w_{i-1}) = \frac{\text{count}(w_{i-2}, w_{i-1}, w_i)}{\text{count}(w_{i-2}, w_{i-1})}
\end{aligned}
$$

但是这有一个问题，当 $n$ 变得很大时，从训练集统计出的 $p(w_i \mid w_{i-1}, w_{i-2},\cdots)$ 往往只占真实语料的很小一部分，有很多搭配可能都没见过，这就会导致**数据稀疏 data sparsity** 的问题，即很多条件概率都是 0 ，一般来讲 Trigram 是一个比较实用的折中

## Evaluation

想法
- 如果一段话本来就很自然、很像真实语言，那好模型应该给它较高概率
- 所以评估时就拿测试集 $D$ 的文本给模型，看它到底有多“看好”这些真实句子

过程
- 先算每个句子的概率 $p(s)$ 
- 把所有句子的概率乘起来 $\prod_{s \in D} p(s)$ 
- 测试集有长有短，长文本词更多，乘的概率项更多，结果天然就更小，所以需要用测试集总词数 $M$ 归一化 $\sqrt[M]{\prod_{s \in D} p(s)}$ 
- 取对数 $\frac{1}{M} \sum_{s \in D} \log p(s)$ 
- 据此定义**困惑度 perplexity**

$$
\text{Perplexity}(D) = 2^{-\frac{1}{M} \sum_{s \in D} \log p(s)}
$$

可以简单理解为模型生成文本时，预测下一个词的时候到有多懵，困惑度越小，说明模型越好

>困惑度只是评估模型续写 / 预测词的能力，不完全涵盖推理能力等

# Deal with Unseen Events

如果测试时出现训练里没见过的词、没见过的 n-gram，直接用 MLE，会导致概率为 0 ，整句概率就直接乘成 0 ，这显然很糟糕
- 需要留出一部分概率质量给“没见过的事件”
- 代价是见过的事件的概率要稍微降一点，因为总概率必须加起来等于 1 

## Back-off

- 有高阶 n-gram 数据就用高阶的，没有就回退到低阶 tri/bigram 
- Google 使用过的 Stupid Back-off ：回退时再乘一个缩放系数
- 好用快速，但不是严格的概率分布（所有概率加起来不一定是 1 ）

## Linear Interpolation

不只使用高阶数据，也综合低阶的

$$
p_{l}(w_i \mid w_{i-2}, w_{i-1}) = \lambda_1 p_{ml}(w_i \mid w_{i-2}, w_{i-1})
+ \lambda_2 p_{ml}(w_i \mid w_{i-1})
+ \lambda_3 p_{ml}(w_i)
$$

## Smoothing

Add-One / Laplace smoothing ：给所有 n-gram （无论是否出现）添加一个伪计数（出现次数 +1）
- 这个方法虽然简单，但给没见过事件分了太多概率，把原来真正常见的事件概率压得太低
- 还有 Good-Turing 、Absolute Discounting 、Kneser-Ney 平滑方法，其中 Kneser-Ney 是传统 n-gram 语言模型中很重要的方法


















