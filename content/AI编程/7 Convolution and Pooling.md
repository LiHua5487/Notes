
# Convolution

为了上下文的完整性，贴个介绍卷积运算的图

![[AI编程/imgs/img7/image.png]]

![[AI编程/imgs/img7/image-1.png]]

神经网络中，输入维度一般为 $[N, C_{in}, H, W]$ ，其中 $N$ 是 batch size ，表示有多少图片，$C_{in}$ 表示每个图片的通道数，$H×W$ 是图像大小，在对一个图片进行卷积时，对不同的通道可以应用不同的 $K×K$ 卷积核，把它们的卷积结果叠加，作为这个图片的卷积结果，其形状为 $H_{out}×W_{out}$ ，这个图片对应的卷积核大小就是 $C_{in}×K×K$ 的

我们可以同时将 $C_{out}$ 个不同的 $C_{in}×K×K$ 卷积核应用于这个图片，就可以得到 $C_{out}$ 个输出，每个输出称为一个通道，把这些卷积核放到一起，形状为 $[C_{out}, C_{in}, K, K]$ ；对于一个 batch 中的多个图片，它们都应用同一套 $[C_{out}, C_{in}, K, K]$ 的卷积核

![[AI编程/imgs/img7/image-2.png]]

pytorch 中的卷积层如下

```python
torch.nn.functional.conv2d(
	input,
	weight,
	bias=None,
	stride=1, # 步长
	padding=0, # 将图像扩展几圈，使用 zero padding
	dilation=1, # 扩张，见注释
	groups=1 # 分组卷积，见下文
)
```

- input ：形状为 $[N, C_{in}, H, W]$
- weight ：形状为 $[C_{out}, C_{in}, K, K]$
- bias ：形状为 $[C_{out}]$ ，默认没有偏置
- output ：形状为 $[N,C_{out},H_{out},W_{out}]$

>扩张 dilation ：将窗口进行扩张，而后每隔一定间距选取元素参与卷积，同时总计算量不变，可用于扩大感受野；默认值为 1 ，表示选取的元素都是相邻的，如果取值为 2 ，代表间隔了 1 个元素

此外还有其它类型的卷积，对于一个 4 通道图片，正常的卷积是对每个通道应用不同的卷积核，将结果叠加到一起输出一个图片；**分组卷积 Group Conv**（假设分 2 组）是对每组内的通道应用相同卷积并叠加，不同组的卷积权重不同，最终得到 2 个图片；**Depth-wise Conv** 是对每个通道应用不同的卷积，但是不叠加，输出 4 个图片

## Convolutions as Matrix Multiplication

以一维情况为例，假设输入为 $[x_1\ ...\ x_5]$ ，卷积核为 $[w_1\ w_2\ w_3]$ ，为了保证输出大小不变，输入两边各填充一个 0 ，输出为 $[y_1\ ...\ y_5]$ ，可得 $y_i$ 是 $x_j$ 和 $w_k$ 的线性组合，这可以写成矩阵形式

一种方式是，把权重展开成一个大矩阵 $\hat W$ （这称为 **Toeplitz矩阵** ），并把输入展平，可见 $\hat W$ 是一个稀疏矩阵，这就可以使用 Sparse matrix-vector multiplication (SpMV) 计算

![[AI编程/imgs/img7/image-3.png]]

对于高维情况，假设只有一个图片，则 $\hat W$ 形状为 $[C_{out}×H_{out}×W_{out},\ C_{in}×H×W]$ ，输入展平后形状为 $[C_{in}×H×W,\ 1]$ 

在反向传播时也可以这么做，值得注意的是权重矩阵中的元素 $w_i$ 会在不止一个窗口中参与运算，计算梯度时需要把它们都考虑到
- 计算 $\frac{\partial L}{\partial X}$ 可以采用  $\frac{\partial L}{\partial X} = \hat{W}^T\cdot \frac{\partial L}{\partial Y}$  ，同样能用 SpMV 计算
- 计算 $\frac{\partial L}{\partial W}$ 可以先计算 $\frac{\partial L}{\partial \hat{W}} = \frac{\partial L}{\partial Y} \cdot X^T$ ，再根据 $\hat W$ 中哪些位置是 $W$ 中的元素 $w_i$ ，把 $\frac{\partial L}{\partial \hat{W}}$ 中这些位置的值累加作为 $w_i$ 处的梯度（这可以通过 segmented reduce 实现）

但是 $\frac{\partial L}{\partial \hat{W}}$ 的大小和 $\hat W$ 相同，这是非常大的，但 $\hat W$ 很多元素是 0 ，可以用稀疏矩阵的方式存储，而 $\frac{\partial L}{\partial \hat{W}}$ 就不一定是这样的了，太费空间了

---

另一种方式是把输入展开为一个大矩阵 $\hat X$ （这个过程称为 **im2col** ），把权重展平，并使用 gemm 计算

![[AI编程/imgs/img7/image-4.png]]

对于高维情况，假设只有一个图片，将每次卷积操作的输入与权重展开为一行/一列，形成特征矩阵 $\hat X$ ，其形状为 $[H_{out}×W_{out},\ C_{in}×K×K]$ ，权重展开后的形状为 $[C_{out},\ C_{in}×K×K]$ （将其转置参与运算）

当涉及到 $N$ 个图片时，$\hat X$ 形状为 $[N×H_{out}×W_{out},\ C_{in}×K×K]$ ，而且这是不稀疏的，会占用很大空间（这种方式称为 Explicit GEMM ，为了解决这个问题，可以使用 Implicit GEMM ，不需要显式的构建这个大矩阵）

![[AI编程/imgs/img7/image-5.png]]

对于反向传播
- $\frac{\partial L}{\partial W} = \frac{\partial L}{\partial Y} \cdot \hat{X}^T$ 可以使用 gemm 计算
- 计算 $\frac{\partial L}{\partial X}$ 可以先通过 gemm 计算 $\frac{\partial L}{\partial \hat{X}} = W^T \cdot \frac{\partial L}{\partial Y}$ ，再根据位置进行还原 $\frac{\partial L}{\partial X} \leftarrow \frac{\partial L}{\partial \hat{X}}$ （这个过程称为 **col2im** ）

由于 im2col 中，一个像素可能复制到多个位置，所以反向传播 col2im 时，需要把在这些位置的梯度累加，可以使用原子操作（原子操作开销较大，这里存在优化空间）

>反卷积 deconvolution/transposed convolution 可以通过交换卷积的正向和反向传播来实现

## Depthwise Conv

在 depthwise conv 中，每个通道是独立进行卷积的，单次计算的计算量减少了，但是总共的计算次数并没有减少，仍会有较频繁的内存访问，此时容易受到内存限制，应该优先从内存方面进行优化

![[AI编程/imgs/img7/image-6.png]]

此外，由于单次卷积只涉及 $K×K$ 范围的元素，直接让一个线程负责计算一个窗口的卷积即可，不需要使用 gemm 

```cpp
__global__ void ConvolutionDepthWiseForward(
    const float* in_data, const float* weight_data, const int in_h, const int in_w, const int height, const int width, const int kernel_h, const int kernel_w, float* out_data, const int threads) 
{
    for (int64_t index = threadIdx.x + blockDim.x * blockIdx.x; 
         index < threads; 
         index += blockDim.x * gridDim.x) 
    {
        // for tensor shape [N, H, W, C]
        const int n = (index / channels / height / width);
        const int h = (index / channels / width) % height;
        const int w = (index / channels) % width;
        const int c = (index % channels);
        // ...
    }
}
```

反向操作时，计算关于权重的梯度比较困难，需要把 $w_i$ 对应的一系列位置的梯度值累加起来，可以利用共享内存和 reduce 操作实现

# Pooling

以最大池化为例，pytorch 的最大池化层如下

```python
torch.nn.functional.max_pool2d(
    input, 
    kernel_size, 
    stride=None, 
    padding=0, 
    dilation=1, 
    ceil_mode=False, # 是否向上取整，见注释
    return_indices=False # 是否提供最大值的索引
)
```

>ceil mode ：一些情况下按照步长移动窗口，窗口的一部分可能超出输入图像的边界，比如输入大小 5×5，`kernel_size=2`，`stride=2`，此时如果采用向上取整模式，会把超出的部分补 0 或其它默认值，输出大小会是 3×3

实现时，只需让每个线程负责在一个窗口内找到最大值即可

```cpp
__global__ void max_pool_forward(
    float* in_data, int nthreads, int num, int channels, int in_h, int in_w, int out_h, int out_w, int kernel_h, int kernel_w, int stride_h, int stride_w, int pad_h, int pad_w, float* out_data, float* out_mask) 
{
    CUDA_KERNEL_LOOP(index, nthreads) {
        int n = index / out_w / out_h / channels;
        int c = (index / out_w / out_h) % channels;
        int ph = (index / out_w) % out_h;
        int pw = index % out_w;
        // implement max pooling for each local window,
        // store the max value and mask
    }
}
```

前向传播时，还可以记录最大值的位置形成的 mask ，便于反向传播计算，只需根据这个 mask 把上游梯度值放到相应位置，其余地方补 0 

![[AI编程/imgs/img7/image-7.png]]

## Stencil Pattern

在 conv 和 pooling 的前向传播时，都是把一个窗口内的多个元素映射到一个元素，这与 gather 一致；反向传播时，conv 要把一个梯度值分散到一系列像素上，而 pooling 可以看作把一个元素映射到一个窗口，这与 scatter 一致

在确定参数后，这些窗口的大小和位置都是固定的，这与 stencil （在固定邻域的 gather ）一致

如果涉及的邻域比较小，可以让一个线程负责一个位置的运算，对于 gather ，每个线程负责输出结果上的一个位置的计算；对于 scatter ，如果让一个线程负责输入上的一个位置的计算，不同输入位置可能会 scatter 到同一个输出位置，这就存在数据竞争，所以要让每个线程负责输出上的一个位置的计算

![[AI编程/imgs/img7/image-8.png]]

# Softmax

对于输入序列 $[a_1,...,a_n]$ ，其 softmax 计算如下

$$S_j = \frac{e^{a_j}}{\sum_{k=1}^N e^{a_k}} \quad \forall j \in 1..N$$

代码实现时，一个简单的想法是直接按照公式计算

```python
def softmax(x):
    exps = np.exp(x) # may overflow
    return exps / np.sum(exps)
```

这有个问题，如果 $x_i$ 非常大，计算其次幂时可能导致数值溢出，所以需要先归一化

```python
def stable_softmax(x):
    d = x - np.max(x)
    exps = np.exp(d)
    return exps / np.sum(exps)
```

对于形状为 $[N,C]$ 的输入，其输出形状也是 $[N,C]$ ，每一个 batch 内部独立进行 softmax ，过程如下
- 计算每一行的最大值（可以用 reduce 操作或者用一个线程直接计算）
- 一行中每个值都减去这一行的最大值（这是个 map 操作）
- 计算每个值的次幂（map）
- 计算每一行的元素和（可以用 reduce 或一个线程计算）
- 每个元素除以这一行的元素和（map）

在反向传播时，可以发现如下的规律，这表明我们不需要先计算每个元素的次幂再代入公式，可以直接利用前向传播中计算好的一系列 $S_j$ 去计算梯度，再乘到上游梯度上即可

![[AI编程/imgs/img7/image-9.png]]

这还能进一步简化，由于 softmax 层一般是在最后面，用于计算交叉熵损失

$$H(y, p) = -\sum_i y_i \log(p_i)$$

```python
def cross_entropy(X, y):
    p = softmax(X)
    m = y.shape[0]
    log_likelihood = -np.log(p[range(m), y])  # map
    loss = np.sum(log_likelihood) / m # reduce mean
    return loss
```

可见我们其实是知道损失函数 $L$ 关于 $p_i$ 的具体表达式的

$$L = -\sum_k y_k \log(p_k)$$

我们又知道 $p_k$ 关于 $a_i$ 的表达式，所以可以不用链式法则，而是直接计算 $\frac{\partial L}{\partial a_i}$

$$
\begin{align}
\frac{\partial L}{\partial a_i} &= -\sum_k y_k \frac{\partial \log(p_k)}{\partial a_i}\\
&= -\sum_k y_k \frac{\partial \log(p_k)}{\partial p_k} \times \frac{\partial p_k}{\partial a_i}\\
&= -\sum_k y_k \frac{1}{p_k} \times \frac{\partial p_k}{\partial a_i}
\end{align}
$$

前面又计算过 $\frac{\partial p_k}{\partial a_i}$ 这一项

$$\frac{\partial p_i}{\partial a_j} =
\begin{cases}
p_i (1 - p_i) & \text{if } i = j \\
-p_j \cdot p_i & \text{if } i \neq j
\end{cases}$$

将其代入，并把 $k=i$ 和 $k\neq i$ 两种情况的求和分开写

$$
\begin{align}
\frac{\partial L}{\partial a_i} &= -y_i (1 - p_i) - \sum_{k \neq i} y_k \frac{1}{p_k} (-p_k \cdot p_i)\\
&= -y_i (1 - p_i) + \sum_{k \neq i} y_k \cdot p_i\\
&= -y_i + y_i \cdot p_i + \sum_{k \neq i} y_k \cdot p_i\\
&= p_i \left( y_i + \sum_{k \neq i} y_k \right) - y_i\\
&= p_i - y_i
\end{align}
$$

可得一个非常简洁的计算公式

$$\frac{\partial L}{\partial a_i} = p_i - y_i$$











