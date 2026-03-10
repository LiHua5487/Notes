
# Matrix Product

## gemm

矩阵乘法定义如下

$$C_{m\times n}=A_{m\times s}\cdot B_{s\times n}, \quad \text{where } c_{ij}=\sum_{k=1}^s a_{ik}b_{kj}$$

更一般的，加上偏置项和系数，就称为 **general matrix multiplication (gemm)**

$$C=\alpha A\cdot B+\beta C$$

在 CPU 上，只需依次遍历计算 $C$ 的每个位置即可，在 GPU 上，一个简单的方式是每个线程负责计算一个位置

如果矩阵的大小不是 32 的整数倍，就会有一部分地方占不满一个 block，那多余的线程就被浪费了，所以一般在设置神经网络中间层的特征维度时取 32 的整数倍

![[AI编程/imgs/img6/image.png]]

对于上面的线程安排，有两种顺序，一种是按列分配给线程，一种是按行分配给线程

![[AI编程/imgs/img6/image-1.png]]

对于按列的情况，考虑相邻的几个线程，对应一列中相邻的几个位置，在写入时显然是不连续的；在遍历计算时，会遍历 A 的连续几行，这看似连续，但实际上在每个遍历步骤，这些线程访问的元素是 A 的同一列的几个元素，这仍然是不连续的

```cpp
__global__ void sgemm_naive( // s表示单精度浮点运算
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    const int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}
```

如果换成按行分配的方式，写入时是连续的，在访问 B 时每个遍历步骤也是连续的

```cpp
__global__ void sgemm_coalesce(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < M && col < N) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += A[col * K + k] * B[k * N + row];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

dim3 gridDim(Ceil(N, 32), Ceil(M, 32)), blockDim(32, 32);
sgemm_naive<<<gridDim, blockDim>>>(M, N, K, alpha, A, B, beta, C);
```

## Roofline Model

在处理器上运算时经常会受到以下两种因素的限制
- 带宽 bandwidth：单位时间内存与处理器能够传输的数据量，代表读写数据的速度，单位 GB/s
- 计算能力：单位时间能进行多少次浮点运算，代表计算速度，单位 FLOP/s

算术强度 Arithmetic Intensity 的定义如下，其衡量每字节数据所执行的计算量

$$\text{Arithmetic Intensity} = \frac{\text{计算量/FLOP}}{\text{内存访问量/Byte}}$$

- 算术强度低，表示每个字节的内存访问对应很少的计算操作，计算单元闲着没事干，但内存访问很频繁，此时容易受到带宽的限制
- 算术强度低，表示每个字节的内存访问对应大量的计算操作，计算单元很忙，此时容易受到计算能力的限制

![[AI编程/imgs/img6/image-2.png]]

图像纵轴 Attainable 是实际可达的计算性能，表示设备在某一算术强度下，能够实现的浮点运算速度，一开始，数据传输量较少，计算单元处理得过来，此时计算速度就取决于传了多少数据；数据传输量过大时，计算单元处理不过来了，计算速度到顶了，这就取决于计算单元的计算能力

GPU 的程序通常处在带宽受限的区间，此时优化的侧重点就应该放在提高内存效率上

## Matrix Product with Shared Memory

根据 roofline model ，考虑使用共享内存加速 gemm 运算，但是共享内存不一定存得下一个位置的计算用到的所有数据，可以考虑分块处理

![[AI编程/imgs/img6/image-3.png]]

>**Tile Quantization** ：将一个很大的数据分成很多小块 tile ，每一块分别处理

对于矩阵乘法，可以让每个 block 负责计算输出矩阵 $C$ 上的一个区块，在计算时，沿着 A/B 的行/列移动窗口，将 $\sum_{k=1}^s a_{ik}b_{kj}$ 分块计算并累加

```cpp
__global__ void sgemm_shared_memory(
    int M, int N, int K, float alpha, const float *A,
    const float *B, float beta, float *C) {
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // the inner row & col that we're accessing in this thread
    const int cRow = blockIdx.x, cCol = blockIdx.y;
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // advance pointers to the starting positions
    A += cRow * BLOCKSIZE * K;  // row=cRow, col=0
    B += cCol * BLOCKSIZE;      // row=0, col=cCol
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;  // row=cRow, col=cCol

    float tmp = 0.0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];
        __syncthreads();

        // execute the dot product on the currently cached block
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] *
                   Bs[dotIdx * BLOCKSIZE + threadCol];
        }
        __syncthreads();

        A += BLOCKSIZE; B += BLOCKSIZE * N;
    }

    C[threadRow * N + threadCol] =
        alpha * tmp + beta * C[threadRow * N + threadCol];
}
```

## Sparse Matrix Product

对于稀疏矩阵，直接存储太费空间了，可以只存储非零元素的信息，一种方式是存储非零元素及其对应的行列坐标，这被称为 **coordinate list (COO)** ；另一种方式对行索引进一步压缩，将行索引换为每行开头对应第几个元素，这被称为 **compressed sparse row (CSR)** 

>通常来看，稀疏矩阵的含 0 量在 90% 以上，如果含 0 量没那么高却仍然用稀疏矩阵的方式存储，占用空间不一定比直接存储小

比如下面的例子中， `ROW_INDEX` 第一个元素是 0 ，就表示第一行的首个非零元素是 V 中下标 0 的元素，即 10 ，其处在的列可以从 `COL_INDEX` 直接获得；`ROW_INDEX` 第二个元素为 2 ，表示第二行的首个非零元素是 V 中下标 2 的元素，即 30 ，处在下标 1 的列

![[AI编程/imgs/img6/image-4.png]]

值得注意的是，CSR 压缩的行索引一共有矩阵行数 +1 个元素，最后一个元素是矩阵非零元素总数；如果矩阵第一行全是 0 ，第二行才有非零元素，则压缩的行索引前两个元素都是 0 

在 pytorch 中，提供了二者的使用方式

```powershell
>>> i = [[0, 1, 1],
         [2, 0, 2]]
>>> v = [3, 4, 5]
>>> s = torch.sparse_coo_tensor(i, v, (2, 3))
>>> s
tensor(indices=tensor([[0, 1, 1],
                       [2, 0, 2]]),
       values=tensor([3, 4, 5]),
       size=(2, 3), nnz=3, layout=torch.sparse_coo)

>>> s.to_dense()
tensor([[0, 0, 3],
        [4, 0, 5]])
```

```powershell
>>> crow_indices = torch.tensor([0, 2, 4])
>>> col_indices = torch.tensor([0, 1, 0, 1])
>>> values = torch.tensor([1, 2, 3, 4])
>>> csr = torch.sparse_csr_tensor(crow_indices, col_indices, values, dtype=torch.float64)
>>> csr
tensor(crow_indices=tensor([0, 2, 4]),
       col_indices=tensor([0, 1, 0, 1]),
       values=tensor([1., 2., 3., 4.]),
       size=(2, 2), nnz=4, dtype=torch.float64)

>>> csr.to_dense()
tensor([[1., 2.],
        [3., 4.]], dtype=torch.float64)
```

## Sparse Matrix-Vector Product

有了以上的简化表示，稀疏矩阵和向量的乘法可以用以下方法计算，比如对于下面这个矩阵，采用 CSR 存储方式

$$\begin{bmatrix}
1 & 0 & 3 \\
2 & 1 & 0 \\
0 & 4 & 3
\end{bmatrix}$$

- `value: [1, 3, 2, 1, 4, 3]`
- `col:   [0, 2, 0, 1, 1, 2]`
- `cRow:  [0, 2, 4, 6]`

其与列向量 $[x,y,z]^T$ 的乘积如下

$$\begin{bmatrix}
1 & 0 & 3 \\
2 & 1 & 0 \\
0 & 4 & 3
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
z
\end{bmatrix}
\to
\begin{bmatrix}
x + 3z \\
2x + y \\
4y + 3z
\end{bmatrix}$$

可以总结出以下计算方法，首先根据 `value` 和 `col` 可以得到结果中出现的所有项，以下标 1 元素为例， `value` 的下标 1 元素是 3 ，说明其系数为 3 ；`col` 的下标 1 元素是 2 ，取乘法中的列向量 `X=[x,y,z]` 的下标 2 元素 $z$ ，二者组合起来就是 $3z$ 

$$[x, 3z, 2x, y, 4y, 3z]$$

>这种运算称为 **map** ，即根据 `value` `col` `X` 计算 scale product

而后根据 `cRow` 进行分组求和 / segmented scan 操作，即可得最终结果

---

但是一些情况下，稀疏矩阵中每行非零元素个数差距可能非常大，如果全都直接进行 segmented scan ，对于非零元素较多的一行，开销会比较大，此时可以根据各行平均非零元素数（可通过 `cRow` 相邻元素差轻易得到每行元素数），把每行劈成两半计算，前一半 reduce ，后一半 segmented scan

![[AI编程/imgs/img6/image-5.png]]

# CUDA Libraries

- CUDA programming language
	- `Thrust` ：基于 CUDA 的 C++ 并行编程库，支持 GPU 加速的 STL 容器和算法
- CUDA deep learning libs
	- `cuDNN`：深度学习框架通用的 GPU 加速库，用于优化神经网络运算
	- `TensorRT` ：高性能深度学习推理优化器和 runtime 库
- CUDA linear algebra and math libs
	- `cuBLAS` ：GPU 加速版 BLAS，用于高效的矩阵运算
	- `cuSPARSE` ：处理系数矩阵操作
	- `cuRAND` ：GPU 加速的随机数生成

## thrust

`thrust` 库提供了 CPU 和 GPU 上的 STL 容器和算法，并提供了很多数据并行处理的 primitives ，如 scan sort reduce ，它们可以被组合用于完成更复杂的运算

---

thrust 提供了 vector ，支持在 CPU 上的 `host_vector` 和在 GPU 上的 `device_vector`

```cpp
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <iostream>

int main(void) {
    // H has storage for 4 integers
    thrust::host_vector<int> H(4);

    // initialize individual elements
    H[0] = 14;    H[1] = 20;    H[2] = 38;

    // H.size() returns the size
    std::cout << H.size() << std::endl;

    // resize H
    H.resize(2);

    // Copy host_vector H to device_vector
    thrust::device_vector<int> D = H;

    // elements of D can be modified
    D[0] = 99;    D[1] = 88;

    // print contents of D
    for (int i = 0; i < D.size(); i++)
        std::cout << D[i] << std::endl;

    // H and D are automatically deleted
}
```

---

thrust 还提供了 transform 操作，即对于 X 的范围内（左闭右开）的元素，分别应用某种操作，并给定一个起始位置 Y 储存结果

```cpp
#include <thrust/transform.h>

thrust::device_vector<int> X(10);
thrust::device_vector<int> Y(10);

// compute Y = -X
thrust::transform(X.begin(), X.end(), Y.begin(), thrust::negate<int>());
```

>`X.end()` 指向最后一个元素的下一个位置

如果 X 和 Y 是 GPU 上的向量，且数组长度为 N ，可以这么写

```cpp
thrust::transform(X, X + N, Y, thrust::negate<int>());
```

利用 transform 还可以实现之前的 map 操作，thrust 为 transform 提供了两个版本的签名，分别对应一个和两个输入序列

```cpp
// 1 input
template <typename InputIterator, typename OutputIterator, typename UnaryFunction>
OutputIterator thrust::transform(InputIterator first, 
                                  InputIterator last, 
                                  OutputIterator result, 
                                  UnaryFunction op);

// 2 input
template <typename InputIterator1, typename InputIterator2, typename OutputIterator, typename BinaryFunction>
OutputIterator thrust::transform(InputIterator first1, 
                                  InputIterator last1, 
                                  InputIterator first2, 
                                  OutputIterator result, 
                                  BinaryFunction op);
```

我们可以自定义一个二元操作 `Saxpy` 来实现每个位置进行的运算，其中同时添加 `__host__` 和 `__device__` 表示既能在 CPU 运行，也能在 GPU 运行

```cpp
class Saxpy {
public:
    const float a;
    saxpy_functor(float _a) : a(_a) {}
    __host__ __device__ float operator()(const float& x, const float& y) const {
        return a * x + y;
    }
};

// Y <= A * X + Y
thrust::transform(X.begin(), X.end(), Y.begin(), Y.begin(), Saxpy(A));
```

---

thrust 还提供了 reduce scan sort 操作

```cpp
#include <thrust/reduce.h>
int sum = thrust::reduce(D.begin(), D.end(), (int) 0, thrust::plus<int>()); // (int) 0 是初始值，强制类型为int
```

```cpp
#include <thrust/scan.h>
int data[6] = {1, 0, 2, 2, 1, 3};
thrust::inclusive_scan(data, data + 6, data);
thrust::exclusive_scan(data, data + 6, data);
```

```cpp
#include <thrust/sort.h>
int A[N] = {1, 4, 2, 8, 5, 7};
thrust::sort(A, A + N);
```

## BLAS

BLAS 全称是 basic linear algebra subprograms ，代表一些基本的线性运算，比如向量加法、数乘、点积、线性组合、矩阵乘法，它们被按照时间复杂度分为 3 个级别，分别为 $O(n^k),\quad k=1,2,3$
- Level 1：向量操作，作用于分段数组 strided arrays ，包括点积、范数、广义向量加法（被称为 "**axpy**"，即 "a x plus y"， $y \gets \alpha x + y$ ）等
- Level 2：矩阵-向量操作，包括广义矩阵-向量乘法（General Matrix-Vector Multiplication,  **gemv**，$y \gets \alpha A x + \beta y$ ）
- Level 3：矩阵-矩阵操作，包括广义矩阵乘法（General Matrix Multiplication, **gemm**，$C \gets \alpha A B + \beta C$ ），其中矩阵 A 和 B 可以选择进行转置

cuBLAS 中提供了一个句柄 `cublasHandle_t` ，封装了 GPU设备上下文、CUDA流 stream 、库内部状态信息、工作空间内存和其他CUDA资源；类似地，cuRAND 中提供了随机数生成器 `curandGenerator_t`

为了避免频繁的声明与释放，最好把它们设为全局变量

>句柄：对某个复杂系统或资源的引用和控制接口，好比一个遥控器，不需要知道电视内部如何工作，只需用遥控器就能控制电视的所有功能

```cpp
// STEP 1: Create cuBLAS Handle
cublasHandle_t handle;
cublasCreate(&handle);

// STEP 2: Call SGEMM
cublasSgemm(handle, ..<options>..);

// STEP 3: Destroy Handle
cublasDestroy(handle);
```

```cpp
// STEP 1: Create cuRAND generator
curandGenerator_t prng;
curandCreateGenerator(&prng);

// STEP 2: Random Seed
curandSetPseudoRandomGeneratorSeed(
    prng, (unsigned long long)clock());

// STEP 3: Call SGEMM
curandGenerateUniform(prng, );

// STEP 4: Destroy generator
curandDestroyGenerator(prng);
```
---

这里讲一下 `cublas` 中的 `sgemm` ，因为 lab 里会用到，其参数如下

```cpp
cublasSgemm(
    cublasHandle_t handle, // cublas句柄
    cublasOperation_t transa, // A是否转置
    cublasOperation_t transb, // B是否转置
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, // A及其leading dim
    const float *B, int ldb, // B及其leading dim
    const float *beta,
    float *C, int ldc // C及其leading dim
);
```

它实现了这样一个运算

$$C\leftarrow \alpha A_{m\times k}\cdot B_{k\times n}+\beta C_{m\times n}$$

- $m,n,k$ 对应 `sgemm` 中的那三个参数
- $lda=m,\quad ldb=n,\quad ldc=m$ ，这里的 $m,n$ 均为转置之前的

值得注意的是，这里的矩阵都是列主序的，即对于一个存储序列 1,2,3...6 ，如果矩阵形状为 2×3，则 cublas 会将其视为

$$
\begin{bmatrix}
1 & 3 & 5\\
2 & 4 & 6
\end{bmatrix}
$$

同样的，对于计算结果的矩阵，其会按列主序将这个矩阵转换为存储序列返回

如果要计算行主序的矩阵乘法 $C=A\cdot B$ ，可以转换为列主序的矩阵乘法 $C^T=B^T\cdot A^T$ ， 这样 $C^T$ 按列主序转换为序列，返回主机按行主序理解时，就是正确的结果 $C$ 了

---

利用上述库，可以实现 FC 层

![[AI编程/imgs/img6/image-6.png]]

在 pytorch 中提供了线性层 `torch.nn.functional.linear(input, weight, bias=None)` ，实现了运算 $y=xA^T+b$
- `input` ；形状为 `(N, *, in_features)` ， `*` 表示任意数量的额外维度
- `weight` ：形状为 `(out_features, in_features)`
- `bias` ：形状为 `(out_features, )`，默认为 None
- 输出张量：形状为 `(N, *, out_features)`

一般来讲，图像的维度存储顺序规定为 $[N, H, W, C]$ 或 $[N, C, H, W]$ ，其中 $N$ 表示 batch 数量，$C$ 表示通道数，$H$ 和 $W$ 表示高和宽

在进入线性层前，先使用 `torch.view` / `torch.reshape` / `torch.flatten` 把图像展平为 $[N, -1]$ ，这可以通过修改 Tensor 类中的 size stride 等轻易实现

















