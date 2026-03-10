
# CUDA

为了加速计算，有两种思路：加快 CPU 的时钟频率，让 CPU 算的更快；将计算分成若干个小单元，并行计算

对于第一种方法， CPU 的频率存在物理学方面的上限，而且增大频率会增大功耗，更难去散热，所以引入并行计算是必要的

- CPU 擅长处理复杂的、串行的任务，更为灵活，注重降低延迟 Latency
- GPU 由很多小核心组成，擅长处理大量简单的、可以并行进行的计算，注重提高吞吐量 Throughput

CUDA 是让 CPU 和 GPU 俩协同工作的桥梁，二者具体关系如下

![[AI编程/imgs/img2/image.png]]

其中 CPU 负责传输数据，并给 GPU 下指令，所以称为 Host，具体如下
- 给数据分配一片内存，使用 `malloc` ，最后释放内存，使用 `free`
- 让 GPU 给数据分配一片内存，使用 `cudaMalloc`
- 实现数据在 CPU 和 GPU 之间的传输，使用 `cudaMemcpy`
- 让 GPU 开始计算，通过定义函数并加载实现

其中分配内存和拷贝数据是非常慢的，所以要尽量避免频繁的数据传输

GPU 则负责进行具体的计算，称为 Device ，CUDA 使用了一个分层结构
- **Thread** ：最基本的执行单元
- **Block** ：一组线程的集合，这些线程可以相互通信和同步
- **Grid**：所有线程块的集合

一个 CUDA 的计算任务称为 **核函数 Kernel**，这是计算时每个线程会干的事，在启动时需要指定上述的层次结构，也就是告诉 GPU 需要启动多少个 Block ，每个 Block 里有多少个 Thread 

一个 CUDA 程序的流程通常如下
1. 在 CPU 上分配和初始化数据
2. 在 GPU 上分配内存
3. 将数据从 CPU 复制到 GPU 
4. 让 GPU 开始执行核函数 Kernel
5. 将结果从 GPU 复制回 CPU 
6. 释放 CPU 和 GPU 上分配的内存

## 一个简单的程序

下面以实现 relu 函数为例，进行具体的代码实现，假设有 N 个输入数据，在 CPU 上，这很容易实现

```cpp
// 定义relu函数
float relu_cpu(float x) {
	return x > 0 ? x : 0;
}

// 串行执行
for (int i = 0; i < N; i++) {
	h_out[i] = relu_cpu(h_in[i]); // h for Host, CPU
}
```

而对于 CUDA ，需要将计算分配给一堆线程，每个线程只需要负责对一个数据进行 relu 计算即可

需要定义核函数 kernel ，使用 `__global__` 关键字

```cpp
__global__ void relu_gpu(float* in, float* out) { 
	int i = threadIdx.x;
	out[i] = in[i] > 0 ? in[i] : 0; 
}
```

>这里 `threadIdx.x` 是一个线程在其 block 内的索引，对于第 i 个线程，只需计算第 i 个输入数据的 relu 即可

按照上述流程，代码如下

```cpp
const int N = 64; // 有64个输入数据
const int size = N * sizeof(float); // 数据总共占用的内存大小

// 1.在CPU分配内存
float* h_in = (float*) malloc(size);
float* h_out = (float*) malloc(size);

// 2.在GPU分配内存
float* d_in = nullptr; // d for Device, GPU
float* d_out= nullptr;
cudaMalloc(&d_in, size);
cudaMalloc(&d_out, size);

// 3.将输入数据复制到GPU
cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

// 4.启动核函数
relu_gpu<<<1, N>>>(d_in, d_out);

// 5.将结果复制回CPU
cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);

// 6.清理内存
cudaFree(d_in); cudaFree(d_out);
free(h_in); free(h_out)
```

>其中 `<<<1, N>>>` 表示一共有 1 个 block ，一个 block 有 N 个线程

## Block 数与线程任务分配

上面的程序只是一个最基本的版本，实际上会有一些调整

一个 block 的最大线程数量取决于具体芯片，一般来讲是 256/512/1024 ，所以一般不将所有线程全放到一个 block ，可以用以下公式计算所需的 block 数

```cpp
int threadsPerBlock = 256; // 每个Block有256个Thread 
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
```

这里不直接 `n/threadsPerBlock` 的原因是，这是一个向下取整，可能导致总线程数不够，需要 `(n/threadsPerBlock) + 1` ，又因为怕整除时多了一个用不上的 block ，所以加一个 `-1` ，这就变成了向上取整

在设置 block 和 grid ，不只可以是一维（只用一个数），还可以用用 CUDA 的 `dim3` 表示 2D 和 3D 情况，比如

```cpp
relu_gpu<<dim3(4, 6, 8), dim3(16, 16)>>
```

这种情况下，计算一个线程的下标 i 的公式就变成了

```cpp
int i = blockIdx.x * blockDim.x+threadIdx.x;
```

> `blockIdx` 代表一个 block 在其 grid 中的索引，`blockDim` 代表一个 block 的维度形状，类似的还有 `gridDim`

之前的设计是让一个线程处理一个数据元素，但如果数据量 N 大于线程总数，就会有一部分数据没有被处理，所以需要让一个线程能处理多个数据

```cpp
#define CUDA_KERNEL_LOOP(i, n)                    \ 
	for(int i =blockIdx.x*blockDim.x+threadIdx.x; \ 
	i<(n);                                        \ 
	i+= blockDim.x * gridDim.x)
```

>其中 `\` 表示两行是相连的
>这里假设了 grid 和 block 都是一维的，所以 `blockDim.x * gridDim.x` 能直接表示线程总数，多维情况需要稍微改一下

而后将 `CUDA_KERNEL_LOOP` 这个宏添加在核函数里

```cpp
__global__ void relu_gpu(float* in, float* out, int n) {
    CUDA_KERNEL_LOOP(i, n) {
        out[i] = in[i] > 0 ? in[i] : 0;
    }
}

relu_gpu <<<blocksPerGrid, threadsPerBlock>>> (d_in, d_out, N);
```

## 显存结构与管理

上面的代码中需要手动分配与释放内存，这太烦人了，而使用 `Tensor` 类封装数据就可以自动分配释放内存

```cpp
// 1.创建Tensor对象，自动分配显存
Tensor<float> tensor_in(shape, Device::GPU);
Tensor<float> tensor_out(shape, Device::GPU);

// 2.上传数据
tensor_in.copy_from(h_in);

// 3.开始计算
relu_gpu<<<1, N>>>(tensor_in.data(), tensor_out.data());

// 4.回传结果
tensor_out.copy_to(h_out);

// 注意：无需手动释放显存，Tensor析构函数会自动释放资源
```

尽管我们说 Tensor 有什么什么样的维度，在内存中实际上是一维连续存储的，同时其具有以下属性

![[AI编程/imgs/img2/image-1.png]]

- `size` ：Tensor 维度大小
- `stride` ：沿各个维度移动到下一个数据，下标变化了多少，某维度的 stride 是从该维度之后的维度上元素个数的乘积
- `dtype` ：存储的数据类型
- `device` ：存在哪个设备上，其中 `cuda` 指明设备类型是 GPU ， `0` 是设备编号，表示存储在第 0 号 GPU 上

>对于 stride ，可以这么想象，假设我们有一个 3 通道的 H×W 的 RGB图片，其大小就是 $(3, H, W)$ ，沿着通道维度 C 移动，会跨过一整张图片；沿着纵向的高度维度 H 移动，会跨过一行；沿着横向的宽度维度 W 移动，就到了下一个像素

假设有下面一个二维 Tensor ，要访问其数据，只需根据 shape 计算 stride ，再根据 stride 计算其展平后对应的一维下标即可，这称为 **内存偏移量 offset**

![[AI编程/imgs/img2/image-2.png]]

这很简单，只用把访问的逻辑下标和 stride 的对应元素相乘再相加即可；对于切片情况，只需根据有数的位置获得 size ，根据 `:` 的位置获得 stride ，把 `:` 视为 0 计算开始的位置 offset 即可

`reshape` 就更容易了，反正实际存储结构没有变，就改一下 size 的数就行，然后重新计算 stride

整体来看，内存结构如下

![[AI编程/imgs/img2/image-3.png]]

每个线程有一个自己的内存 local memory ；一个线程块的内存是 shared memory ，其内部的线程的内存是共享的，可以相互访问；所有线程还共享一个全局的内存 global memory 

访问速度：local memory > shared memory >> global memory >> CPU memory

对于下面这个程序，定义时啥也不加，就是在 local memory ；使用关键字 `__shared__` 可以定义 shared memory 中的变量；由于核函数是 global 的，函数参数 `x` 和 `y` 就是在 global memory 中的

```cpp
__global__ foo(float* x, float* y) { 
int i = threadIdx.x;      // local memory 
float t;               // local memory 
__shared__ float s[128];  // shared memory 
__shared__ float a, b, c; // shared memory

// 由快到慢
s[i] = t; // 读取局部写入共享
b = a;    // 读取共享写入共享
t = *x;   // 读取全局写入局部
*y = *x;  // 读取全局写入全局
}
```

全局内存访问太慢了，但可以用 **合并内存访问 Coalesced Memory Access** 进行优化，核心思想是当一组线程（通常是 32 个，即一个 Warp）在同一个指令周期内访问全局内存时，如果它们访问的内存地址是连续的且对齐的，GPU 可以将这些多个内存访问请求合并为一次或少数几次大的内存事务 Transaction

![[AI编程/imgs/img2/image-4.png]]

访问速度：连续的内存访问 > 跨步的内存访问 > 随机的内存访问

>这很好理解，想象你要从图书馆借 32 本书
>1. 合并访问：你要借的书是连续放在一个书架上的（地址连续），图书管理员可以一次帮你把这一摞书都拿下来，很高效
>2. 非合并访问：你要借的书分散在整个图书馆的各个角落（地址分散），图书管理员不得不跑遍整个图书馆，一本一本地找，就很慢

对于下面这个程序，连续的访问就属于合并访问，而有跨步的就属于非合并访问

```cpp
__global__ foo(float* x) { 
int i = threadIdx.x; 
float s, t; 

t = x[i];   // 合并
x[i+1] = s; // 合并，只是有一个偏移
x[i*2] = t; // 非合并，stride为2
}
```

总结下来，需要尽量避免 CPU 和 GPU 间的频繁的数据传输；使用共享内存以减少对全局内存的访问；如果要访问全局内存，尽量使用合并访问


