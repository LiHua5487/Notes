
# GPU Hardware

## Streaming Multiprocessor

GPU 上有一个重要结构，**Streaming Multiprocessor (SM) 多流处理器** ，其组成如下
- 一堆 GPU core ：每个 core 负责运行一个线程
- 寄存器 Register ：用于存放每个 core 的 local memory 
- 共享内存：用于存放每个 block 的 shared memory 

>线程编号在启动核函数时就已经被自动分配好了，当 SM 开始执行线程块时，会领取这些已经定义好索引的线程，并将其映射到 SM 内的 core 上

一个 SM 上可以执行多个 block （具体数量取决于线程与内存的资源分配），但一个 block 必须在同一个 SM 内执行（若 SM 剩余空间不足，则进行等待）

![[AI编程/imgs/img3/image-2.png]]

程序员负责定义 block ，而 GPU 自动将 block 分配到 SM 上，CUDA 对于一个线程具体是什么时候开始运行、在哪运行没有过多的要求，这虽然很灵活，但有时候会导致一些问题，比如执行以下核函数

```cpp
__global__ void hello() {
    printf("Hello world! I'm a thread"
           " in block %d\n", blockIdx.x);
}

int main(int argc, char **argv) {
    hello<<<16, 1>>>(); // 16个block，每个block一个线程
    cudaDeviceSynchronize();
    printf("That's all!\n");
    return 0;
}
```

其中 `cudaDeviceSynchronize()` 是让 CPU 等到 GPU 的核函数执行完了，再继续执行后续部分，不然由于 CPU 和 GPU 是异步执行，二者默认不会相互等待，如果不进行同步就会在一开始输出 `That's all!` 

由于 CUDA对于并行计算没有过多限制，所以不一定哪个 block 先开始执行，输出顺序不一定是 0,1,...,15 ，而是会有 16! 种输出顺序

CUDA 可以保证：一个 block 中的线程都在同一个 SM ，且一个核函数的所有线程都执行完了，才会执行下一个核函数

## Warp

GPU 的调度并不是以一个线程为单位，而是以一个 **线程束 Warp** 为单位的（处在一个 SM ，且线程编号连续），在 NVIDIA GPU 上，一个 warp 包含 32 个线程

>可以这样比喻：一个 SM 就像一个车间，每个 core 就是里面的一个工人，分配一个 block 就相当于对工人进行分组，但分组和执行任务都要以 32 个人（Warp）为单位进行，如果任务（线程）不足 32 个，剩下的人就干等着

当一个 Warp 的实际执行的线程束不足 32 时，其它 core 会被禁用，这就造成了资源的浪费

线程的随机性与 block 类似，但并不完全一样

```cpp
__global__ void hello() {
    printf("Hello world! I'm thread "
           "%d, f=%f\n", threadIdx.x);
}

int main(int argc, char **argv) {
    hello<<<1, 5>>>();
    cudaDeviceSynchronize();
    printf("That's all!\n");
    return 0;
}
```

比如这个程序输出顺序的结果数并不是 5! ，因为存在 Warp ，这 5 个线程会被分配到一个 Warp 里，而对于 `printf` `malloc` `cudaMalloc` 这种操作，如果多个线程同时调用可能产生冲突
CUDA 实现了一个内部的序列化机制，当 Warp 中的多个线程同时遇到 `printf` 时，它们并不会真的同时往你的屏幕上打印字符，CUDA 运行时会将这些并发的打印请求捕获下来，放入一个内部队列，再按照一定的顺序（通常是线程编号顺序）进行执行

---

由于是以 warp 为单位，所以同一个 warp 收到的指令是一样的，这就是 **SIMT (Single Instruction, Multiple Threads) 单指令多线程** 的架构

但这也会导致一些问题，比如说要执行一个 if else 语句

```cpp
__global__ void simpleKernel() {
    if (threadIdx.x % 2 == 0) {
        printf("0\n");
    } else {
        printf("1\n");
    }
}

int main(int argc, char **argv) {
    hello<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
```

一开始，一个 Warp 内的所有线程都会进行条件判断，一部分进入 `printf("0\n")` 分支，一部分进入 `printf("1\n")` 分支，这就产生了 **分支分歧 Branch Divergence** ，GPU硬件无法让一个 Warp 同时走两条路，它的解决方法是先让进入分支 1 的线程同时执行，其它线程干等着，然后再让进入分支 2 的线程同时执行，其它线程干等着，等到所有分支都执行完了，再继续往下执行后续部分

![[AI编程/imgs/img3/image.png]]

所以上面的程序会先输出一堆 1 ，再输出一堆 0 （具体先 1 还是先 0 取决于编译器和硬件的优化策略，但常见的行为是先执行条件为 false 的线程，再执行条件为 true 的线程）

这对于循环来说也是类似的

![[AI编程/imgs/img3/image-1.png]]

## Barrier

线程的并行会导致一些问题，比如多个线程同时对同一块内存进行访问（且其中存在写入操作），由于 CUDA 无法保证顺序，这就可能乱了（因为在写入操作前后进行读取，结果是不一样的），这就是 **数据竞争 Data Race** 问题

为了避免这种乱序现象，需要手动设定一个同步点，所有线程先进行读取，都读完了再进行写入，这个手动设置的同步点就是 **Barrier** ，可以使用 `__syncthreads()` 实现

比如我们想实现的效果是对于数组中的每个数，都加上它后面一个数

```cmd
[1, 2, 3,       [3, 5, 7,
 4, 5, 6,  -->   9, 11, 13,
 7, 8, 9]        15, 17, 9]
```

对于下面这个核函数

```cpp
__global__ void shift_sum(float* array) {
    int idx = threadIdx.x;
    if (idx < N-1) {
        float tmp = array[idx] + array[idx+1];
        __syncthreads();
        array[idx] = tmp;
    }
}
```

如果不加 `__syncthreads()` ，可能会出现某个线程先算完并写入，然后其运算结果被另一个线程读取并当作”后一个数“（但实际上应该是运算前的数），这就导致结果有问题（这种现象并不是必然出现）

上面的核函数在计算时是直接访问全局内存的，这部分可以用共享内存优化，先把数据读到共享内存（每个线程负责读一个数），再从共享内存获取数据计算

```cpp
__global__ void shift_sum(float* array) {
    __shared__ float shared[N];
	
    int idx = threadIdx.x;
    shared[idx] = array[idx];
    __syncthreads(); // 这里需要同步一下，不然容易出问题
	
    if (idx < N-1) {
        array[idx] = shared[idx] + shared[idx+1];
    }
    shared[idx] = 3.14;
}
```

值得注意的是，由于共享内存相当于一个局部变量，程序运行完，这片内存就会归还给系统，所以最后一步 `shared[idx] = 3.14` 写入的东西并不会被保存

## 原子操作

**原子操作 Atomic Operation** ：一个不可分割的操作，它保证在执行过程中不会被任何其他线程或进程中断，要么完全执行完毕，要么根本不执行，不会出现执行到一半被切换走的情况，常见操作有
- `ADD R1, R2, R3` 将寄存器 R2 和 R3 相加，结果存入 R1
- `LDG R1, [R2]` 从 R2 寄存器保存的地址读取数据到 R1
- `BRA Target` 跳转到目标地址

比如我们想实现的效果是，对于一块内存上的变量，会有多个线程对其进行 +1 操作，希望其结果是每有一个线程执行增加，其结果便 +1 

但如果像下面这么写，会导致数据读取和写入顺序乱了，即便加上线程同步，也会导致一开始读的都是 0 ，加完写入都会变成 1 

```cpp
__global__ void increment_atomic(int *g) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    i = i % kArraySize;
    g[i] = g[i] + 1; // g[i] 初始化为0
}
```

使用 `atomicAdd(&g[i], 1)` 替换，这会让同一块内存上的增加操作变成串行，即对于一块内存，同时只有一个线程对其进行读写等操作，不同内存上的操作仍然是并行

但这仍然保证不了哪个线程先读与后读，只能保证同时只有一个线程进行操作，导致结果不一定可复现，而且由于引入串行化，会减慢速度

## GPU 测速

怎么测量一个核函数的执行时间呢？ `ctime` 库提供了对一段程序计时的方法

```cpp
#include <ctime>
std::clock_t start = std::clock(); 
////////////////////////////
// put your C++ code here //
////////////////////////////
std::clock_t end = std::clock();
double time_elapsed = (end - start) / CLOCKS_PER_SEC;
```

但对于核函数不能这么用，因为 CPU 和 GPU 是异步的，CPU 在执行完启动核函数的指令后，就会立马往下执行，导致计时器的时间实际上是启动核函数用的时间（不过可以加上 `cudaDeviceSynchronize()` ，但更多采用下面的方式）

```cpp
cudaEvent_t start = cudaEventCreate(&start);
cudaEvent_t stop = cudaEventCreate(&stop);

cudaEventRecord(start, 0);
///////////////////////////////
// put your CUDA kernel here //
///////////////////////////////
cudaEventRecord(stop, 0);
cudaEventSynchronize(stop); // 等待执行结束

float elapsed;
cudaEventElapsedTime(&elapsed, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);
```

在 pytorch 中也是类似的

```python
import torch
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
# your code here
end.record()
torch.cuda.synchronize() # 等待执行结束
print(start.elapsed_time(end))
```

# Parallel Communication Patterns

线程对于内存访问的映射方式有以下几种

![[AI编程/imgs/img3/image-3.png]]

Scatter 就是 Gather 的逆过程，而 Stencil 是一种特殊的 Gather

在 Transpose 中，不只可以实现矩阵转置，还可以进行数据转置，把一片数据类型混合存储的内存分离开

![[AI编程/imgs/img3/image-4.png]]

但是二者不一定谁更好，AoS (Array of Structures) 优点是对单个类实例的所有属性操作更直观高效，而 SoA (Structure of Arrays) 优点是利于 SIMD

>**SIMD (Single Instruction Multiple Data)** ：一种并行计算技术，让一条 CPU 指令同时操作多个数据元素，而不需要循环

