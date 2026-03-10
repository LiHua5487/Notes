
# Reduction

对于数组求和操作，在 CPU 上用遍历实现

```cpp
float sum = 0;
for (int i = 0; i < N; i++) {
	sum += h_in[i];
}
```

其运算量 **work complexity** 是 $O(N)$ ，运算的时间复杂度（这里叫 **step complexity**）也是 $O(N)$ ，但这对于 GPU 来说太低效了，既然运算量改不了（都得加那么多数），需要减小时间复杂度，尽量并行化

可以采用以下的方式进行，每一步都把数组中的元素两两相加，这样时间复杂度就变成了 $O(logN)$ ，这种方式被称为 **并行归约 reduce**

![[AI编程/imgs/img4/image.png]]

还可以进一步改进，让内存访问变成连续的

![[AI编程/imgs/img4/image-1.png]]

在实际实现时，可以使用共享内存进一步加速，但是如果输入的数组很长，一个 block 的共享内存放不下，就需要多个 block 

```cpp
__global__ void reduce_shared(float *d_out, const float *d_in, int N) {
    int tid = threadIdx.x; // 线程在block中的索引
    int idx = threadIdx.x + blockDim.x * blockIdx.x; //线程全局索引
    
    extern __shared__ float shared[];
    shared[tid] = idx < N ? d_in[idx] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    
    // 最后block中索引0处就是结果，写入全局内存
    if (tid == 0) { d_out[blockIdx.x] = shared[0]; }
}
```

- 参数中 `din` 为输入数组， `dout` 为输出数组，N 为输入数组的长度
- `extern` 作用是延迟定义数组大小，等到核函数启动时再确定，具体大小由 `kernel<<<blocks, threads, shared_memory_size>>>();` 中的第三个参数决定
- `s>>=1` 即 `s = s >> 1` ，通过位运算来实现 `s /= 2` 的效果

但是上面的运算中，每个 block 都会有一个结果，但 block 之间的共享内存是独立的，无法互相访问，还得把这些结果收集起来重复上面的计算，这就需要在输入和输出数组之外再引入一个数组，用于倒腾运算结果

```cpp
void reduce(float *d_out, float *d_in, int N) {
    float *d_tmp; // 额外引入的数组
    cudaMalloc((void **)&d_tmp, CudaGetBlocks(N) * sizeof(float));
	
    int num = N;
    float *ptr_in = d_in;
    float *ptr_out = d_tmp; // 暂时将输出结果储存在d_tmp
    
    // 计算共享内存大小，保险起见假设block中每个线程都会存一个数
    int kShared = kThreadsNum * sizeof(float);
    
    // 重复计算直到最后只剩1个block的结果
    while (num > 1) {
        int blocks = CudaGetBlocks(num);
        reduce_shared<<<blocks, kThreadsNum, kShared>>>
			    (ptr_out, ptr_in, num);
        num = blocks;
        std::swap(ptr_in, ptr_out); // 交换输入输出数组
    }
    
    cudaMemcpy(d_out, ptr_in, sizeof(float), 
			    cudaMemcpyDeviceToDevice);
    cudaFree(d_tmp);
}
```

- `CudaGetBlocks(n)` 为自定义的函数，计算输入数组长度为 n 时，需要多少 block

但这可能带来误差，因为当很大的数与很小的数相加时，由于存储精度，这个很小的数会被稀释掉，但是假设我们先把许多很小的数加起来变成一个不那么小的数，再和大数相加，结果就不一样了

上述 Reduce 的思想不仅适用于求和操作，只要是二元且满足结合律的操作都可以，比如 `*` `max` `min` `and` `or` 等

## Parallel Histogram

上面的 Reduce 包括后面的 Scan 等是非常 primitive 的思想，即是一种思考与解决问题的思路，常常应用在很多方面

在直方图的统计中，我们有一堆数，想把这些数按照取值所在的区间分为几类（每一类叫一个 bin），如果直接并行处理，每个线程判断一个数要放到哪个 bin 里，并使对应的数量 +1 ，这会导致数据竞争，因为会有很多线程对一个 bin 所含元素的数量进行读取并增加，即便改成原子操作，那一个 bin 的这么多的增加操作全会变成串行，大大降低效率

但是我们可以让每个 block 分别负责统计一部分数据（使用原子操作），每个 block 都会得到一个统计结果，再利用 Reduce 将这些结果相加即可

# Scan

考虑一种特殊的数组求和，比如对于数组 `[1,2,3,4]` ，需要求得其一系列前缀和，即前 $k$ 个元素的和
- inclusive scan ：从 $k=1$ 开始算，结果是 `[1,3,6,10]`
- exclusive scan ：从 $k=0$ 开始算，结果是 `[0,1,3,6]` ，这相当于把最后一个元素排除了

这种计算虽然在串行编程不多见，却是并行编程中很重要的一类算法

其在 CPU 的实现方式如下，只需要依次遍历并存储当前和即可， work complexity 和 step complexity 都是 $O(n)$

```cpp
float sum = 0;
for (int i = 0; i < N; i++) {
	sum += array[i];
	out[i] = sum;
}
```

对于并行计算，有以下两种实现方式

## Hillis/Steele Inclusive Scan

参考 Reduce 的实现思路，我们可以把这个求和树中缺少的部分补全

![[AI编程/imgs/img4/image-2.png]]

总结成计算图就是这样，每一步的求和步长是 $2^{\text{step}}$ ，直到步长超过了数组长度

![[AI编程/imgs/img4/image-3.png]]

这种方式的运行步数是对数级的，每一步进行的运算次数与 $n$ 成正比，则 step complexity 是 $O(\text{log}n)$ ，work complexity 是 $O(n\text{log}n)$

## Blelloch Scan

另一种方式进一步简化了计算量

![[AI编程/imgs/img4/image-4.png]]

这种方式的 step complexity 是 $O(\text{log}n)$ ，计算量是 $O(n)$

## Parallel Compact

在 pytorch 中，经常可以见到一种操作，我们有一个 mask ，可以使用它来取出一个 tensor 中在 mask 上激活的部分

```python
a = torch.rand(5, 2)
b = torch.tensor([1,0,0,1,0], dtype=bool)
c = a[b] # Compact
```

以一维情况为例，现有数组 `[3,1,8,4,6,5,2,7]` ，以及 mask `[1,0,1,0,1,0,1,0]` ，预期的 compact 结果是 `[3,8,6,2]`

其并行计算过程如下，先利用 exclusive scan 操作计算 mask 的前缀和，得到结果 `[0,1,1,3,3,3,3,4]` ，可以发现，mask 激活处的元素在这个前缀和数组中的对应位置的值，就是其在取出的元素构成的数组中的位置下标

![[AI编程/imgs/img4/image-5.png]]

## Segmented Scan

Segmented Scan ：对数据分段进行 scan 操作
- Input ： `[[1, 2], [6, 7, 1], [1, 2, 3, 4]]`
- Exclusive scan ： `[[0, 1], [0, 6, 13], [0, 1, 3, 6]]`
可以利用一个 Flag 表示这种分段关系，把每段的开头位置设为 1 ，其余为 0 
- Flag = `[1, 0, 1, 0, 0, 1, 0, 0, 0]`
- Data = `[1, 2, 6, 7, 1, 1, 2, 3, 4]`

这个运算的 step complexity 也是 $O(\text{log}n)$

# Transpose

要把一个二维数据转置，即把 $(i,j)$ 处的元素放到 $(j,i)$ 

![[AI编程/imgs/img4/image-6.png]]

在 CPU 上，只需要遍历执行 `out[i * N + j] = in[j * N + i]` 

在 GPU 上，可以每个线程负责转移一个元素，这种方式的读取时是 coalesced 的，但写入时却不是（尽管列方向连续，但在内存中是隔开的），我们希望写入也能变成 coalesced

但是如果读取是连续的（即行连续），那转置后写入时就变成列连续了，这看似是不可能办到的，但如果先把数据存到共享内存里，再从共享内存写入就可以了，因为共享内存访问速度较快，不要求非得是连续的

我们可以把一个矩阵分为许多区块，每个 block 处理其中的一块（以 N×N 矩阵分为 K× K 的小块为例），把这一块存到共享内存中，在写入时，按照行连续的顺序，只需要写入共享内存中关于对角线对称过去的位置的元素就行

```cpp
__global__ void transpose_tiled(float in[], float out[]) {
    // (i, j) are the tile corners
    int i = blockIdx.x * K, j = blockIdx.y * K;
    int x = threadIdx.x, y = threadIdx.y;

    // coalesced read from global memory
    __shared__ float tile[K][K];
    tile[x][y] = in[(j + y) * N + (i + x)];
    __syncthreads();

    // coalesced write to global memory
    out[(i + y) * N + (j + x)] = tile[y][x];
}

// launch kernel
dim3 blocks(N / K, N / K); dim3 threads(K, K);
transpose_tiled<<<blocks, threads>>>(d_in, d_out);
```








