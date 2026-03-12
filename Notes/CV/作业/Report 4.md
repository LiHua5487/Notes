
# 1 Basic Stereo Matching Algorithm

从参考图像的左上角开始，逐行逐像素地处理整个图像，对于每个像素位置$(x,y)$，在第二图像的同一行上寻找最佳匹配

在给定的视差范围内，依次测试每个可能的视差值 d ，对于每个视差值d，在第二图像的同一行上，从位置 $(x-d)$ 处提取对应窗口，根据给定的相似度函数进行计算，并从中选择最匹配的窗口

对于边界处的部分，我采用了窗口截断的方法，即只计算处在图像内的部分

我的参数设置如下

```
window_sizes = [3, 5, 7]
disparity_range = (0,16) and (0, 32)
matching_functions = ['SSD', 'SAD', 'normalized_correlation']
```

在遍历匹配的部分，我用 `tqdm` 构建了一个进度条，从中可以获得运行时间的信息，结果如下

![[CV/作业/imgs/img4/image-1.png]]

![[CV/作业/imgs/img4/image.png]]

可以发现，窗口越大时间越长（但是这个影响相对不那么明显）；设置更大的 `disparity_range` ，由于搜索空间变大，时间也会增长；对于匹配度函数，SSD 和 SAD 时间差不多，都比较快，而 NCC 却很慢，可能因为其进行了更复杂的运算

对于 SSD 和 NCC ，窗口大小为 7 的效果会好一些，噪声没那么大；而 SAD 窗口大小为 3 会好一些，可能是因为 SAD 只是简单的将各处差异的绝对值累加，对每个像素给予同等权重，随着窗口增大，不匹配区域的误差会线性累积，即使只有一小部分区域不匹配，也会显著增加总代价，所以窗口变大时匹配效果变差

当最大深度设为 32 时，视差图看起来会更暗淡，与 gt 的颜色差异较大，而设为 16 就差不多了

对于 `tsukuba` 场景，使用 SSD 的效果会好一些，我又将窗口大小增大了一下，参数与结果如下

```
window_sizes = 11
disparity_range = (0,16)
matching_functions = 'normalized_correlation'
```

![[task1_tsukuba_11_0_16_SSD.png]]

可以发现，得到的视差图噪声较大，存在很多空洞区域，且在物体边缘处很混乱，这种方法假设局部窗口内视差恒定，忽略了场景的几何结构和全局约束，导致在复杂场景中表现不佳；同时 half occlusion 现象的存在会导致空洞，但这种方法并没有应对措施

# 2 Depth from Disparity

使用 `cv2.StereoBM` 匹配结果还原出的点云及相关参数如下，与 `pointcloud_example.ply` 对比，二者基本重合

```
baseline = 10
focal_length = 1000
```

![[CV/作业/imgs/img4/image-2.png]]

对于 task1 中的算法的匹配结果还原的点云，其存在明显的分层现象，而且支离破碎的，我稍微调整了一下参数使其看起来稍微凝聚一点

```
baseline = 1
focal_length = 500
```

![[CV/作业/imgs/img4/image-3.png]]

这可能是因为最基本的算法没有额外添加任何平滑性约束，应该符合预期

# Stereo Matching with Dynamic Programming

在使用 dp 进行左右两图的 scanline 匹配时，先计算出代价矩阵 $e(i,j)$ ，我采用了 SSD 函数，并维护一个窗口

而后考虑状态转移方程，对于一个位置，要么是从上面过来的，要么是从左边过来的，要么是从左上方过来的

```
C(i,j) = min([
	C(i-1,j-1) + e(i,j),
	C(i-1,j) + occlusionConstant,
	C(i,j-1) + occlusionConstant, ])
```

将其记录到一个矩阵 $B$ 中，分别标记这些情况为 1 2 3 ，回溯时只需在 B 中查找即可 

后处理时，根据周围的像素进行填充，由于以左图为基准，就沿着 scanline 在其左边找到最近的未被遮挡的像素，把它的值复制过来

可得参数设置、运行时间与结果如下

```
max_disparity = 16
occlusionConstant = 100
window_size = 5
```

![[CV/作业/imgs/img4/image-5.png]]

![[task3_tsukuba.png]]

转换为点云如下

![[CV/作业/imgs/img4/image-4.png]]

得到的视差图与 basic stereo matching algorithm 相比干净了很多，但是有许多横向的线条，这是后处理部分导致的（如果不加后处理，则在每个线条左端会有很多黑点）




