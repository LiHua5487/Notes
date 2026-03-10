
# DragGAN 概述

DragGAN 的核心目标是：让用户能够通过选定并拖动图片上的几个点（指定目标位置），来精准地改变图像内容，比如调整姿势、形状、表情

DragGAN 并不是直接对原图像进行编辑的，而是调整一个图像在 StyleGAN 的潜在编码 $w$ ，把调整后的向量 $w'$ 输入到 StyleGAN 的生成器得到调整后的图像，其中调整过程是迭代的进行下面步骤
- 运动监督：改变图像，使点更接近目标位置
- 点追踪：在改变后的图像上，找到选定的点移动到了哪里

# Part A -  Paper Reproduction

## 代码分析

关于运动监督与点追踪的代码在官方仓库的 `./vis/render.py` 中的 `Renderer` 类的 `_render_drag_impl` 函数中

### 点追踪

相关代码片段如下

```python
# Point tracking with feature matching
with torch.no_grad():
	for j, point in enumerate(points):
		r = round(r2 / 512 * h)
		up = max(point[0] - r, 0)
		down = min(point[0] + r + 1, h)
		left = max(point[1] - r, 0)
		right = min(point[1] + r + 1, w)
		feat_patch = feat_resize[:,:,up:down,left:right]
		L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
		_, idx = torch.min(L2.view(1,-1), -1)
		width = right - left
		point = [idx.item() // width + up, idx.item() % width + left]
		points[j] = point

res.points = [[point[0], point[1]] for point in points]
```

---

在代码中，首先计算半径

```python
r = round(r2 / 512 * h)
```

- 这里除以 512 再乘 h 是因为参数 `r2` 是在 512×512 图像上定义的，因为 StyleGAN 通常生成512x512图像
- 运动监督中的 `r1` 和停止阈值 `max(2 / 512 * h, 2)` 也是这个原因

而后计算 $\Omega_2$ 区域的边界

```python
up = max(point[0] - r, 0)
down = min(point[0] + r + 1, h)
left = max(point[1] - r, 0)
right = min(point[1] + r + 1, w)
```

并从变换到图像大小后的特征图 `feat_resize` 中取出这个区域

```python
feat_patch = feat_resize[:,:,up:down,left:right]
```

而后根据公式计算特征值差距

```python
L2 = torch.linalg.norm(feat_patch - self.feat_refs[j].reshape(1,-1,1,1), dim=1)
```

- `self.feat_refs[j]` 就是 $f_j$ （公式里采用 $p_i$ ，而代码里采用 $p_j$ 的记号）
- `self.feat_refs[j]` 的形状是 `[C,]` ，通过 `.reshape(1,-1,1,1)` 变为 `[1,C,1,1]` ，通过广播便能与 `feat_patch` 的形状 `[1,C,H,W]` 匹配
- 使用 `torch.linalg.norm` 计算 L2 范数，指定 `dim=1` 表示沿通道维度计算

>`reshape` 中的 -1 表示自动推断该维度的数值，使得总元素数量一致，只允许出现一个 -1 ；后面 `view` 中的 -1 也是同理

选出最小的位置，并将一维索引转换为图像中的二维坐标

```python
_, idx = torch.min(L2.view(1,-1), -1)
width = right - left
point = [idx.item() // width + up, idx.item() % width + left]
points[j] = point
```

- `L2` 形状为 `[1,H,W]` ，通过 `.view(1,-1)` 展平为 `[1,H*W]` 
- 通过  `min` 找到最小值及其位置，`dim=-1` 表示在最后一个维度找
- 找到的位置是在 `H*W` 中的一维索引，需要转换为二维坐标
- 这里使用 `.item()` ，是因为 `min` 返回的位置是一个张量，所以 `idx` 实际上是一个单元素张量，通过 `.item()` 得到数字

最后把这些点收集起来

```python
res.points = [[point[0], point[1]] for point in points]
```

### 运动监督

相关代码片段如下

```python
# Motion supervision
loss_motion = 0
res.stop = True
for j, point in enumerate(points):
	direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
	if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
		res.stop = False
	if torch.linalg.norm(direction) > 1:
		distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
		relis, reljs = torch.where(distance < round(r1 / 512 * h))
		direction = direction / (torch.linalg.norm(direction) + 1e-7)
		gridh = (relis+direction[1]) / (h-1) * 2 - 1
		gridw = (reljs+direction[0]) / (w-1) * 2 - 1
		grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
		target = F.grid_sample(feat_resize.float(), grid, aligned_corners=True).squeeze(2)
		loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)

loss = loss_motion
if mask is not None:
	if mask.min() == 0 and mask.max() == 1:
		mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
		loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
		loss += lambda_mask * loss_fix

loss += reg * F.l1_loss(ws, self.w0)  # latent code regularization
if not res.stop:
	self.w_optim.zero_grad()
	loss.backward()
	self.w_optim.step()
```

---

第一部分计算 loss 中的第一项，先计算方向向量，此时还没有归一化，是为了进行下面的终止条件判断

```python
direction = torch.Tensor([targets[j][1] - point[1], targets[j][0] - point[0]])
```

而后判断是否需要继续，只要有某个点到目标的距离大于一个阈值，就继续迭代；只有当所有点到目标的距离都小于等于阈值，才停止

```python
if torch.linalg.norm(direction) > max(2 / 512 * h, 2):
    res.stop = False
```

代码在算一个点的 loss 贡献之前，先进行了一个额外的判断 `if torch.linalg.norm(direction) > 1:` ，这是一个工程上的优化，如果一个点离目标足够近，就不计算其 loss 了，避免频繁震荡与无效调整，并提高效率

计算 loss 时，先计算出 $\Omega_1$ 区域，并对方向向量归一化

```python
distance = ((xx.to(self._device) - point[0])**2 + (yy.to(self._device) - point[1])**2)**0.5
relis, reljs = torch.where(distance < round(r1 / 512 * h))
direction = direction / (torch.linalg.norm(direction) + 1e-7)
```

- `xx` 和 `yy` 是事先计算好的图像中每个位置的行 / 列索引，据此得到一个 `distance` 图，表示各处到目标的距离
- 使用 `.where()` 选出范围内的部分

而后计算目标方向的特征值

```python
gridh = (relis+direction[1]) / (h-1) * 2 - 1
gridw = (reljs+direction[0]) / (w-1) * 2 - 1
grid = torch.stack([gridw,gridh], dim=-1).unsqueeze(0).unsqueeze(0)
target = F.grid_sample(feat_resize.float(), grid, aligned_corners=True).squeeze(2)
```

- `gridh` 和 `gridw` 是归一到 $[0,1]$ 区间的坐标值，供后面 `.grid_sample()` 使用
- 把它俩用 `stack` 拼起来，并用 `unsqueeze` 添加维度，得到的形状为 `[1,1,n,2]` ，其中 n 为区域内像素数量
- 使用 `.grid_sample()` ，其根据 `grid [N,H_out,W_out,2]` 从 `feat_resize [N,C,H,W]` 插值采样，输出形状为 `[N,C,H_out,W_out]` ，在这里输出形状就是  `[1,C,1,n]` 
- 通过 `.squeeze(2)` 抹除 `dim=2` 的维度，变为 `[1,C,n]`

计算 loss 的第一项

```python
loss_motion += F.l1_loss(feat_resize[:,:,relis,reljs].detach(), target)
```

- `feat_resize[:,:,relis,reljs]` 取出 $\Omega_1$ 内的像素的特征值，形状为 `[1,C,n]` 
- 用 `.detach()` 去除梯度，表示不更新这个东西

---

第二部分计算 loss 中的第二项，这部分是因为 DragGAN 还允许用户选择图像的一个区域，只对这个区域进行改变，记为一个 mask ，代码中没有进行 $1-M$ 的操作，是因为这里的 mask 中 1 为固定区域，与论文相反

```python
mask_usq = mask.to(self._device).unsqueeze(0).unsqueeze(0)
loss_fix = F.l1_loss(feat_resize * mask_usq, self.feat0_resize * mask_usq)
loss += lambda_mask * loss_fix
```

---

代码中还额外添加了 latent code regularization 一项，这是因为 StyleGAN 的潜空间有一个特性：远离训练数据分布的区域很可能会生成低质量图像

```python
loss += reg * F.l1_loss(ws, self.w0)
```

- `ws` 和 `self.w0` 分别为当前和初始的 latent code

---

最后如果不需要停止，就进行一步优化更新

```python
if not res.stop:
	self.w_optim.zero_grad()
	loss.backward()
	self.w_optim.step()
```

## 结果复现

下图是使用 DragGAN 对图像进行编辑的结果，使用 `stylegan2-afhqwild-512x512.pkl` 模型，`latent seed = 11`

![[draggan-tiger.png]]

我们还尝试了 mask 功能，使用 `stylegan2-afhqcat-512x512.pkl` 模型，`latent seed = 3` ，编辑结果如下，仔细对比可以发现背景部分并非完全没有变化，只是变化较小

![[draggan-cat-mask.png]]

我们也尝试了真实图像编辑，使用 PTI 输出的模型（基于 `ffhq` 模型调整，并将输出的模型转换为 `.pkl` 格式） `stylegan2_custom_lrx.pkl` 与 latent code `./checkpoints/lrx.pt`

但是效果并不理想，这可能是由于 PTI 过程中，在得到 latent code 之外还会对模型进行调整，以得到一个能生成自定义图像的模型，这个过程中可能会导致自定义图像的失真与模型性能的改变

![[draggan-lrx.png]]

# Part B - Innovation & Extension

# DragDiffusion

## 领域内输入

我们选择了人脸图像来进行对比，这个图像是使用 DragGAN 的预训练模型根据 latent code 自动生成的，而非真实图像，相关参数如下
- DragGAN ：使用 `stylegan2-ffhq-512x512.pkl` 模型，`latent seed = 1` 
- DragDiffusion ：Diffusion Model 与 VAE 均使用 `anything-v5` 模型，提示词为 `a smiling boy` 

![[compare-boy.png]]

可以发现，DragGAN 的编辑结果中，人脸存在较大的变形，且背景部分变得非常混乱；而 DragDiffusion 对于人脸与背景的特征保存的更完整

我们还尝试编辑真实图像，相关参数如下
- DragGAN ：使用 PTI 输出的模型（基于 `ffhq` 模型调整）与 latent code
- DragDiffusion ：Diffusion Model 与 VAE 均使用 `anything-v5` 模型，提示词为 `an image of a person` 

![[compare-lrx.png]]

除了在 GAN inversion 过程中会造成失真以外，DragGAN 对于图像的编辑结果变形严重；而 DragDiffusion 则保持正常效果

## 领域外输入

我们选择了卡通风格的图像作为输入，相关参数如下
- DragGAN ：使用 PTI 输出的模型（基于 `ffhq` 模型调整）与 latent code
- DragDiffusion ：Diffusion Model 与 VAE 均使用 `anything-v5` 模型，提示词为 `an anime-style girl with long light blue hair tied in twin braids and yellow eyes, wearing a white blouse` 

![[compare-xiangzi.png]]

由于 `ffhq` 模型是基于真实人脸训练的，所以 DragGAN 的编辑效果存在明显的失真；而 DragDiffusion 表现正常

## 问题分析

受限于设备性能（单 GPU 设备），DragDiffusion 的编辑速度比 DragGAN 要慢很多，但是泛化能力与编辑质量明显强于 DragGAN 

一方面，Diffusion 模型一般是基于大规模、多样化的数据集训练的，而 GAN 模型的训练集较为专一；另一方面，Diffusion 模型在结构上也更有优势，其学习的是去噪函数，更加泛用，而 GAN 中的生成器与判别器往往会局限于训练集的图像，较难泛化到领域外的图像上

---

论文链接

[DragGAN](https://arxiv.org/pdf/2305.10973)

[DragDiffusion](https://arxiv.org/pdf/2306.14435)






