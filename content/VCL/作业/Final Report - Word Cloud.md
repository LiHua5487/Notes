
# 项目简介

我选择了 F. Visualization 中的 Word Cloud 项目，实现了以下功能
- **基于位图的单词碰撞检测**（精细到不同字体中每个字母的形状）
- 简单的**螺旋线布局**（在没有 mask 的情况下）
- **Shape Wordle 算法**（在有 mask 的情况下，使得单词按照特定形状分布）
	- 若 mask 大小与设置的图像大小不同，会进行插值处理
	- 支持多区域 mask 
	- 支持根据 mask 各区域颜色设置单词颜色
- **二次填充**

项目主要文件（夹）如下
- `./assets/fonts` ：字体文件
- `./assets/words` ：用于生成词云的词频信息文件，格式为每行一个单词和其出现次数，用一个空格隔开
- `./assets/masks` ：遮罩文件，用于让单词在特定区域分布
- `./src/VCX/Labs/Lab-Final` ：源代码路径
- `./build/.../release/assets/wordles` ：生成的词云图片的保存路径

使用以下指令以运行本项目

```
xmake run lab-final
```

# 实现过程

## 算法总览

本项目的词云生成总体流程如下s
- **螺旋线布局**：使用**螺旋线 / Shape Wordle 算法**，沿着曲线生成一系列候选位置点（预先计算并存储，而不是每次遍历时现场计算），遍历侯选位置并放置单词（频率 top 20，根据频率设置字体大小）
- **二次填充**：使用扫描线算法，遍历 mask 中每个像素，将频率较小（top 20 ~ 200）的词填充到剩余的空隙中，统一设为一个较小的大小与透明度

## 阿基米德螺旋线

阿基米德螺旋线的极坐标公式为

$$
r(\theta) = m\theta + b
$$

对于一个离散的步数 $n$ ，螺旋线上第 $n$ 个点的直角坐标 $(x_n, y_n)$ 为

$$
\begin{aligned}
\theta_n &= \alpha \cdot n \\
r_n &= \beta \cdot n \\
x_n &= x_c + r_n \cdot \cos(\theta_n) \\
y_n &= y_c + r_n \cdot \sin(\theta_n)
\end{aligned}
$$

- $\alpha$ ：超参数，角度步长，决定旋转速度
- $\beta$ ：超参数，半径步长，决定往外扩张的速度
- $(x_c, y_c)$ ：图像中心点坐标

但是阿基米德螺旋线无法应对指定的分布形状

## Shape Wordle

[Shape Wordle](https://enac.hal.science/hal-02399743/file/shapeSSS.pdf) 算法能够为给定区域计算出一个与之形状适配的螺旋线（形状感知螺旋线），该算法会事先计算出一个距离场 $\phi(x,y)$ （如相对于区域边缘的 SDF），在任意点 $p = (x, y)$ 处，螺旋线下一步的移动量 $(dx, dy)$ 为

$$
\begin{pmatrix}
dx \\
dy
\end{pmatrix}
= \frac{mR\,d\eta}{r}\mathbf{N} + R\,d\eta\,\mathbf{T}
$$

- $m$ ：超参数，控制螺旋线各圈之间的间距
- $d\eta$：超参数，角度步长
- $R$：点 $p$ 处距离场等值线的局部曲率半径
- $r$：在点 $p$ 处，一个用于参考的等效半径，可近似为 $\sqrt{x^2 + y^2}$（在以起点为中心的极坐标系下）
- $\mathbf{N}$：点 $p$ 处距离场 $\phi$ 的单位法向量
- $\mathbf{T}$：与 $\mathbf{N}$ 垂直的单位切向量

## 多区域 mask 

一个 mask 中可能含有多个互不连通的区域，算法的处理过程如下
- 使用 BFS 找到各个区域，并按照区域大小排序
- 为每个区域分别计算螺旋线
- 在放置单词时，依次尝试放到每个区域中，如果都放不下就丢弃（可通过调整参数避免这种情况）

下图是 `bird.png` 的 mask 的形状感知螺旋线的计算结果（红色为螺旋线，绿色为 mask 轮廓线）

![[bird-spiral.png]]

# 实现结果

![[wordle-heart.png]]

![[wordle-bird.png]]

>具体参数说明请见 `App.cpp` 中的代码注释
































