import numpy as np

## 创建
np.array([1, 2, 3]) # 创建数组
np.zeros((3, 4)) # 3*4的全零数组
np.ones((3, 4)) # 全1数组
np.arange(7, 3) # 从7到3递变的数组
np.linspace(0, 1, 5) # 从0到1等差5元数组
np.random.rand(2, 4) # 2*4的数据随机的数组


## 基本信息
a = np.zeros((3, 4), dtype=np.int32) # 指定数据类型，默认np.float64
a.astype(int) # 转换数据类型，暂时修改
print(a.shape) # 数组的大小，返回tuple


## 运算

# 数组大小相同
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
print(a / b) # 大小相同，对应位置分别运算
print(np.dot(a, b)) # 向量点乘

# 数组大小不同
# 广播：数组形状不同，可通过广播机制匹配
# 1.如果两个数组在某些维度上形状不同，则数组的形状会从右向左对齐。
# 2.在对齐过程中，如果其中一个数组在某个维度上为 1，则会沿该维度进行扩展（复制数据）。
# 3.如果两个数组在某个维度的大小既不匹配，又都不为 1，那么广播失败，会报错

a = a = np.array([1, 2, 3])
b = 2
print(a * b) # 数乘

a = np.array([[1, 2, 3], 
              [4, 5, 6]])
b = np.array([10, 20, 30])
print(a + b) # a的每一行都加上 b

a = np.array([1, 2, 3])
b = np.array([[10], [20]])
print(a + b)  # 广播：a 扩展为 [[1, 2, 3], [1, 2, 3]]，第1行+10，第2行+20

a = np.array([1, 2, 3])
b = np.array([10]) # 长度为1，可以广播
c = np.array([10, 20])  # 形状不兼容
print(a + b)
# print(a + c)  # 报错：ValueError: operands could not be broadcast together

# 矩阵乘法
a = np.array([1, 2])
b = np.array([[1, 1],
              [0, 1]])
print(np.matmul(a, b)) # 也可直接 a @ b

# 针对所有元素运算
a = np.array([1, 4, 9])
np.sqrt(a) # 所有元素开方
np.power(a, 2) # 所有元素乘方
# 还可以sin, cos, log等

# 取最大/最小值索引（展开至1维）
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(a.argmin())
print(a.argmax())

# 统计
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(a.sum()) # 总和
print(a.mean()) # 平均值
print(np.median(a)) # 中位数（展开至1维）
print(a.var()) # 方差
print(a.std()) # 标准差

# axis参数
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(a.sum(axis = 0)) # 每行元素求和，返回array
print(a.sum(axis = 1)) # 每列元素求和
# 其余函数同理


## 读取
a = np.array([[1, 2, 3], 
              [4, 5, 6]])
print(a[0, 1])

a = np.array([1, 3, 6, 5, 2, 4])
print(a[a < 3]) # 获取小于3的元素
print(a[::-1]) # 翻转