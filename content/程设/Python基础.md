
# 补充的杂项

---

保留小数位数
```python
x = 123.45678
a = round(x, 2) # 保留两位小数
b = round(x, -1) # 保留到十位
c = f"{x:.2f}" # 保留两位小数，位数不能是负数
```

计算最大公因数
```python
def gcd(m, n):
    while n != 0:
        m, n = n, m % n
    return m
```

---

关于数据类型的获取与判断
```python
type(a) # 获取数据类型
isinstance(a, str) # 判断某个变量是不是某个类型
```

---

`is` 和 `==` 的区别
- `is` 判断两个变量是否指向同一个地方
- `==` 判断两个变量指向的东西是不是相同

`a = b` 会使得 a 指向 b 的地方
- 对 `int` `float` `complex` `str` `tuple` 类型的变量，只需关注 `a == b` 是否成立，关注 `a is b` 无意义，因这些数据本身都不会更改
- 对 `list` `dict` 等其他类型的变量，`a == b` 和 `a is b` 都需要关注，因为这些数据本身会改变

比如对于这段代码
```python
a, b =(1,2,3), (4,5,6)
a = b
b = (4,5,6,7)
print(a)
```
输出的仍然是 `(4,5,6)` ，即便执行了 `a = b` ，但后面 `b = (4,5,6,7)` 又让 b 指向了 `(4,5,6,7)` 这片地方，那 a 和 b 指向的地方就又不一样了

而这段代码
```python
a, b = [1,2,3], [4,5,6]
a = b
b.append(7)
print(a)
```
输出的是 `[4,5,6,7]` ，因为 `a = b` 后二者指向的地方一样，改 b 也会 改 a <br>
值得注意的是，列表的每个元素也可以视为一个“指针”

---

字符串和字节流<br>
字节流：一系列字符实际存储时的编码

字符串与字节流的转换
```python
str = "hello"
s = bytes(str, encoding="utf8")
str = string(s, encoding="utf8")

s = str.encode("utf8")
str = s.decode("utf8)
```

---

要实现 cpp 中的 `while(cin>>n)` 这种的效果，可以这么写
```python
while True:
    try:
        n = int(input())
        print(n)
    except EOFError:
        pass
```

---

一些特殊的排列，比如先按分数降序，再按姓名升序排列
```python
data = [
    ("Alice", 85),
    ("Bob", 90),
    ("Charlie", 85),
    ("David", 72)
]

data.sort(key=lambda x: (-x[1], x[0]))
```

对于列表，`+=` 是直接对列表操作，`l = l + [num]` 是新建个列表再赋给 `l`<br>
这就导致了如下情况
```python
a = [1,2,3]
b = a
a += [4] # 直接改a，b也跟着改了
a = a + [4] # a变成一个新列表，b还是[1,2,3]
```

元组不能修改，无 `sort` ，只能用 `sorted(tuple)` 得到新的排好序的元组

---

`set` 的一些函数
- `set.add(x)` 添加元素
- `s2.update(s1)` 将集合s1并到s2
- `s.discard(x)` 删除元素，x不存在不会引发异常
- `s.remove(x)` 删除元素，x不存在会引发异常
- `s1.issubset(s2)` 判断s1是不是s2的子集

---

# 类与重载

## 成员函数

| 函数            | 功能               | 函数             | 功能                    |
| ------------- | ---------------- | -------------- | --------------------- |
| `__init__`    | 构造函数             | `__del__`      | 析构函数                  |
| `__eq__`      | equal: `==`<br>  | `__ne__`       | not equal: `!=`       |
| `__lt__`      | less than `<`    | `__le__`       | less or equal `<=`    |
| `__gt__`      | greater than `>` | `__ge__`       | greater or equal `>=` |
| `__add__`     | `+` 对象在左         | `__radd__`     | `+` 对象在右              |
| `__sub__`     | `-`              | `__mul__`      | `*`                   |
| `__pow__`     | `**`             | `__truediv__`  | `/`                   |
| `__mod__`     | `%`              | `__floordiv__` | `//`                  |
| `__iadd__`    | `+=`             | `__hash__`     | 返回对象的哈希值              |
| `__getitem__` | `[]`             | `__call__`     | 将对象视为函数               |
| `__lshift__`  | `«`左移            | `__rshift__`   | `»`右移                 |
| `__or__`      | 或                | `__and__`      | and与                  |
| `__xor__`     | ^异或              | `__invert__`   | ~否                    |
| `__str__`     | 强制转字符串           | `__repr__`     | 强制转换可执行字符串            |

注：
- `setitem` 的重载格式为 `def __setitem__(self, key, val):`
- 当 `print(obj)` 或 `str(obj)` 时需要重载 `__str__`

一个类的构造和析构函数只能有一个

一个对象是**可哈希**的，必须满足以下两个条件
- 有固定的哈希值：即调用 `hash(obj)` 时一直返回相同的结果
- 它可以与其他对象使用 `==` 做比较

一般来说不可变元素可哈希，可变元素不可哈希

如果只重载 `eq` 的话，对象是不可哈希的，因为默认情况下，Python 会将 `__hash__` 方法显式设置为 `None`

使用对象作为字典的key时，实际上是以对象的地址，而非内容作为关键字

而如果要使用对象的属性作为key，需要同时重载 `__eq__` 与 `__hash__`

```python
class  A:
    def __init__(self,x):
        self.x = x
    def __hash__(self):
        return 0 # 所有实例哈希值都为0，会导致哈希冲突
    def __eq__(self,other): # 哈希冲突时使用eq判断是不是同一个key
        return self.x == other.x

a = A(3)
b = A(3)
d = {A(5): 10, A(3): 20}
print(d[a])
print(d[b])
```

在创建 `d` 时，进行了以下过程
- 对 `A(5)` 进行哈希计算，`__hash__` 返回 `0` ，存储到哈希桶
- 对 `A(3)` 进行哈希计算，哈希值也是 `0`，与第一个对象哈希冲突
- 出现哈希冲突的情况下，使用 `__eq__` 比较：`A(3).__eq__(A(5))` 返回 `False`，因为 `3 != 5`，所以 `A(3)` 被作为一个新键存储

此时，字典 `d` 的存储内容为
- 键：`A(5)`，对应值 `10`
- 键：`A(3)`，对应值 `20`

在使用对象a进行访问时，进行一下过程
- 进行哈希计算：`a.__hash__()` 返回 `0`，从字典中找到哈希为 `0` 的所有键
- 遍历冲突的哈希桶，使用 `__eq__` 判断相等性
    - `a.__eq__(A(5))` 返回 `False`，跳过
    - `a.__eq__(A(3))` 返回 `True`，因此找到对应的值 `20`<br>
而用b访问时，与之相同，所以也会输出 `20`


## 类属性、类方法、静态方法

```python
class Employee:
    company = "TechCorp"  # 静态成员变量/类属性，所有实例共享
    def __init__(self, name):
        self.name = name

    @classmethod # 类方法
    def change_company(cls, new_company):
        cls.company = new_company  # 修改类属性
        return f"Company changed to {new_company}"

    @staticmethod # 静态方法，独立于类
    def calculate_bonus(salary):
        return salary * 0.1

    def get_info(self):
        return f"{self.name} works at {self.company}"

print(Employee.company) # 获取类属性
emp = Employee("Alice") # 声明
print(Employee.change_company("NewTech"))  # 调用类方法
bonus = Employee.calculate_bonus(50000) # 调用静态方法
emp.bonus = bonus # 可随时添加成员变量
```

|        | 普通方法             | 类方法                          | 静态方法                       |
| ------ | ---------------- | ---------------------------- | -------------------------- |
| 格式     | `def func(self)` | `@classmethod def func(cls)` | `@staticmethod def func()` |
| 调用     | 类名/对象            | 类名/对象                        | 类名/对象                      |
| 访问类属性  | 不能，间接调用`A.n`     | 可以，直接调用`cls.n`               | 不能，间接调用`A.n`               |
| 访问成员变量 | `self.n`         | 不能                           | 不能                         |
| 用途     | 操作类属性            | 操作类属性，为类创建预处理实例              | 工具函数、与类相关但不需要类数据的函数        |

**公有私有变量**
- `x`：公有变量
- `_x`：约定私有变量，类对象和子类可以访问，`import`无法访问，通过`__all__ = ["_x"]`显式地允许`import`
- `__x`：完全私有变量，会触发Python的名称重整机制，不可直接访问，可以在类内或在外部通过`_ClassName__x`访问

`property`：将函数封装为属性，能像成员变量一样被访问
- 允许在访问或修改属性时，加入逻辑（如检查值是否合法、自动计算等）
- 可以轻松实现只读/只写属性

```python
class Person:
    def __init__(self, name):
        self._name = name

    # 定义 getter 方法，将函数封装为属性
    @property
    def name(self):
        return self._name

    # 定义 setter 方法，允许修改属性
    @name.setter
    def name(self, value):
        if not isinstance(value, str): # 检查赋值是否合法
            raise ValueError("name must be string.")
    self._name = value

    # 定义 deleter 方法，删除属性
    @name.deleter
    def name(self):
        print("name attr deleted.")
        del self._name
```

泛型：由于Python函数不会显式地声明类型，所以任何函数都是相当于模板

继承与多态：本来对象类型就是运行时确定，没有明显的多态，其余类似于cpp

```python
class A:
    def __init__(self, x):
        self.x = x
class B(A):
    def __init__(self, x, y):
        A.__init__(self, x) # 调用基类构造函数
        self.b = b

a, b = A(), B()
print(isinstance(a, A)) # >>> True
print(isinstance(b, A)) # >>> True
print(isinstance(a, B)) # >>> False
print(isinstance(b, B)) # >>> True
```


# 迭代器与生成器

可迭代对象：可以用`for i in x:`形式遍历的对象，必须实现迭代器协议，即
- `__iter__()`：返回对象本身（迭代器）
- `__next__()`：返回下一个元素

在for循环中，Python将自动调用 `x.__iter__()` 获得迭代器p，自动调用`__next__(p)` 获取元素，并且检查StopIteration异常，碰到就结束循环

在类内实现一个简单的迭代器
```python
class MyRange:
    def __init__(self, n):
        self.idx = 0 # 当前下标
        self.n = n # 总长度
    def __iter__(self):
        return self
    def __next__(self):
        if self.idx < self.n: # 保证不越界
            val = self.idx
            self.idx += 1
            return val
        raise StopIteration() # 越界则raise错误
```

其中 `raise` 千万不要写成 `return` ，不然就一直给你输出错误停不下来

迭代器还可以作为单独一个类去实现
```python
class MyRange:
    def __init__(self, n):
        self.n = n
    def __iter__(self):
        return MyRangeIteration(self.n)

class MyRangeIteration:
    def __init__(self, n):
        self.i = 0
        self.n = n
    def __iter__(self):
        return self
    def __next__(self):
        if self.i < self.n:
            i = self.i
            self.i += 1
            return i
        raise StopIteration
```

**生成器**：一种延时求值的特殊迭代器，内部包含计算过程，真正需要时才去计算

一个简单的生成器长这样
```cpp
a = (i * i for i in range(5))
print(a)
# >>> <generator object <genexpr> at ...>
for x in a:
    print(x, end=" ")
# >>> 0 1 4 9 16
```

与列表推导式不同，这是用 `()` 扩起的<br>
也可以用 `next(generator)` 显式的逐步进行

`yield`：将所在函数变成一个生成器，调用时不会立刻执行，而是以yield作为“断点”，遇到yield就暂停，并且返回 yield 的值，相当于有暂停功能的 `return`

```python
# 实现斐波那契数列
def fibonacci(n):  # 求斐波那契数列前 n 项
    a, b, counter = 0, 1, 0
    while counter <= n:
        yield a
        a, b = b, a + b
        counter += 1

f = fibonacci(10)
while True:
    try:
        print(next(f), end=" ")
    except StopIteration:
        break
```

但是，如果异想天开写一个 `x = yield 4` 这种的东西，yield 暂停了，再次执行时，剩下的 `x = ` 卡一半了，这时候就需要在 `next` 前使用 `send(num)` 来给一个初始值
```python
def counter(start=0):
    current = start # <-- 初次调用从这里执行
    while True:
        received = yield current  # 暂停并返回当前值，等待外部发送值
        if received is not None:  # 如果接收到新的值，将其作为新的起点
            current = received
        else:
            current += 1

gen = counter(10)
print(next(gen))  # >>> 10 初次调用
print(next(gen))  # >>> 11
print(gen.send(20))  # >>> 20 接收传入值 20，并从该值继续运行
print(next(gen))  # >>> 21
```

没有执行`next`或`send(None)`前，不能`send(x)` ，因为此时x还不是None，只有yield完了之后x才变成None，不然会报错 `TypeError`

初次执行 `send(None)` ，相当于执行 `next` ，因为这同样会让生成器从一开始处执行，直到碰到yield或结束

# 闭包与装饰器

在讲装饰器之前，先讲一个概念

**闭包**：闭包是嵌套函数的一种特性
- 保持状态：闭包能访问并保存外层函数的变量，执行后不会立即释放
- 延迟计算：闭包可以携带环境变量，在需要的时候动态计算结果

一个最基本的闭包长这样
```python
def outer_function(x):  # 外部函数
    def inner_function(y):  # 内部函数
        return x + y
    return inner_function  # 返回内部函数

closure = outer_function(10)  # x = 10
print(closure(5)) # >>> 15
print(closure(20)) # >>> 30
```

闭包还能修改外部变量，需要加上 `nonlocal`
```python
def make_counter():
    count = 0
    def counter():
        nonlocal count  # 允许修改外部变量
        count += 1
        return count
    return counter

# 创建两个独立的计数器
counter1 = make_counter()
counter2 = make_counter()
print(counter1())  # >> 1
print(counter1())  # >> 2
print(counter2())  # >> 1
print(counter2())  # >> 2
```

这里举一个作业中的例子，假设我们提前定义了 `f1(x)` ... `f5(x)` <br>
现在我们想实现函数组合功能，例如输入 `f1 f2 f3` ，再输入 `4` ，就可以输出 `f3(f2(f1(4)))` <br>
已知的代码如下，需补全 `accfunc`
```python
def accfunc(f):
    # === TO DO ===

while True:
    try:
        s = input()
        n = int(input())
        s = s.split()
        k = accfunc
        for x in s:
            k = k(eval(x))
        print(k()(n))
    except:  #读到 EOF 产生异常
        break
```

一种解决方案如下
```python
def accfunc(f):
    def inner(g=None):
        if g is None:
            return f
        else:
            new_func = accfunc(lambda x: g(f(x)))
            return new_func
    return inner
```

当我们按照例子输入后，首先，对于循环执行 `k = k(eval(x))` 的部分

一开始 `k = accfunc, x = f1` ，执行 `k(eval(x))` 即 `accfunc(f1)` <br>
这会返回一个 `inner` ，其记住了外层变量 `f1` <br>
我们不妨将其记为 `inner with f1`

第二步时 `k = inner with f1, x = f2` ，执行 `inner(f2)` <br>
这会进入执行 `accfunc(lambda x: f2(f1(x)))` 的分支<br>
执行后 `new_func` 就是一个 `inner` 函数，携带外部变量 `f2(f1)`

第三步与第二步同理，最后 `k = inner with f3(f2(f1))`

当输出 `k()(n)` 时，先执行 `k()` ，由于输入为空，所以会直接返回外部变量，即 `f3(f2(f1))` 而后应用于参数 `n` ，就得到 `f3(f2(f1(n)))`

---

现在讲装饰器，假设要写很多函数，但这些函数有一部分都是差不多的，如果每次写一个新的函数都要再写这么一段，太烦人了

于是装饰器应运而生，把这个公共部分提取出来
```python
def log_wrapper(func):
    def wrapper(*args, **kwargs): # 接收具体函数的参数
        print(f"executing function {func.__name__}.")
        result = func(*args, **kwargs)  # 调用被装饰的函数
        print(f"function {func.__name__} done.")
        return result  # 调用具体函数时的实际返回值
    return wrapper

@log_wrapper
def add(a, b):
    return a + b

@log_wrapper
def multiply(a, b):
    return a * b
```

这个例子中，我们希望每次执行add和multiply都能打印日志信息，于是将其提取出来放到装饰器<br>
当使用 `@log_wrapper` 装饰函数时，函数就会自动被装饰器包装

现在，对于每个函数，其日志信息想要分成不同的类别，比如 `WARN` `DEBUG` `INFO` 等，那就需要含参的装饰器
```python
# 定义带参数的装饰器
def log_with_prefix(prefix):
    # 这个是装饰器的真正实现
    def decorator(func):
        def wrapper(*args, **kwargs):
            print(f"[{prefix}] executing {func.__name__}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

@log_with_prefix("DEBUG")
def add(a, b):
    return a + b

@log_with_prefix("INFO")
def multiply(a, b):
    return a * b
```

除了函数，还可以用类当作装饰器
```python
class CallCounter:
    def __init__(self, func):
        self.func = func
        self.call_count = 0

    def __call__(self, *args, **kwargs): # 当作装饰器被调用时
        self.call_count += 1
        print(f"{self.func.__name__} has been called {self.call_count} times.")
        return self.func(*args, **kwargs)  # 真正调用目标函数

@CallCounter
def greet(name):
    print(f"Hello, {name}!")
```

在用 `@CallCounter` 装饰时，会创建一个类实例，并传入被装饰的函数名进行初始化

补充一句，python不支持重载，前一个函数会被后一个覆盖掉，只有最后定义的函数被装饰并计入调用次数




