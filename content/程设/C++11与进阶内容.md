# C++11 特性

## 统一的初始化方式

可以使用大括号对数组和容器等进行初始化，而不用 `=`

```cpp
int arr[3]{1, 2, 3};
vector<int> iv{1, 2, 3};
map<int, string> mp{ {1, "a"}, {2, "b"} };
int *p = new int[20]{1, 2, 3};
```

还可以用于类和函数

```cpp
struct A {
    int i, j;
    A(int m, int n): i(m), j(n) { }
};

// 对于有构造函数的类，可以这样初始化
A *pa = new A{3, 7};
A a[] = { {1, 2}, {3, 4}, {5, 6} }; // 不能用圆括号

// 还可以初始化返回值，隐式调用构造函数
A func(int m, int n) { return {m, n}; }
```

成员变量在声明时可以直接指定初始值

```cpp
class A {
public:
    int n = 0;
};
```

### initializer list

C++11引入了模板类 `std::initializer_list` ，支持通过列表`{}`来初始化变量或传递参数
它使得函数可以接受一组用大括号`{}`括起来的不定数量的参数，而无需提前定义数据的个数

```cpp
double Sum(initializer_list<double> il) {
    double sum = 0.0;
    for (auto p = il.begin(); p != il.end(); p++) {
        sum += *p;
    }
    return sum;
}

double res1 = SumByIntialList({1.14, 5.14});
double res2 = SumByIntialList({2.5, 3.1, 4});
```

- `initializer_list`支持只读访问，无法修改其内容
- 提供成员函数 `begin()` 和 `end()` ，以及函数 `size()` 访问数据

## auto 与 decltype

auto 自动匹配变量类型
decltype 可获取变量类型并用于声明变量

```cpp
const A* p1 = new A(); 
decltype(a) p2; // p2就是A*类型的
```

- 用 auto 声明变量时要提供初始化，以便判断类型

## 智能指针shared_ptr

提供了一种安全的内存管理机制，用于避免手动内存管理可能带来的内存泄漏和悬垂指针问题
`std::shared_ptr` 是其中一个常用的智能指针类型，实现了共享所有权的语义

- 头文件 `<memory>`
- 用于托管一个 new 出来的指针 `shared_ptr p(new T);`
- 使用 `make_shared<T>(val)` 指定托管对象
- 多个 `shared_ptr` 对象可以同时托管一个指针，系统会维护一个托管计数，当无 `shared_ptr` 托管该指针时,，就 delete 该指针
- 不能托管指向动态分配的数组的指针 

| 成员函数              | 功能                                     |
| ----------------- | -------------------------------------- |
| `shared_ptr()`    | 创建一个空的 `shared_ptr`                    |
| `shared_ptr(ptr)` | 构造一个新的 `shared_ptr`，管理动态分配的对象 `ptr`    |
| `reset()`         | 释放当前所管理的对象，若为最后一个引用则销毁该对象              |
| `reset(ptr)`      | 释放当前所管理的对象，转为托管 `ptr`                  |
| `swap()`          | 交换两个 `shared_ptr` 所管理的对象               |
| `use_count()`     | 返回对象的引用计数                              |
| `get()`           | 获取所管理对象的原生指针（不会影响引用计数，谨慎使用）            |
| `unique()`        | 检查是否是唯一持有该对象的 `shared_ptr` （引用计数是否为 1） |

当多个智能指针托管同一个指针时，注意以下情况

```cpp
A* p = new A();
shared_ptr<A> ptr1(p);  // 正确
shared_ptr<A> ptr2(p);  // 错误！两个 shared_ptr 独立管理同一对象
shared_ptr<A> ptr2.reset(p); // 这样也不行
```

这种情况下，`ptr2` 的出现并不会增加 `ptr` 中对 `p` 的引用计数
存在两个 `shared_ptr`（`ptr` 和 `ptr2`）都认为自己是该资源的唯一拥有者
即这两个智能指针的引用计数都是 1

最终，`ptr` 和 `ptr2` 互不知晓对方的存在，而都在析构时尝试释放 `p` 指向的资源，也就是说 `delete p` 会被调用两次，会出问题

正确的做法是

```cpp
shared_ptr<A> ptr1(p); // 创建第一个 shared_ptr
shared_ptr<A> ptr2 = ptr1; // 创建新 shared_ptr，引用计数增加
```

这样两个指针就共享了托管资源，引用计数都是 2

## 右值引用

- 左值（lvalue）
	- 表示有明确的地址的对象，可被引用，也可以被修改,，即出现赋值语句的左侧，持久存在，直到超出作用域
- 右值（rvalue）
	- 通常是临时值，没有明确的地址，不能赋值，生命周期仅限于表达式
	- 如字面常量 `42` 表达式 `x + y` 返回的临时对象 `return A();`

右值引用 `&&`

```cpp
A a = A();
A& a1 = a; // 这里a是左值

A&& a2 = A(); // 这里A()是临时对象，是右值
```

`std::move` 
- 头文件 `<utility>`
- 将一个对象显式地转换为右值引用，从而实现移动语义
- 从原对象中窃取资源（例如动态内存），而后清空原对象（或将原对象置于一种“合法但无意义”的状态，方便其销毁）

通过移动操作替代拷贝操作，避免不必要的内存分配，提升效率

```cpp
vector<int> v1 = {1, 2, 3, 4};
vector<int> v2 = std::move(v1); // 之后v1变成空的了
```

移动构造函数

```cpp
class Resource {
    string data;
public:
    // 构造函数
    Resource(const string& initData) : data(initData) {}

    // 复制构造函数（会深拷贝）
    Resource(const Resource& other) : data(other.data) {}

    // 移动构造函数（会移动而不是拷贝）
    Resource(Resource&& other) noexcept: data(move(other.data)) {}
};

int main() {
    Resource r1("Original");
    Resource r2 = r1; // 调用复制构造函数
    Resource r3 = move(r1); // 调用移动构造函数
    return 0;
}
```

其中 `noexcept` 表示该函数不会抛出异常
如果移动构造函数没有标记为 `noexcept`，对于一些 STL 容器，可能会选择执行深拷贝而不是移动，即使移动已经足够安全，这导致性能下降

移动构造函数中，`other` 的变量类型是右值引用，但它是左值，所以需要 `move(other.data)` 将其转换为右值进行移动

return 不同类型的对象时，触发的构造函数如下

| return 的类型   | 局部对象 | 全局对象 | move(全局对象) |
| ------------ | ---- | ---- | ---------- |
| 只写复制构造函数     | 复制   | 复制   | N/A        |
| 只写移动构造函数     | 移动   | 默认复制 | 移动         |
| 同时写复制和移动构造函数 | 移动   | 复制   | 移动         |

右值引用的一个特殊用途
想要实现一个函数对象 `S` ，支持如下的使用方式，进行求和

```cpp
S sum(0);
cout << sum(1)(2) << endl; // 输出3 (0+1+2)
cout << sum(3, 4) << endl; // 输出7 (0+3+4)
cout << sum(5, 6)(7) << endl; // 输出18
```

```cpp
class S {
public:
    int curr_value;
    friend ostream& operator << (ostream& o, const S& s) = delete;
    friend ostream& operator << (ostream& o, S& s) {
        o << "total: " << s.curr_value;
        return o;
    }
    S(int n = 0) : curr_value(n) {}

    // 在此处补充你的代码

};
```

这里加了 `const` 的传参被禁了，也就是说只能用 `<<` 输出左值

如果这么做，返回的是左值，可以正常输出
但执行另一个计算时，`curr_value` 就不是从 0 开始了，所以不能这么做
```cpp
S& operator()(int a) {
    curr_value += a;
    return *this;
}
S& operator()(int a, int b) {
    curr_value += a + b;
    return *this;
}
```

如果这么做，返回的是临时对象，是右值，就不能直接输出
```cpp
S operator()(int a) {
    return S(curr_value + a);
}
S operator()(int a, int b) {
    return S(curr_value + a + b);
}
```

但可以用右值引用捕捉上面返回的右值
```cpp
friend ostream& operator<<(ostream& o, S&& s) {
    return o << s;
}
```

在这个函数里，`s` 类型是右值引用，但属于左值，所以其中的 `o << s `会触发传入左值的 `<<` 的重载，就可以输出了

## lambda 表达式

Lambda 表达式是 C++11 引入的一种轻量级语法糖，它可以用来定义匿名函数（即没有名字的函数）并直接使用，这样就不用再单独定义函数

```cpp
[捕获列表](参数列表) mutable -> 返回值类型 {
    函数体
};
```
- 捕获列表 `[]`
    - 用于定义 Lambda 表达式可以捕获的外部变量
    - 捕获的变量可以在 Lambda 表达式内部使用
- 参数列表 `()`
    - 表示 Lambda 表达式接受的参数，语法与普通函数一样
- mutable（可选）
    - 默认情况下，Lambda 表达式内部不能修改被捕获的变量
    - 如果想修改，需使用 `mutable` 关键字
- 返回值类型 `-> typename`（可选）
    - 用于显式指定 Lambda 的返回值类型
    - 如果能通过返回值推导，则可以省略

捕获形式
`[]` ：不使用任何外部变量
`[=]` ：以传值的形式使用所有外部变量
`[&]` ：以引用的形式使用所有外部变量
`[x, &y]` ：x 以传值形式使用，y 以引用形式使用
`[=, &x, &y]` ：x, y 以引用形式使用，其余变量以传值形式使用
`[&, x, y]` ：x, y 以传值的形式使用，其余变量以引用形式使用

用 lambda 表达式实现递归调用
```cpp
#include <functional>
function<int(int)> fib = [&fib](int n) {
    return n <= 2 ? 1 : fib(n - 1) + fib(n - 2);

};
```

其中 `<functional>` 提供了函数包装器 `function<返回值类型(参数类型)>`

- `fib` 是 `std::function<int(int)>` 的一个实例，它可以用来存储任意类型的函数，只要这个函数的签名满足传入一个 `int` 参数并返回 `int` 类型的值
- 因为 Lambda 表达式是匿名的，没有显式的名字，而递归需要函数调用自己，所以需要通过捕获外部的 `fib` 变量实现

或者不使用 `<functional>` ，而是额外传一个自身的引用
```cpp
auto fib = [](int n, const auto& fib_r) -> int {
    return n <= 2 ? 1 : fib_r(n - 1, fib_r) + fib_r(n - 2, fib_r);
};  

cout << fib(5, fib);
```

但是不能这样，因为在 `[&fib]` 时，`fib` 还没定义完
```cpp
auto fib = [&fib](int n) -> int {
    return n <= 2 ? 1 : fib_ref(n - 1) + fib_ref(n - 2);
};
```


# C++进阶内容

## cast 运算符

由于C语言类型转换中语义模糊和固有的危险陷阱，C语言不去判断所要操作的类型转换是否合理

`static_cast` `dynamic_cast` `reinterpret_cast` `const_cast`

```cpp
xxx_cast<new_type>(表达式)
```

`static_cast`
- 执行编译时的类型转换
- 用于进行比较 " 自然" 和低风险的转换，如整型、浮点型、字符型的转换
- 不能用于不同类型的指针和引用的转换
- 不能用于指针与整型的转换

`reinterpret_cast`
- 不同类型的指针和引用的转换
- 指针与整型（能容纳得下指针）的转换
- 执行的是逐bit拷贝操作

`const_cast`
- 去除 `const` 属性
- 将 const 指针/引用转换成同类型的非 const 指针/引用

`dynamic_cast`
- 执行运行时的类型转换
- 用于将多态基类的指针/引用转换为派生类的指针/引用  
- 不能用于非多态基类到派生类的转换
- 运行时检查转换的安全性，如果基类的指针引用的不是派生类对象，则会返回 `nullptr`（指针）或抛出 `std::bad_cast` 异常（引用）

## 异常检测

```cpp
try {
    // 可能会发生异常的代码
    if (/*某种错误条件判断*/) {
        throw exception;  // 抛出异常
    }
} catch (ExceptionType e) {
    // 捕获并处理 ExceptionType 类型的异常
} catch(...) {    
    // 捕获任意类型的异常
}
```

异常不一定非要用 `throw` 显示抛出，某处代码有问题也会抛出
如果异常没有被处理，会程序调用默认的异常处理机制 `std::terminate()`，程序异常终止

如果异常没有在本层捕获，会向外层传递

```cpp
void func() {
    try {
        throw 10;  // 抛出整型异常
    } catch (float e) {  // 捕获 float 类型异常
        cout << "Caught float in func()" << endl;
    }
    // 未捕获的整型异常继续向上传递
}

try {
    func();
} catch (int e) {  // 捕获整型异常
    cout << "Caught int in main(): " << e << endl;
}
```

异常类型

头文件 `<stdexcept>`

| **异常类**                    | **作用 / 说明**                                                   |
| -------------------------- | ------------------------------------------------------------- |
| **std::exception**         | 所有异常类的基类，提供基本的异常接口，例如 `what()` 方法返回错误描述字符串                    |
| **std::bad_typeid**        | 当使用 `typeid` 获取对象类型时，如果类型信息不可用（如访问指向空指针的多态对象的 `typeid`）会引发此异常 |
| **std::bad_cast**          | 使用 `dynamic_cast` 进行不安全的向下转换                                  |
| **std::bad_alloc**         | 使用 `new` 进行动态内存分配时内存不足                                        |
| **std::ios_base::failure** | 输入/输出流（`iostream`）操作失败                                        |
| **std::logic_error**       | 表示程序中逻辑性问题的异常，通常可以通过修改代码避免该类错误，其本身是一个异常基类，派生类有其他具体的逻辑错误异常     |
| **std::out_of_range**      | 从 `std::logic_error` 派生，表示容器访问越界                              |

## 预编译

在 C/C++ 编译过程中，编译器会先对源代码进行一个预处理步骤，处理源代码中的预编译指令（以 `#` 开头的指令，如 `#include``#define`），对代码进行文本替换或其他必要的准备工作，形成一个临时源程序文件，但不会修改原程序 

主要包括
- 头文件包含：`#include` 指令会将头文件内容插入到当前代码文件中
- 宏定义：`#define` 可以定义宏
	- 宏是一种简单的文本替换机制，比如常量替换或函数式替换
- 条件编译

### 条件编译

根据特定的条件选择性地编译某些代码块，或者忽略其他代码块

| **指令**    | **含义**                            |
| --------- | --------------------------------- |
| `#ifdef`  | 如果定义了某个宏则包含对应代码块                  |
| `#ifndef` | 如果没有定义某个宏则包含对应代码块                 |
| `#if`     | 如果满足某个布尔条件则包含对应代码块                |
| `#else`   | 如果之前的条件不满足，则执行 `#else` 下的代码块      |
| `#elif`   | 如果满足当前条件，则执行对应代码块（类似于 `else if` ） |
| `#endif`  | 用于结束一个条件编译块                       |
| `#define` | 定义一个宏                             |
| `#undef`  | 取消定义一个宏，使其失效                      |

```cpp
#include <iostream>
#define PLATFORM 1  // 定义平台，值为 1

int main() {
#if PLATFORM == 1
    std::cout << "Compiling for Windows" << std::endl;
#elif PLATFORM == 2
    std::cout << "Compiling for Linux" << std::endl;
#else
    std::cout << "Unknown platform" << std::endl;
#endif

    return 0;
}
```

因此编写.h文件时, 往往这样写

```cpp
#ifndef SOMEHEAD_H 
#define SOMEHEAD_H 
头文件内容
#endif 
```

这样能避免在多个文件的工程中, 有的头文件可能被重复包含的问题





