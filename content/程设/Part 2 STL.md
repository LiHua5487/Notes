# 输入输出流与文件操作

![[程设/img/image 1.png|539x353]]

- cin cout 与标准输入/输出设备相连，对应于标准输入/输出流，用于从键盘读取数据/向屏幕输出数据，也可以被重定向为从文件中读取/写入数据
- cerr （无缓冲） 
	- 数据会立即输出到设备，如终端或日志文件，不经过缓冲区
	- 适用于紧急错误、调试信息（确保即使程序崩溃也能输出）
- clog（带缓冲）
	- 数据先存入缓冲区，缓冲区满或手动刷新时才输出
	- 适用于非紧急日志记录，性能更高（减少频繁I/O操作）
	- 也可用于常规日志记录（如程序运行状态、非关键警告），能延迟输出

`ios::binary` 用于读写二进制文件

#  函数模板和类模板


```cpp
#include <iostream>
using namespace std;
// ====== TO DO ======
template<typename T>
class MyLess {
public:
	bool operator()(T a, T b) {
		return a < b;
	}
};

template<typename T, typename Cmp1 = MyLess<T>>
class MaxFinder {
	T* begin;
	T* end;
public:
	MaxFinder(T* b, T* e): begin(b), end(e) {}
	T* getMax() {
		T* maxp = begin;
		Cmp1 cmp;
		for (T* p = begin + 1; p != end; p++) {
			if (cmp(*maxp, *p))
				maxp = p;
		}
		return maxp;
	}
	template<typename Cmp2>
	T* getMax(Cmp2 cmp) {
		T* maxp = begin;
		for (T* p = begin + 1; p != end; p++) {
			if (cmp(*maxp, *p))
				maxp = p;
		}
		return maxp;
	}
};
// ====== TO DO ======

bool cmp(int a, int b) {
	return a % 10 < b % 10;
}
struct op {
	bool operator()(int a, int b) {
		return a % 7 < b % 7;
	}
};
int main() {
	int a[8];
	for (int i = 0; i < 8; ++i)
		cin >> a[i];
	MaxFinder<int> mf1(a, a + 8);
	int cmd;
	cin >> cmd;
	switch (cmd) {
	case 0:
		cout << *mf1.getMax() << endl;
		break;
	case 10:
		cout << *mf1.getMax(cmp) << endl;
		break;
	case 7:
		cout << *mf1.getMax(op()) << endl;
	}
	MaxFinder<int, op> mf2(a, a + 8);
	cout << *mf2.getMax() << endl;
	return 0;
}
```

初始化 T 类型的变量：`T t = T()` 
声明函数指针 `func` ： `ReturnType(*func)(ArgType)`

- 当函数模板和一般函数同名时，普通函数匹配优先级高于函数模板，只有当普通函数无法匹配时，才会尝试实例化模板
- 函数模板可以被重载，而两个类模板名字不能相同

# 基本概念

- 容器：可容纳各种数据类型的通⽤数据结构，是类模板 
- 迭代器：可⽤于依次存取容器中元素，类似于指针
	普通的C++指针就是⼀种迭代器
- 算法：⽤来操作容器中的元素的函数模板
	如`sort()`对⼀个数组中的数据进⾏排序，`find()`搜索⼀个数组中的对象 
	算法本⾝与他们操作的数据的类型⽆关，因此他们可以在从简单数组到⾼度复杂容器的任何数据结构上使⽤

# 容器

- 第一类容器
	- 顺序容器/序列容器 vector, deque, list 
	- 关联容器/有序容器 set, multiset, map, multimap 
- 容器适配器 stack, queue, priority_queue

- 对象被插⼊容器中时, 被插⼊的是对象的⼀个复制品 
- 放⼊容器的对象所属的类, 往往还应该重载 `==` 和 `<` 运算符用于排序

STL 中的相等
- 有时， "x和y相等" 等价于 "x == y 为真"
	在未排序的区间上进⾏的算法
	如顺序查找 `find`
- 有时， "x和y相等" 等价于 "x < y 和 y < x 均为假"
	在排好序的区间上进⾏查找，合并等操作的算法
	与 `==` 运算符⽆关 
	如折半查找算法 `binary_search` ，关联容器⾃⾝的成员函数 `find` 


所有容器共有的成员函数
- 相当于按词典顺序⽐较的比较运算符
- `empty` ：判断容器中是否有元素
- `max_size` ：系统资源允许时，容器理论上能够存储的最大元素数量
- `size` ： 容器中元素个数
- `capacity` ：容器目前分配的内存量能够存储的元素数量
    这个值表示容器在不重新分配内存的情况下能够存储的元素的最大数量当容器的大小 `size` 超过当前容量时，容器会自动重新分配更大的内存空间
- `swap` ：交换两个容器的内容

只在第一类容器的成员函数
- `begin` ：返回指向容器中第⼀个元素的迭代器 
- `end` ：返回指向容器中最后⼀个元素后⾯的位置的迭代器 
- `rbegin` ：返回指向容器中最后⼀个元素的迭代器 
- `rend` ：返回指向容器中第⼀个元素前⾯的位置的迭代器 
- `erase` ：移除单个或范围内元素，返回指向被删除区域后一个元素的迭代器
- `clear` ：从容器中删除所有元素
- `count` ：统计为某个值的元素的个数

![[程设/img/image-3.png]]

## 顺序容器

vector / deque / list 
- 容器不自动排序, 元素的插⼊位置同元素的值⽆关

| 函数           | 适用范围              | 功能                                                  |
| ------------ | ----------------- | --------------------------------------------------- |
| `front`      | 所有顺序容器            | 返回第一个元素的引用                                          |
| `back`       | 所有顺序容器            | 返回最后一个元素的引用                                         |
| `push_back`  | 所有顺序容器            | 在末尾添加元素                                             |
| `pop_back`   | 所有顺序容器            | 弹出末尾元素                                              |
| `insert`     | 所有顺序容器            | 在指定位置插入元素                                           |
| `push_front` | `deque`, `list`   | 在头部添加元素                                             |
| `pop_front`  | `deque`, `list`   | 弹出头部元素                                              |
| `sort`       | `vector`, `deque` | 对区间 `[first, last)` 内的元素排序；<br>`list` 要使用自己的 `sort` |

### vector

- 动态数组
- 随机存取任何元素都能在常数时间完成
- 在尾端增删元素具有较佳的性能

![[程设/img/image.png]]

### deque

- 双向队列，元素在内存连续存放
- 随机存取任何元素都能在常数时间完成（但次于vector） 
- 在两端增删元素具有较佳的性能（⼤部分情况下是常数时间）

![[程设/img/image-1.png]]

⽐vector的优点：头部删除/添加元素性能也很好 

### list

- 双向链表，元素在内存不连续存放 
- 在任何位置增删元素都能在常数时间完成 
- 不⽀持随机存取

![[程设/img/image-2.png]]

自带的函数
- `sort` ：自带的排序算法，`list` 不支持 STL 的算法 `sort` 
- `remove` ：删除和指定值相等的所有元素
- `unique` ：删除所有和前一个元素相同的元素（要做到无重复元素，还需先 `sort`）
- `merge` ：合并两个链表，并清空被合并的那个
- `reverse` ：颠倒链表
- `splice` ：在指定位置前插入另一个链表中的一个或多个元素，并在另一个链表中删除被插入的元素

```cpp
list<int> mylist; 
mylist.sort(compare); // compare 函数可以⾃⼰定义
mylist.sort(); // ⽆参数版本, 按 < 排序
```

## 关联容器

set / multiset / map / multimap 
- 自动对元素排序，新元素插入的位置取决于它的值 
- 查找速度快，通常以平衡⼆叉树⽅式实现,，插⼊和检索的时间都是 $O(logN)$

支持函数

`iterator find(iterator first, iterator last, const T& val)`
- 在范围内查找值为 `val` 的元素，返回其迭代器，找不到返回 `end()`
 
`iterator insert(const T & val); `
- 将 `val` 插入到容器中并返回其迭代器 
-  set/map 的返回类型是 `pair<iterator, bool>` ，`bool` 表示是否插入成功

`void insert(iterator first, iterator last);` 
- 将另一个容器的区间 `[first, last)` 的元素插入这个容器 
- 此时会调用比较函数进行自动排序

`iterator lower_bound(const T & val); `
- 查找 it ，使得 `[begin(), it)` 中所有的元素都比 `val` 小 

`iterator upper_bound(const T & val); `
- 查找 it ，使得 `[it, end())` 中所有的元素都比 `val` 大 

`pair equal_range(const T & val);` 
- 同时求得 `lower_bound` `upper_bound`，范围内的值都是 `val`
- `pair.first` 与 `pair.second` 分别指向 `lower_bound` 与  `upper_bound`

## set / multiset

不能修改元素，只能插入或删除元素

multiset 元素可以重复
set 元素不重复

- 插入set中已有的元素时, 插入不成功 
- 与multiset的区别： 是否允许重复元素 
- 与map的区别：是否显示定义 key ，set/multiset 使用元素本身作为 key

下面以 multiset 为例

```cpp
multiset<class T, class Pred = less<T>, class A = allocator<T> >
```
- `T` 表示元素类型
- `Pred` 决定排列顺序，默认 `less<T>` 为升序排列
- `A` 是内存分配器，用于管理容器中元素的动态内存分配方式

```cpp
template<class T>
struct less : public binary_function<T,  T,  bool> {
    bool operator()(const T& x, const T& y) const {
        return x < y;
    }
};
```

- 如果用类作为元素类型，那这个类必须重载 `<` ，因为 less 是用 `<` 比较的
- 自定义排序模板时，要和 `less` 一样重载 `()`

## map / multimap

- 储存一堆 pair 键值对，按照键排序
- 如果 multimap 键相同，保留插入的相对顺序
- map 不允许值重复，而 multimap 可以

```cpp
multimap<class Key, class T, class Pred = less<Key>, class A = allocator<T> >
```
- `Key` `T` 为键和值的类型

对于 map，可以用 `map[key]` 访问值
若没有对应 `key` 则插入，并返回值的引用

对于 multimap，默认访问同键的第一个值


## 容器适配器

可以用某种顺序容器来让已有的顺序容器以栈/队列的方式工作

- stack 栈：后进先出 
- queue 队列：先进先出 
- priority_queue 优先队列：优先弹出最高优先级的元素

不能修改元素，只能增删元素

| 函数      | 适用范围                      | 功能                                            |
| ------- | ------------------------- | --------------------------------------------- |
| `push`  | 所有容器适配器                   | 添加元素                                          |
| `pop`   | 所有容器适配器                   | 弹出元素                                          |
| `top`   | `stack`, `priority_queue` | 返回栈顶元素（`stack`）或优先级最高的元素（`priority_queue`）的引用 |
| `front` | `queue`                   | 返回队首元素的引用                                     |
| `back`  | `queue`                   | 返回队尾元素的引用                                     |

# 迭代器

- 指向第⼀类容器中的元素，有 const 和⾮ const 两种
	- 通过迭代器可以读取它指向的元素 
	- 通过⾮ const 迭代器还能修改其指向的元素 
- 迭代器⽤法和指针类似

```cpp
// 声明 
容器类名::iterator 变量名;
容器类名::const_iterator 变量名;
// 访问 
*迭代器变量名
```

- 迭代器上可以执⾏ `++` 操作，以使其指向容器中的下⼀个元素 
- 如果迭代器到达了容器中的最后⼀个元素的后⾯ 
	- 迭代器变成 `past‐the‐end` 值 
	- 此时再使⽤它，就会出错（类似于使⽤ `NULL` 或未初始化的指针）

- 不同容器上⽀持的迭代器功能强弱有所不同
	- 这决定了该容器是否⽀持STL中的某种算法
	- 只有第⼀类容器能⽤迭代器遍历
	- 排序算法需要通过随机迭代器来访问容器中的元素，所以有的容器就不⽀持排序算法

STL中的迭代器按功能由弱到强分为5种 
1. 输⼊：只读访问 
2. 输出：只写访问 
3. 正向：支持读写，并能⼀次⼀个地向前推进迭代器 
4. 双向：支持读写，并能⼀次⼀个地向前和向后移动 
5. 随机访问：支持读写，并能在数据中随机移动 

迭代器支持的操作
- 所有迭代器: `++p`  `p++` 
- 输⼊迭代器: `*p`  `p == p1`  `p != p1` （指向元素是否相同）
- 输出迭代器: `*p = p1` 
- 正向迭代器: 上⾯全部 
- 双向迭代器: 上⾯全部，以及 `‐‐p`  `p‐‐`
- 随机访问迭代器: 上⾯全部，以及 
	- 移动i个单元: `p += i`  `p‐= i` `p + i`  `p– i` 
	- ⼤⼩⽐较: `p < p1`  `p <= p1`  `p > p1`  `p >= p1` （指向的先后位置）
	- 数组下标 `p[i]` : p 后⾯的第 i 个元素的引⽤

| 容器             | 迭代器类别  |
| -------------- | ------ |
| vector         | 随机     |
| deque          | 随机     |
| list           | 双向     |
| set / multiset | 双向     |
| map / multimap | 双向     |
| stack          | 不支持迭代器 |
| queue          | 不支持迭代器 |
| priority_queue | 不支持迭代器 |
有的算法, 例如 `sort` `binary_search` 需要通过随机访问迭代器来访问容器的元素，那 list 以及关联容器就不⽀持该算法

同时，由于 list 支持的是双向迭代器，有些遍历方式就不能用

```cpp
list<int> v;
list<int>::const_iterator p;
// 正确的遍历方式
for(p = v.begin(); p != v.end(); p ++)
    cout << *p;

// 错误的遍历方式
for(p = v.begin(); p < v.end(); p ++) //双向迭代器不⽀持 <
    cout << *p;  

for(int i = 0; i < v.size(); i++)
    cout << v[i]; //双向迭代器不⽀持[]
```


# STL 算法

STL 提供能在各种容器中通⽤的算法
- 算法就是⼀个个函数模板，⼤多数在 `<algorithm>` 中定义 
- 算法通过迭代器来操纵容器中的元素
- 许多算法可以对容器中的⼀个局部区间进⾏操作，因此需要起始元素的迭代器，终⽌元素的后⾯⼀个元素的迭代器这两个参数，如排序和查找
- 有的算法返回⼀个迭代器，⽐如 `find()` 算法，在容器中查找⼀个元素，并返回⼀个指向该元素的迭代器
- 算法可以处理容器，也可以处理C语⾔普通数组

STL 的 `remove()` 和 list 自带的不同，实际上是把元素放到后面，并重置 `end()` ，相当于“重新移动”

## find

```cpp
template<class InIt, class T>
InIt find(InIt first, InIt last, const T& val);
```

- 查找区间
	- `first` 和 `last` 都是容器的迭代器，它们给出了容器中的查找区间起点和终点 `[first, last)`
	- 这个区间是一个左闭右开的区间，即区间的起点是位于查找范围之中的，而终点不是
- 查找对象
	- 在 `[first, last)` 查找等于 `val` 的元素，用 `==` 运算符判断相等
- 返回值是一个迭代器
	-  如果找到，则该迭代器指向被找到的元素
	- 如果找不到，则该迭代器等于 `last`

- 关联容器自带 find
- 顺序容器可以使用 STL 的find

## copy

```cpp
template <class T1, class T2>
T2 Copy(T1 s, T1 e, T2 x) {
    for (; s != e; ++s, ++x)
        *x = *s;
    return x;
}
```

- `copy(first, last, newbegin);` 用于将 `[first, last)` 内的元素复制到以 `newbegin` 开始的另一个范围，这些迭代器指向的元素类型应该兼容

`copy` 也可以用来输入输出
- `copy (v.begin(), v.end(), output);` 将 `v` 的内容通过 `output` 输出
- `ostream_iterator output(cout, " ");` 是一个输出迭代器，将数据写⼊到 cout 中，在每次写⼊数据后插⼊的分隔符 " "
- `istream_iterator inputInt(cin);` 用于读取输⼊流

```cpp
istream_iterator<int> inputInt(cin);
int n1, n2;
n1 = * inputInt; // 读⼊n1
inputInt++;
n2 =*inputInt; // 读⼊n2

ostream_iterator<int> outputInt(cout);
* outputInt = n1 + n2;

int a[5] ={1, 2, 3, 4, 5};
copy(a, a+5, outputInt); // 输出整个数组
```

要想实现 `ostream_iterator` 的功能，要重载 `++` `=` `*` ，且输出最好放在 `=`

```cpp
template<class T>
class My_ostream_iterator{
private:
    string sep; // 分隔符
    ostream & os;
public:
    My_ostream_iterator(ostream & o, string s): sep(s), os(o){ }
    void operator ++() { } // 执行输出任务不需要额外改变状态，有定义即可
    My_ostream_iterator& operator * () {
        return *this;
    }
    My_ostream_iterator& operator = ( const T & val ) {
        os << val << sep;
        return *this;
    }
};
```

## for_each

```cpp
template<typename InputIt, typename Func>
Func for_each(InputIt first, InputIt last, Func f) {
    for (; first != last; ++first) {
        f(*first);  // 对每个元素调用函数对象
    }
    return f;  // 返回最终状态的函数对象
}
```

如果 f 传入的是函数对象，会传入对象的副本，影响不到原对象的状态
如果想要修改原对象，需要在复制构造时传递指针，或用智能指针

执行结束后会返回 f ，触发复制构造函数

# 函数对象

```cpp
class Prod{
    int value;
public:
    Prod(int _v): value(_v) {}
    //重载 () 运算符
    int operator()(int other){
        return value * other;
    }
};

Prod p =Prod(2);
cout << p(1) << endl;
```

## accumulate

头文件 `<numeric>`

`std::accumulate(first, last, init);`
- `first` ：累加范围的起始迭代器
- `last` ：累加范围的结束迭代器
- `init` ：累加的初始值，结果会累加到这个初始值上

`std::accumulate(first, last, init, binary_op);`
- `binary_op` ：一个函数对象或 lambda 表达式，自定义的二元累加操作

```cpp
template<typename InputIt,  typename Tp,  typename BinaryOp>
Tp accumulate(InputIt first,  InputIt last, Tp init, BinaryOp binary_op) {
    for( ; first != last; ++first)
        init = binary_op(init,  *first);
    return init;
}
```

## STL 的函数对象类模板

- 如 `equal_to`  `greater`  `less` 
- 根据函数对象参数个数，STL算法⽤到的主要有三类基类（0/1/2个参数）

```cpp
template<class T>
struct greater : public binary_function<T, T, bool> {
    bool operator()(const T& x, constT& y) const {
        return x > y; // 降序排列
    }
};
```

## sort

list 的 sort
```cpp
void sort(); // 升序排列
void sort(Compare comp); // 自定义排列
```

通用的 sort （需要支持随机访问）
```cpp
#include <algorithm>

std::sort(first, last);
std::sort(first, last, comp);
```

自定义的 comp
```cpp
struct comp {
    bool operator()( constT & a1, const T& a2) {
        // 若a1应该在a2前⾯，则返回true
        // 否则返回false
    }
};
```

- 关联容器要排序应该在声明时设置，如 `map<T1, T2, greater<int>>`

