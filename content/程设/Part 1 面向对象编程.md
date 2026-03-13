# 类和对象

## 构造函数

- 构造函数可以设为 private （单例模式，工厂模式）
- 类的成员函数可以访问同类的其他对象的`private`成员

**单例模式**：确保一个类只有一个实例，提供全局访问点获取该实例，常用于数据库连接、配置管理、日志系统等

```cpp
class Singleton {
private:
    // 1. 私有构造函数（防止外部创建）
    Singleton() {}
    // 2. 私有静态实例
    static Singleton* instance;
public:
    // 3. 删除拷贝构造函数和赋值操作
    Singleton(const Singleton&) = delete;
    void operator=(const Singleton&) = delete;
    // 4. 全局访问点
    static Singleton* getInstance() {
        if (!instance) {
            instance = new Singleton();
        }
        return instance;
    }
    void showMessage() {
        cout << "I'm the one and only Singleton!" << endl;
    }
};
// 初始化静态成员
Singleton* Singleton::instance = nullptr;
int main() {
    // 获取单例对象
    Singleton* s1 = Singleton::getInstance();
    s1->showMessage();
    // 再次获取 - 仍是同一个对象
    Singleton* s2 = Singleton::getInstance();
    // 验证是同一个实例
    if (s1 == s2) {
        cout << "same instance" << endl;
    }
    return 0;
}
```

**工厂模式**：根据传入的参数不同创建不同的产品对象

```cpp
// 产品接口
class Shape {
public:
    virtual void draw() = 0;
    virtual ~Shape() = default;
};
// 具体产品
class Circle : public Shape {
public:
    void draw() override {
        std::cout << "Drawing a Circle" << std::endl;
    }
};
class Rectangle : public Shape {
public:
    void draw() override {
        std::cout << "Drawing a Rectangle" << std::endl;
    }
};

// 简单工厂
class ShapeFactory {
public:
    // 静态工厂方法
    static std::unique_ptr<Shape> createShape(const std::string& type) {
        if (type == "Circle") {
            return std::make_unique<Circle>();
        } else if (type == "Rectangle") {
            return std::make_unique<Rectangle>();
        }
        return nullptr;
    }
};
int main() {
    // 使用工厂创建对象
    auto circle = ShapeFactory::createShape("Circle");
    auto rect = ShapeFactory::createShape("Rectangle");
    circle->draw(); // Drawing a Circle
    rect->draw();   // Drawing a Rectangle
    return 0;
}
```

### 复制构造函数

```cpp
class A {
private:
    char* data;
public:
    // 构造函数
    A(const char* str = "") {
        data = new char[strlen(str) + 1];
        strcpy(data, str);
    }

    // 复制构造函数（深拷贝）
    A(const A& other) {
        data = new char[strlen(other.data) + 1];
        strcpy(data, other.data);
    }

    // 析构函数
    ~A() {
        delete[] data;
    }
};
```
- 以 `A& a` 为参数
- 调用时机
 1. 用类实例初始化时
```cpp
A a2(a1);
A a2 = a1;
``` 
- 赋值语句 `a2 = a1;` 不会调用

2. 作为函数参数
```cpp
void func(A a) {}
func(a);
```

3. 作为返回值
```cpp
A createObject() {
    A temp;
    return temp;
}
createObject();
```
- `return` 执行后，`temp` 销毁，返回复制出来的临时变量

- 复制构造函数接收 `const A&` ，这样支持临时对象

### 类型转换构造函数

1. 隐式，和正常构造函数一样
```cpp
class A {
    int val;
public:
    A(int n): val(n) {}
};
A a = num;
```

2. 显示，添加 `explicit` 关键字
```cpp
class A {
    int val;
public:
    explicit A(int n): val(n) {}
};
A a = A(num);
```

## this 指针

- 指向类实例自身
- 调用非静态成员函数会隐式的作为参数传入，类似于 python 的 `self`
- 不会实际存在于类实例的内存空间里，本质上只是一个传入的参数
-  `*this` 表示类实例自身的引用 

## static

静态成员变量
- 所有类实例共享，通过类名直接访问
- 在类内定义，但要在外边初始化

```cpp
class MyClass {
public:
    static int staticVar;
};

int MyClass::staticVar = 0; // 通过类名访问，推荐
obj.publicStatic = 0; // 通过对象访问
ptr->publicStatic = 0; // 通过对象的指针访问
```

注： C++17 开始允许类内初始化带 inline 的静态变量

```cpp
class MyClass {
public:
    inline static int staticVar = 100;
};
```

静态成员函数 `static void func() {}`
- 在类内声明，可以在类内或内外实现，通过类名直接调用而不用实例化
- 不传入 `this` 指针，所以不能调用非静态成员变量与函数
- 所有类实例共享，本质上是全局的

## const

`const` 指针与引用
- 不能通过 `const` 指针与引用修改变量，但可以通过其它方式修改
- 不能将 `const` 指针/引用赋值给非 `const` 指针/引用，但反过来可以

```cpp
int a = 10;
const int* ptr = &a; // 错误，不能通过 ptr 修改 a 的值
a = 5; // 通过变量名 a 直接修改，这是合法的
```


使用引用传参，由于不是传入副本，能避免不必要的拷贝
常引用 `const T&` 可以用来接收临时变量， `void func(const T& a)` 除了接收左值，还可以接受临时对象（右值）作为参数

引用可以用来返回类的成员变量的引用
```cpp
class MyClass {
    int value;
public:
    int& getValueRef() {
        return value;
    }
};
```

初始化 const 成员与引用类型的成员时，要在初始化列表进行
```cpp
class A{
private:
    const int num;
    int& ref;
public:
    A(int n, int& r): num(n),  ref(r) {}                    
};
```

`const` 函数 `void func() const {}`
- 不能修改非静态的成员变量、不能调用非 `const` 成员函数，但可以调用静态成员函数
- 用 `mutable` 前缀修饰，则可以修改这个成员变量
- `const` 放在前面，意思就变成了返回值类型为常量
- 对于同名同参数的两个函数，一个是 `const` ，一个不是，属于重载

`const` 对象
- 成员变量不能被修改
- 只能使用构造函数、析构函数、`const` 函数

## 友元

- 可以访问该类的私有成员
- 类内实现的友元函数，本质上是全局函数

## 继承与派生 

- 派生类转换为基类时，会调用复制构造函数
- protected 成员对于自身及自身的派生类可见，对于外部不可见，只能通过基类或派生类的方法间接访问
	- 派生类可以访问自身的基类的 protected 成员，但对于其它基类就不行
	- 基类不能访问派生类的 protected 成员

构造顺序
- 派生类：先基类，再派生类
- 封闭类：先包含的类，再自身

## 多态与虚函数

虚函数：成员函数前面加 `virtual` 关键字
- 只用在类内声明写，实现函数体时不用
- 基类必须写，派生类可以不用
- 基类与派生类的虚函数同名同参数
- 静态成员函数不能作为虚函数，因为没有 `this` 指针

在构造函数或析构函数中调用虚函数时，不会表现出多态行为
- 在对象的构造函数执行过程中，派生类的构造函数尚未被完全执行，派生类部分的虚表还未初始化
- 在对象的析构函数执行过程中，派生类部分已经被销毁
- 因此，即使调用虚函数，执行的也都是基类的版本

析构函数可以是虚函数，但构造函数不行

多态
- 派生类指针/引用可以赋给基类指针/引用，用基类指针/引用调用基类/派生类对象，会调用相应的虚函数
- 无论调用路径如何（直接调用/通过成员函数间接调用），只要通过基类指针或引用访问虚函数，且虚函数被重写，就会触发多态

```cpp
class Base {
public:
    void show() { bar(); }
    virtual void bar() { std::cout << "Base::bar\n"; }
};

class Derived : public Base {
public:
    void bar() override { std::cout << "Derived::bar\n"; }
};

int main() {
    Base* obj = new Derived(); // 基类指针指向派生类对象
    obj->show(); // 输出 "Derived::bar"
    delete obj;
    return 0;
}
```

其中 `override` 显式地指定派生类中的成员函数重写基类中的虚函数

与覆盖的区别
- 覆盖是对于派生类和基类的同名函数，使用不同类型的指针调用时会执行各自的函数
- 如果没有声明为虚函数，即使使用指向派生类对象的基类指针，调用的仍然是基类的函数

动态联编：程序实际运行时才能确定执行哪段代码

虚函数表 
- 虚表结构
    - 每个包含虚函数的类有一个虚表，存储该类所有虚函数的地址
    - 派生类继承基类的虚表，并可以覆盖重写的虚函数地址
- 对象内存布局
	- 每个对象内存开头含一个指向这个类的虚表的指针 `vptr`
	- 调用虚函数时，通过 `vptr` 找到虚表，再根据函数偏移量定位具体实现

纯虚函数：一种特殊的虚函数，函数体被赋值为 `0`

```cpp
class Base {
public:
    virtual void pureVirtualFunc() = 0;
};
```

- 纯虚函数没有具体的实现（在基类中没有定义函数体），它只是规定了函数的接口，要求派生类必须根据自己的情况来实现这个函数
- 如果一个类包含纯虚函数，那么这个类就是抽象类
- 不能创建抽象类的对象
- 析构函数可以声明为纯虚函数，但此时必须在派生类提供具体实现，基类作为抽象类

# 运算符重载

- 重载运算符 `()` `[]` `->` `=` 时，运算符重载函数必须声明为类的成员函数
- 不允许操作数都不是对象

## 重载基本运算符

```cpp
class MyString {
    char *p;
public:
    MyString(const char *s = "") {
        if (s) { // 确保非空再初始化
            p = new char[strlen(s) + 1];
            strcpy(p, s);
        } else {
            p = nullptr;
        }
    }

    ~MyString() {
        if (p)
            delete[] p;
    }

    MyString(const MyString &other) {
        if (other.p) {
            p = new char[strlen(other.p) + 1];
            strcpy(p, other.p);
        } else {
            p = nullptr;
        }
    }

    MyString& operator=(const MyString &other) {
        if (this == &other) // 防止自赋值
            return *this;
        if (p) // 如果非空，就先清空
            delete[] p;
        if (other.p) {
            p = new char[strlen(other.p) + 1];
            strcpy(p, other.p);
        } else {
            p = nullptr;
        }
        return *this;
    }

    MyString operator+(const MyString &other) const {
        int len1 = p ? strlen(p) : 0;
        int len2 = other.p ? strlen(other.p) : 0;
        char *newStr = new char[len1 + len2 + 1];
        if (p)
            strcpy(newStr, p);
        else
            newStr[0] = '\0';
        if (other.p)
            strcat(newStr, other.p);
        MyString result(newStr);
        delete[] newStr;
        return result;

    }

    MyString& operator+=(const char *s) {
        *this = *this + MyString(s);
        return *this;
    }

    char& operator[](int index) {
        return p[index];
    }

    friend istream& operator>>(istream& is, MyString& s) {
        char buffer[1024]; // 临时缓冲区（假设最大长度为1023）
        is >> buffer;
        s = MyString(buffer);
        return is;
    }

    friend ostream& operator<<(ostream &os, const MyString &s) {
        if (s.p)
            os << s.p;
        return os;
    }
};
```

返回对象与引用的区别
- 返回对象
	- 生成新对象作为结果，不修改操作数本身
		- 算术运算符：`+`, `-`, `*`, `/`, `%`
		- 关系运算符：`==`, `!=`, `<`, `>`, `<=`, `>=`
		- 逻辑运算符：`&&`, `||`, `!`
	- 需要返回临时对象
		- 后置++/- -（先创建临时对象，再自增/自减，返回临时对象）
- 返回引用
	- 支持链式调用/修改自身
		- 赋值运算符 `=`
		- 复合赋值运算符：`+=`, `-=`, `*=`, `/=` 等
		- 前置++/- -
		- 输入输出流运算符：`<<` , `>>`
- 特殊情况
```cpp
class Number {
    double value;
public:
    Number(double v) : value(v) {}
    // 返回引用，直接修改第一个操作数
    Number& operator+=(const Number& other) {
        this->value += other.value;
        return *this;
    }
    // 使用 operator+= 实现 operator+
    Number& operator+(const Number& other) {
        return *this += other;
    }
};

int main() {
    Number a(1), b(2), c(3), d(4);
    a + b + c + d;  // 等价于：((a += b) += c) += d
    std::cout << a.value << std::endl; // a 的值被改变，结果为 10
    return 0;

}
```
- 出现 `a + b + c + d;` ，等价于 `((a += b) += c) += d`
- 不建议直接重载 `+`，因为会破坏其原有功能

## 重载强制类型转换

```cpp
class MyInt {
private:
    int value;
public:
    MyInt(int v) : value(v) {}
    operator int() const {
        return value;
    }
};
MyInt myObj(42);
int num = myObj;
```

## 重载自增自减

```cpp
class Counter {
private:
    int value;
public:
    Counter(int v = 0) : value(v) {}
    
    // 重载前置自增运算符 (++obj)
    Counter& operator++() {
        ++value; // 先增加值
        return *this; // 返回自身
    }

    // 重载后置自增运算符 (obj++)
    Counter operator++(int) { // 为了区分前置后置，要写一个 int
        Counter temp = *this; // 保存当前值
        value++;              // 后增加值
        return temp;          // 返回旧值
    }
};
```

## 重载 ()

```cpp
class Multiplier {
private:
    int factor;
public:
    Multiplier(int f) : factor(f) {}
    int operator()(int x) const {
        return x * factor;
    }
};
Multiplier times2(2); // 创建一个乘以 2 的函数对象
int n = times2(10);
```
- 允许对象可以像函数一样被使用















