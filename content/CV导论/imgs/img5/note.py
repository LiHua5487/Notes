# =====================================================================================================


# this is just a py note, not a program
### python之禅 ###
import this # simply run this line


# =====================================================================================================


### 输入输出 ###
# print()
msg = "hello world"
print(msg)
# 注：print()自带换行，调整结尾用print(msg, end="")

#input(prompt)
msg = input("please input your message: ") # 默认读入字符串
num = int(input("please input a number: ")) # 读入整数


# =====================================================================================================


### 字符串 ###

# 声明
# py中''与""均可表示字符串
message = "tHis iS a tEst"

# 大小写替换（暂时修改）
print(message.title()) # 每个单词首字母大写其余小写
print(message.upper()) # 全都大写
print(message.lower()) # 全都小写
print(message.capitalize()) # 第一个字符大写，其余小写
print(message.swapcase()) # 大小写互换

# 去除空格（暂时修改）
message = "   3<-space num->4    "
print(message.rstrip()) # 去除左端的空格
print(message.lstrip()) # 去除右端的空格
print(message.strip()) # 去除首尾空格

# 去除前后缀（暂时修改）
message = "https://www.bilibili.com"
print(message.removeprefix("https://")) # 删除前缀
print(message.removeprefix("https://www.").removesuffix(".com")) # 删除前后缀

# 计数
message = "strawberry"
print(message.count("r")) 

# 替换（暂时修改）
message = "iron_pickaxe diamond_sword"
print(message.replace("iron", "diamond")) 

# f字符串(format)
first = "Don"
last = "Quixote"
full = f"{first} balabala {last}" # 在字符串中插入变量
print(full)
print(f"My name is {full}")
print(f"Parts of Name:\n\tfirst:{first}\n\tlast:{last}")

# 注意单双引号混合
# "this isn't wrong" is okay
# 'this isn't right' is not okay 
person = "lu xun"
print(f"{person} said: \"I didn't say this!\"") # 使用\"避免歧义


# =====================================================================================================


### 数 ###
a, b, c = 3, 2, 1.0 # 多个变量同时赋值
d2 = 0b1001001 # 2进制
d8 = 0o1234567 # 8进制
d16 = 0xccb # 16进制
print(a ** b) # 乘方a^b
print(a/b) # 除法结果为浮点数
print(a + c) # 带浮点数运算，结果为浮点数 

big_num = 123_456_789 # 可用_分隔数字
print(big_num) # 打印时不带_

# 注1：py没有常量类型，可用全大写变量名代表常量
# 注2：py没有++运算符，但有+=


# =====================================================================================================


### 列表 ###
# 有序，可以重新赋值、增删元素

# 声明 list()转换
numbers = [1, 2, 3, 4, 5]

# 读取
print(numbers[-1]) # 取倒数第1个
print(len(numbers))
print(max(numbers))
print(min(numbers))
print(sum(numbers))

# 增删元素
numbers.append(6) # 在末尾添加6
numbers.insert(0, -1) # 在索引0位置插入-1
del numbers[0] # 删除索引0的元素
numbers.remove(3) #按照变量值删除元素（只删除第一个为该值的元素）

# 弹出元素
popped_num = numbers.pop() # 弹出最后一个元素并将其赋给popped_num
print(popped_num) 
print(numbers.pop(0)) # 弹出索引0的元素并将其输出

# 排序
numbers = [3, 5, 1, 4, 2]
numbers.sort() # 升序排列（永久修改）
numbers.sort(reverse=True) # 降序排列

print(sorted(numbers)) # 升序排列（暂时修改）
print(sorted(numbers, reverse=True)) # 降序排列

# 反转（永久修改）
numbers.reverse() # 反转

# 清空
numbers.clear()
# or simply numbers = []

# 遍历
for num in numbers:
    print(num)

# range(start, stop, step)函数
# 默认从0开始，步长为1
for num in range(1, 10): # 从1到9，到10停止所以不会执行
    print(num)
numbers = list(range(1, 10, 2)) # 生成1-9的奇数的列表
# 注意range()函数返回的是一个迭代器，需要转换为列表

# 列表推导式
numbers = [num**2 for num in range(5)] # 生成0-4的平方的列表
numbers = [num for num in range(10) if num % 3 == 0] # 生成0-9的3的倍数的列表

numbers=[int(num) for num in input("please input a series of numbers:").split(',')]
# split()函数分割成列表，默认为字符串
print(f"the numbers are {numbers}")

# 列表切片
numbers = [0, 1, 2, 3, 4]
print(numbers[0:3]) # 取索引0-2（到索引3停止所以不执行）
print(numbers[:3]) # 取前3个（即索引0-2）
print(numbers[2:]) # 取索引2到末尾
print(numbers[-3:]) # 取后3个

# 复制列表
origin = [1, 2, 3]
same = origin # same与origin指向同一列表
copy = origin[:] # 将origin的内容复制到另一个列表copy


# =====================================================================================================


### 元组 ###
# 有序，不可重新赋值、增删元素，但可以重新赋值整个元组

# 声明 tuple()转换
numbers = (1, 2, 3, 4, 5)

# 其余可用方法与列表类似


# =====================================================================================================


### 集合 ###
# 无序，不可重新赋值元素，可以增删元素，元素不重复

# 声明 set()转换
my_set = {1, 2, 3}


# =====================================================================================================


### if ###

banned = ["Steve", "Alex"]
if banned: # 列表非空
    if "Dream" not in banned:
        banned.append("Dream")
else:
    print("this list is empty")

age = 18
if age >= 18 and age < 65:
    print("You can vote!")
elif age >= 0:
    print("hey kid!")
else:
    print("what are you!")


# =====================================================================================================


### 字典 ###

# 声明
alien = {
    'color': 'green', 
    'points': 5 # add a ',' here is okay
    }

# 读取
print(alien['color'])
print(alien.items()) # 所有键值对
print(alien.keys()) # 所有键
# 注：.items()和.keys()返回的是遍历器

# 增删
alien['speed'] = 1
del alien['color']

# get(key, default)访问
favourite = {
    'Bob': 5,
    'Carmen': 114514,
    'Alice': 1,
    'Dream': 114514
    }

name = 'Alice'
if favourite.get(name, 'NAN') != 'NAN': 
    print(f"{name}'s favourite number is {favourite[name]}")
else:
    print(f'{name} is not in list')

# 遍历(for循环)
for pair in favourite.items(): # 遍历键值对
    print(pair)
for key, value in favourite.items(): # 将键和值分别赋予
    print(f"{key}'s favourite number is {value}")
for key in favourite.keys(): # 遍历键
    print(f"{key} is included")
for value in favourite.values(): # 遍历值
    print(f"{value} is included")

for name, num in sorted(favourite.items()): # 按顺序遍历
    print(f"{name}'s favourite number is {num}")

for num in set(favourite.values()): # 用集合(set)去重
    print(f"{num} is included")

# 注：for variable in dict: 是遍历键，但不安全
my_dict = {'a': 1, 'b': 2, 'c': 3}

# 错误示例：遍历过程中直接修改字典会导致 RuntimeError
# for key in my_dict:
#     if key == 'b':
#         del my_dict[key]

# 正确示例：使用 keys() 方法修改字典
for key in list(my_dict.keys()):
    if key == 'b':
        del my_dict[key]
print(my_dict)  # 输出: {'a': 1, 'c': 3}


# 排序
my_dict = {"b": 3, "a": 1, "c": 2}

sorted_dict_by_value = dict(sorted(my_dict.items(), key=lambda x: x[0])) # 按键排
sorted_dict_by_value = dict(sorted(my_dict.items(), key=lambda x: x[1])) # 按值排
sorted_dict = dict(sorted(my_dict.items(), key=lambda x: (x[1], x[0]))) # 先按值，再按键


# =====================================================================================================


### while循环 ###
# break, continue

# flag
prompt = "please input your message"
prompt += "\ninput 'quit' to stop. your message: " 
msg = ""
flag = True
while flag:
    msg = input(prompt)
    if msg.lower() == "quit":
        flag = False
    else:
        print(msg)

# 处理列表 
waiting_usrs = input("input a series of names: ").split()
checked_usrs = []
while waiting_usrs: # 当waiting_usrs非空
    current_usr = waiting_usrs.pop(0)
    if current_usr.title() != current_usr:
        print(f"{current_usr}: format invalid, do you mean {current_usr.title()}?")
    checked_usrs.append(current_usr)    
# 示例输入输出：
# input a series of names: Alex BoB Carmen drEaM
# BoB: format invalid, do you mean Bob?
# drEaM: format invalid, do you mean Dream?

food = ['pizza', 'beef', 'shit', 'cake', 'shit', 'pasta', 'shit']
while 'shit' in food: # 当food列表含有'shit'
    food.remove('shit') # 移除首次出现的'shit'
# 注：variable in list 是遍历，const in list 是条件


# =====================================================================================================


### 函数 ###

# 基本格式
def greet(name):
    """say Hello to someone""" # 文档字符串，描述函数功能
    print(f"Hello, {name.title()}!")

greet("Dream")

# 传递参数
def give(name, item="nothing"): # 默认值
    """print giving message"""
    print(f"{name.title()} got {item}.")

give("Dream", "a book") # 位置实参，按序对应
give(name="Steve", item="a diamond") # 关键字实参，顺序随便
# 注：指定默认值、传递关键字实参时，等号两边约定不加空格

# 可选实参（空默认值）
def format(first, last, middle=""):
    """return formatted name"""
    if middle:
        full = f"{first} {middle} {last}"
    else:
        full = f"{first} {last}"
    return full.title()

print(format("doN", "QuixoTE"))
print(format("Harry", "Potter", "James"))

# 传递列表
def check_format(waiting_usrs, checked_usrs=[]):
    """check the format of name(s)"""
    while waiting_usrs:
        current_usr = waiting_usrs.pop(0)
        if current_usr != current_usr.title():
            print(f"{current_usr}: format invalid, do you mean {current_usr.title()}?")
        checked_usrs.append(current_usr)

waiting_usrs = ['Alex', 'BoB', 'Carmen', 'drEaM']
check_format(waiting_usrs[:]) # 传递副本，不会修改原列表
print(waiting_usrs)
check_format(waiting_usrs) # 传递原列表，会修改
print(waiting_usrs)

# 传递数量未知的实参
def make_pizza(size, *toppings): # 创建一个名为toppings的元组接收后续实参
    """print the recipe of the pizza you want to make."""
    print(f"\nMaking a {size}-inch pizza with the following toppings:")
    for topping in toppings:
        print(f"-{topping}")

make_pizza(12, 'mushroom', 'cheese', 'beef', 'green peppers')

def build_profile_dict(first, last, **info): # 创建一个名为info的字典接收后续关键字实参，关键字视为字符串
    """build a dict containing the info of a person"""
    # info创建时已经包含后续参数，所以只需添加first与last
    info['first_name'] = first.title()
    info['last_name'] = last.title()

print(build_profile_dict('albert', 'einstein', field='physics', location='princeton'))

# 储存并导入模块
# folder
# ⌊__ module.py
#       |def function_1():
#       |   """this is a function"""
#       |   print("function_1 called")
#       |
#       |def function_2():
#       |   """this is another function"""
#       |   print("function_2 called")    
# 
# ⌊__ main.py
#       |# 导入模块
#       |import module
#       |module.function_1()
#       |module.function_2()
#       |
#       |# 导入模块中的函数
#       |from module import function_1, function_2
#       |# 导入所有函数：from module import *
#       |function_1()
#       |function_2()
#       |
#       |# 指定别名
#       |import module as m
#       |m.function_1()
#       |
#       |from module import function_1 as f1
#       |f1()


# =====================================================================================================


### 类 ###

# 声明
class Inventory:
    """just an Inventory"""

    def __init__(self):
        self.inventory = []
    
    def get_inventory(self):
        if self.inventory:
            print(self.inventory)
        else:
            print("inventory empty")
    
    def add_item(self, *items):
        for item in items:
            self.inventory.append(item)

class Mob:
    """just a Mob"""
    # 若在此处声明变量，则为所有实例共用

    def __init__(self, name, type, HP=20): # __init__()方法，创建实例时自动运行
        """initialization"""
        # self代表实例对象，这里为实例对象添加name属性
        # 注：若不包含"self."，声明的变量作用域限制在此函数
        self.name = name
        self.type = type
        self.HP = HP
        self.inventory = Inventory() # 内置属性，将类实例作为属性

    def intro(self):
        """print basic info"""
        # 通过self访问实例对象的属性，会自动传递
        print(f"{self.name} is a {self.type}, with {self.HP} HP")

    def say(self, message): 
        """print message"""
        print(f"{self.name}: {message}")

    def take_damage(self, damage): # 利用方法修改实例属性
        """take damage and update HP"""
        if damage <= 0:
            print(f"{self.name} didn't take any damage.")
        else:
            self.HP -= damage

        if self.HP <= 0:
            print(f"{self.name} dies.")

player0 = Mob("Steve", "player", 30)
player0.intro()

player0.say(f"i am {player0.name}")

player0.inventory.add_item("apple", "diamond")
player0.inventory.get_inventory()

# 继承
class Zombie(Mob):
    """just a Zombie"""

    def __init__(self, name, HP=20):
        """initialization"""
        super().__init__(name, "zombie", HP) # 使用super()调用父类方法
        self.target = "" # 子类独有的属性

    def say(self, message=""): # 重写父类方法
        """zombie"can't speak"""
        if message:
            print("a zombie can't speak")
        else:
            print(f"{self.name} the zombie: rararar")

zombie0 = Zombie("Lawson", 25)
zombie0.intro()

zombie0.say("Hello")
zombie0.say()    

zombie0.inventory.add_item("iron_sword", "leather_helmet")
zombie0.inventory.get_inventory()

zombie0.take_damage(30)


# 排序
class Student:
    def __init__(self, name, score):
        self.name = name
        self.score = score

students = [
    Student("Alice", 90),
    Student("Bob", 80),
    Student("Cindy", 95),
    Student("Eve", 70)
]
sorted_students = sorted(students, key=lambda student: student.score)

# 储存并导入类
# folder
# ⌊__ mob.py
#       |class Mob:
#       |   --snip--   
#       |
#       |class Another_class:
#       |   --snip--
#       | 
# ⌊__ zombie.py
#       |class Zombie(Mob):
#       |   --snip-- 
# 
# ⌊__ main.py
#       |# 导入模块
#       |import mob as m
#       |import zombie as z
#       |player0 = m.Mob("Steve", "player")
#       |zombie0 = z.Zombie("Lawson")
#       |
#       |# 导入模块中的类
#       |from module import Mob as M
#       |player0 = M("Steve", "player")
#       |
#       |# 导入所有类：from mob import *


# =====================================================================================================


### 处理文件 ###

from pathlib import Path # 导入Path类

file = Path('data.txt') # 创建Path实例指向文件
# 注：也可使用绝对路径 path = Path('D:/Files/Program/python/data.txt')

print(file.read_text()) # 读取所有内容
# 注：read_text读到文件结尾会返回一个空字符串，结果会多出一个空行，可用.rstrip()去除

# 读取各行
contents = file.read_text()
lines = contents.splitlines() # 用splitlines()按行分割为列表
pi_str = ''
for line in lines:
    pi_str += line.lstrip()
print(pi_str)

# 写入文件（只能写入字符串）
file.write_text('this is a writen line.') # 会覆盖原本内容

# 不覆盖原本内容，在末尾追加
with open('data.txt', mode='a') as file:
    file.write('\nthis is a new line.')
# with open('data.txt', 'a') as file 是一个上下文管理器，它会：
# 1.尝试打开 data.txt 文件，如果不存在，则自动创建
# 2.在追加模式下打开文件
# 3.创建一个文件对象 file
# 4.执行操作后，自动关闭文件


# =====================================================================================================


### 存储数据 ###

from pathlib import Path
import json # json格式支持字典、列表

file = Path('save.txt')
data = {
    "name": "John",
    "age": 30,
    "is_student": True,
    "hobbies": ["reading", "swimming"]
}

# 写入
json_str = json.dumps(data) # 转换为json格式，返回字符串
file.write_text(json_str)

# 读取
json_str = file.read_text()
data = json.loads(json_str) # 将json格式字符串转换为python对象
print(data)

# 另一种方式
with open('data.json', 'w') as file: # w: 覆写模式 
    json.dump(data, file) # 将data转换为json格式并写入file

with open('data.json', 'r') as file: # r: 读取模式
    data = json.load(file) # 将储存在文件中的json数据转换为python对象
    print(data)


# =====================================================================================================


### 异常 ###

# try-except-else
try:
    ans = int(input("input the first num: "))/int(input("input the second num: "))
except ValueError: # 若出现对应异常，则执行相应代码
    print("[error]: your input is not a number.") # 若使用pass可以进行静默
except ZeroDivisionError:
    print("[error]: can't divide by 0.")
else:
    print(ans)


from pathlib import Path

def count_words(file):
    """count the number of words in a file"""
    try:
        contents = file.read_text(encoding='utf-8')
    except FileNotFoundError as fne: # 指定异常别名
        print(f"{fne}, the file '{file}' not found.")
    else:
        words = contents.split()
        print(f"the file '{file}' has {len(words)} words.")

filepaths = ['data.txt', '404.txt']
for path in filepaths:
    file = Path(path)
    count_words(file)

# 常见异常类型
# NameError 名称错误：使用未定义的变量、类或函数名
# TypeError 类型错误：对数据进行不支持的操作，如 1 + '2'
# ValueError 值错误：传入参数类型正确，但值不符合要求，如int("abab") 
# IndexError 索引错误：索引超出范围
# KeyError 键错误：使用字典中不存在的键
# FileNotFoundError 文件未找到错误：尝试打开一个不存在的文件
# ZeroDivisionError 除零错误：用一个数除以零
# AttributeError 属性错误：访问一个对象不存在的属性或方法


# =====================================================================================================


### 测试（以pytest为例） ###

# 测试函数
# folder
# ⌊__ main.py # 待测
#       |def format_name(first, last, middle=''):
#       |   """a function to test"""
#       |   if middle:
#       |       full = f"{first} {middle} {last}"
#       |   else:
#       |       full = f"{first} {last}"
#       |   return full.title()   
#       |
# ⌊__ test_main.py # 以test_开头，cmd运行 pytest 以执行
#       |from main import format_name
#       |
#       |def test_first_last_name():
#       |   """test the function"""
#       |   formatted = format_name('doN', 'quiXote')
#       |   assert formatted == 'Don Quixote' # 断言，符合条件则通过测试
#       |
#       |def test_first_middle_last_name():
#       |   """test the function"""
#       |   formatted = format_name('wolfgang', 'mozart', 'amadeus')
#       |   assert formatted == 'Wolfgang Amadeus Mozart'

# 测试类同理

# 常用断言
# assert a == b
# assert a != b
# assert a (a == True)
# assert not a (a == False)
# assert element in list
# assert element not in list


# =====================================================================================================


## 补充
# isinstance
a = (1, 2, 3)
print(isinstance(a, tuple)) # 判断数据类型

# is 和 == 区别
# is: 是否指向同一内存 
# ==: 内容是否相同
# 示例 1: 相同的值，但不是同一对象
a = [1, 2, 3]
b = [1, 2, 3]

print(a == b)  # True，因为两个列表的值相等
print(a is b)  # False，因为 a 和 b 是不同的对象，地址不同

# 示例 2: 同一对象
x = [4, 5, 6]
y = x  # 将y设为x的引用
print(x == y)  # True，因为 x 和 y 的值相等
print(x is y)  # True，因为 x 和 y 是同一个对象，地址相同

# 对int ,float,complex, str,tuple类型的变量a和b，只需关注 a == b是否成立
# 对list,dict等其他类型的变量a和b，a == b和a is b的结果都需要关注

# 浮点数判等
import sys
def equal_float(a,b):
    return abs(a-b) <= sys.float_info.epsilon #最小浮点数间隔,32位机为 2e-16
# print(sys.float_info) # 相关信息
print(equal_float(1.0,1.0))

# 高精度浮点数计算
import decimal
from decimal import *
a = decimal.Decimal(98766)
b = decimal.Decimal("123.223232323432424244858484096781385731294")
print(b)
c = decimal.Decimal(a + b) # 默认精度缺省为小数点后面28位
print(c)
getcontext().prec = 50 # 设置精度为小数点后面50位
print(c) # 还是28位
c = decimal.Decimal(a + b) # 要重新算一下才变成50位
print(c)

# 输入输出重定向
import sys
f = open("t.txt","r")
g = open("d.txt","w")
sys.stdin = f
sys.stdout = g
s = input()
print(s)
f.close()
g.close()