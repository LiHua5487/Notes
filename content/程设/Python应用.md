# Beautiful Soup

`beautiful soup` 库用于在html文件中查找具体内容

在爬虫中，需要先通过网页的url获取其html内容，而后进行查找解析

先了解一下啥是html，一个简单的html结构单元长这样
```html
 <a href="www.sohu.com" id='mylink'>
    搜狐网
 </a>
```
这一整块被称作一个 `tag` ，可以理解为“功能区”
- `a` 是 tag 的名字，表示这个 tag 是啥类别的
- `<a>` 表示开头，`</a>` 表示结尾
- `href` `id` 是 tag 的属性 (attr) ，用于配置具体行为、提供附加数据或修改显示方式等，可以理解为“参数”
- `搜狐网` 是 tag 的正文，也就是具体的内容

下面是一个html文档
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>页面标题</title>
</head>
<body>
    <h1>一级标题</h1>
    <p>这是个段落</p>
    <a href="https://www.example.com">这是个链接</a>
</body>
</html>

```
有下面这些主要的组成部分

开头用`<!DOCTYPE html>` 声明这是个html类型的文档

整个文档的内容放在根元素 `<html>` 里面，其 `lang` 属性指定语言

头部 `<head>` 存放元数据，不显示出来，一般是一些基本信息
- `<meta charset="UTF-8">` 字符编码
- `<title>` 浏览器标签页标题
- `<style>` CSS样式，设置页面内容长啥样
- `<link>` 引入外部CSS/图标
- `<script>` JavaScript代码或引用

主体 `<body>` 存放显示出来的内容，比如

| 标签                    | 类型     | 示例                                      |
| --------------------- | ------ | --------------------------------------- |
| `<h1>...<h6>`         | 1~6级标题 | `<h1>一级标题</h1>`                         |
| `<p>`                 | 段落     | `<p>这是一个段落</p>`                         |
| `<a>`                 | 超链接    | `<a href="https://example.com">链接</a>`  |
| `<img>`               | 图片     | `<img src="img.jpg" alt="描述">`          |
| `<div>`               | 块级容器   | `<div>内容区块</div>`                       |
| `<span>`              | 行内容器   | `<span>行内样式</span>`                     |

对于一个网页的html，我们爬取时一般只关注特定tag的信息，但是如果手动去找太费劲了，可以使用`beautiful soup` 库 (bs4)

这是基本的使用流程
- 将html文档装入一个BeautifulSoup对象X  
- 用X对象的 `find` `find_all` 等成员函数去找想要的tag
- 对找到的tag，还可以在其内部进一步查找，如嵌套的tag，一些属性的内容

首先，需要安装并import
```python
>>> pip install beautifulsoup4
>>> import bs4
```

下面是一个简单的使用示例，其中html内容直接存在了一个string里
```python
str = '''
<div id="siteHeader" class="wrapper">
    <h1 class="logo"></h1>
    <div id="topsearch">
        <ul id="userMenu">
            <li><a href="http://openjudge.cn/">首页</a></li>
        </ul>
    </div>
</div>
'''

soup = bs4.BeautifulSoup(str,"html.parser") # 构建bs4对象
```
这里的 `'html.parser'` 是python内置的html解释器，指定如何去处理这个html

而后，使用成员函数去查找具体内容

`find` ：查找第一个符合的tag，返回一个对象，具体类型为 `bs4.element.Tag`

```python
# 查找第一个名字为 "ul" 的元素
ul = soup.find("ul")
print(ul)
# >>> <ul id="userMenu">
# >>> <li><a href="http://openjudge.cn/">首页</a></li>
# >>> </ul>

# 查找 id 为 "siteHeader" 的元素
header = soup.find(id="siteHeader")

# 查找 class 为 "logo" 的元素
logo = soup.find(class_="logo")  # 注意 class 后面有下划线
```

对于这个对象，还可以进一步获取其成员变量

| 方法        | 用途     | 示例                                                     |
| --------- | ------ | ------------------------------------------------------ |
| `.name`   | 获取标签名  | `soup.find("li").name → 'li'`                          |
| `['属性名']` | 获取属性值  | `soup.find("a")['href'] → 'http://openjudge.cn/'`      |
| `.get()`  | 安全获取属性 | `soup.find("div").get('id', 'default') → 'siteHeader'` |
| `.attrs`  | 获取所有属性 | `soup.find("h1").attrs → {'class': ['logo']}`          |
| `.text`   | 获取内容   | `soup.find("li").text → '首页'`                          |

这里补充一下 `get` 函数，假设 a 是个字典，那么 `a.get(key, default_val)` 会查找 a 的键 key 对应的值，如果没找到，就返回 `default_val` ，比 `[]` 访问更安全

`find_all` ：查找所有符合的tag，返回这些对象组成的列表，找不到就是空的

```python
# 查找所有包含 href 属性的 <a> 标签
links = soup.find_all('a', href=True)

# 查找所有 class 为 "menu" 的 <li> 标签
menu_items = soup.find_all('li', class_="menu")
```

还可以用 `limit` 参数指定最多找多少个

`select` ：通过 CSS 选择器查找所有符合的元素，返回列表，更高端，一般用于复杂查找

```python
# 查找所有 <h1> 标签
h1_tags = soup.select('h1')

# 查找 class 为 "logo" 的元素
logo = soup.select('.logo')

# 查找 id 为 "userMenu" 的元素
menu_items = soup.select('#userMenu')

# 查找所有包含 href 属性的 <a> 标签
links = soup.select('a[href]')
```

除此以外，还可以指定层级顺序

```python
# 查找所有 <div> 中的所有 <ul>
ul_tags = soup.select('div > ul')

# 查找 class 为 "active" 的 <li> 中的 <a>
active_link = soup.select('li.active > a')

# 查找 id 为 "userMenu" 的元素下的所有 <li>
menu_items = soup.select('#userMenu > li')
```

要查找属于多个 `class` 的tag，可以这样
```python
highlight_links = soup.select(".external-link.highlight")
```

---

这里放一道 gpt 出的题目（由于往年没有python部分所以只能让gpt出题了）

一个html文档内容如下
```html
<!DOCTYPE html>
<html>
    <head>
        <title>示例网页</title>
    </head>
    <body>
        <div class="content">
            <h1>文章标题</h1>
            <p id="intro">这是一段简介内容。</p>
            <p id="main-content">这里是主体内容，包含<b>加粗文本</b>。</p>
            <a href="https://example.com/page1" class="external-link">链接到页面1</a>
            <a href="https://example.com/page2" class="external-link highlight">链接到页面2</a>
        </div>
        <div class="footer">
            <p>版权所有 © 2023</p>
        </div>
    </body>
</html>
```

任务要求
- 使用 `find` 方法找到 h1 标题，并输出其内容
- 使用 `find_all` 方法获取所有 p 标签的内容，并输出成一个列表
- 使用 `select` 方法找到所有带有 `class="external-link"` 的超链接，提取它们的 href 属性，并输出网址列表
- 对于使用 `select` 方法找到的超链接，进一步筛选出 `class="highlight"` 的链接，并输出该链接的文本

---

那么怎么获取这个html呢？

假设有一个html文件，名为 `example.html` ，可以这么读取
```python
with open("example.html", "r", encoding="utf-8") as file:
    html_content = file.read()  # 读取文件内容为字符串

soup = BeautifulSoup(html_content, "html.parser")
```

或者如果内容是一串字符，也可以直接读取
```python
soup = bs4.BeautifulSoup(open("c:\\tmp\\test.html", "r", encoding="utf-8"), "html.parser")
```

但更多的时候，html要从网页获取，可以使用 `requests` 或 `pyppeteer` 库

由于这部分不考，而且比较固定，所以只提供对应的获取html的函数

`requests` ：爬取静态网页

```python
import requests

def getHtml(url):
    fakeHeaders = {
        'User-Agent':
           'Mozilla/5.0 (Windows NT 10.0; Win64; x64)  \
           AppleWebKit/537.36 (KHTML, like Gecko)  \
           Chrome/81.0.4044.138 Safari/537.36 Edg/81.0.416.77',
        'Accept': 'text/html,application/xhtml+xml,*/*'
    } # 伪造请求头，骗过简单的反爬机制
    try:
        r = requests.get(url, headers = fakeHeaders)
        r.encoding = r.apparent_encoding
        return r.text
    except Exception as e:
        print(e)
        return None
```

`pyppeteer` ：可爬动态网页

```python
import asyncio
import pyppeteer as pyp

def getHtmlByPyppeteer(url):
    async def asGetHtml(url): # 协程函数
        browser = await pyp.launch(
            executablePath = r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe", # 浏览器路径
            headless=False)
        page = await browser.newPage()
        await page.setUserAgent(
            'Mozilla/5.0 (Windows NT 6.1; Win64; \
            x64) AppleWebKit/537.36 (KHTML, like Gecko) \
            Chrome/78.0.3904.70 Safari/537.36')
        await page.evaluateOnNewDocument(
            '() =>{ Object.defineProperties(navigator, \
            { webdriver:{ get: () => false } }) }')
        await page.goto(url)
        text = await page.content()
        await browser.close()
        return text
    loop = asyncio.new_event_loop() # 创建事件循环
    asyncio.set_event_loop(loop) # 设为当前线程的事件循环
    html = loop.run_until_complete(asGetHtml(url))
    return html
```

启动一个浏览器Chromium，用浏览器装入网页，浏览器可以用无头模式 headless ，即隐藏模式启动，也可以显式启动

从浏览器可以获取网页源代码，若网页有javascript程序，获取到的是javascript被浏览器执行后的网页源代码

可以向浏览器发送命令，模拟用户在浏览器上键盘输入/鼠标点击等操作，让浏览器转到其它网页


# Pandas

`pandas` 库用于数据分析，尤其是对于二维表格的数据进行各种操作

其可以在二维表格上进行操作，需要`numpy`支持，如果有`openpyxl/xlrd/xlmt`支持，也可以读写`excel`文档

其中涉及到 `Series` 与 `DataFrame` 两个关键的类

`Series` ：单列数据，由索引和值组成，类似于字典

```python
import pandas as pd

# 从列表创建 Series
fruits = pd.Series(['苹果', '香蕉', '橙子', '葡萄'])
print(fruits)
# >>> 0 苹果
# >>> 1 香蕉
# >>> 2 橙子
# >>> 3 葡萄
# >>> dtype: object

# 创建带自定义索引的 Series
fruit_prices = pd.Series([5.5, 3.2, 4.0, 6.8], # 值
            index=['苹果', '香蕉', '橙子', '葡萄'], # 索引
            name='水果价格') # 序列名称
print(fruit_prices)
# >>> 苹果    5.5
# >>> 香蕉    3.2
# >>> 橙子    4.0
# >>> 葡萄    6.8
# >>> Name: 水果价格, dtype: float64
```

访问时，既支持按照索引名称访问，也支持按照位置顺序访问，切片同理
```python
fruit_prices['橙子'] # >>> 4.0
fruit_prices.loc['橙子'] # 按索引访问的另一种写法
fruit_prices[1] # >>> 3.2，不推荐，会警告
fruit_prices.iloc[1] # 推荐的做法

fruit_prices['苹果':'橙子'] # 苹果、香蕉、橙子
fruit_prices[0:2] # 苹果、香蕉
```
注意按索引切片是左闭右闭

与字典类似，Series包含 `index` 与 `value`
```python
fruit_prices.value[0] # >>> 5.5
fruit_prices.index[0] # >>> 苹果
```

增删元素
```python
fruit_prices['西瓜'] = 7.0 # 在尾部添加元素
fruit_prices.pop('橙子') # 删除元素
```

一些类似于 numpy 的操作
```python
# 向量化运算
discounted = fruit_prices + 1 # 所有值+1

# 条件筛选
cheap_fruits = fruit_prices[fruit_prices < 5]
```

以及一些其它的函数
- `Series.tolist()`：把值的部分转为列表，如果直接用 `list()` 也是这样
- `Series.sum()` `Series.min()` `Series.max()` `Series.mean()` `Series.median()`：和、最小值、最大值、平均值、中位数
- `Series.idxmax()`、`Series.argmax()`：最大元素的标签和下标
- `pd.concat()`：拼接两个Series，不改变前者

---

`DataFrame` ：带行列标签的二维表格，每一行/列都是一个Series

```python
# 从字典创建 DataFrame
data = {
    '水果': ['苹果', '香蕉', '橙子', '葡萄'],
    '价格': [5.5, 3.2, 4.0, 6.8],
    '库存': [120, 200, 150, 80]
}

fruit_df = pd.DataFrame(data)
print(fruit_df)
# >>>    水果   价格   库存
# >>> 0  苹果  5.5  120
# >>> 1  香蕉  3.2  200
# >>> 2  橙子  4.0  150
# >>> 3  葡萄  6.8   80
```

但是，输出的表格并不对齐，可以用这句话
```python
pd.set_option('display.unicode.east_asian_width',True)
```

这么设置后输出就变成了这样
```python
# 直接创建 DataFrame
df = pd.DataFrame(data=[['男',108,115,97],['女',115,87,105],['女',100,60,130],['男',112,80,50]], # 每行内容
            index = ['刘一','王二','张三','李四'], # 行索引
            columns = ['性别','语文','数学','英语']) # 列索引
print(df)
# >>>      性别  语文   数学  英语
# >>> 刘一   男   108   115    97
# >>> 王二   女   115    87   105
# >>> 张三   女   100    60   130
# >>> 李四   男   112    80    50
```

DataFrame 包含 `value` (2D) 和 `index` `columns` (1D) 

可以通过行/列索引直接获取一行/一列，类型是 Series
```python
# 获取一列
print(df['数学'])
# >>> 刘一    115
# >>> 王二     87
# >>> 张三     60
# >>> 李四     80
# >>> Name: 数学, dtype: int64

# 获取一行
print(df.loc['李四']) # 必须用.loc
# >>> 性别     男
# >>> 语文    112
# >>> 数学     80
# >>> 英语     50
# >>> Name: 李四, dtype: object
```

以及各种成员函数，要用直接搜，考试提供参考文档，真的不想罗列了

---

一般来讲，表格是需要从文档读取的，读取后是 DataFrame 类型

对于 `.xlsx` 格式的表格，可以这么读取
```python
dt = pd.read_excel("sample.xlsx", sheet_name=['Sheet1', 1], index_col=0)
```
- `"sample.xlsx"` 即读取的文件
- `sheet_name=['Sheet1', 1]` 表示读取名字为 `Sheet1` 的表格，和 `index=1` 的（即第二个）表格
- `index_col=0` 指定表格中的第一列被用作行索引 

也可以将结果写入表格
```python
df.to_excel("result.xlsx", sheet_name="Sheet1")
```

要写入多个表格，需要用 `pd.ExcelWriter`
```python
writer = pd.ExcelWriter("new.xlsx")
df.to_excel(writer, sheet_name="Sheet1")
df.to_excel(writer, sheet_name="Sheet2")
writer.save()
```

还可以读写 `.csv` 格式的表格
```python
# 读
df = pd.read_csv("result.csv")

# 写
df.to_csv("result.csv", sep=",", na_rep='NA', float_format="%.2f", encoding="gbk")
```




