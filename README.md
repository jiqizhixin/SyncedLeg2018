# SyncedLeg2018
[![npm](https://img.shields.io/npm/l/express.svg)]()
[![npm](https://img.shields.io/badge/python-3.x-blue.svg)]()
	

机器之腿是源于机器之心内部 Hackathon 之后产品化的成果，可以基于微信历史文章与相应的流量数据、分析统计出热点词汇。

# Get Started

- 安装依赖

```shell
pip install -r requirements.txt
```
- 路径设置

```python
dict_dir = '../dict'  # 词典目录
stop_words_path = '../dict/stop_words.dat'  # 停用词文件路径
excel_path = '../input/posts.xls'  # 输入 excel 路径
selected_data_save_path = '../input/select_data2018.xls'  # 筛选后的输入数据路径文件名(具体到文件名哦), 可为 None
output_dir = '../output'  # 保存热词结果 excel 的目录
comment = "20181029"  # 热词结果文件名的添加的前缀
topN = 10000  # 保留前 n 个热词
```

- 进行数据筛选(select_param)和参数选择(param_grid).
- 执行下列命令，运行脚本
```python
python Analyser.py
```


# 运行说明

### 项目结构说明

[code](code) 为该项目的源代码。

[dict](dict) 目录存放**词典文件**和**停用词文件**. 
 - 词典文件用来协助 jieba 进行分词以及来给热词结果添加标签，某个热词添加的标签为改词所属的词典文件名,因此请注意词典文件名的设置。
 - 停用词文件后缀设置为'.dat',以方便与词典文件区分。
 
[input](input) 目录存放输入文件。
 
[output](output) 存放生成的结果。
 

### 输入文件格式要求

输入的 excel 文件应当放在 input 目录中，并包含以下字段：
- **title**
- **content**
- **readNum**
- **likeNum**
- **follower** (optional)

### 数据筛选

通过修改 `select_param` 变量的取值完成数据的筛选。目前包含5个维度的数据筛选：
- **data_range**: 起始日期和结束日期。例如 \['2018-01-01', '2018-10-11'\]
- **msgIdx**: 例如 \[2, 3, 4, 5\]只保留数据中 msgIdx 字段为2，3,4和5的值。
- **sourceUrl**: 例如\[True\]只保留该字段不为空的数据，\[True, False\]则不做筛选。
- **readNum_range**: 阅读量的范围。
- **likeNum_range**: 点赞量的范围。

```python
select_param = {
    'date_range': ['2018-01-01', '2018-10-11'],  # 起始日期,结束日期
    'msgIdx': [2, 3, 4, 5, 6],  # 文章位置(例如2,3,4表示非头条文章)
    'sourceUrl': [True, False],  # 有无sourceUrl
    'readNum_range': [0, 4000000],  # 阅读数范围
    'likeNum_range': [0, 2000000]  # 点赞数范围
}
```

### 参数选择

通过修改 `param_grid` 完成参数的选择。
```python
param_grid = {
        'like_weight': [0.6, 0.8, 1.0],  # 在计算热度的时候，点赞量的权重；阅读量的权重为(1-like_weight)
        'title_weight': [0.6, 0.8, 1.0],  # 在计算文章关键字的权重时，标题关键字的权重
        'cut_method': ['JTextRank', 'tdidf'],  # 计算关键字权重的算法，支持'JTextRank'和'tfidf'
        'hot_method': ['avg', 'sum'],  # 同一热词出现在多篇文章的时候，采用 'sum' 或 'avg'
        'normalize_rd_lk': [True, False],  # 是否对阅读数，点赞数的归一化
        'normalize_title_content': [True, False],  # 对标题，全文关键字权重的归一化
        'follower': [True, False],  # 是否有 follower
        'remove': ['公告 ', '活动 ', '报名 '] # 如果关键字出现在标题中，则剔除(建议后面加空格)
    }
```
上述参数设置会产生3\*3\*2\*2\*2\*2\*2种参数组合。

经过实验，下列是较为理想的参数组合, 会产生4中不同的结果。
```python
param_grid = {
        'like_weight': [0.8],
        'title_weight': [0.8, 1.0],
        'cut_method': ['JTextRank', 'tdidf'],
        'hot_method': ['avg'],
        'normalize_rd_lk': [True],
        'normalize_title_content': [True],
        'follower': [True]  # 根据实际情况，如果数据有 follower 字段则设置为True
    }
```

### 标签添加

- 需要将所有的词典文件放置在 `dict` 目录下，并以 .txt 作为后缀名。词典一是用来方便更准确地切词，二是后续为热词添加标签，热词的标签为该热词所属词典的文件名前缀。例如'julia'出现在'tech.txt'中，那么julia会被标记'tech'标签。
- **新增的字典**文件只要放在`dict`目录下即可。
- **停用词文件**也放在`dict`目录下，建议以 .dat 作为停用词的文件后缀，以示区分。
- 如果词典存在**优先级**，则可以在词典文件名添加数字表示优先级，数字越大优先级越高, 例如'5tech.txt', '4org.txt', 则前者优先级高。

# 算法设计

### 文章热度计算

提供follower数据时，文章热度的计算公式为：

![公式](https://latex.codecogs.com/svg.latex?\Large&space;article\\_hot=\frac{readNum}{follower}*(1-like\\_weight)+\frac{likeNum}{follower}*like\\_weight)

未提供follower数据时，文章热度的计算公式为：

![](https://latex.codecogs.com/svg.latex?\Large&space;article\\_hot=readNum*(1-like\\_weight)+likeNum*like\\_weight)


### 文章关键词热度计算

使用JTextRank/tfidf计算文章关键词的权重 weight，然后使用如下公式计算关键词的热度：

![](https://latex.codecogs.com/svg.latex?\Large&space;word\\_hot=article\\_hot*weight)


### 标题全文关键词权重的计算

标题、全文均使用 jieba.cut 进行切词，关键词权重的计算可以选用`JTextRank`和`tfidf`. 根据经验，JTextRank算法保留归一化前分数 >=1 的词. 而 tfidf 算法保留归一化前分数 >=0.03 的词。

上述步骤完成后，会分别得到标题关键词的权重以及全文关键词的权重，接着通过title\_weight作为权重对两者关键字权重进行合并。

### 同一关键字在多篇文章重复出现情况的处理

提供3种方法：
- sum: 对所有文章中该关键词的词热进行累加。
- avg: 对所有文章中该关键词的词热进行平均。
- medium: 对所有文章中该关键词的词热取中位数。


### 归一化方法

#### 阅读/点赞数归一化

normalize_rd_lk = True,使用高斯分布对阅读/点赞数进行拟合,得到0~1的归一化后的数值:

![](https://latex.codecogs.com/svg.latex?\Large&space;\int_{-\infty}^{x}\frac{1}{\sqrt{2\pi}\sigma}\exp\\{-\frac{(x-\mu)^2}{2\sigma^2}\\}dx)

其中x为文章的阅读数/点赞数。

由于考虑到阅读/点赞数对文章热度的贡献更大，因此在计算的时候采用下面的方式计算关键词的热度：

![](https://latex.codecogs.com/svg.latex?\Large&space;word\\_hot=e^{10*article\\_hot}*word\\_weight)


#### 标题/全文关键字权重归一化

使用 softmax 对标题/全文的关键字权重进行归一化。

# Contributors
Chao Wen, VXenomac, JJ Weng, Mos Zhang, Chain Zhang

# License
[MIT](http://opensource.org/licenses/MIT)

Copyright (c) 2018-present, Synced

