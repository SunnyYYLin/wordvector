# 项目说明

## 项目概述

本项目基于连续词袋模型（CBOW）实现了Word2Vec的词向量训练。项目包括数据预处理、词频分析、文本清洗、词向量训练与模型评估等功能，适用于中英文数据。项目的主要目的是通过CBOW模型生成高质量的词向量，并对文本进行语义分析和可视化展示。

## 提交说明

1. 提交的模型文件与爬取的数据在`to_submit`文件夹下
2. 实验报告markdown与pdf版本均在`docs`文件夹下
3. 为保证能够上传，其余数据均未上传，如需测试请按`main.ipynb`思路从爬取数据开始
4. 如果需要单独测试模型性能，使用`Word2Vec.load(foldername)`可一并加载CBOW模型与词汇表

## 文件结构

```
├── LICENSE                      # 开源许可文件
├── README.md                    # 项目说明
├── data                         # 数据文件夹
│   ├── cn                       # 中文数据
│   │   ├── tokenized.txt        # 分词后的中文数据
│   │   └── washed.json          # 清洗后的中文数据
│   ├── cn_stopwords.txt         # 中文停用词
│   └── en                       # 英文数据
│       ├── tokenized.txt        # 分词后的英文数据
│       └── washed.json          # 清洗后的英文数据
├── docs                         # 文档与可视化结果
├── logs                         # 训练日志
├── main.ipynb                   # 主实验流程的 Jupyter Notebook
├── models                       # 模型定义和数据集处理
│   ├── cbow.py                  # CBOW 模型实现
│   ├── dataset.py               # 数据集加载与处理
│   ├── vocab.py                 # 词汇表管理
│   └── word2vec.py              # Word2Vec模型及其训练方法
├── news_crawler                 # 新闻爬虫相关代码
│   ├── crawler_requests.py      # 基于requests的爬虫实现
│   ├── news_crawler             # 基于Scrapy的爬虫实现
│   │   ├── items.py
│   │   ├── middlewares.py
│   │   ├── pipelines.py
│   │   ├── settings.py
│   │   └── spiders
│       └── news_spider.py       # 新闻爬虫的Spider定义
├── to_submit                    # 提交内容
│   ├── cn_baseline              # 中文基线模型保存
│   ├── dates.json               # 提取的日期信息
│   └── en_baseline              # 英文基线模型保存
├── utils                        # 辅助工具代码
│   ├── SimHei.ttf               # 中文字体，用于词云可视化
│   ├── cleaning.py              # 文本清洗工具
│   ├── date.py                  # 日期提取工具
│   ├── plot.py                  # 训练曲线绘制工具
│   ├── tokenization.py          # 中英文文本的分词工具
│   └── wordfreq_viz.py          # 词频可视化工具（词云）
└── weights                      # 模型权重保存
```

## 模块说明

### 1. `main.ipynb`
这是项目的主实验流程，展示了从数据加载、预处理、训练CBOW模型到模型评估的完整流程。该Notebook包含了代码和解释，是项目的核心实验脚本。

### 2. `models/`
- **`word2vec.py`**：包含Word2Vec类的定义，基于CBOW模型进行词向量的训练和推理。包括模型的训练、验证、最近邻查找等功能。
- **`cbow.py`**：实现了CBOW模型的定义，支持词袋的前向传播、损失计算和预测。
- **`vocab.py`**：管理词汇表，将词汇映射到索引，支持词频统计和词汇表保存与加载。
- **`dataset.py`**：处理数据集，生成训练所需的上下文-中心词对，支持数据的负采样与划分。

### 3. `news_crawler/`
- **`crawler_requests.py`**：基于`requests`实现的新闻爬虫，用于抓取在线新闻数据。
- **`news_spider.py`**：基于`Scrapy`的爬虫定义，抓取特定网站的新闻内容。

### 4. `utils/`
- **`wordfreq_viz.py`**：提供词频可视化工具，生成词云图。
- **`plot.py`**：从TensorBoard日志中提取并绘制训练曲线，展示模型的训练损失和准确率。
- **`cleaning.py`**：提供中英文文本清洗功能，统一标点符号，去除无关字符。
- **`date.py`**：从文本中提取日期信息。
- **`tokenization.py`**：分词工具，支持中文的`jieba`分词和英文的简单分词。

### 5. `logs/`
存储训练过程中的TensorBoard日志文件，记录模型训练的损失、准确率等信息。通过这些日志可以对模型进行调试和性能分析。

### 6. `weights/`
存储训练好的模型权重和词汇表文件。不同实验条件（如窗口大小、负样本数量、词汇表大小等）的模型会分别存储在不同的子文件夹中。

## 环境要求

1. 项目使用Python3.12，或许兼容更低的版本
2. 使用Conda安装环境：
```bash
conda env create -f environment.yml
```