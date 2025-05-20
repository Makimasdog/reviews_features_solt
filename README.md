# 应用评论分析与特征提取工具集

这个项目是一个用于从用户评论中提取应用功能特征的工具集合，包含多种分析方法和模型。

## 项目结构

```
├── code/
│   ├── KEFE/                  # 关键特征提取模块
│   │   ├── feature_extraction.py    # 特征提取
│   │   ├── feature_identification.py # 特征识别
│   │   └── main.py                  # 主程序
│   ├── BTM+BST/               # 主题模型与情感分析
│   │   └── review_analyzer.py       # 评论分析器
│   ├── 评论分类/               # 评论分类模块
│   │   └── 评论分类.py              # 评论分类器
│   └── 聚类/                  # 评论聚类模块
│       ├── 聚类_Kmeans.py           # K-means聚类
│       └── 聚类_LDA.py              # LDA主题聚类
├── reviews.txt               # 示例评论数据
└── requirements.txt          # 项目依赖
```

## 功能模块

### KEFE (关键特征提取)

从应用描述和用户评论中提取关键功能特征，并进行重要性排序。

使用方法：

```bash
# 同时执行特征提取和识别
python code/KEFE/main.py --mode both --description "应用描述文本" --reviews reviews.txt
```

### BTM+BST (主题模型与情感分析)

使用双词主题模型(BTM)和基于双词的情感-主题模型(BST)对评论进行分析。

### 评论分类

将用户评论分类为不同类别，如功能请求、错误报告、用户体验等。

### 评论聚类

使用K-means或LDA对评论进行聚类，发现评论中的主要主题。

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用示例

```python
# 使用KEFE模块提取特征
from code.KEFE.feature_extraction import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features("这是一款功能强大的音乐播放器应用...")
print(features)
```

## 许可证

MIT