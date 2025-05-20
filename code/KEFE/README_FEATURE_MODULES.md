# 应用特征提取与识别模块

## 简介

本模块用于从应用描述和用户评论中提取和识别应用的功能特征，帮助开发者了解用户最关注的功能点。

## 模块组成

### 特征提取 (feature_extraction.py)

从应用描述中提取功能特征，使用了以下技术：

- 自然语言处理 (NLP) 技术进行文本分析
- 关键词提取算法 (TF-IDF, TextRank)
- 句法分析识别功能短语
- 语义理解模型提取功能描述

### 特征识别 (feature_identification.py)

从用户评论中识别关键功能，并与应用描述中的功能进行匹配，使用了以下技术：

- 文本分类模型判断评论是否与功能相关
- 特征匹配算法关联评论与功能
- 重要性评分算法对功能进行排序

## 工作流程

1. **特征提取**：从应用描述中提取功能特征列表
2. **评论处理**：分析用户评论，识别与功能相关的评论
3. **特征匹配**：将评论与功能特征进行匹配
4. **重要性评分**：根据匹配结果对功能特征进行重要性排序
5. **结果输出**：输出排序后的功能特征列表

## 技术细节

### 特征提取技术

- **关键词提取**：使用TF-IDF和TextRank算法提取关键词
- **短语提取**：使用句法分析提取名词短语和动词短语
- **功能识别**：使用规则和模型判断短语是否描述功能
- **语义聚类**：将相似功能合并

### 特征识别技术

- **评论分类**：使用分类模型判断评论是否与功能相关
- **特征匹配**：使用语义相似度算法匹配评论与功能
- **重要性计算**：综合考虑评论数量、情感倾向和匹配度

## 依赖库

- jieba: 中文分词
- numpy: 数值计算
- tensorflow: 深度学习框架
- pyltp: 语言技术平台

## 使用方法
### 使用命令行工具

项目提供了一个命令行工具 `main.py`，可以方便地使用特征提取和识别功能：

```bash
# 仅执行特征提取
python main.py --mode extract --description "这是一款功能强大的音乐播放器应用..."

# 仅执行特征识别
python main.py --mode identify --description "这是一款功能强大的音乐播放器应用..." --reviews reviews.txt

# 同时执行特征提取和识别
python main.py --mode both --description app_description.txt --reviews reviews.txt --output result.txt
```

参数说明：
- `--mode`：运行模式，可选 `extract`(仅提取)、`identify`(仅识别)、`both`(提取和识别)
- `--description`：应用描述文本或包含描述的文件路径
- `--reviews`：评论文件路径
- `--output`：输出文件路径
- `--top_n`：返回的特征数量，默认为10

### 在代码中使用

```python
# 特征提取
from feature_extraction import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract_features("这是一款功能强大的音乐播放器应用...")
print(features)

# 特征识别
from feature_identification import FeatureIdentifier

identifier = FeatureIdentifier()
key_features = identifier.process_app_data(
    "这是一款功能强大的音乐播放器应用...", 
    reviews_file="reviews.txt"
)
print(key_features)
```

## 输出示例

```
提取的功能特征:
1. 音乐播放
2. 歌单管理
3. 在线搜索
4. 歌词显示
5. 均衡器调节

识别的关键功能:
1. 音乐播放 (重要性: 0.8765)
2. 歌词显示 (重要性: 0.7654)
3. 在线搜索 (重要性: 0.6543)
4. 歌单管理 (重要性: 0.5432)
5. 均衡器调节 (重要性: 0.4321)
```