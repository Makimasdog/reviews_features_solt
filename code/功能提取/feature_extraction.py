# -*- coding: utf-8 -*-
"""
特征提取模块
用于从用户评论中提取应用功能点
"""

import re
import jieba
import jieba.posseg as pseg
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 加载自定义词典
jieba.load_userdict('user_dict.txt')

# 功能相关词汇
FEATURE_KEYWORDS = [
    '功能', '特性', '选项', '设置', '模式', '工具', '操作',
    '添加', '增加', '支持', '希望', '建议', '能够', '应该',
    '同步', '分享', '导出', '导入', '上传', '下载', '备份',
    '界面', 'UI', '按钮', '菜单', '页面', '窗口', '显示',
    '播放', '暂停', '快进', '倒退', '搜索', '查找', '筛选',
    '通知', '提醒', '消息', '推送', '提示', '警告', '更新'
]


def preprocess_text(text):
    """
    文本预处理函数
    
    Args:
        text: 输入文本
        
    Returns:
        处理后的文本
    """
    if not isinstance(text, str):
        return ""
    
    # 去除特殊字符和数字
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z]', ' ', text)
    
    return text


def extract_features_rule_based(text):
    """
    基于规则的特征提取
    
    Args:
        text: 输入文本
        
    Returns:
        提取的特征列表
    """
    features = []
    
    # 预处理文本
    text = preprocess_text(text)
    
    # 词性标注
    words = pseg.cut(text)
    
    # 提取名词和动词
    nouns = []
    verbs = []
    for word, flag in words:
        if flag.startswith('n') and len(word) > 1:
            nouns.append(word)
        elif flag.startswith('v') and len(word) > 1:
            verbs.append(word)
    
    # 基于关键词的特征提取
    for keyword in FEATURE_KEYWORDS:
        if keyword in text:
            # 查找关键词周围的名词
            keyword_index = text.find(keyword)
            for noun in nouns:
                noun_index = text.find(noun)
                # 如果名词在关键词附近，认为是一个特征
                if abs(noun_index - keyword_index) < 10 and noun != keyword:
                    feature = noun + keyword
                    features.append(feature)
    
    # 基于动词-名词搭配的特征提取
    for verb in verbs:
        for noun in nouns:
            if text.find(verb) < text.find(noun) and text.find(noun) - text.find(verb) < 5:
                feature = verb + noun
                features.append(feature)
    
    return list(set(features))


def extract_features_tfidf(reviews, top_n=20):
    """
    基于TF-IDF的特征提取
    
    Args:
        reviews: 评论列表
        top_n: 返回前N个特征
        
    Returns:
        提取的特征列表
    """
    # 预处理文本
    processed_reviews = [preprocess_text(review) for review in reviews]
    
    # 分词
    segmented_reviews = [' '.join(jieba.cut(review)) for review in processed_reviews]
    
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=100)
    tfidf_matrix = vectorizer.fit_transform(segmented_reviews)
    
    # 获取特征名称
    feature_names = vectorizer.get_feature_names_out()
    
    # 计算每个特征的平均TF-IDF值
    avg_tfidf = np.array(tfidf_matrix.mean(axis=0)).flatten()
    
    # 获取前N个特征
    top_indices = avg_tfidf.argsort()[-top_n:][::-1]
    top_features = [feature_names[i] for i in top_indices]
    
    return top_features


def cluster_reviews_by_features(reviews, n_clusters=5):
    """
    基于特征聚类评论
    
    Args:
        reviews: 评论列表
        n_clusters: 聚类数量
        
    Returns:
        聚类结果和每个聚类的关键特征
    """
    # 预处理文本
    processed_reviews = [preprocess_text(review) for review in reviews]
    
    # 分词
    segmented_reviews = [' '.join(jieba.cut(review)) for review in processed_reviews]
    
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(segmented_reviews)
    
    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    # 获取每个聚类的关键特征
    feature_names = vectorizer.get_feature_names_out()
    cluster_features = {}
    
    for i in range(n_clusters):
        # 获取该聚类的所有评论索引
        cluster_indices = np.where(clusters == i)[0]
        
        # 如果该聚类没有评论，跳过
        if len(cluster_indices) == 0:
            continue
        
        # 计算该聚类中每个特征的平均TF-IDF值
        cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
        cluster_tfidf = np.array(cluster_tfidf).flatten()
        
        # 获取前10个特征
        top_indices = cluster_tfidf.argsort()[-10:][::-1]
        top_features = [feature_names[j] for j in top_indices]
        
        cluster_features[i] = {
            'size': len(cluster_indices),
            'features': top_features,
            'reviews': [reviews[j] for j in cluster_indices[:5]]  # 示例评论
        }
    
    return clusters, cluster_features


def main():
    """
    主函数
    """
    # 示例评论
    reviews = [
        "希望能增加夜间模式功能，晚上看太刺眼了",
        "界面设计很精美，但是缺少暗黑模式",
        "经常崩溃，无法正常使用，希望开发者尽快修复",
        "希望能支持文件导出为PDF格式",
        "能否添加自动保存功能？经常忘记保存导致数据丢失"
    ]
    
    # 基于规则提取特征
    print("基于规则的特征提取:")
    for review in reviews:
        features = extract_features_rule_based(review)
        print(f"评论: {review}")
        print(f"特征: {features}\n")
    
    # 基于TF-IDF提取特征
    print("\n基于TF-IDF的特征提取:")
    top_features = extract_features_tfidf(reviews)
    print(f"Top特征: {top_features}")
    
    # 聚类
    print("\n基于特征的评论聚类:")
    clusters, cluster_features = cluster_reviews_by_features(reviews, n_clusters=2)
    for cluster_id, info in cluster_features.items():
        print(f"聚类 {cluster_id}:")
        print(f"大小: {info['size']}")
        print(f"特征: {info['features']}")
        print(f"示例评论: {info['reviews']}\n")


if __name__ == "__main__":
    main()