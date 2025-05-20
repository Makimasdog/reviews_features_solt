# -*- coding: utf-8 -*-
"""
基于K-means的评论聚类模块
用于将评论按主题进行聚类分析
"""

import os
import jieba
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import re

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 停用词列表
STOPWORDS_PATH = 'stopwords.txt'
stopwords = set()
if os.path.exists(STOPWORDS_PATH):
    with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])


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
    
    # 分词
    words = jieba.cut(text)
    
    # 去除停用词
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    
    return ' '.join(filtered_words)


class ReviewClusterer:
    """
    评论聚类器类
    用于对评论进行主题聚类
    """
    
    def __init__(self, n_clusters=5, random_state=42):
        """
        初始化聚类器
        
        Args:
            n_clusters: 聚类数量
            random_state: 随机种子
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        self.cluster_keywords = None
        self.cluster_reviews = None
        
    def fit(self, reviews):
        """
        训练聚类模型
        
        Args:
            reviews: 评论文本列表
        """
        # 预处理文本
        processed_reviews = [preprocess_text(review) for review in reviews]
        
        # TF-IDF向量化
        tfidf_matrix = self.vectorizer.fit_transform(processed_reviews)
        
        # KMeans聚类
        self.kmeans.fit(tfidf_matrix)
        
        # 获取聚类结果
        clusters = self.kmeans.predict(tfidf_matrix)
        
        # 获取每个聚类的关键词
        self._extract_cluster_keywords(tfidf_matrix, clusters, reviews)
        
        return clusters
    
    def predict(self, reviews):
        """
        预测评论的聚类
        
        Args:
            reviews: 评论文本列表
            
        Returns:
            聚类结果
        """
        # 预处理文本
        processed_reviews = [preprocess_text(review) for review in reviews]
        
        # TF-IDF向量化
        tfidf_matrix = self.vectorizer.transform(processed_reviews)
        
        # 预测聚类
        clusters = self.kmeans.predict(tfidf_matrix)
        
        return clusters
    
    def _extract_cluster_keywords(self, tfidf_matrix, clusters, reviews, top_n=10):
        """
        提取每个聚类的关键词
        
        Args:
            tfidf_matrix: TF-IDF矩阵
            clusters: 聚类结果
            reviews: 原始评论
            top_n: 每个聚类提取的关键词数量
        """
        feature_names = self.vectorizer.get_feature_names_out()
        self.cluster_keywords = {}
        self.cluster_reviews = {}
        
        for i in range(self.n_clusters):
            # 获取该聚类的所有评论索引
            cluster_indices = np.where(clusters == i)[0]
            
            # 如果该聚类没有评论，跳过
            if len(cluster_indices) == 0:
                continue
            
            # 保存该聚类的评论
            self.cluster_reviews[i] = [reviews[j] for j in cluster_indices]
            
            # 计算该聚类中每个特征的平均TF-IDF值
            cluster_tfidf = tfidf_matrix[cluster_indices].mean(axis=0)
            cluster_tfidf = np.array(cluster_tfidf).flatten()
            
            # 获取前N个特征
            top_indices = cluster_tfidf.argsort()[-top_n:][::-1]
            top_keywords = [feature_names[j] for j in top_indices]
            
            self.cluster_keywords[i] = top_keywords
    
    def evaluate(self, tfidf_matrix, clusters):
        """
        评估聚类效果
        
        Args:
            tfidf_matrix: TF-IDF矩阵
            clusters: 聚类结果
            
        Returns:
            轮廓系数
        """
        silhouette_avg = silhouette_score(tfidf_matrix, clusters)
        return silhouette_avg
    
    def visualize_clusters(self, save_path=None):
        """
        可视化聚类结果
        
        Args:
            save_path: 图表保存路径
        """
        if not self.cluster_reviews:
            print("请先运行fit方法训练模型")
            return
        
        # 统计每个聚类的评论数量
        cluster_sizes = {i: len(reviews) for i, reviews in self.cluster_reviews.items()}
        
        # 绘制饼图
        plt.figure(figsize=(10, 6))
        labels = [f'主题{i+1}: {self.cluster_keywords[i][0]}' for i in cluster_sizes.keys()]
        sizes = list(cluster_sizes.values())
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('评论主题分布')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


def main():
    """
    主函数
    """
    # 示例评论
    reviews = [
        "界面设计很精美，视觉效果出色",
        "推荐算法很精准，总能找到我喜欢的内容",
        "视频播放时经常卡顿，影响观看体验",
        "希望能增加更多的个性化设置选项",
        "内容更新太慢，缺少最新资源",
        "会员价格太贵，性价比不高",
        "客服态度很好，解决问题很及时",
        "离线模式下无法使用大部分功能，很不方便",
        "广告太多，严重影响使用体验",
        "整体用户体验很好，功能布局合理",
        "最新版本严重卡顿，希望尽快修复",
        "内容质量参差不齐，希望加强审核",
        "界面太复杂了，找不到想要的功能",
        "使用体验非常流畅，操作简单直观",
        "希望能增加夜间模式功能",
        "经常崩溃，无法正常使用"
    ]
    
    # 创建聚类器
    clusterer = ReviewClusterer(n_clusters=4)
    
    # 预处理文本
    processed_reviews = [preprocess_text(review) for review in reviews]
    
    # TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(processed_reviews)
    
    # 聚类
    clusters = clusterer.fit(reviews)
    
    # 评估
    silhouette_avg = clusterer.evaluate(tfidf_matrix, clusters)
    print(f"轮廓系数: {silhouette_avg:.4f}")
    
    # 输出聚类结果
    for i, keywords in clusterer.cluster_keywords.items():
        print(f"\n主题 {i+1}:")
        print(f"关键词: {', '.join(keywords[:5])}")
        print(f"评论数: {len(clusterer.cluster_reviews[i])}")
        print(f"示例评论:")
        for review in clusterer.cluster_reviews[i][:3]:
            print(f"- {review}")
    
    # 可视化
    clusterer.visualize_clusters()


if __name__ == "__main__":
    main()