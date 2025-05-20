# -*- coding: utf-8 -*-
"""
评论分类模块
用于将用户评论分类为不同类型（功能请求、错误报告、用户体验等）
"""

import os
import pandas as pd
import numpy as np
import jieba
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import joblib

# 停用词列表
STOPWORDS_PATH = os.path.join(os.path.dirname(__file__), 'stopwords.txt')
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


class ReviewClassifier:
    """
    评论分类器类
    用于训练和预测评论类别
    """
    
    def __init__(self):
        """
        初始化分类器
        """
        self.pipeline = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=5000)),
            ('clf', SVC(kernel='linear', probability=True))
        ])
        self.label_encoder = LabelEncoder()
        self.classes = None
        
    def train(self, X, y):
        """
        训练分类器
        
        Args:
            X: 评论文本列表
            y: 评论类别列表
        """
        # 编码标签
        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_
        
        # 训练模型
        self.pipeline.fit(X, y_encoded)
        
    def predict(self, X):
        """
        预测评论类别
        
        Args:
            X: 评论文本列表
            
        Returns:
            预测的类别列表
        """
        y_pred = self.pipeline.predict(X)
        return self.label_encoder.inverse_transform(y_pred)
    
    def predict_proba(self, X):
        """
        预测评论属于各类别的概率
        
        Args:
            X: 评论文本列表
            
        Returns:
            概率矩阵
        """
        return self.pipeline.predict_proba(X)
    
    def save_model(self, model_path):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
        """
        model_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'classes': self.classes
        }
        joblib.dump(model_data, model_path)
    
    @classmethod
    def load_model(cls, model_path):
        """
        加载模型
        
        Args:
            model_path: 模型路径
            
        Returns:
            加载的分类器实例
        """
        model_data = joblib.load(model_path)
        classifier = cls()
        classifier.pipeline = model_data['pipeline']
        classifier.label_encoder = model_data['label_encoder']
        classifier.classes = model_data['classes']
        return classifier


def main():
    """
    主函数
    """
    # 示例数据
    data = {
        'review': [
            "希望能增加夜间模式功能",
            "经常崩溃，无法正常使用",
            "界面设计很精美，视觉效果出色",
            "如何设置通知提醒？",
            "谢谢开发者提供这么好的应用"
        ],
        'category': [
            "功能请求",
            "错误报告",
            "用户体验",
            "信息查询",
            "其他"
        ]
    }
    
    df = pd.DataFrame(data)
    
    # 预处理文本
    df['processed_text'] = df['review'].apply(preprocess_text)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['category'], test_size=0.2, random_state=42
    )
    
    # 训练分类器
    classifier = ReviewClassifier()
    classifier.train(X_train, y_train)
    
    # 预测
    y_pred = classifier.predict(X_test)
    
    # 评估
    print(classification_report(y_test, y_pred))
    
    # 保存模型
    os.makedirs('models', exist_ok=True)
    classifier.save_model('models/review_classifier.pkl')


if __name__ == "__main__":
    main()