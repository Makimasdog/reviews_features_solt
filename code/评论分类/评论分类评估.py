# -*- coding: utf-8 -*-
"""
评论分类评估模块
用于评估评论分类模型的性能
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


class ClassificationEvaluator:
    """
    分类评估器类
    用于评估分类模型性能并生成可视化报告
    """
    
    def __init__(self, true_labels, predicted_labels, label_names=None):
        """
        初始化评估器
        
        Args:
            true_labels: 真实标签列表
            predicted_labels: 预测标签列表
            label_names: 标签名称列表
        """
        self.true_labels = true_labels
        self.predicted_labels = predicted_labels
        self.label_names = label_names
        
        # 计算评估指标
        self.accuracy = accuracy_score(true_labels, predicted_labels)
        self.report = classification_report(true_labels, predicted_labels, output_dict=True)
        self.conf_matrix = confusion_matrix(true_labels, predicted_labels)
        
        # 如果没有提供标签名称，则从报告中获取
        if self.label_names is None:
            self.label_names = list(self.report.keys())
            # 移除非类别标签
            for item in ['accuracy', 'macro avg', 'weighted avg']:
                if item in self.label_names:
                    self.label_names.remove(item)
    
    def print_summary(self):
        """
        打印评估摘要
        """
        print(f"准确率: {self.accuracy:.4f}")
        print("\n分类报告:")
        print(classification_report(self.true_labels, self.predicted_labels))
    
    def plot_confusion_matrix(self, save_path=None):
        """
        绘制混淆矩阵
        
        Args:
            save_path: 图表保存路径
        """
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.conf_matrix, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.label_names, yticklabels=self.label_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_classification_metrics(self, save_path=None):
        """
        绘制分类指标图表
        
        Args:
            save_path: 图表保存路径
        """
        # 提取每个类别的精确率、召回率和F1值
        categories = []
        precision = []
        recall = []
        f1_score = []
        
        for category in self.label_names:
            if category in self.report:
                categories.append(category)
                precision.append(self.report[category]['precision'])
                recall.append(self.report[category]['recall'])
                f1_score.append(self.report[category]['f1-score'])
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        x = np.arange(len(categories))
        width = 0.25
        
        plt.bar(x - width, precision, width, label='精确率')
        plt.bar(x, recall, width, label='召回率')
        plt.bar(x + width, f1_score, width, label='F1值')
        
        plt.xlabel('类别')
        plt.ylabel('分数')
        plt.title('分类指标对比')
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def plot_category_distribution(self, save_path=None):
        """
        绘制类别分布图
        
        Args:
            save_path: 图表保存路径
        """
        # 统计每个类别的数量
        true_counts = pd.Series(self.true_labels).value_counts().sort_index()
        pred_counts = pd.Series(self.predicted_labels).value_counts().sort_index()
        
        # 创建DataFrame
        df = pd.DataFrame({
            '真实分布': true_counts,
            '预测分布': pred_counts
        })
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        df.plot(kind='bar', figsize=(12, 6))
        plt.xlabel('类别')
        plt.ylabel('数量')
        plt.title('类别分布对比')
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def generate_report(self, output_dir):
        """
        生成完整评估报告
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存图表
        self.plot_confusion_matrix(os.path.join(output_dir, '混淆矩阵.png'))
        self.plot_classification_metrics(os.path.join(output_dir, '分类指标对比.png'))
        self.plot_category_distribution(os.path.join(output_dir, '类别分布对比.png'))
        
        # 保存评估指标到CSV
        metrics_df = pd.DataFrame(self.report).transpose()
        metrics_df.to_csv(os.path.join(output_dir, '评估指标.csv'))
        
        # 生成摘要报告
        with open(os.path.join(output_dir, '评估摘要.txt'), 'w', encoding='utf-8') as f:
            f.write(f"准确率: {self.accuracy:.4f}\n\n")
            f.write("分类报告:\n")
            f.write(classification_report(self.true_labels, self.predicted_labels))


def main():
    """
    主函数
    """
    # 示例数据
    true_labels = [
        "功能请求", "功能请求", "错误报告", "用户体验", "信息查询",
        "功能请求", "错误报告", "错误报告", "用户体验", "其他"
    ]
    
    predicted_labels = [
        "功能请求", "功能请求", "错误报告", "用户体验", "用户体验",
        "功能请求", "错误报告", "用户体验", "用户体验", "其他"
    ]
    
    # 创建评估器
    evaluator = ClassificationEvaluator(true_labels, predicted_labels)
    
    # 打印摘要
    evaluator.print_summary()
    
    # 绘制图表
    evaluator.plot_confusion_matrix()
    evaluator.plot_classification_metrics()
    evaluator.plot_category_distribution()
    
    # 生成报告
    evaluator.generate_report('evaluation_results')


if __name__ == "__main__":
    main()