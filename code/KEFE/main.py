# -*- coding: utf-8 -*-
import os
import sys
import argparse

# 添加当前目录到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入特征提取和特征识别模块
from feature_extraction import FeatureExtractor
from feature_identification import FeatureIdentifier

def main():
    """
    主函数：演示如何使用特征提取和特征识别模块
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='应用特征提取和识别工具')
    parser.add_argument('--mode', type=str, choices=['extract', 'identify', 'both'], default='both',
                        help='运行模式: extract(仅提取), identify(仅识别), both(提取和识别)')
    parser.add_argument('--description', type=str, help='应用描述文本或包含描述的文件路径')
    parser.add_argument('--reviews', type=str, help='评论文件路径')
    parser.add_argument('--output', type=str, help='输出文件路径')
    parser.add_argument('--top_n', type=int, default=10, help='返回的特征数量')
    
    args = parser.parse_args()
    
    # 获取当前文件所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 停用词文件路径
    stopwords_file = os.path.join(current_dir, 'stopwords-master', 'cn_stopwords.txt')
    
    # 获取应用描述
    app_description = ""
    if args.description:
        if os.path.isfile(args.description):
            try:
                with open(args.description, 'r', encoding='utf-8') as f:
                    app_description = f.read()
            except Exception as e:
                print(f"读取描述文件失败: {e}")
                return
        else:
            app_description = args.description
    
    if not app_description:
        print("请提供应用描述文本或描述文件路径")
        return
    
    # 设置输出文件
    output_file = args.output if args.output else os.path.join(current_dir, 'feature_analysis_result.txt')
    
    # 根据模式执行相应功能
    if args.mode in ['extract', 'both']:
        # 特征提取
        print("\n执行特征提取...")
        extractor = FeatureExtractor(stopwords_file=stopwords_file)
        features = extractor.extract_features(app_description, top_n=args.top_n)
        
        print("\n提取的功能特征:")
        for i, feature in enumerate(features, 1):
            print(f"{i}. {feature}")
        
        # 保存提取结果
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("应用描述:\n")
            f.write(app_description)
            f.write("\n\n提取的功能特征:\n")
            for i, feature in enumerate(features, 1):
                f.write(f"{i}. {feature}\n")
        
        print(f"\n特征提取结果已保存到: {output_file}")
    
    if args.mode in ['identify', 'both']:
        # 特征识别
        print("\n执行特征识别...")
        identifier = FeatureIdentifier(stopwords_file=stopwords_file)
        
        # 处理应用数据
        key_features = identifier.process_app_data(
            app_description, 
            reviews_file=args.reviews, 
            top_n=args.top_n
        )
        
        print("\n识别的关键功能:")
        for i, (feature, score) in enumerate(key_features, 1):
            print(f"{i}. {feature} (重要性: {score:.4f})")
        
        # 保存识别结果
        with open(output_file, 'w' if args.mode == 'identify' else 'a', encoding='utf-8') as f:
            if args.mode == 'identify':
                f.write("应用描述:\n")
                f.write(app_description)
                f.write("\n\n")
            
            f.write("识别的关键功能:\n")
            for i, (feature, score) in enumerate(key_features, 1):
                f.write(f"{i}. {feature} (重要性: {score:.4f})\n")
        
        print(f"\n特征识别结果已保存到: {output_file}")

if __name__ == "__main__":
    main()