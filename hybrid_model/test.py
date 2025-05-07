#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import pandas as pd
import joblib
from fastparquet import ParquetFile
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
from sklearn.preprocessing import label_binarize

from models.hybrid_model import HybridModel
from data.dataset import NetworkTrafficDataset
from utils.data_utils import prepare_transformer_input
from visualization.visualization_utils import (
    plot_confusion_matrix, plot_roc_curves,
    plot_f1_scores
)
from config.config import *

def load_test_data():
    """加载测试数据"""
    print("正在加载测试数据...")
    df_test = ParquetFile(TEST_SET).to_pandas()
    
    # 提取特征和标签
    X_test = df_test.iloc[:, :43].copy()
    y_test_cat = df_test['attack_cat'].copy()
    
    # 删除无关特征
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    if 'attack_cat' in X_test.columns:
        X_test = X_test.drop(['attack_cat'], axis=1)
    
    # 加载预处理对象
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    
    # 标准化数值特征
    numeric_cols = X_test.select_dtypes(include=['float32', 'float64', 'int16', 'int32', 'int64', 'int8']).columns
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 处理分类特征
    categorical_cols = X_test.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
        X_test = X_test.drop(categorical_cols, axis=1)
        X_test = pd.concat([X_test, X_test_cat], axis=1)
    
    # 对齐特征列
    feature_columns = joblib.load(FEATURE_COLUMNS_PATH)
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_columns]
    
    # 编码标签
    y_test_encoded = label_encoder.transform(y_test_cat)
    
    # 转换为numpy数组
    X_test_np = X_test.values.astype(np.float32)
    
    # 加载特征选择信息
    selected_features = joblib.load(SELECTED_FEATURES_PATH)
    
    # 创建测试数据集
    test_dataset = NetworkTrafficDataset(
        prepare_transformer_input(X_test_np, selected_features)['input_ids'],
        X_test_np[:, selected_features],
        y_test_encoded
    )
    
    # 创建数据加载器
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return test_loader, y_test_encoded, label_encoder

def evaluate_model(model, test_loader, label_encoder):
    """评估模型性能"""
    print("\n正在评估模型...")
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs_trans, inputs_rf, labels in tqdm(test_loader):
            inputs_trans = inputs_trans.to(device)
            inputs_rf = inputs_rf.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs_trans, inputs_rf)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    print("\n测试集性能指标:")
    print(f"准确率: {accuracy:.4f}")
    print(f"精确率: {precision:.4f}")
    print(f"召回率: {recall:.4f}")
    print(f"F1分数: {f1:.4f}")
    
    # 保存评估指标
    test_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    with open(os.path.join(MODEL_DIR, 'test_metrics.txt'), 'w') as f:
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # 生成混淆矩阵
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    plot_confusion_matrix(
        all_labels, all_preds, class_mapping,
        '混合模型测试集混淆矩阵',
        CONFUSION_MATRIX_PATH
    )
    
    # 生成ROC曲线
    y_score = np.array(all_probs)
    y_test_bin = label_binarize(all_labels, classes=range(len(class_mapping)))
    plot_roc_curves(y_test_bin, y_score, class_mapping, ROC_CURVES_PATH)
    
    # 生成F1分数图
    report = classification_report(
        all_labels, all_preds,
        target_names=class_mapping.values(),
        output_dict=True
    )
    
    categories = []
    f1_scores = []
    for category, metrics in report.items():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            categories.append(category)
            f1_scores.append(metrics['f1-score'])
    
    plot_f1_scores(categories, f1_scores, F1_SCORES_PATH)
    
    return test_metrics

def main():
    try:
        # 加载测试数据
        test_loader, y_test_encoded, label_encoder = load_test_data()
        
        # 加载模型
        print("\n正在加载模型...")
        num_classes = len(label_encoder.classes_)
        model = HybridModel(num_classes=num_classes)
        model.load_state_dict(torch.load(MODEL_PATH))
        
        # 如果有GPU，将模型移到GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # 训练随机森林模型
        print("\n正在训练随机森林模型...")
        from fastparquet import ParquetFile
        df_train = ParquetFile(TRAIN_SET).to_pandas()
        
        # 准备训练数据
        X_train = df_train.iloc[:, :43].copy()
        y_train_cat = df_train['attack_cat'].copy()
        
        # 删除无关特征
        if 'id' in X_train.columns:
            X_train = X_train.drop(['id'], axis=1)
        if 'attack_cat' in X_train.columns:
            X_train = X_train.drop(['attack_cat'], axis=1)
        
        # 处理分类特征
        categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
        if len(categorical_cols) > 0:
            X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
            X_train = X_train.drop(categorical_cols, axis=1)
            X_train = pd.concat([X_train, X_train_cat], axis=1)
        
        # 加载预处理对象
        scaler = joblib.load(SCALER_PATH)
        selected_features = joblib.load(SELECTED_FEATURES_PATH)
        
        # 标准化数值特征
        numeric_cols = X_train.select_dtypes(include=['float32', 'float64', 'int16', 'int32', 'int64', 'int8']).columns
        X_train[numeric_cols] = scaler.transform(X_train[numeric_cols])
        
        # 转换为numpy数组并选择特征
        X_train_np = X_train.values.astype(np.float32)
        X_train_np = X_train_np[:, selected_features]
        
        # 编码标签
        y_train_encoded = label_encoder.transform(y_train_cat)
        
        # 训练随机森林
        model.train_rf(X_train_np, y_train_encoded)
        print("随机森林模型训练完成")
        
        # 评估模型
        test_metrics = evaluate_model(model, test_loader, label_encoder)
        
    except KeyboardInterrupt:
        print("\n测试被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 