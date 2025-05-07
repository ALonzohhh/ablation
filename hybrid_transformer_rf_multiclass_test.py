#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
混合模型测试脚本
用于在测试集上评估模型性能
"""

import os
import numpy as np
import pandas as pd
import torch
import joblib
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize

from hybrid_transformer_rf_multiclass_with_validation import (
    HybridModel, NetworkTrafficDataset, prepare_transformer_input
)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def load_test_data():
    """加载测试数据"""
    print("正在加载测试数据...")
    from fastparquet import ParquetFile
    test_set = 'data/UNSW_NB15_testing-set.parquet'
    pf_test_set = ParquetFile(test_set)
    df_test = pf_test_set.to_pandas()
    
    # 提取特征和标签
    X_test = df_test.iloc[:, :43].copy()
    y_test_cat = df_test['attack_cat'].copy()
    
    # 删除无关特征
    if 'id' in X_test.columns:
        X_test = X_test.drop(['id'], axis=1)
    if 'attack_cat' in X_test.columns:
        X_test = X_test.drop(['attack_cat'], axis=1)
    
    # 加载预处理对象
    scaler = joblib.load('models/hybrid_transformer_rf_multiclass/scaler.joblib')
    label_encoder = joblib.load('models/hybrid_transformer_rf_multiclass/label_encoder.joblib')
    
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
    feature_columns = joblib.load('models/hybrid_transformer_rf_multiclass/feature_columns.joblib')
    for col in feature_columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[feature_columns]
    
    # 编码标签
    y_test_encoded = label_encoder.transform(y_test_cat)
    
    # 转换为numpy数组
    X_test_np = X_test.values.astype(np.float32)
    
    # 加载特征选择信息
    selected_features = joblib.load('models/hybrid_transformer_rf_multiclass/selected_features.joblib')
    
    # 创建测试数据集
    test_dataset = NetworkTrafficDataset(
        prepare_transformer_input(X_test_np, selected_features)['input_ids'],
        X_test_np[:, selected_features],
        y_test_encoded
    )
    
    # 创建数据加载器
    BATCH_SIZE = 128
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return test_loader, y_test_encoded, label_encoder

def evaluate_model(model, test_loader, label_encoder):
    """评估模型性能"""
    print("\n正在评估模型...")
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs_trans, inputs_rf, labels in tqdm(test_loader):
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
    
    with open('models/hybrid_transformer_rf_multiclass/test_metrics.txt', 'w') as f:
        for metric, value in test_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    # 生成混淆矩阵
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_mapping.values())
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title('混合模型测试集混淆矩阵')
    plt.xticks(rotation=43, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/hybrid_transformer_rf_multiclass/test_confusion_matrix.png')
    
    # 生成ROC曲线
    y_score = np.array(all_probs)
    y_test_bin = label_binarize(all_labels, classes=range(len(class_mapping)))
    
    plt.figure(figsize=(10, 8))
    for i in range(len(class_mapping)):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_mapping[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('多分类ROC曲线 (测试集)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/hybrid_transformer_rf_multiclass/test_roc_curves.png')
    
    return test_metrics

def main():
    try:
        # 加载测试数据
        test_loader, y_test_encoded, label_encoder = load_test_data()
        
        # 加载模型
        print("\n正在加载模型...")
        num_classes = len(label_encoder.classes_)
        model = HybridModel(num_classes=num_classes)
        model.load_state_dict(torch.load('models/hybrid_transformer_rf_multiclass/hybrid_model_best.pth'))
        
        # 训练随机森林模型
        print("\n正在训练随机森林模型...")
        from fastparquet import ParquetFile
        train_set = 'data/UNSW_NB15_training-set-new.parquet'
        pf_train_set = ParquetFile(train_set)
        df_train = pf_train_set.to_pandas()
        
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
        scaler = joblib.load('models/hybrid_transformer_rf_multiclass/scaler.joblib')
        selected_features = joblib.load('models/hybrid_transformer_rf_multiclass/selected_features.joblib')
        
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