#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.utils import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report
)
import joblib
import pandas as pd
from fastparquet import ParquetFile
from tqdm import tqdm

from models.hybrid_model import HybridModel
from data.dataset import NetworkTrafficDataset
from utils.data_utils import prepare_transformer_input, get_feature_importance
from utils.training_utils import EarlyStopping, FocalLoss
from visualization.visualization_utils import (
    plot_confusion_matrix, plot_roc_curves,
    plot_training_curves, plot_f1_scores
)
from config.config import *

def create_directories():
    """创建必要的目录"""
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(VIS_DIR, exist_ok=True)

def load_and_preprocess_data():
    """加载和预处理数据"""
    print("正在加载数据...")
    df_train = ParquetFile(TRAIN_SET).to_pandas()
    df_val = ParquetFile(VAL_SET).to_pandas()
    df_test = ParquetFile(TEST_SET).to_pandas()
    
    print(f"训练集大小: {df_train.shape}")
    print(f"验证集大小: {df_val.shape}")
    print(f"测试集大小: {df_test.shape}")
    
    # 提取特征和标签
    X_train = df_train.iloc[:, :43].copy()
    y_train_cat = df_train['attack_cat'].copy()
    X_val = df_val.iloc[:, :43].copy()
    y_val_cat = df_val['attack_cat'].copy()
    X_test = df_test.iloc[:, :43].copy()
    y_test_cat = df_test['attack_cat'].copy()
    
    # 删除无关特征
    for df in [X_train, X_val, X_test]:
        if 'id' in df.columns:
            df.drop(['id'], axis=1, inplace=True)
        if 'attack_cat' in df.columns:
            df.drop(['attack_cat'], axis=1, inplace=True)
    
    # 标准化数值特征
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=['float32', 'float64', 'int16', 'int32', 'int64', 'int8']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 保存标准化器
    joblib.dump(scaler, SCALER_PATH)
    
    # 处理分类特征
    categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
        X_val_cat = pd.get_dummies(X_val[categorical_cols], drop_first=True)
        X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
        
        # 确保验证集和测试集有相同的列
        for col in X_train_cat.columns:
            if col not in X_val_cat.columns:
                X_val_cat[col] = 0
            if col not in X_test_cat.columns:
                X_test_cat[col] = 0
        
        X_val_cat = X_val_cat[X_train_cat.columns]
        X_test_cat = X_test_cat[X_train_cat.columns]
        
        X_train = X_train.drop(categorical_cols, axis=1)
        X_val = X_val.drop(categorical_cols, axis=1)
        X_test = X_test.drop(categorical_cols, axis=1)
        
        X_train = pd.concat([X_train, X_train_cat], axis=1)
        X_val = pd.concat([X_val, X_val_cat], axis=1)
        X_test = pd.concat([X_test, X_test_cat], axis=1)
    
    # 保存特征列名
    joblib.dump(X_train.columns.tolist(), FEATURE_COLUMNS_PATH)
    
    # 编码标签
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_cat)
    y_val_encoded = label_encoder.transform(y_val_cat)
    y_test_encoded = label_encoder.transform(y_test_cat)
    
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    
    # 转换为numpy数组
    X_train_np = X_train.values.astype(np.float32)
    X_val_np = X_val.values.astype(np.float32)
    X_test_np = X_test.values.astype(np.float32)
    
    # 特征选择
    sorted_idx, importance = get_feature_importance(X_train_np, y_train_encoded)
    selected_features = sorted_idx[:50]
    joblib.dump(selected_features, SELECTED_FEATURES_PATH)
    
    # 准备Transformer输入
    train_trans = prepare_transformer_input(X_train_np, selected_features)
    val_trans = prepare_transformer_input(X_val_np, selected_features)
    test_trans = prepare_transformer_input(X_test_np, selected_features)
    
    # 创建数据集
    train_dataset = NetworkTrafficDataset(
        train_trans['input_ids'],
        X_train_np[:, selected_features],
        y_train_encoded
    )
    val_dataset = NetworkTrafficDataset(
        val_trans['input_ids'],
        X_val_np[:, selected_features],
        y_val_encoded
    )
    test_dataset = NetworkTrafficDataset(
        test_trans['input_ids'],
        X_test_np[:, selected_features],
        y_test_encoded
    )
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader, test_loader, label_encoder, X_train_np, selected_features, y_train_encoded

def train_model(train_loader, val_loader, label_encoder, X_train_np, selected_features, y_train_encoded):
    """训练模型"""
    # 初始化模型
    num_classes = len(label_encoder.classes_)
    model = HybridModel(num_classes=num_classes)
    
    # 如果有GPU，将模型移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 训练随机森林模型
    model.train_rf(X_train_np[:, selected_features], y_train_encoded)
    
    # 初始化损失函数和优化器
    criterion = FocalLoss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,
        T_mult=2,
        eta_min=1e-6
    )
    
    # 初始化早停
    early_stopping = EarlyStopping(
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=EARLY_STOPPING_MIN_DELTA
    )
    
    # 训练循环
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        # 训练阶段
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}'):
            inputs_trans, inputs_rf, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(inputs_trans, inputs_rf)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                inputs_trans, inputs_rf, labels = [b.to(device) for b in batch]
                outputs = model(inputs_trans, inputs_rf)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # 打印训练信息
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {correct/total*100:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_correct/val_total*100:.2f}%')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'模型已保存: 验证损失从 {best_val_loss:.4f} 改善到 {val_loss:.4f}')
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break
        
        # 更新学习率
        scheduler.step()
    
    # 绘制训练曲线
    plot_training_curves(train_losses, val_losses, TRAINING_CURVES_PATH)
    
    return model

def evaluate_model(model, test_loader, label_encoder):
    """评估模型"""
    model.eval()
    device = next(model.parameters()).device
    
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in test_loader:
            inputs_trans, inputs_rf, labels = [b.to(device) for b in batch]
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
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    try:
        # 创建必要的目录
        create_directories()
        
        # 加载和预处理数据
        train_loader, val_loader, test_loader, label_encoder, X_train_np, selected_features, y_train_encoded = load_and_preprocess_data()
        
        # 训练模型
        model = train_model(train_loader, val_loader, label_encoder, X_train_np, selected_features, y_train_encoded)
        
        # 评估模型
        metrics = evaluate_model(model, test_loader, label_encoder)
        
        # 保存评估指标
        with open(os.path.join(MODEL_DIR, 'test_metrics.txt'), 'w') as f:
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
        
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 