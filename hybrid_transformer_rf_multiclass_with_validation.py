#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基于Transformer和随机森林的混合多分类模型（使用验证集版本）
用于识别UNSW-NB15数据集中的9种攻击类型和1种正常流量
使用Transformer提取序列特征，随机森林提取统计特征，通过动态融合层进行融合
"""

# ### 导入必要的库
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
from tqdm import tqdm

# PyTorch相关导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Transformers相关导入
from transformers import BertModel, BertTokenizer

# sklearn相关导入
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.utils import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay,
    roc_curve, auc, roc_auc_score
)

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False 

# ### 创建目录
if not os.path.exists('models/hybrid_transformer_rf_multiclass'):
    os.makedirs('models/hybrid_transformer_rf_multiclass')

if not os.path.exists('visualizations/hybrid_transformer_rf_multiclass'):
    os.makedirs('visualizations/hybrid_transformer_rf_multiclass')

# ### 模型定义
class HybridModel(nn.Module):
    def __init__(self, num_classes, trans_dim=768, rf_dim=200):
        super().__init__()
        
        # Transformer模块
        self.transformer = BertModel.from_pretrained(
            'bert-base-chinese',
            output_hidden_states=True,  # 可以添加
            return_dict=True            # 可以添加
        )
        
        # 随机森林特征提取器
        self.rf = RandomForestClassifier(
            n_estimators=300,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42
        )
        self.rf_trained = False
        
        # 动态融合层
        self.fusion = FusionLayer(
            trans_dim=trans_dim,
            rf_dim=rf_dim,
            num_classes=num_classes,
            dropout=0.2
        )
    
    def train_rf(self, X, y):
        """训练随机森林模型"""
        print("正在训练随机森林模型...")
        self.rf.fit(X, y)
        self.rf_trained = True
        print("随机森林模型训练完成")
        
    def forward(self, x_trans, x_rf):
        # Transformer特征提取
        trans_out = self.transformer(x_trans)[0]  # [batch_size, seq_len, hidden_dim]
        trans_out = trans_out.mean(dim=1)  # [batch_size, hidden_dim]
        
        # 随机森林特征提取
        if not self.rf_trained:
            raise RuntimeError("随机森林模型尚未训练，请先调用train_rf方法")
        rf_prob = torch.tensor(self.rf.predict_proba(x_rf.cpu().numpy()), dtype=torch.float32).to(x_trans.device)
        
        # 特征融合
        output = self.fusion(trans_out, rf_prob)
        return output

class FusionLayer(nn.Module):
    def __init__(self, trans_dim, rf_dim, num_classes, dropout=0.3):
        super().__init__()
        self.trans_adjust = nn.Sequential(
            nn.Linear(trans_dim, trans_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(trans_dim//2, num_classes)
        )
        self.attention = nn.Sequential(
            nn.Linear(num_classes, num_classes//2),
            nn.ReLU(),
            nn.Linear(num_classes//2, 1)
        )
        self.fc = nn.Linear(num_classes * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, trans_out, rf_prob):
        # 调整Transformer输出的维度
        trans_out = self.trans_adjust(trans_out)
        # 计算注意力权重
        alpha = torch.sigmoid(self.attention(trans_out))
        # 特征融合
        fused = torch.cat([alpha * trans_out, (1-alpha) * rf_prob], dim=1)
        fused = self.dropout(fused)
        return self.fc(fused)

# ### 数据集类
class NetworkTrafficDataset(Dataset):
    def __init__(self, features_trans, features_rf, labels):
        self.features_trans = torch.tensor(features_trans, dtype=torch.long)
        self.features_rf = torch.tensor(features_rf, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features_trans)
    
    def __getitem__(self, idx):
        return self.features_trans[idx], self.features_rf[idx], self.labels[idx]

# ### 数据加载和预处理
print("正在加载数据...")
from fastparquet import ParquetFile
train_set = 'data/UNSW_NB15_training-set-new.parquet'
val_set = 'data/UNSW_NB15_validation-set.parquet'
test_set = 'data/UNSW_NB15_testing-set.parquet'

pf_train_set = ParquetFile(train_set)
pf_val_set = ParquetFile(val_set)
pf_test_set = ParquetFile(test_set)

df_train = pf_train_set.to_pandas()
df_val = pf_val_set.to_pandas()
df_test = pf_test_set.to_pandas()

# 显示数据集大小
print(f"训练集大小: {df_train.shape}")
print(f"验证集大小: {df_val.shape}")
print(f"测试集大小: {df_test.shape}")

# 提取特征和标签
print("\n正在准备数据...")
X_train = df_train.iloc[:, :43].copy()
y_train_cat = df_train['attack_cat'].copy()
X_val = df_val.iloc[:, :43].copy()
y_val_cat = df_val['attack_cat'].copy()
X_test = df_test.iloc[:, :43].copy()
y_test_cat = df_test['attack_cat'].copy()

# 删除无关特征
print("\n删除无关特征...")
if 'id' in X_train.columns:
    X_train = X_train.drop(['id'], axis=1)
    X_val = X_val.drop(['id'], axis=1)
    X_test = X_test.drop(['id'], axis=1)
    print("已删除'id'特征")

if 'attack_cat' in X_train.columns:
    X_train = X_train.drop(['attack_cat'], axis=1)
    X_val = X_val.drop(['attack_cat'], axis=1)
    X_test = X_test.drop(['attack_cat'], axis=1)
    print("已删除'attack_cat'特征")

# 特征预处理
print("\n正在处理特征...")

# 标准化数值特征
with tqdm(total=100, desc="标准化数值特征") as pbar:
    scaler = StandardScaler()
    numeric_cols = X_train.select_dtypes(include=['float32', 'float64', 'int16', 'int32', 'int64', 'int8']).columns
    X_train[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
    X_val[numeric_cols] = scaler.transform(X_val[numeric_cols])
    X_test[numeric_cols] = scaler.transform(X_test[numeric_cols])
    
    # 保存标准化器
    joblib.dump(scaler, 'models/hybrid_transformer_rf_multiclass/scaler.joblib')
    pbar.update(100)

# 处理分类特征
with tqdm(total=100, desc="处理分类特征") as pbar:
    categorical_cols = X_train.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
        X_val_cat = pd.get_dummies(X_val[categorical_cols], drop_first=True)
        X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
        
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
    pbar.update(100)
    # 保存特征列名
    joblib.dump(X_train.columns.tolist(), 'models/hybrid_transformer_rf_multiclass/feature_columns.joblib')

# 编码标签
with tqdm(total=100, desc="编码标签") as pbar:
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train_cat)
    y_val_encoded = label_encoder.transform(y_val_cat)
    y_test_encoded = label_encoder.transform(y_test_cat)
    
    joblib.dump(label_encoder, 'models/hybrid_transformer_rf_multiclass/label_encoder.joblib')
    
    class_mapping = {i: label for i, label in enumerate(label_encoder.classes_)}
    with open('models/hybrid_transformer_rf_multiclass/class_mapping.txt', 'w') as f:
        for idx, label in class_mapping.items():
            f.write(f"{idx}: {label}\n")
    
    print(f"类别数量: {len(label_encoder.classes_)}")
    print("类别映射:", class_mapping)
    pbar.update(100)

# 转换为numpy数组
X_train_np = X_train.values.astype(np.float32)
X_val_np = X_val.values.astype(np.float32)
X_test_np = X_test.values.astype(np.float32)

# 特征重要性排序
def get_feature_importance(X, y):
    """使用随机森林获取特征重要性"""
    print("正在计算特征重要性...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    return sorted_idx, importance

# 在数据预处理后添加特征选择
print("\n正在选择重要特征...")
sorted_idx, importance = get_feature_importance(X_train_np, y_train_encoded)
# 选择前30个最重要的特征
selected_features = sorted_idx[:50]
print(f"选择了前{len(selected_features)}个最重要的特征")

# 保存特征选择信息
joblib.dump(selected_features, 'models/hybrid_transformer_rf_multiclass/selected_features.joblib')

# 准备Transformer输入
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
max_length = 128
BATCH_SIZE = 500

def prepare_transformer_input(features, selected_features=None):
    """将数值特征转换为字符串格式，用于Transformer输入"""
    print("正在准备Transformer输入...")
    all_input_ids = []
    
    if selected_features is not None:
        features = features[:, selected_features]
    
    # 分批处理数据
    for i in tqdm(range(0, len(features), BATCH_SIZE), desc="处理数据批次"):
        batch = features[i:i+BATCH_SIZE]
        # 将数值特征转换为字符串，并用空格分隔
        texts = [' '.join(map(str, row)) for row in batch]
        # 对每个批次进行tokenization
        encoded = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        all_input_ids.append(encoded['input_ids'])
        # 及时释放内存
        del encoded
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 合并所有批次的输入
    return {'input_ids': torch.cat(all_input_ids, dim=0)}

# 创建数据加载器
TRAIN_BATCH_SIZE = 32
train_dataset = NetworkTrafficDataset(
    prepare_transformer_input(X_train_np, selected_features)['input_ids'],
    X_train_np[:, selected_features],  # 只使用选定的特征
    y_train_encoded
)
val_dataset = NetworkTrafficDataset(
    prepare_transformer_input(X_val_np, selected_features)['input_ids'],
    X_val_np[:, selected_features],
    y_val_encoded
)
test_dataset = NetworkTrafficDataset(
    prepare_transformer_input(X_test_np, selected_features)['input_ids'],
    X_test_np[:, selected_features],
    y_test_encoded
)

train_loader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False)

# ### 模型训练和评估
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, verbose=True):
        """
        早停类
        Args:
            patience: 容忍的验证损失不下降的轮数
            min_delta: 最小改善值
            verbose: 是否打印信息
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'早停计数器: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_and_evaluate():
    # 初始化模型
    num_classes = len(label_encoder.classes_)
    model = HybridModel(num_classes=num_classes)
    
    # 如果有GPU，将模型移到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # 训练随机森林模型
    model.train_rf(X_train_np[:, selected_features], y_train_encoded)
    
    # 训练参数
    epochs = 100  # 减少训练轮数
    best_val_loss = float('inf')
    
    # 初始化损失历史记录
    train_losses = []
    val_losses = []
    
    # 类别权重计算
    class_weights = compute_class_weight(
        'balanced',  # 使用正确的参数值
        classes=np.unique(y_train_encoded),
        y=y_train_encoded
    )

    # 使用焦点损失
    class FocalLoss(nn.Module):
        def __init__(self, alpha=1, gamma=2):
            super().__init__()
            self.alpha = alpha
            self.gamma = gamma

        def forward(self, inputs, targets):
            ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
            pt = torch.exp(-ce_loss)
            focal_loss = (self.alpha * (1-pt)**self.gamma * ce_loss).mean()
            return focal_loss

    # 损失函数
    criterion = FocalLoss(alpha=0.5, gamma=2)
    
    # 定义优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # 增加学习率
        weight_decay=0.01,  # 减小权重衰减
        betas=(0.9, 0.999)
    )
    
    # 使用余弦退火学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20,  # 增加重启周期
        T_mult=2,
        eta_min=1e-6
    )
    
    # 初始化学习率历史记录
    lr_history = []
    
    # 记录开始时间
    start_time = time.time()
    
    # 初始化早停
    early_stopping = EarlyStopping(patience=5, min_delta=0.001, verbose=True)
    
    # 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch in progress_bar:
            inputs_trans, inputs_rf, labels = batch
            # 将数据移到GPU
            inputs_trans = inputs_trans.to(device)
            inputs_rf = inputs_rf.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs_trans, inputs_rf)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 及时释放内存
            del outputs, loss, predicted
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            # 更新进度条
            progress_bar.set_postfix({
                'loss': f'{total_loss/(progress_bar.n+1):.4f}',
                'acc': f'{correct/total*100:.2f}%'
            })
        
        # 计算并记录训练损失
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs_trans, inputs_rf, labels = batch
                # 将数据移到GPU
                inputs_trans = inputs_trans.to(device)
                inputs_rf = inputs_rf.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs_trans, inputs_rf)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(labels.cpu().numpy())
                
                # 及时释放内存
                del outputs, loss
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # 计算并记录验证损失
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # 早停检查
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("早停触发，停止训练")
            break
            
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'models/hybrid_transformer_rf_multiclass/hybrid_model_best.pth')
            print(f'模型已保存: 验证损失从 {best_val_loss:.4f} 改善到 {val_loss:.4f}')
        
        # 打印训练和验证指标
        print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, '
              f'Train Acc: {correct/total*100:.2f}%, '
              f'Val Loss: {val_loss:.4f}, Val Acc: {val_correct/val_total*100:.2f}%')
        
        # 记录学习率
        lr_history.append(optimizer.param_groups[0]['lr'])
        
        # 更新学习率
        scheduler.step()
    
    # 记录结束时间
    end_time = time.time()
    training_time = end_time - start_time
    print(f"模型训练完成，用时: {training_time:.2f}秒")
    
    # 绘制训练和验证损失
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', label='训练损失')
    plt.plot(val_losses, marker='o', label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('visualizations/hybrid_transformer_rf_multiclass/training_validation_loss.png')

    # 加载最佳模型进行评估
    model.load_state_dict(torch.load('models/hybrid_transformer_rf_multiclass/hybrid_model_best.pth'))
    model = model.to(device)  # 确保模型在正确的设备上
    
    # 在验证集上评估
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs_trans, inputs_rf, labels in val_loader:
            # 将数据移到正确的设备上
            inputs_trans = inputs_trans.to(device)
            inputs_rf = inputs_rf.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs_trans, inputs_rf)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())  # 将结果移回CPU
            all_labels.extend(labels.cpu().numpy())    # 将标签移回CPU
            
            # 及时释放内存
            del outputs
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # 计算评估指标
    val_accuracy = accuracy_score(all_labels, all_preds)
    val_precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    val_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    print("\n验证集性能指标:")
    print(f"准确率: {val_accuracy:.4f}")
    print(f"精确率: {val_precision:.4f}")
    print(f"召回率: {val_recall:.4f}")
    print(f"F1分数: {val_f1:.4f}")
    
    # 保存评估指标
    val_metrics = {
        'accuracy': val_accuracy,
        'precision': val_precision,
        'recall': val_recall,
        'f1': val_f1,
        'training_time': training_time,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1
    }
    
    with open('models/hybrid_transformer_rf_multiclass/validation_metrics.txt', 'w') as f:
        for metric, value in val_metrics.items():
            f.write(f"{metric}: {value}\n")
    
    print("\n验证集分类报告:")
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    val_class_report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(val_class_report)
    
    with open('models/hybrid_transformer_rf_multiclass/validation_classification_report.txt', 'w') as f:
        f.write(val_class_report)

    # 生成混淆矩阵
    val_cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=class_mapping.values())
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title('混合模型验证集混淆矩阵')
    plt.xticks(rotation=43, ha='right')
    plt.tight_layout()
    plt.savefig('visualizations/hybrid_transformer_rf_multiclass/validation_confusion_matrix.png')
    
# 可视化每个类别的F1分数
    print("\n生成每个类别的F1分数图...")
    plt.figure(figsize=(12, 8))

    # 修改生成分类报告的部分
    print("\n验证集分类报告:")
    class_names = [class_mapping[i] for i in range(len(class_mapping))]
    val_report = classification_report(all_labels, all_preds, 
                                     target_names=class_names,  # 添加类别名称
                                     output_dict=True, 
                                     zero_division=0)
    print(classification_report(all_labels, all_preds, 
                              target_names=class_names,
                              zero_division=0))

    # 修改可视化部分
    categories = []
    f1_scores = []

    # 直接遍历分类报告中的所有类别（除了'accuracy','macro avg','weighted avg'）
    for category, metrics in val_report.items():
        if category not in ['accuracy', 'macro avg', 'weighted avg']:
            categories.append(category)
            f1_scores.append(metrics['f1-score'])
            print(f"类别 {category}: F1分数 = {metrics['f1-score']}")

    # 排序以便更好的可视化
    sorted_indices = np.argsort(f1_scores)
    sorted_categories = [categories[i] for i in sorted_indices]
    sorted_f1_scores = [f1_scores[i] for i in sorted_indices]

    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_categories, sorted_f1_scores, color='skyblue')
    plt.xlabel('F1分数')
    plt.title('各攻击类别的F1分数 (验证集)')
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig('visualizations/hybrid_transformer_rf_multiclass/validation_category_f1_scores.png')
    

    # 生成ROC曲线
    model.eval()
    all_scores = []
    with torch.no_grad():
        for inputs_trans, inputs_rf, _ in val_loader:
            inputs_trans = inputs_trans.to(device)
            inputs_rf = inputs_rf.to(device)
            outputs = model(inputs_trans, inputs_rf)
            scores = torch.softmax(outputs, dim=1)
            all_scores.append(scores.cpu().numpy())
    
    y_score = np.vstack(all_scores)
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
    plt.title('多分类ROC曲线 (验证集)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('visualizations/hybrid_transformer_rf_multiclass/validation_roc_curves.png')
    
    # 可以添加更多的评估指标
    metrics = {
        'accuracy': accuracy_score,
        'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted'),
        'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
        'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted'),
        'roc_auc': lambda y_true, y_pred: roc_auc_score(y_true, y_pred, multi_class='ovr')
    }
    
    return model, val_metrics

# ### 主程序
if __name__ == "__main__":
    try:
        train_and_evaluate()
    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n发生错误: {e}")
        import traceback
        traceback.print_exc() 