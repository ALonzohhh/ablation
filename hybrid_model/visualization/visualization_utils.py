#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

def plot_confusion_matrix(y_true, y_pred, class_mapping, title, save_path):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_mapping.values())
    disp.plot(cmap='Blues', values_format='.0f')
    plt.title(title)
    plt.xticks(rotation=43, ha='right')
    plt.tight_layout()
    plt.savefig(save_path)

def plot_roc_curves(y_true, y_score, class_mapping, save_path):
    """绘制ROC曲线"""
    plt.figure(figsize=(10, 8))
    for i in range(len(class_mapping)):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{class_mapping[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假正率 (False Positive Rate)')
    plt.ylabel('真正率 (True Positive Rate)')
    plt.title('多分类ROC曲线')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)

def plot_training_curves(train_losses, val_losses, save_path):
    """绘制训练和验证损失曲线"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, marker='o', label='训练损失')
    plt.plot(val_losses, marker='o', label='验证损失')
    plt.title('训练和验证损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)

def plot_f1_scores(categories, f1_scores, save_path):
    """绘制各类别的F1分数"""
    plt.figure(figsize=(12, 8))
    bars = plt.barh(categories, f1_scores, color='skyblue')
    plt.xlabel('F1分数')
    plt.title('各攻击类别的F1分数')
    plt.xlim(0, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7)

    # 添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path) 