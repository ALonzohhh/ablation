#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from transformers import BertModel

class HybridModel(nn.Module):
    def __init__(self, num_classes, trans_dim=768, rf_dim=200):
        super().__init__()
        
        # Transformer模块
        self.transformer = BertModel.from_pretrained(
            'bert-base-chinese',
            output_hidden_states=True,
            return_dict=True
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