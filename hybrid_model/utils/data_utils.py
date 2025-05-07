#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import BertTokenizer

def prepare_transformer_input(features, selected_features=None, max_length=128, batch_size=500):
    """将数值特征转换为字符串格式，用于Transformer输入"""
    print("正在准备Transformer输入...")
    all_input_ids = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    if selected_features is not None:
        features = features[:, selected_features]
    
    # 分批处理数据
    for i in tqdm(range(0, len(features), batch_size), desc="处理数据批次"):
        batch = features[i:i+batch_size]
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

def get_feature_importance(X, y):
    """使用随机森林获取特征重要性"""
    print("正在计算特征重要性...")
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)
    importance = rf.feature_importances_
    sorted_idx = np.argsort(importance)[::-1]
    return sorted_idx, importance 