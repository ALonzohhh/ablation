#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch.utils.data import Dataset

class NetworkTrafficDataset(Dataset):
    def __init__(self, features_trans, features_rf, labels):
        self.features_trans = torch.tensor(features_trans, dtype=torch.long)
        self.features_rf = torch.tensor(features_rf, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self):
        return len(self.features_trans)
    
    def __getitem__(self, idx):
        return self.features_trans[idx], self.features_rf[idx], self.labels[idx] 