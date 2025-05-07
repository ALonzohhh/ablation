#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# 数据路径配置
DATA_DIR = 'data'
TRAIN_SET = os.path.join(DATA_DIR, 'UNSW_NB15_training-set-new.parquet')
VAL_SET = os.path.join(DATA_DIR, 'UNSW_NB15_validation-set.parquet')
TEST_SET = os.path.join(DATA_DIR, 'UNSW_NB15_testing-set.parquet')

# 模型保存路径配置
MODEL_DIR = 'models/hybrid_transformer_rf_multiclass'
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
FEATURE_COLUMNS_PATH = os.path.join(MODEL_DIR, 'feature_columns.joblib')
SELECTED_FEATURES_PATH = os.path.join(MODEL_DIR, 'selected_features.joblib')
MODEL_PATH = os.path.join(MODEL_DIR, 'hybrid_model_best.pth')

# 可视化保存路径配置
VIS_DIR = 'visualizations/hybrid_transformer_rf_multiclass'
CONFUSION_MATRIX_PATH = os.path.join(VIS_DIR, 'confusion_matrix.png')
ROC_CURVES_PATH = os.path.join(VIS_DIR, 'roc_curves.png')
TRAINING_CURVES_PATH = os.path.join(VIS_DIR, 'training_curves.png')
F1_SCORES_PATH = os.path.join(VIS_DIR, 'f1_scores.png')

# 训练参数配置
BATCH_SIZE = 32
MAX_LENGTH = 128
NUM_EPOCHS = 100
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 0.01
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.001

# 模型参数配置
TRANSFORMER_DIM = 768
RF_DIM = 200
DROPOUT = 0.2
FOCAL_LOSS_ALPHA = 0.5
FOCAL_LOSS_GAMMA = 2

# 随机森林参数配置
RF_N_ESTIMATORS = 300
RF_MAX_DEPTH = 30
RF_MIN_SAMPLES_SPLIT = 5
RF_MIN_SAMPLES_LEAF = 2
RF_MAX_FEATURES = 'sqrt'
RF_RANDOM_STATE = 42 