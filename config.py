"""
Inference Configuration (PRO)
=============================
Минимальная конфигурация для инференса
"""

import os
import random
import numpy as np
import torch

# ============================================================================
# SEED
# ============================================================================
SEED = 993

def set_seed(seed: int = SEED):
    """Set seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# DEVICE
# ============================================================================
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
FEATURES_DIR = os.path.join(BASE_DIR, 'cached_features')

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FEATURES_DIR, exist_ok=True)

# ============================================================================
# EMBEDDING MODELS
# ============================================================================
EMBEDDING_MODELS = {
    'qwen3': 'Qwen/Qwen3-Embedding-0.6B',
}

# ============================================================================
# RERANKER MODELS
# ============================================================================
RERANKER_MODELS = {
    'bge_v2_m3': 'BAAI/bge-reranker-v2-m3',
}

# ============================================================================
# FEATURE ENGINEERING PARAMS
# ============================================================================
TFIDF_PARAMS = {
    'max_features': 5000,
    'ngram_range': (1, 2),
    'min_df': 2,
    'sublinear_tf': True,
}

EMBEDDING_BATCH_SIZE = 64
RERANKER_BATCH_SIZE = 32

