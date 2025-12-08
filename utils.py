"""
Inference Utilities (PRO)
=========================
Вспомогательные функции для инференса
"""

import numpy as np
import pandas as pd
import re


def clean_text(text: str) -> str:
    """Clean and normalize text"""
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing values in dataframe"""
    df = df.copy()
    df['product_title'] = df['product_title'].fillna('').astype(str)
    df['product_description'] = df['product_description'].fillna('').astype(str)
    df['product_bullet_point'] = df['product_bullet_point'].fillna('').astype(str)
    df['product_brand'] = df['product_brand'].fillna('unknown_brand').astype(str)
    df['product_color'] = df['product_color'].fillna('').astype(str)
    df['query'] = df['query'].fillna('').astype(str)
    df['product_locale'] = df['product_locale'].fillna('us').astype(str)
    return df


def jaccard_similarity(text1: str, text2: str) -> float:
    """Compute Jaccard similarity between two texts"""
    words1 = set(str(text1).lower().split())
    words2 = set(str(text2).lower().split())
    if len(words1) == 0 or len(words2) == 0:
        return 0.0
    return len(words1 & words2) / len(words1 | words2)


def word_match_share(text1: str, text2: str) -> float:
    """Compute fraction of words from text1 found in text2"""
    words1 = str(text1).lower().split()
    words2 = set(str(text2).lower().split())
    if len(words1) == 0:
        return 0.0
    return len([w for w in words1 if w in words2]) / len(words1)

