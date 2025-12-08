"""
Inference Feature Engineering (PRO)
===================================
Extended features with fuzzy matching, n-grams, positional features, etc.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional, Dict
import torch
from tqdm import tqdm
import gc
import os
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from difflib import SequenceMatcher

from config import (
    SEED, DEVICE, EMBEDDING_MODELS, RERANKER_MODELS,
    TFIDF_PARAMS, EMBEDDING_BATCH_SIZE, RERANKER_BATCH_SIZE,
    FEATURES_DIR, set_seed
)
from utils import clean_text, jaccard_similarity, word_match_share

set_seed(SEED)


# ============================================================================
# TEXT SIMILARITY FUNCTIONS
# ============================================================================

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Compute Levenshtein similarity ratio (0-1)"""
    return SequenceMatcher(None, str(s1).lower(), str(s2).lower()).ratio()


def longest_common_substring_ratio(s1: str, s2: str) -> float:
    """Ratio of longest common substring to shorter string"""
    s1, s2 = str(s1).lower(), str(s2).lower()
    if not s1 or not s2:
        return 0.0
    
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_len = 0
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
                max_len = max(max_len, dp[i][j])
    
    return max_len / min(m, n) if min(m, n) > 0 else 0.0


def ngram_overlap(text1: str, text2: str, n: int = 2) -> float:
    """Compute n-gram overlap ratio"""
    def get_ngrams(text, n):
        words = str(text).lower().split()
        if len(words) < n:
            return set()
        return set(tuple(words[i:i+n]) for i in range(len(words) - n + 1))
    
    ngrams1 = get_ngrams(text1, n)
    ngrams2 = get_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union) if union else 0.0


def char_ngram_overlap(text1: str, text2: str, n: int = 3) -> float:
    """Compute character n-gram overlap"""
    def get_char_ngrams(text, n):
        text = str(text).lower().replace(' ', '')
        if len(text) < n:
            return set()
        return set(text[i:i+n] for i in range(len(text) - n + 1))
    
    ngrams1 = get_char_ngrams(text1, n)
    ngrams2 = get_char_ngrams(text2, n)
    
    if not ngrams1 or not ngrams2:
        return 0.0
    
    intersection = ngrams1 & ngrams2
    union = ngrams1 | ngrams2
    return len(intersection) / len(union) if union else 0.0


def query_position_in_text(query: str, text: str) -> dict:
    """Find position features of query in text"""
    query = str(query).lower()
    text = str(text).lower()
    
    result = {
        'found': 0,
        'position_ratio': 1.0,
        'first_word_pos': -1,
    }
    
    if not query or not text:
        return result
    
    pos = text.find(query)
    if pos != -1:
        result['found'] = 1
        result['position_ratio'] = pos / len(text) if len(text) > 0 else 1.0
    
    query_words = query.split()
    if query_words:
        first_word = query_words[0]
        text_words = text.split()
        for i, word in enumerate(text_words):
            if first_word in word:
                result['first_word_pos'] = i / len(text_words) if text_words else -1
                break
    
    return result


def count_query_words_in_text(query: str, text: str) -> dict:
    """Count how many query words appear in text"""
    query_words = set(str(query).lower().split())
    text_words = set(str(text).lower().split())
    
    if not query_words:
        return {'count': 0, 'ratio': 0.0}
    
    matches = query_words & text_words
    return {
        'count': len(matches),
        'ratio': len(matches) / len(query_words)
    }


def clean_query(query: str) -> str:
    """Clean query from noise characters"""
    query = str(query).lower()
    query = query.replace('#', '')
    query = re.sub(r'[,;!?]', ' ', query)
    query = re.sub(r'\s+', ' ', query).strip()
    return query


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text"""
    numbers = re.findall(r'\d+\.?\d*', str(text))
    return [float(n) for n in numbers if n]


def is_placeholder_text(text: str) -> bool:
    """Check if text is a placeholder"""
    text = str(text).lower().strip()
    placeholders = {'none', 'n/a', 'na', '', 'null', 'no description', 'no info'}
    return text in placeholders or len(text) < 3


def fuzzy_color_match(query: str, color: str) -> float:
    """Check if color appears in query (fuzzy)"""
    query = str(query).lower()
    color = str(color).lower()
    
    if not color or color in ['', 'unknown']:
        return 0.0
    
    if color in query:
        return 1.0
    
    color_words = color.split()
    query_words = query.split()
    
    matches = sum(1 for cw in color_words if any(cw in qw or qw in cw for qw in query_words))
    return matches / len(color_words) if color_words else 0.0


def detect_query_type(query: str) -> dict:
    """Detect query type features"""
    query = str(query).lower()
    
    size_patterns = [r'\b(xs|s|m|l|xl|xxl|xxxl)\b', r'\b\d+\s*(oz|ml|inch|cm|mm|ft|lb|kg)\b']
    has_size = any(re.search(p, query) for p in size_patterns)
    
    has_number = bool(re.search(r'\d+', query))
    
    word_count = len(query.split())
    is_short_query = word_count <= 2
    is_long_query = word_count >= 5
    is_optimal_length = word_count == 3
    
    gender_pattern = r'\b(women|woman|womens|men|man|mens|girl|girls|boy|boys|kids|baby|unisex)\b'
    has_gender = bool(re.search(gender_pattern, query))
    
    material_pattern = r'\b(cotton|leather|silk|wool|polyester|plastic|metal|wood|glass|stainless|steel)\b'
    has_material = bool(re.search(material_pattern, query))
    
    negative_words = ['without', 'not', 'no', 'free', 'less']
    is_negative_query = any(nw in query.split() for nw in negative_words)
    
    category_patterns = {
        'electronics': r'\b(iphone|phone|case|charger|cable|battery|wireless|bluetooth|usb|hdmi)\b',
        'clothing': r'\b(shirt|dress|pants|shoes|jacket|coat|sweater|socks|underwear)\b',
        'home': r'\b(kitchen|bathroom|bedroom|furniture|decor|lamp|pillow|blanket)\b',
        'toys': r'\b(toy|game|puzzle|lego|doll|action figure)\b',
    }
    
    query_category = 'other'
    for cat, pattern in category_patterns.items():
        if re.search(pattern, query):
            query_category = cat
            break
    
    return {
        'has_size': int(has_size),
        'has_number': int(has_number),
        'is_short_query': int(is_short_query),
        'is_long_query': int(is_long_query),
        'is_optimal_length': int(is_optimal_length),
        'has_gender': int(has_gender),
        'has_material': int(has_material),
        'is_negative_query': int(is_negative_query),
        'query_word_count_cat': min(word_count, 10),
        'query_category': query_category
    }


# ============================================================================
# MAIN FEATURE GENERATOR CLASS
# ============================================================================

class FeatureGenerator:
    """Feature generator for inference (PRO version)"""
    
    def __init__(self, use_cache: bool = False):
        self.use_cache = use_cache
        self.tfidf_vectorizer = None
        self.tfidf_vectorizer_desc = None
        self.embedding_model = None
        self.reranker_model = None
        self.feature_names = []
        self.cache_suffix = '_pro'
        os.makedirs(FEATURES_DIR, exist_ok=True)
    
    def generate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate basic text-based features"""
        print("📝 Generating basic features...")
        features = pd.DataFrame()
        
        features['query_len'] = df['query'].str.len()
        features['title_len'] = df['product_title'].str.len()
        features['desc_len'] = df['product_description'].str.len()
        features['bullet_len'] = df['product_bullet_point'].str.len()
        features['brand_len'] = df['product_brand'].str.len()
        
        features['query_word_count'] = df['query'].str.split().str.len().fillna(0)
        features['title_word_count'] = df['product_title'].str.split().str.len().fillna(0)
        features['desc_word_count'] = df['product_description'].str.split().str.len().fillna(0)
        features['bullet_word_count'] = df['product_bullet_point'].str.split().str.len().fillna(0)
        
        features['title_query_len_ratio'] = features['title_len'] / (features['query_len'] + 1)
        features['desc_query_len_ratio'] = features['desc_len'] / (features['query_len'] + 1)
        features['bullet_query_len_ratio'] = features['bullet_len'] / (features['query_len'] + 1)
        
        features['query_len_log'] = np.log1p(features['query_len'])
        features['title_len_log'] = np.log1p(features['title_len'])
        features['desc_len_log'] = np.log1p(features['desc_len'])
        
        features['title_query_wc_ratio'] = features['title_word_count'] / (features['query_word_count'] + 1)
        features['desc_query_wc_ratio'] = features['desc_word_count'] / (features['query_word_count'] + 1)
        
        features['title_char_density'] = features['title_word_count'] / (features['title_len'] + 1)
        features['desc_char_density'] = features['desc_word_count'] / (features['desc_len'] + 1)
        
        return features
    
    def generate_overlap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate word overlap features"""
        print("🔤 Generating overlap features...")
        features = pd.DataFrame()
        
        df = df.copy()
        df['query_clean'] = df['query'].apply(clean_text)
        df['title_clean'] = df['product_title'].apply(clean_text)
        df['desc_clean'] = df['product_description'].apply(clean_text)
        df['bullet_clean'] = df['product_bullet_point'].apply(clean_text)
        
        features['query_title_jaccard'] = [
            jaccard_similarity(q, t) 
            for q, t in zip(df['query_clean'], df['title_clean'])
        ]
        features['query_desc_jaccard'] = [
            jaccard_similarity(q, d) 
            for q, d in zip(df['query_clean'], df['desc_clean'])
        ]
        features['query_bullet_jaccard'] = [
            jaccard_similarity(q, b) 
            for q, b in zip(df['query_clean'], df['bullet_clean'])
        ]
        
        features['query_title_match'] = [
            word_match_share(q, t) 
            for q, t in zip(df['query_clean'], df['title_clean'])
        ]
        features['query_desc_match'] = [
            word_match_share(q, d) 
            for q, d in zip(df['query_clean'], df['desc_clean'])
        ]
        features['query_bullet_match'] = [
            word_match_share(q, b) 
            for q, b in zip(df['query_clean'], df['bullet_clean'])
        ]
        
        features['query_in_title_exact'] = [
            1 if str(q).lower() in str(t).lower() else 0 
            for q, t in zip(df['query'], df['product_title'])
        ]
        features['query_in_desc_exact'] = [
            1 if str(q).lower() in str(d).lower() else 0 
            for q, d in zip(df['query'], df['product_description'])
        ]
        
        features['brand_in_query'] = [
            1 if str(b).lower() in str(q).lower() and b != 'unknown_brand' else 0 
            for q, b in zip(df['query'], df['product_brand'])
        ]
        features['color_in_query'] = [
            1 if str(c).lower() in str(q).lower() and c != '' else 0 
            for q, c in zip(df['query'], df['product_color'])
        ]
        
        features['has_brand'] = (df['product_brand'] != 'unknown_brand').astype(int)
        features['has_color'] = (df['product_color'] != '').astype(int)
        features['has_description'] = (df['product_description'] != '').astype(int)
        features['has_bullets'] = (df['product_bullet_point'] != '').astype(int)
        
        return features
    
    def generate_fuzzy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate fuzzy matching features"""
        print("🔮 Generating fuzzy matching features...")
        features = pd.DataFrame()
        
        features['query_title_levenshtein'] = [
            levenshtein_ratio(q, t)
            for q, t in tqdm(zip(df['query'], df['product_title']), 
                            total=len(df), desc="Levenshtein")
        ]
        
        features['query_title_lcs'] = [
            longest_common_substring_ratio(q, t)
            for q, t in tqdm(zip(df['query'], df['product_title']),
                            total=len(df), desc="LCS")
        ]
        
        return features
    
    def generate_ngram_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate n-gram overlap features"""
        print("📊 Generating n-gram features...")
        features = pd.DataFrame()
        
        features['query_title_bigram'] = [
            ngram_overlap(q, t, n=2)
            for q, t in tqdm(zip(df['query'], df['product_title']),
                            total=len(df), desc="Bigrams")
        ]
        
        features['query_title_trigram'] = [
            ngram_overlap(q, t, n=3)
            for q, t in tqdm(zip(df['query'], df['product_title']),
                            total=len(df), desc="Trigrams")
        ]
        
        features['query_title_char3gram'] = [
            char_ngram_overlap(q, t, n=3)
            for q, t in tqdm(zip(df['query'], df['product_title']),
                            total=len(df), desc="Char 3-grams")
        ]
        
        return features
    
    def generate_positional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate positional features"""
        print("📍 Generating positional features...")
        features = pd.DataFrame()
        
        title_positions = [
            query_position_in_text(q, t)
            for q, t in tqdm(zip(df['query'], df['product_title']),
                            total=len(df), desc="Position (title)")
        ]
        features['query_in_title_found'] = [p['found'] for p in title_positions]
        features['query_in_title_pos_ratio'] = [p['position_ratio'] for p in title_positions]
        features['query_first_word_pos_title'] = [p['first_word_pos'] for p in title_positions]
        
        title_counts = [
            count_query_words_in_text(q, t)
            for q, t in zip(df['query'], df['product_title'])
        ]
        features['query_words_in_title_count'] = [c['count'] for c in title_counts]
        features['query_words_in_title_ratio'] = [c['ratio'] for c in title_counts]
        
        desc_counts = [
            count_query_words_in_text(q, d)
            for q, d in zip(df['query'], df['product_description'])
        ]
        features['query_words_in_desc_count'] = [c['count'] for c in desc_counts]
        features['query_words_in_desc_ratio'] = [c['ratio'] for c in desc_counts]
        
        return features
    
    def generate_advanced_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate advanced text analysis features"""
        print("🔬 Generating advanced text features...")
        features = pd.DataFrame()
        
        df = df.copy()
        df['query_clean'] = df['query'].apply(clean_query)
        
        features['clean_query_title_jaccard'] = [
            jaccard_similarity(q, t)
            for q, t in zip(df['query_clean'], df['product_title'].apply(lambda x: str(x).lower()))
        ]
        
        features['desc_is_placeholder'] = df['product_description'].apply(is_placeholder_text).astype(int)
        features['bullet_is_placeholder'] = df['product_bullet_point'].apply(is_placeholder_text).astype(int)
        
        query_types = [detect_query_type(q) for q in tqdm(df['query'], desc="Query types")]
        features['query_has_size'] = [qt['has_size'] for qt in query_types]
        features['query_has_number'] = [qt['has_number'] for qt in query_types]
        features['query_is_short'] = [qt['is_short_query'] for qt in query_types]
        features['query_is_long'] = [qt['is_long_query'] for qt in query_types]
        features['query_is_optimal_len'] = [qt['is_optimal_length'] for qt in query_types]
        features['query_has_gender'] = [qt['has_gender'] for qt in query_types]
        features['query_has_material'] = [qt['has_material'] for qt in query_types]
        features['query_is_negative'] = [qt['is_negative_query'] for qt in query_types]
        features['query_word_count_cat'] = [qt['query_word_count_cat'] for qt in query_types]
        
        categories = ['electronics', 'clothing', 'home', 'toys', 'other']
        for cat in categories:
            features[f'query_cat_{cat}'] = [1 if qt['query_category'] == cat else 0 for qt in query_types]
        
        features['color_in_query_fuzzy'] = [
            fuzzy_color_match(q, c)
            for q, c in zip(df['query'], df['product_color'])
        ]
        
        query_numbers = [extract_numbers(q) for q in df['query']]
        title_numbers = [extract_numbers(t) for t in df['product_title']]
        
        features['query_has_numbers'] = [1 if nums else 0 for nums in query_numbers]
        features['title_has_numbers'] = [1 if nums else 0 for nums in title_numbers]
        features['number_match'] = [
            1 if set(qn) & set(tn) else 0
            for qn, tn in zip(query_numbers, title_numbers)
        ]
        
        features['query_special_char_count'] = df['query'].apply(
            lambda x: len(re.findall(r'[#,;!?@]', str(x)))
        )
        
        features['brand_in_title'] = [
            1 if str(b).lower() != 'unknown_brand' and str(b).lower() in str(t).lower() else 0
            for b, t in zip(df['product_brand'], df['product_title'])
        ]
        
        features['product_completeness'] = (
            (df['product_brand'] != 'unknown_brand').astype(int) +
            (df['product_color'].fillna('') != '').astype(int) +
            (~df['product_description'].apply(is_placeholder_text)).astype(int) +
            (~df['product_bullet_point'].apply(is_placeholder_text)).astype(int)
        ) / 4.0
        
        desc_lens = df['product_description'].fillna('').str.len()
        features['desc_is_very_short'] = (desc_lens < 10).astype(int)
        features['desc_is_short'] = ((desc_lens >= 10) & (desc_lens < 100)).astype(int)
        features['desc_is_medium'] = ((desc_lens >= 100) & (desc_lens < 500)).astype(int)
        features['desc_is_long'] = (desc_lens >= 500).astype(int)
        
        bullet_lens = df['product_bullet_point'].fillna('').str.len()
        features['has_bullet_points'] = (bullet_lens > 5).astype(int)
        features['bullet_is_detailed'] = (bullet_lens > 200).astype(int)
        
        top_brands = ['nike', 'apple', 'adidas', 'hanes', 'samsung', 'amazon', 'under armour', 'lego']
        features['is_top_brand'] = df['product_brand'].str.lower().apply(
            lambda x: 1 if any(tb in str(x) for tb in top_brands) else 0
        )
        
        top_colors = ['black', 'white', 'blue', 'red', 'silver', 'grey', 'gray', 'pink', 'green']
        features['has_common_color'] = df['product_color'].fillna('').str.lower().apply(
            lambda x: 1 if any(tc in str(x) for tc in top_colors) else 0
        )
        
        features['common_color_in_query'] = [
            1 if any(tc in str(q).lower() for tc in top_colors) else 0
            for q in df['query']
        ]
        
        def word_overlap_ratio(query, title):
            query_words = set(str(query).lower().split())
            title_words = set(str(title).lower().split())
            if not query_words:
                return 0.0
            return len(query_words & title_words) / len(query_words)
        
        word_overlaps = [
            word_overlap_ratio(q, t) 
            for q, t in zip(df['query'], df['product_title'])
        ]
        features['word_overlap_ratio'] = word_overlaps
        features['word_overlap_0_25'] = [1 if wo <= 0.25 else 0 for wo in word_overlaps]
        features['word_overlap_25_50'] = [1 if 0.25 < wo <= 0.50 else 0 for wo in word_overlaps]
        features['word_overlap_50_75'] = [1 if 0.50 < wo <= 0.75 else 0 for wo in word_overlaps]
        features['word_overlap_75_100'] = [1 if wo > 0.75 else 0 for wo in word_overlaps]
        
        top_query_words = ['for', 'without', 'women', 'men', 'with', 'case', 'iphone', 'kids', 'baby']
        for word in top_query_words:
            features[f'query_has_{word}'] = df['query'].str.lower().str.contains(
                rf'\b{word}\b', regex=True
            ).astype(int)
        
        gender_words = ['women', 'woman', 'womens', 'men', 'man', 'mens', 'girl', 'girls', 'boy', 'boys', 'kids', 'baby']
        
        def gender_match(query, title):
            query = str(query).lower()
            title = str(title).lower()
            query_genders = [g for g in gender_words if g in query]
            if not query_genders:
                return -1
            title_genders = [g for g in gender_words if g in title]
            if not title_genders:
                return 0
            return 1 if any(qg in title for qg in query_genders) else 0
        
        features['gender_match'] = [
            gender_match(q, t) for q, t in zip(df['query'], df['product_title'])
        ]
        
        return features
    
    def generate_locale_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate locale features"""
        print("🌍 Generating locale features...")
        features = pd.DataFrame()
        
        df = df.copy()
        locale_series = df['product_locale'].fillna('us').astype(str)
        df['locale_normalized'] = locale_series.str.lower().str.split('_').str[-1]
        features['is_us_locale'] = (df['locale_normalized'] == 'us').astype(int)
        
        return features
    
    def generate_bm25_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate BM25 scores"""
        print("📚 Generating BM25 features...")
        features = pd.DataFrame()
        
        df = df.reset_index(drop=True)
        
        titles = [str(t).lower().split() for t in df['product_title']]
        bm25 = BM25Okapi(titles)
        
        bm25_scores = []
        for i in tqdm(range(len(df)), desc="BM25 title"):
            query_tokens = str(df.iloc[i]['query']).lower().split()
            scores = bm25.get_scores(query_tokens)
            bm25_scores.append(scores[i])
        features['bm25_title'] = bm25_scores
        
        descs = [str(d).lower().split() for d in df['product_description']]
        bm25_desc = BM25Okapi(descs)
        
        bm25_desc_scores = []
        for i in tqdm(range(len(df)), desc="BM25 desc"):
            query_tokens = str(df.iloc[i]['query']).lower().split()
            scores = bm25_desc.get_scores(query_tokens)
            bm25_desc_scores.append(scores[i])
        features['bm25_desc'] = bm25_desc_scores
        
        bullets = [str(b).lower().split() for b in df['product_bullet_point']]
        bm25_bullet = BM25Okapi(bullets)
        
        bm25_bullet_scores = []
        for i in tqdm(range(len(df)), desc="BM25 bullet"):
            query_tokens = str(df.iloc[i]['query']).lower().split()
            scores = bm25_bullet.get_scores(query_tokens)
            bm25_bullet_scores.append(scores[i])
        features['bm25_bullet'] = bm25_bullet_scores
        
        return features
    
    def generate_tfidf_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate TF-IDF similarity features"""
        print("📊 Generating TF-IDF features...")
        
        if self.tfidf_vectorizer is None:
            self.tfidf_vectorizer = TfidfVectorizer(**TFIDF_PARAMS)
            all_texts = pd.concat([df['query'], df['product_title']])
            self.tfidf_vectorizer.fit(all_texts)
        
        if self.tfidf_vectorizer_desc is None:
            self.tfidf_vectorizer_desc = TfidfVectorizer(**TFIDF_PARAMS)
            all_desc = pd.concat([df['query'], df['product_description']])
            self.tfidf_vectorizer_desc.fit(all_desc)
        
        features = pd.DataFrame()
        
        query_tfidf = self.tfidf_vectorizer.transform(df['query'])
        title_tfidf = self.tfidf_vectorizer.transform(df['product_title'])
        
        sims = []
        for i in tqdm(range(len(df)), desc="TF-IDF title"):
            sim = cosine_similarity(query_tfidf[i], title_tfidf[i])[0][0]
            sims.append(sim)
        features['tfidf_sim'] = sims
        
        query_tfidf_desc = self.tfidf_vectorizer_desc.transform(df['query'])
        desc_tfidf = self.tfidf_vectorizer_desc.transform(df['product_description'])
        
        sims_desc = []
        for i in tqdm(range(len(df)), desc="TF-IDF desc"):
            sim = cosine_similarity(query_tfidf_desc[i], desc_tfidf[i])[0][0]
            sims_desc.append(sim)
        features['tfidf_sim_desc'] = sims_desc
        
        return features
    
    @torch.no_grad()
    def generate_semantic_features(self, df: pd.DataFrame, 
                                    model_name: str = None) -> pd.DataFrame:
        """Generate semantic embedding features"""
        print("🤖 Generating semantic features...")
        
        if model_name is None:
            model_name = list(EMBEDDING_MODELS.values())[0]
        
        from sentence_transformers import SentenceTransformer, util
        
        model = SentenceTransformer(model_name, device=DEVICE)
        features = pd.DataFrame()
        
        all_title_sims = []
        all_title_dists = []
        all_desc_sims = []
        
        for start_idx in tqdm(range(0, len(df), EMBEDDING_BATCH_SIZE), desc="Embeddings"):
            end_idx = min(start_idx + EMBEDDING_BATCH_SIZE, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            queries = batch['query'].tolist()
            titles = batch['product_title'].tolist()
            descs = [d[:500] for d in batch['product_description'].tolist()]
            
            query_embs = model.encode(queries, convert_to_tensor=True, normalize_embeddings=True)
            title_embs = model.encode(titles, convert_to_tensor=True, normalize_embeddings=True)
            desc_embs = model.encode(descs, convert_to_tensor=True, normalize_embeddings=True)
            
            title_sims = util.cos_sim(query_embs, title_embs).diagonal().cpu().numpy()
            title_dists = torch.norm(query_embs - title_embs, dim=1).cpu().numpy()
            desc_sims = util.cos_sim(query_embs, desc_embs).diagonal().cpu().numpy()
            
            all_title_sims.extend(title_sims)
            all_title_dists.extend(title_dists)
            all_desc_sims.extend(desc_sims)
            
            del query_embs, title_embs, desc_embs
            if DEVICE == 'cuda':
                torch.cuda.empty_cache()
        
        features['sem_cos_sim'] = all_title_sims
        features['sem_euclidean_dist'] = all_title_dists
        features['sem_cos_sim_desc'] = all_desc_sims
        features['sem_combined'] = (np.array(all_title_sims) + np.array(all_desc_sims)) / 2
        
        del model
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        return features
    
    @torch.no_grad()
    def generate_reranker_features(self, df: pd.DataFrame,
                                    model_name: str = None) -> pd.DataFrame:
        """Generate reranker scores"""
        print("🎯 Generating reranker features...")
        
        if model_name is None:
            model_name = list(RERANKER_MODELS.values())[0]
        
        from FlagEmbedding import FlagReranker
        
        model = FlagReranker(model_name, use_fp16=True, device=DEVICE)
        features = pd.DataFrame()
        
        all_scores = []
        for start_idx in tqdm(range(0, len(df), RERANKER_BATCH_SIZE), desc="Reranking title"):
            end_idx = min(start_idx + RERANKER_BATCH_SIZE, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            pairs = [[str(q), str(t)] for q, t in zip(batch['query'], batch['product_title'])]
            batch_scores = model.compute_score(pairs, normalize=True)
            
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            all_scores.extend(batch_scores)
        
        features['reranker_score'] = all_scores
        
        all_scores_full = []
        for start_idx in tqdm(range(0, len(df), RERANKER_BATCH_SIZE), desc="Reranking full"):
            end_idx = min(start_idx + RERANKER_BATCH_SIZE, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            combined = [f"{t} {d[:500]}" for t, d in 
                       zip(batch['product_title'], batch['product_description'])]
            pairs = [[str(q), c] for q, c in zip(batch['query'], combined)]
            batch_scores = model.compute_score(pairs, normalize=True)
            
            if isinstance(batch_scores, float):
                batch_scores = [batch_scores]
            all_scores_full.extend(batch_scores)
        
        features['reranker_score_full'] = all_scores_full
        
        del model
        gc.collect()
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()
        
        return features
    
    def generate_query_aggregate_features(self, df: pd.DataFrame, 
                                          base_features: pd.DataFrame) -> pd.DataFrame:
        """Generate query-level aggregate features"""
        print("📈 Generating query aggregate features...")
        features = pd.DataFrame()
        
        df = df.reset_index(drop=True)
        base_features = base_features.reset_index(drop=True)
        
        query_counts = df.groupby('query_id').size()
        features['n_candidates'] = df['query_id'].map(query_counts).values
        
        numeric_cols = [
            'sem_cos_sim', 'sem_cos_sim_desc', 'sem_combined',
            'reranker_score', 'reranker_score_full',
            'bm25_title', 'bm25_desc', 'bm25_bullet',
            'tfidf_sim', 'tfidf_sim_desc',
            'query_title_jaccard', 'query_title_levenshtein',
            'query_title_bigram'
        ]
        
        for col in numeric_cols:
            if col in base_features.columns:
                temp_df = df[['query_id']].copy()
                temp_df[col] = base_features[col].values
                
                query_stats = temp_df.groupby('query_id')[col].agg(['mean', 'max', 'min', 'std', 'median'])
                query_stats.columns = [f'{col}_query_{stat}' for stat in ['mean', 'max', 'min', 'std', 'median']]
                
                temp_df = temp_df.merge(query_stats, left_on='query_id', right_index=True)
                
                features[f'{col}_query_mean'] = temp_df[f'{col}_query_mean'].values
                features[f'{col}_query_max'] = temp_df[f'{col}_query_max'].values
                features[f'{col}_query_min'] = temp_df[f'{col}_query_min'].values
                features[f'{col}_query_rank'] = temp_df.groupby('query_id')[col].rank(ascending=False, method='min').values
                features[f'{col}_query_rank_pct'] = temp_df.groupby('query_id')[col].rank(ascending=False, pct=True).values
                
                features[f'{col}_rel_to_mean'] = base_features[col].values - features[f'{col}_query_mean']
                features[f'{col}_rel_to_max'] = base_features[col].values - features[f'{col}_query_max']
                features[f'{col}_rel_to_min'] = base_features[col].values - features[f'{col}_query_min']
        
        return features
    
    def generate_all_features(self, df: pd.DataFrame,
                               use_semantic: bool = True,
                               use_reranker: bool = True
                               ) -> Tuple[np.ndarray, List[str]]:
        """Generate all features for inference"""
        print("="*80)
        print("🚀 GENERATING FEATURES FOR INFERENCE (PRO)")
        print("="*80)
        
        all_features = []
        feature_names = []
        
        # 1. Basic features
        basic = self.generate_basic_features(df)
        all_features.append(basic)
        feature_names.extend(basic.columns.tolist())
        
        # 2. Overlap features
        overlap = self.generate_overlap_features(df)
        all_features.append(overlap)
        feature_names.extend(overlap.columns.tolist())
        
        # 3. Fuzzy features
        fuzzy = self.generate_fuzzy_features(df)
        all_features.append(fuzzy)
        feature_names.extend(fuzzy.columns.tolist())
        
        # 4. N-gram features
        ngram = self.generate_ngram_features(df)
        all_features.append(ngram)
        feature_names.extend(ngram.columns.tolist())
        
        # 5. Positional features
        pos = self.generate_positional_features(df)
        all_features.append(pos)
        feature_names.extend(pos.columns.tolist())
        
        # 6. Advanced text features
        advanced = self.generate_advanced_text_features(df)
        all_features.append(advanced)
        feature_names.extend(advanced.columns.tolist())
        
        # 7. Locale features
        locale = self.generate_locale_features(df)
        all_features.append(locale)
        feature_names.extend(locale.columns.tolist())
        
        # 8. BM25 features
        bm25 = self.generate_bm25_features(df)
        all_features.append(bm25)
        feature_names.extend(bm25.columns.tolist())
        
        # 9. TF-IDF features
        tfidf = self.generate_tfidf_features(df)
        all_features.append(tfidf)
        feature_names.extend(tfidf.columns.tolist())
        
        # 10. Semantic features
        if use_semantic:
            sem = self.generate_semantic_features(df)
            all_features.append(sem)
            feature_names.extend(sem.columns.tolist())
        
        # 11. Reranker features
        if use_reranker:
            rerank = self.generate_reranker_features(df)
            all_features.append(rerank)
            feature_names.extend(rerank.columns.tolist())
        
        # Concatenate all features
        features_df = pd.concat(all_features, axis=1)
        X = features_df.values.astype(np.float32)
        
        # 12. Query aggregate features
        agg = self.generate_query_aggregate_features(df, features_df)
        feature_names.extend(agg.columns.tolist())
        X = np.hstack([X, agg.values.astype(np.float32)])
        
        # Handle NaN/Inf
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        print(f"\n✅ Feature generation complete!")
        print(f"   Features: {X.shape}")
        print(f"   Feature count: {len(feature_names)}")
        
        self.feature_names = feature_names
        
        return X, feature_names

