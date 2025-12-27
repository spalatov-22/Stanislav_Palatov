"""
CatBoost Solution for Dynamic Pricing Competition
==================================================
Загрузка предобученных моделей и создание предсказаний.

Seed: 322
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from catboost import CatBoostRegressor
import os
import pickle

warnings.filterwarnings('ignore')

# Папка с сохраненными моделями
MODEL_DIR = 'models_catboost_tuned'
SEED = 322
np.random.seed(SEED)


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

class FeatureEngineer:
    """Feature Engineering для CatBoost"""
    
    def __init__(self):
        self.product_stats = None
        self.product_lags = None
        self.product_embeddings = None
        self.category_stats = {}
        self.cluster_info = {}
        self.global_stats = {}
        self.feature_cols = []
        
    def _transform_internal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['dt'] = pd.to_datetime(df['dt'])
        
        # Календарные
        df['is_weekend'] = (df['dow'] >= 5).astype(float)
        df['is_monday'] = (df['dow'] == 0).astype(float)
        df['is_friday'] = (df['dow'] == 4).astype(float)
        df['is_month_start'] = (df['day_of_month'] <= 3).astype(float)
        df['is_month_end'] = (df['day_of_month'] >= 28).astype(float)
        df['quarter'] = df['dt'].dt.quarter
        df['week_of_year'] = df['dt'].dt.isocalendar().week.astype(int)
        
        # Циклические
        df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dom_sin'] = np.sin(2 * np.pi * df['day_of_month'] / 31)
        df['dom_cos'] = np.cos(2 * np.pi * df['day_of_month'] / 31)
        
        # Merges
        df = df.merge(self.product_stats, on='product_id', how='left')
        df = df.merge(self.product_lags, on='product_id', how='left')
        df = df.merge(self.product_embeddings, on='product_id', how='left')
        
        for col, stats in self.category_stats.items():
            df = df.merge(stats, on=col, how='left')
        
        df = df.merge(self.cluster_info['product_clusters'], on='product_id', how='left')
        df = df.merge(self.cluster_info['stats'], on='product_cluster', how='left')
        
        # Interactions
        df['activity_n_stores'] = df['activity_flag'] * df['n_stores']
        df['n_stores_sq'] = df['n_stores'] ** 2
        df['n_stores_log'] = np.log1p(df['n_stores'])
        
        # Weather
        df['weather_index'] = (
            df['avg_temperature'] * 0.3 + 
            df['avg_humidity'] * 0.3 + 
            df['precpt'] * 0.2 + 
            df['avg_wind_level'] * 0.2
        )
        df['is_rainy'] = (df['precpt'] > 0.5).astype(float)
        
        self._fill_missing(df)
        
        return df
    
    def _fill_missing(self, df: pd.DataFrame):
        cat_hierarchy = ['third_category_id', 'second_category_id', 
                        'first_category_id', 'management_group_id']
        
        for col in ['prod_emb_p05', 'prod_price_p05_mean', 'prod_last_p05']:
            if col not in df.columns:
                continue
            for cat_col in cat_hierarchy:
                cat_col_name = f'{cat_col}_price_p05_mean'
                if cat_col_name in df.columns:
                    df[col] = df[col].fillna(df[cat_col_name])
            df[col] = df[col].fillna(self.global_stats['p05_mean'])
            
        for col in ['prod_emb_p95', 'prod_price_p95_mean', 'prod_last_p95']:
            if col not in df.columns:
                continue
            for cat_col in cat_hierarchy:
                cat_col_name = f'{cat_col}_price_p95_mean'
                if cat_col_name in df.columns:
                    df[col] = df[col].fillna(df[cat_col_name])
            df[col] = df[col].fillna(self.global_stats['p95_mean'])
            
        for col in ['prod_emb_center', 'prod_center_mean', 'prod_last_center']:
            if col not in df.columns:
                continue
            for cat_col in cat_hierarchy:
                cat_col_name = f'{cat_col}_center_mean'
                if cat_col_name in df.columns:
                    df[col] = df[col].fillna(df[cat_col_name])
            df[col] = df[col].fillna(self.global_stats['center_mean'])
            
        for col in ['prod_emb_width', 'prod_width_mean', 'prod_last_width']:
            if col not in df.columns:
                continue
            for cat_col in cat_hierarchy:
                cat_col_name = f'{cat_col}_width_mean'
                if cat_col_name in df.columns:
                    df[col] = df[col].fillna(df[cat_col_name])
            df[col] = df[col].fillna(self.global_stats['width_mean'])
        
        for col in df.columns:
            if df[col].isna().any():
                if df[col].dtype in ['float64', 'float32', 'int64', 'int32']:
                    median_val = df[col].median()
                    df[col] = df[col].fillna(median_val if not np.isnan(median_val) else 0)
                    
        if 'product_cluster' in df.columns:
            df['product_cluster'] = df['product_cluster'].fillna(0).astype(int)
            
        if 'prod_emb_confidence' in df.columns:
            df['prod_emb_confidence'] = df['prod_emb_confidence'].fillna(0)
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        df_fe = self._transform_internal(df)
        X = df_fe[self.feature_cols].fillna(0).values
        return X.astype(np.float32)
    
    @classmethod
    def load(cls, path: str) -> 'FeatureEngineer':
        """Загрузка FeatureEngineer"""
        fe = cls()
        with open(path, 'rb') as f:
            state = pickle.load(f)
        fe.product_stats = state['product_stats']
        fe.product_lags = state['product_lags']
        fe.product_embeddings = state['product_embeddings']
        fe.category_stats = state['category_stats']
        fe.cluster_info = state['cluster_info']
        fe.global_stats = state['global_stats']
        fe.feature_cols = state['feature_cols']
        print(f"✅ FeatureEngineer loaded from {path}")
        return fe


# ============================================================================
# CATBOOST MODEL
# ============================================================================

class CatBoostModel:
    """CatBoost модель для предсказания ценовых интервалов"""
    
    def __init__(self, fe: FeatureEngineer):
        self.fe = fe
        self.models_p05_mae = []
        self.models_p05_quantile = []
        self.models_p95_mae = []
        self.models_p95_quantile = []
        self.calibration_params = []
        
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Предсказание"""
        all_p05 = []
        all_p95 = []
        
        for fold in range(len(self.models_p05_mae)):
            pred_p05_mae = self.models_p05_mae[fold].predict(X)
            pred_p05_q = self.models_p05_quantile[fold].predict(X)
            pred_p95_mae = self.models_p95_mae[fold].predict(X)
            pred_p95_q = self.models_p95_quantile[fold].predict(X)
            
            shift_p05, shift_p95, scale, w05, w95 = self.calibration_params[fold]
            
            pred_p05 = w05 * pred_p05_mae + (1 - w05) * pred_p05_q
            pred_p95 = w95 * pred_p95_mae + (1 - w95) * pred_p95_q
            
            pred_p05 = pred_p05 * scale + shift_p05
            pred_p95 = pred_p95 * scale + shift_p95
            
            all_p05.append(pred_p05)
            all_p95.append(pred_p95)
        
        final_p05 = np.mean(all_p05, axis=0)
        final_p95 = np.mean(all_p95, axis=0)
        
        # Гарантируем, что p95 > p05
        mask = final_p95 < final_p05
        if mask.any():
            center = (final_p05[mask] + final_p95[mask]) / 2
            final_p05[mask] = center - 0.001
            final_p95[mask] = center + 0.001
        
        return final_p05, final_p95
    
    @classmethod
    def load(cls, model_dir: str) -> 'CatBoostModel':
        """Загрузка всех моделей и параметров"""
        # Загружаем FeatureEngineer
        fe = FeatureEngineer.load(os.path.join(model_dir, 'feature_engineer.pkl'))
        
        model = cls(fe)
        
        # Загружаем параметры калибровки
        calibration_path = os.path.join(model_dir, 'calibration_params.pkl')
        with open(calibration_path, 'rb') as f:
            calibration_data = pickle.load(f)
        model.calibration_params = calibration_data['calibration_params']
        n_folds = calibration_data['n_folds']
        
        # Загружаем CatBoost модели
        for fold in range(n_folds):
            model_p05_mae = CatBoostRegressor()
            model_p05_mae.load_model(os.path.join(model_dir, f'model_p05_mae_fold{fold}.cbm'))
            model.models_p05_mae.append(model_p05_mae)
            
            model_p05_q = CatBoostRegressor()
            model_p05_q.load_model(os.path.join(model_dir, f'model_p05_quantile_fold{fold}.cbm'))
            model.models_p05_quantile.append(model_p05_q)
            
            model_p95_mae = CatBoostRegressor()
            model_p95_mae.load_model(os.path.join(model_dir, f'model_p95_mae_fold{fold}.cbm'))
            model.models_p95_mae.append(model_p95_mae)
            
            model_p95_q = CatBoostRegressor()
            model_p95_q.load_model(os.path.join(model_dir, f'model_p95_quantile_fold{fold}.cbm'))
            model.models_p95_quantile.append(model_p95_q)
        
        print(f"✅ All models loaded from {model_dir}/")
        print(f"   - {n_folds} folds x 4 models = {n_folds * 4} CatBoost models")
        
        return model


# ============================================================================
# CREATE SUBMISSION
# ============================================================================

def create_submission(submission: pd.DataFrame) -> str:
    """
    Создание файла submission.csv в папку results
    """
    os.makedirs('results', exist_ok=True)
    submission_path = 'results/submission.csv'
    submission.to_csv(submission_path, index=False)
    
    print(f"✅ Submission файл сохранен: {submission_path}")
    
    return submission_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Главная функция программы
    """
    print("=" * 60)
    print("🚀 Dynamic Pricing Competition - CatBoost Solution")
    print("=" * 60)
    
    # 1. Загрузка тестовых данных
    print("\n📂 Загрузка тестовых данных...")
    test_df = pd.read_csv('data/test.csv')
    print(f"   Test shape: {test_df.shape}")
    
    row_ids = test_df['row_id'].values
    
    # 2. Загрузка предобученных моделей
    print("\n📦 Загрузка предобученных моделей...")
    model = CatBoostModel.load(MODEL_DIR)
    
    # 3. Feature Engineering
    print("\n🔧 Преобразование признаков...")
    X_test = model.fe.transform(test_df)
    print(f"   X_test shape: {X_test.shape}")
    
    # 4. Предсказание
    print("\n🎯 Генерация предсказаний...")
    pred_p05, pred_p95 = model.predict(X_test)
    
    print(f"\n📊 Статистика предсказаний:")
    print(f"   price_p05: mean={pred_p05.mean():.4f}, std={pred_p05.std():.4f}")
    print(f"   price_p95: mean={pred_p95.mean():.4f}, std={pred_p95.std():.4f}")
    print(f"   width: mean={(pred_p95 - pred_p05).mean():.4f}")
    
    # 5. Создание submission
    print("\n💾 Создание submission файла...")
    submission = pd.DataFrame({
        'row_id': row_ids,
        'price_p05': pred_p05,
        'price_p95': pred_p95
    })
    
    # Проверка на отсутствие NaN
    if submission.isna().any().any():
        print("⚠️ Обнаружены NaN значения, заполняем средними...")
        submission['price_p05'] = submission['price_p05'].fillna(submission['price_p05'].mean())
        submission['price_p95'] = submission['price_p95'].fillna(submission['price_p95'].mean())
    
    create_submission(submission)
    
    print("\n" + "=" * 60)
    print("✅ Выполнение завершено успешно!")
    print("=" * 60)


if __name__ == "__main__":
    main()
