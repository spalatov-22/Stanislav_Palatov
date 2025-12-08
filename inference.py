"""
CatBoost PRO Inference
======================
Загружает обученные модели и генерирует предсказания.

Использование:
    python inference.py --input test.csv --output submission.csv
    python inference.py --input test.csv --output submission.csv --models_dir models
    python inference.py --input test.csv --best_only  # Только лучшая модель
"""

import numpy as np
import pandas as pd
import os
import json
import argparse
from glob import glob
import catboost as cb

from config import MODELS_DIR, SEED, set_seed
from utils import fill_missing
from features import FeatureGenerator

set_seed(SEED)


class InferenceEngine:
    """Движок для инференса CatBoost PRO моделей"""
    
    def __init__(self, models_dir: str = None):
        self.models_dir = models_dir or MODELS_DIR
        self.models = []
        self.feature_names = None
        self.best_params = None
        
    def load_models(self, use_best_only: bool = False) -> int:
        """Загрузить модели из директории
        
        Args:
            use_best_only: Если True, загружает только лучшую модель.
                          Если False, загружает все fold-модели (ensemble).
        """
        print(f"📂 Loading models from {self.models_dir}...")
        
        if use_best_only:
            # Загружаем только лучшую модель
            best_path = os.path.join(self.models_dir, 'catboost_pro_best.cbm')
            if not os.path.exists(best_path):
                raise FileNotFoundError(f"Best model not found: {best_path}")
            
            model = cb.CatBoost()
            model.load_model(best_path)
            self.models = [model]
            print(f"   ✅ Loaded best model: {os.path.basename(best_path)}")
        else:
            # Загружаем все fold-модели
            model_paths = sorted(glob(os.path.join(self.models_dir, 'catboost_pro_fold_*.cbm')))
            
            if not model_paths:
                raise FileNotFoundError(f"No models found in {self.models_dir}")
            
            self.models = []
            for path in model_paths:
                model = cb.CatBoost()
                model.load_model(path)
                self.models.append(model)
                print(f"   ✅ Loaded: {os.path.basename(path)}")
        
        print(f"   Total models: {len(self.models)}")
        
        # Загрузить feature names
        feature_names_path = os.path.join(self.models_dir, 'feature_names_pro.json')
        if os.path.exists(feature_names_path):
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
            print(f"   ✅ Loaded feature names: {len(self.feature_names)} features")
        
        # Загрузить best params
        params_path = os.path.join(self.models_dir, 'catboost_pro_best_params.json')
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.best_params = json.load(f)
            print(f"   ✅ Loaded best params")
        
        return len(self.models)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Предсказать с усреднением по всем моделям"""
        if not self.models:
            raise ValueError("No models loaded! Call load_models() first.")
        
        predictions = np.zeros(len(X))
        
        for i, model in enumerate(self.models):
            pred = model.predict(X)
            predictions += pred / len(self.models)
            print(f"   Model {i+1}/{len(self.models)} predicted")
        
        return predictions
    
    def generate_submission(self, test_df: pd.DataFrame, predictions: np.ndarray,
                           output_path: str = 'submission.csv') -> pd.DataFrame:
        """Создать файл submission (формат: id, prediction)"""
        submission = pd.DataFrame({
            'id': test_df['id'],
            'prediction': predictions
        })
        
        submission.to_csv(output_path, index=False)
        
        return submission


def main():
    parser = argparse.ArgumentParser(description='CatBoost PRO Inference')
    parser.add_argument('--input', '-i', type=str, required=True,
                       help='Path to input CSV file (like test.csv)')
    parser.add_argument('--output', '-o', type=str, default='submission.csv',
                       help='Path to output submission CSV')
    parser.add_argument('--models_dir', '-m', type=str, default=None,
                       help='Directory with trained models')
    parser.add_argument('--best_only', action='store_true',
                       help='Use only best model instead of ensemble')
    
    args = parser.parse_args()
    
    print("="*80)
    print("🚀 CATBOOST PRO INFERENCE")
    print("="*80)
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Model mode: {'Best only' if args.best_only else 'Ensemble (all folds)'}")
    print("="*80)
    
    # 1. Загрузить модели
    engine = InferenceEngine(args.models_dir)
    n_models = engine.load_models(use_best_only=args.best_only)
    
    if n_models == 0:
        print("❌ No models found!")
        return
    
    # 2. Загрузить данные
    print(f"\n📂 Loading data from {args.input}...")
    test_df = fill_missing(pd.read_csv(args.input))
    print(f"   Rows: {len(test_df)}")
    print(f"   Columns: {list(test_df.columns)}")
    
    # 3. Генерация фичей
    print("\n🔧 Generating features...")
    generator = FeatureGenerator(use_cache=False)
    
    X, feature_names = generator.generate_all_features(
        test_df,
        use_semantic=True,
        use_reranker=True
    )
    
    print(f"   Features shape: {X.shape}")
    
    # 4. Предсказания
    print("\n🔮 Generating predictions...")
    predictions = engine.predict(X)
    
    print(f"\n📈 Prediction Statistics:")
    print(f"   Min: {predictions.min():.4f}")
    print(f"   Max: {predictions.max():.4f}")
    print(f"   Mean: {predictions.mean():.4f}")
    print(f"   Std: {predictions.std():.4f}")
    
    # 5. Создание submission
    print(f"\n💾 Creating submission: {args.output}")
    submission = engine.generate_submission(test_df, predictions, args.output)
    
    print(f"\n✅ Submission saved!")
    print(f"   Rows: {len(submission)}")
    print(f"   Columns: {list(submission.columns)}")
    
    print("\n" + "="*80)
    print("🎉 INFERENCE COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    main()

