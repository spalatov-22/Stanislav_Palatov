"""
Основной файл с решением соревнования.
Запускает полноценный инференс и сохраняет submission.
"""

import os
from typing import Optional
import pandas as pd

from config import SEED, set_seed
from utils import fill_missing
from inference import InferenceEngine
from features import FeatureGenerator

# Путь к тестовым данным ищем по умолчанию в двух местах
DEFAULT_TEST_PATHS = [
    os.path.join("data", "test.csv"),
    "test.csv",
]
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_SUBMISSION_NAME = "submission.csv"


def _resolve_test_path(custom_path: Optional[str] = None) -> str:
    """Определить путь к тестовому файлу."""
    if custom_path:
        if os.path.exists(custom_path):
            return custom_path
        raise FileNotFoundError(f"Не найден указанный test файл: {custom_path}")

    for path in DEFAULT_TEST_PATHS:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        "Не найден test.csv. Поместите файл в корень проекта "
        "или в папку data/test.csv."
    )


def create_submission(
    test_df: pd.DataFrame,
    predictions,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    filename: str = DEFAULT_SUBMISSION_NAME,
) -> str:
    """
    Создает файл submission в формате SAMPLE SUBMISSION.
    """
    os.makedirs(output_dir, exist_ok=True)
    submission = pd.DataFrame(
        {
            "id": test_df["id"],
            "prediction": predictions,
        }
    )
    submission_path = os.path.join(output_dir, filename)
    submission.to_csv(submission_path, index=False)

    print(f"Submission файл сохранен: {submission_path}")
    return submission_path


def main(
    test_path: Optional[str] = None,
    use_best_only: bool = False,
    use_semantic: bool = True,
    use_reranker: bool = True,
):
    """
    Полный цикл инференса: загрузка данных -> генерация фичей -> предсказания.

    Вы можете менять параметры, но обязательно вызывается create_submission().
    """
    print("=" * 80)
    print("🚀 Запуск решения соревнования")
    print("=" * 80)

    set_seed(SEED)

    resolved_test_path = _resolve_test_path(test_path)
    print(f"📂 Используем тестовый файл: {resolved_test_path}")
    test_df = fill_missing(pd.read_csv(resolved_test_path))
    print(f"   Загружено строк: {len(test_df)}")

    # 1. Загрузка моделей
    engine = InferenceEngine()
    engine.load_models(use_best_only=use_best_only)

    # 2. Генерация фичей
    generator = FeatureGenerator(use_cache=False)
    X, feature_names = generator.generate_all_features(
        test_df,
        use_semantic=use_semantic,
        use_reranker=use_reranker,
    )
    print(f"   Сформирован набор фич: {X.shape}, всего фичей: {len(feature_names)}")

    # 3. Предсказания
    predictions = engine.predict(X)
    print("   Предсказания готовы")

    # 4. Создание submission
    create_submission(test_df, predictions)

    print("=" * 80)
    print("✅ Выполнение завершено успешно!")
    print("=" * 80)


if __name__ == "__main__":
    main()
