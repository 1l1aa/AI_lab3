import pytest
import yaml
from pathlib import Path
from ultralytics import YOLO

def test_yaml_paths():
    """Проверяет, что все папки из yaml существуют."""
    with open("data/ppe_data.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    
    base = Path("data")
    assert (base / cfg["train"]).exists(), "Папка train не найдена"
    assert (base / cfg["val"]).exists(), "Папка val не найдена"
    assert (base / cfg["test"]).exists(), "Папка test не найдена"

def test_model_prediction():
    """Проверяет, что модель загружается и делает предсказание."""
    model_path = "runs/detect/runs/ppe/e1_baseline-2/weights/best.pt"
    # Если файла нет, тест пропускается (чтобы не падал)
    if not Path(model_path).exists():
        pytest.skip("Модель не найдена, пропускаем тест")
        
    model = YOLO(model_path)
    # Берем первую попавшуюся картинку из теста
    test_img = next(Path("data/test/images").glob("*.jpg"))
    
    results = model.predict(source=test_img, verbose=False)
    
    # Проверяем, что вернулся список (даже если пустой)
    assert isinstance(results, list)
    assert len(results) > 0
