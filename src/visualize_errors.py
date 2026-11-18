from ultralytics import YOLO
from pathlib import Path
import cv2
import random

def run_visualization():
    # Путь к лучшей модели E1 (из твоего лога)
    model_path = "runs/detect/runs/ppe/e1_baseline-2/weights/best.pt"
    
    # Проверяем, существует ли модель
    if not Path(model_path).exists():
        print(f"Ошибка: модель не найдена по пути {model_path}")
        return
        
    model = YOLO(model_path)
    
    # Берем 10 картинок из теста
    test_dir = Path("data/test/images")
    all_images = list(test_dir.glob("*.jpg"))
    
    if len(all_images) < 10:
        print("В тестовой папке меньше 10 картинок!")
        return
        
    random.seed(42)
    sample_images = random.sample(all_images, 10)
    
    out_dir = Path("reports/figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Генерация галереи ошибок...")
    
    # Прогоняем модель
    results = model.predict(source=sample_images, conf=0.25, save=False)
    
    for i, r in enumerate(results):
        # Рисуем предсказания на картинке
        im_array = r.plot()
        out_path = out_dir / f"error_gallery_{i}.jpg"
        cv2.imwrite(str(out_path), im_array)
        
    print(f"Готово! Сохранено {len(results)} картинок в {out_dir}")

if __name__ == "__main__":
    run_visualization()