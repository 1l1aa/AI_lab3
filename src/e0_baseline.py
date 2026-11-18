import random
from pathlib import Path
from ultralytics import YOLO

def run_e0():
    # 1. Загружаем стандартную модель
    model = YOLO("yolov8n.pt")
    
    # 2. Берем 10 случайных картинок из теста
    test_images_dir = Path("data/test/images")
    all_images = list(test_images_dir.glob("*.jpg"))
    
    if len(all_images) < 10:
        print("В тестовой папке меньше 10 картинок!")
        return
        
    random.seed(42)
    sample_images = random.sample(all_images, 10)
    
    # 3. Делаем предсказание и сохраняем
    print("Запуск E0 на 10 изображениях...")
    results = model.predict(source=sample_images, save=True, project="runs", name="e0_predictions")
    
    print("Результаты сохранены в: runs/e0_predictions")

if __name__ == "__main__":
    run_e0()