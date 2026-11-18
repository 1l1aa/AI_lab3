from ultralytics import YOLO

def train_e1():
    # Загружаем предобученную модель
    model = YOLO("yolov8n.pt")
    
    # Запускаем обучение
    results = model.train(
        data="data/ppe_data.yaml", # путь к твоему yaml
        epochs=50,                 # количество эпох
        imgsz=640,                 # размер изображения
        batch=16,                  # размер батча (если не хватит памяти, поставь 8)
        seed=42,                   # для воспроизводимости
        project="runs/ppe",        # куда сохранять результаты
        name="e1_baseline",         # имя эксперимента
        device=0
    )
    
    print("Обучение E1 завершено! Веса сохранены в runs/ppe/e1_baseline/weights/best.pt")

if __name__ == "__main__":
    train_e1()