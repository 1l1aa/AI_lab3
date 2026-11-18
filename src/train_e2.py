from ultralytics import YOLO

def train_e2():
    model = YOLO("yolov8n.pt")
    
    results = model.train(
        data="data/ppe_data.yaml",
        epochs=50,
        imgsz=1280,      # <--- ЕДИНСТВЕННОЕ ИЗМЕНЕНИЕ (было 640)
        batch=8,         # <--- Уменьшили, чтобы влезло в память
        seed=42,
        project="runs/ppe",
        name="e2_imgsz1280",
        device=0
    )
    
    print("Обучение E2 завершено!")

if __name__ == "__main__":
    train_e2()