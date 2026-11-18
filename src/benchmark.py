import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO

def benchmark():
    # Путь к модели E1
    model_path = "runs/detect/runs/ppe/e1_baseline-2/weights/best.pt"
    model = YOLO(model_path)
    
    test_dir = Path("data/test/images")
    images = list(test_dir.glob("*.jpg"))[:100] # Берем 100 картинок
    
    print(f"Замер задержки на {len(images)} изображениях...")
    
    times = []
    # Первый прогон (разогрев)
    model.predict(source=images[0], verbose=False)
    
    for img in images:
        start = time.perf_counter()
        model.predict(source=img, verbose=False)
        end = time.perf_counter()
        times.append((end - start) * 1000) # в миллисекундах
        
    times = np.array(times)
    median = np.median(times)
    p95 = np.percentile(times, 95)
    
    print(f"Медиана: {median:.2f} мс")
    print(f"95-й процентиль: {p95:.2f} мс")
    print(f"Среднее: {np.mean(times):.2f} мс")
    
    # Сохраняем в файл
    out_path = Path("reports/latency.txt")
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"Latency benchmark (100 images)\n")
        f.write(f"Median: {median:.2f} ms\n")
        f.write(f"95th percentile: {p95:.2f} ms\n")
        f.write(f"Mean: {np.mean(times):.2f} ms\n")
    
    print(f"Результаты сохранены в {out_path}")

if __name__ == "__main__":
    benchmark()