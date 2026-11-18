import shutil
import random
from pathlib import Path
from collections import defaultdict

def get_group(filename: str) -> str:
    # Берем все до первого подчеркивания как имя группы (видео/сцены)
    return filename.split('_')[0]

def make_smart_split():
    data_dir = Path("data")
    all_images = []
    # Собираем все картинки из исходных папок
    for split in ["train", "val", "test"]:
        img_dir = data_dir / split / "images"
        if img_dir.exists():
            all_images.extend(list(img_dir.glob("*.jpg")))
    
    print(f"Всего найдено картинок: {len(all_images)}")
    
    groups = defaultdict(list)
    for img_path in all_images:
        group_name = get_group(img_path.name)
        groups[group_name].append(img_path)
    
    print(f"Всего уникальных групп: {len(groups)}")
    
    group_names = list(groups.keys())
    random.seed(42)
    random.shuffle(group_names)
    
    n_total = len(group_names)
    n_train = int(n_total * 0.70)
    n_val = int(n_total * 0.15)
    
    train_groups = group_names[:n_train]
    val_groups = group_names[n_train:n_train + n_val]
    test_groups = group_names[n_train + n_val:]
    
    # Создаем ВРЕМЕННЫЕ папки, чтобы не трогать исходники
    for split in ["train_new", "val_new", "test_new"]:
        (data_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (data_dir / split / "labels").mkdir(parents=True, exist_ok=True)
    
    def copy_files(groups_list, split_name):
        count = 0
        for g in groups_list:
            for img_path in groups[g]:
                # Копируем в НОВУЮ папку
                dst_img = data_dir / split_name / "images" / img_path.name
                shutil.copy2(img_path, dst_img)
                
                # Ищем разметку (в исходной папке)
                lbl_path = img_path.parent.parent / "labels" / (img_path.stem + ".txt")
                if lbl_path.exists():
                    dst_lbl = data_dir / split_name / "labels" / lbl_path.name
                    shutil.copy2(lbl_path, dst_lbl)
                count += 1
        return count

    print("Копирование файлов в новые папки...")
    c_train = copy_files(train_groups, "train_new")
    c_val = copy_files(val_groups, "val_new")
    c_test = copy_files(test_groups, "test_new")
    
    print(f"Готово! Распределение:")
    print(f"  Train: {c_train} картинок ({len(train_groups)} групп)")
    print(f"  Val:   {c_val} картинок ({len(val_groups)} групп)")
    print(f"  Test:  {c_test} картинок ({len(test_groups)} групп)")
    
    # Теперь переименовываем папки (удаляем старые и ставим новые)
    for old, new in [("train", "train_new"), ("val", "val_new"), ("test", "test_new")]:
        old_dir = data_dir / old
        new_dir = data_dir / new
        if old_dir.exists():
            shutil.rmtree(old_dir) # Удаляем старую папку
        new_dir.rename(old_dir)   # Переименовываем новую
    
    print("Папки обновлены! Старые данные удалены, новые на месте.")

if __name__ == "__main__":
    make_smart_split()