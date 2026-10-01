"""
Підготовка FracAtlas до тренування YOLO.

FracAtlas уже має анотації у форматі YOLO (Annotations/YOLO) та офіційний розподіл
(Utilities/Fracture Split/{train,valid,test}.csv), тому конвертація не потрібна —
скрипт лише розкладає знімки й мітки у структуру, яку очікує Ultralytics.

Офіційний розподіл містить тільки знімки з переломами; знімки без переломів
у YOLO-датасет не потрапляють.

Вихід (--out, типово data/fracatlas/yolo):
    images/{train,val,test}/
    labels/{train,val,test}/
    data.yaml, classes.txt

Використання:
    python scripts/prepare_fracatlas_yolo.py
"""

import argparse
import csv
import shutil
from pathlib import Path

from yolo_utils import REPO_ROOT, reset_split_dirs, write_data_yaml

DATASET_DIR = REPO_ROOT / "data" / "fracatlas"

# назва частини в YOLO -> файл офіційного розподілу
SPLIT_FILES = {"train": "train.csv", "val": "valid.csv", "test": "test.csv"}


def read_split(csv_path: Path):
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        return [row["image_id"].strip() for row in csv.DictReader(f) if row["image_id"].strip()]


def find_image(images_dir: Path, image_name: str):
    # IMG0003375.jpg та IMG0003376.jpg лежать в обох папках; за dataset.csv це переломи,
    # тому Fractured перевіряється першою.
    for subdir in ("Fractured", "Non_fractured"):
        path = images_dir / subdir / image_name
        if path.exists():
            return path
    return None


def prepare(raw_dir: Path, out_dir: Path):
    images_dir = raw_dir / "images"
    labels_dir = raw_dir / "Annotations" / "YOLO"
    split_dir = raw_dir / "Utilities" / "Fracture Split"
    class_names = (labels_dir / "labels.txt").read_text(encoding="utf-8").split()

    reset_split_dirs(out_dir, SPLIT_FILES)

    counts = dict.fromkeys(SPLIT_FILES, 0)
    missing = []
    for split, split_file in SPLIT_FILES.items():
        for image_name in read_split(split_dir / split_file):
            image_path = find_image(images_dir, image_name)
            label_path = labels_dir / (Path(image_name).stem + ".txt")
            if image_path is None or not label_path.exists():
                missing.append(image_name)
                continue
            shutil.copy2(image_path, out_dir / "images" / split / image_name)
            shutil.copy2(label_path, out_dir / "labels" / split / label_path.name)
            counts[split] += 1

    yaml_path = write_data_yaml(out_dir, class_names, SPLIT_FILES)

    print("Готово. " + ", ".join(f"{split}: {n}" for split, n in counts.items()) + " знімків")
    print(f"Класи ({len(class_names)}): {class_names}")
    print(f"Конфіг збережено: {yaml_path}")
    if missing:
        print(f"УВАГА: не знайдено зображення або мітку для {len(missing)} знімків, напр.: {missing[:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=DATASET_DIR / "raw",
                        help="Папка з розпакованим FracAtlas (images/, Annotations/, Utilities/)")
    parser.add_argument("--out", type=Path, default=DATASET_DIR / "yolo",
                        help="Куди зберегти YOLO-датасет (images/, labels/, data.yaml)")
    args = parser.parse_args()

    prepare(args.raw_dir, args.out)
