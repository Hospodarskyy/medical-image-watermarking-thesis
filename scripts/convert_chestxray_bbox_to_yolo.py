"""
Конвертер анотацій NIH ChestX-ray14 (BBox_List_2017.csv) у датасет формату YOLO.

Вхідний формат (BBox_List_2017.csv), офіційні колонки NIH:
    Image Index, Finding Label, Bbox [x, y, w, h], (+ кілька порожніх/службових колонок)
де x, y — координати лівого верхнього кута bbox у пікселях,
    w, h — ширина і висота bbox у пікселях.
Розмір зображень NIH ChestX-ray14: 1024x1024 (більшість; код бере реальний розмір з файлу).

Вихід (--out, типово data/chestxray14/yolo):
    images/{train,val}/  — копії лише анотованих знімків (знімок без .txt YOLO вважає фоном,
                           тому тренувати на всій папці raw/images не можна)
    labels/{train,val}/  — один .txt на зображення:
                           <class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>
    data.yaml            — конфіг для Ultralytics

Train/val ділиться по пацієнтах (ID пацієнта — префікс імені файлу), а не по знімках,
щоб знімки одного пацієнта не потрапили в обидві частини.

Використання (усі шляхи мають типові значення відносно кореня репозиторію):
    python scripts/convert_chestxray_bbox_to_yolo.py
    python scripts/convert_chestxray_bbox_to_yolo.py --val-fraction 0.2 --seed 0
"""

import argparse
import csv
import random
import shutil
from pathlib import Path

from yolo_utils import REPO_ROOT, reset_split_dirs, write_data_yaml

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Потрібен Pillow: pip install Pillow")

DATASET_DIR = REPO_ROOT / "data" / "chestxray14"


def load_bboxes(bbox_csv_path: Path):
    """Зчитує BBox_List_2017.csv і групує bbox-и за зображенням."""
    rows_by_image = {}
    with open(bbox_csv_path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader)  # заголовок
        # Очікувані індекси колонок у офіційному файлі NIH:
        # 0: Image Index, 1: Finding Label, 2: Bbox [x, 3: y, 4: w, 5: h]
        for row in reader:
            if not row or not row[0].strip():
                continue
            image_name = row[0].strip()
            label = row[1].strip()
            x, y, w, h = (float(row[2]), float(row[3]), float(row[4]), float(row[5]))
            rows_by_image.setdefault(image_name, []).append((label, x, y, w, h))
    return rows_by_image


def build_class_map(rows_by_image):
    labels = sorted({label for boxes in rows_by_image.values() for label, *_ in boxes})
    return {label: idx for idx, label in enumerate(labels)}


def patient_id(image_name: str) -> str:
    return image_name.split("_")[0]


def pick_val_patients(image_names, val_fraction: float, seed: int):
    """Обирає пацієнтів для val.

    Рахується по всіх пацієнтах з CSV, а не лише по наявних на диску знімках,
    тому розподіл не змінюється, коли довантажуються нові архіви.
    """
    patients = sorted({patient_id(name) for name in image_names})
    random.Random(seed).shuffle(patients)
    return set(patients[: round(len(patients) * val_fraction)])


def to_yolo_line(class_id: int, x, y, w, h, img_w, img_h):
    """Обрізає bbox по межах зображення і повертає рядок YOLO (або None, якщо bbox вироджений)."""
    x1, y1 = max(x, 0.0), max(y, 0.0)
    x2, y2 = min(x + w, img_w), min(y + h, img_h)
    if x2 <= x1 or y2 <= y1:
        return None
    x_center = (x1 + x2) / 2 / img_w
    y_center = (y1 + y2) / 2 / img_h
    return f"{class_id} {x_center:.6f} {y_center:.6f} {(x2 - x1) / img_w:.6f} {(y2 - y1) / img_h:.6f}"


def convert(bbox_csv: Path, images_dir: Path, out_dir: Path, val_fraction: float, seed: int):
    rows_by_image = load_bboxes(bbox_csv)
    class_map = build_class_map(rows_by_image)
    val_patients = pick_val_patients(rows_by_image, val_fraction, seed)

    splits = ("train", "val")
    reset_split_dirs(out_dir, splits)

    counts = dict.fromkeys(splits, 0)
    missing, bad_boxes = 0, 0
    for image_name, boxes in rows_by_image.items():
        image_path = images_dir / image_name
        if not image_path.exists():
            missing += 1
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        label_lines = []
        for label, x, y, w, h in boxes:
            line = to_yolo_line(class_map[label], x, y, w, h, img_w, img_h)
            if line is None:
                bad_boxes += 1
                continue
            label_lines.append(line)
        if not label_lines:
            continue

        split = "val" if patient_id(image_name) in val_patients else "train"
        shutil.copy2(image_path, out_dir / "images" / split / image_name)
        label_path = out_dir / "labels" / split / (image_path.stem + ".txt")
        with open(label_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(label_lines) + "\n")
        counts[split] += 1

    yaml_path = write_data_yaml(out_dir, class_map, splits)

    print(f"Готово. train: {counts['train']}, val: {counts['val']} знімків")
    print(f"Класи ({len(class_map)}): {list(class_map)}")
    print(f"Конфіг збережено: {yaml_path}")
    if bad_boxes:
        print(f"УВАГА: пропущено вироджених bbox: {bad_boxes}")
    if missing:
        print(
            f"УВАГА: {missing} з {len(rows_by_image)} анотованих знімків не знайдено в {images_dir} "
            "— розпакуйте решту архівів images_0XX.tar.gz і запустіть скрипт ще раз."
        )
    if not counts["train"] or not counts["val"]:
        raise SystemExit("Помилка: train або val порожній — недостатньо знімків для тренування.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox-csv", type=Path, default=DATASET_DIR / "raw" / "BBox_List_2017.csv",
                        help="Шлях до BBox_List_2017.csv")
    parser.add_argument("--images-dir", type=Path, default=DATASET_DIR / "raw" / "images",
                        help="Папка з розпакованими .png зображеннями")
    parser.add_argument("--out", type=Path, default=DATASET_DIR / "yolo",
                        help="Куди зберегти YOLO-датасет (images/, labels/, data.yaml)")
    parser.add_argument("--val-fraction", type=float, default=0.2, help="Частка пацієнтів у val")
    parser.add_argument("--seed", type=int, default=0, help="Seed для розподілу пацієнтів")
    args = parser.parse_args()

    convert(args.bbox_csv, args.images_dir, args.out, args.val_fraction, args.seed)
