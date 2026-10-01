"""
Конвертер анотацій NIH ChestX-ray14 (BBox_List_2017.csv) у формат YOLO.

Вхідний формат (BBox_List_2017.csv), офіційні колонки NIH:
    Image Index, Finding Label, Bbox [x, y, w, h], (+ кілька порожніх/службових колонок)
де x, y — координати лівого верхнього кута bbox у пікселях,
    w, h — ширина і висота bbox у пікселях.
Розмір зображень NIH ChestX-ray14: 1024x1024 (більшість; код бере реальний розмір з файлу).

Вихідний формат YOLO (один .txt на зображення, у тому ж імені):
    <class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>
усі значення нормалізовані до [0, 1] відносно розміру зображення.

Використання:
    python convert_chestxray_bbox_to_yolo.py \
        --bbox-csv BBox_List_2017.csv \
        --images-dir ./images \
        --labels-out ./labels \
        --classes-out ./classes.txt
"""

import argparse
import csv
import os
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    raise SystemExit("Потрібен Pillow: pip install Pillow")


def load_bboxes(bbox_csv_path: str):
    """Зчитує BBox_List_2017.csv і групує bbox-и за зображенням."""
    rows_by_image = {}
    with open(bbox_csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
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


def convert(bbox_csv: str, images_dir: str, labels_out: str, classes_out: str):
    rows_by_image = load_bboxes(bbox_csv)
    class_map = build_class_map(rows_by_image)

    os.makedirs(labels_out, exist_ok=True)
    with open(classes_out, "w", encoding="utf-8") as f:
        for label, idx in sorted(class_map.items(), key=lambda kv: kv[1]):
            f.write(f"{label}\n")

    converted, skipped = 0, 0
    for image_name, boxes in rows_by_image.items():
        image_path = Path(images_dir) / image_name
        if not image_path.exists():
            skipped += 1
            continue

        with Image.open(image_path) as img:
            img_w, img_h = img.size

        label_lines = []
        for label, x, y, w, h in boxes:
            class_id = class_map[label]
            x_center = (x + w / 2) / img_w
            y_center = (y + h / 2) / img_h
            w_norm = w / img_w
            h_norm = h / img_h

            # Захист від виходу за межі [0, 1] через похибки округлення
            x_center = min(max(x_center, 0.0), 1.0)
            y_center = min(max(y_center, 0.0), 1.0)
            w_norm = min(max(w_norm, 0.0), 1.0)
            h_norm = min(max(h_norm, 0.0), 1.0)

            label_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}")

        label_filename = Path(labels_out) / (image_path.stem + ".txt")
        with open(label_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(label_lines) + "\n")
        converted += 1

    print(f"Готово. Конвертовано: {converted}, пропущено (зображення не знайдено): {skipped}")
    print(f"Класи ({len(class_map)}): {list(class_map.keys())}")
    print(f"Файл класів збережено: {classes_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bbox-csv", required=True, help="Шлях до BBox_List_2017.csv")
    parser.add_argument("--images-dir", required=True, help="Папка з .png зображеннями")
    parser.add_argument("--labels-out", required=True, help="Куди зберегти .txt анотації YOLO")
    parser.add_argument("--classes-out", default="classes.txt", help="Куди зберегти список класів")
    args = parser.parse_args()

    convert(args.bbox_csv, args.images_dir, args.labels_out, args.classes_out)
