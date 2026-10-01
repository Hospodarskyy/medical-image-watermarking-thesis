"""
Конвертер VinDr-SpineXR (DICOM + CSV-анотації) у датасет формату YOLO.

Вхід (--raw-dir, типово data/vindr-spinexr/raw):
    train_images/*.dicom, test_images/*.dicom
    annotations/train.csv, annotations/test.csv
    Колонки CSV: study_id, series_id, image_id, rad_id, lesion_type, xmin, ymin, xmax, ymax
    де xmin..ymax — кути bbox в абсолютних пікселях; у рядків lesion_type="No finding"
    поля bbox порожні. Розмір зображення в CSV не вказаний — береться з DICOM.

Вихід (--out, типово data/vindr-spinexr/yolo):
    images/{train,val,test}/*.png  — 8-бітні знімки після VOI LUT (MONOCHROME1 інвертується)
    labels/{train,val,test}/*.txt  — <class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>;
                                     для "No finding" файл порожній (негативний приклад)
    classes.txt, data.yaml

Офіційний розподіл уже patient-level (study_id у train.csv і test.csv не перетинаються):
test.csv цілком стає test — незалежним набором для фінальної оцінки, а train.csv ділиться
на train і val по study_id, щоб знімки одного дослідження не потрапили в обидві частини.

Кожен знімок розмічений рівно одним радіологом (rad_id), тому Weighted Box Fusion тут
не зливає думки кількох радіологів, а усуває дублікати анотацій одного радіолога:
bbox-и одного класу з IoU > --iou-thr (той самий bbox, введений двічі) стають одним.
Окремі ураження одного класу на одному знімку (multi-instance, напр. Osteophytes на
різних рівнях хребта) майже не перекриваються і лишаються окремими bbox. У train.csv
з 3403 груп "знімок + клас" із кількома bbox 97.71% мають max IoU < 0.1 і лише 1.62% —
max IoU > 0.5. Кути bbox обрізаються по межах зображення до злиття і до перерахунку
в center/width/height.

Використання (усі шляхи мають типові значення відносно кореня репозиторію):
    python scripts/convert_vindr_spinexr_to_yolo.py
    python scripts/convert_vindr_spinexr_to_yolo.py --val-fraction 0.1 --seed 0
    python scripts/convert_vindr_spinexr_to_yolo.py --limit 100 --out data/vindr-spinexr/yolo-sanity
"""

import argparse
import csv
import os
import random
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from yolo_utils import REPO_ROOT, reset_split_dirs, write_data_yaml

try:
    import numpy as np
    import pydicom
    from ensemble_boxes import weighted_boxes_fusion
    from PIL import Image
    from pydicom.pixels import apply_modality_lut, apply_voi_lut
except ImportError as e:
    raise SystemExit(f"{e}\nПотрібні пакети: pip install pydicom ensemble-boxes Pillow numpy "
                     "(для JPEG 2000 DICOM — Pillow з підтримкою OpenJPEG або pylibjpeg)")

DATASET_DIR = REPO_ROOT / "data" / "vindr-spinexr"

SPLITS = ("train", "val", "test")

NO_FINDING = "No finding"
CLASS_NAMES = [
    "Disc space narrowing",
    "Foraminal stenosis",
    "Osteophytes",
    "Other lesions",
    "Spondylolysthesis",
    "Surgical implant",
    "Vertebral collapse",
]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_annotations(csv_path: Path):
    """Зчитує CSV і групує bbox-и за зображенням: image_id -> [(rad_id, class_id, xmin, ymin, xmax, ymax)].

    Зображення лише з рядками "No finding" отримує порожній список.
    Другим значенням повертає image_id -> study_id.
    """
    boxes_by_image, study_by_image = {}, {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            image_id = row["image_id"].strip()
            study_by_image[image_id] = row["study_id"].strip()
            boxes = boxes_by_image.setdefault(image_id, [])
            label = row["lesion_type"].strip()
            if label == NO_FINDING:
                continue
            corners = tuple(float(row[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
            boxes.append((row["rad_id"].strip(), CLASS_IDS[label], *corners))
    return boxes_by_image, study_by_image


def split_annotations(annotations_dir: Path, val_fraction: float, seed: int):
    """Повертає частина YOLO -> (офіційна частина, {image_id: bbox-и}).

    val відбирається з train.csv по study_id; test.csv не чіпається.
    """
    train_boxes, study_by_image = load_annotations(annotations_dir / "train.csv")
    test_boxes, _ = load_annotations(annotations_dir / "test.csv")

    studies = sorted(set(study_by_image.values()))
    random.Random(seed).shuffle(studies)
    val_studies = set(studies[: round(len(studies) * val_fraction)])

    parts = {"train": {}, "val": {}}
    for image_id, boxes in train_boxes.items():
        parts["val" if study_by_image[image_id] in val_studies else "train"][image_id] = boxes
    return {"train": ("train", parts["train"]), "val": ("train", parts["val"]), "test": ("test", test_boxes)}


def repair_window_tags(ds):
    """Виправляє WindowCenter/WindowWidth, записані з порушенням формату.

    У частини файлів значення має зайву кому ("32688,"), тому pydicom лишає його рядком
    і apply_voi_lut падає на порівнянні рядка з числом.
    """
    for keyword in ("WindowCenter", "WindowWidth"):
        value = ds.get(keyword)
        if isinstance(value, str):
            numbers = [float(part) for part in value.replace("\\", ",").split(",") if part.strip()]
            setattr(ds, keyword, numbers[0] if len(numbers) == 1 else numbers)


def dicom_to_uint8(ds):
    """Повертає 8-бітне зображення: modality LUT -> VOI LUT -> інверсія MONOCHROME1 -> [0, 255]."""
    repair_window_tags(ds)
    pixels = apply_voi_lut(apply_modality_lut(ds.pixel_array, ds), ds).astype(np.float32)
    if ds.PhotometricInterpretation == "MONOCHROME1":
        pixels = pixels.max() - pixels
    pixels -= pixels.min()
    peak = pixels.max()
    if peak > 0:
        pixels *= 255.0 / peak
    return pixels.round().astype(np.uint8)


def fuse_boxes(boxes, img_w, img_h, iou_thr):
    """Обрізає кути bbox по межах зображення й усуває дублікати анотацій через WBF.

    Зливаються лише bbox-и одного класу з IoU > iou_thr; окремі ураження одного класу
    лишаються окремими. Bbox-и групуються за rad_id, бо WBF приймає по списку на "модель";
    у цьому датасеті на знімок припадає один rad_id, тобто один список.
    Повертає список (class_id, x1, y1, x2, y2) у нормалізованих координатах [0, 1].
    """
    by_rad = {}
    for rad_id, class_id, xmin, ymin, xmax, ymax in boxes:
        x1, y1 = min(max(xmin, 0.0), img_w), min(max(ymin, 0.0), img_h)
        x2, y2 = min(max(xmax, 0.0), img_w), min(max(ymax, 0.0), img_h)
        if x2 <= x1 or y2 <= y1:
            continue
        by_rad.setdefault(rad_id, []).append((class_id, x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h))
    if not by_rad:
        return []

    rads = sorted(by_rad)
    fused_boxes, _, fused_labels = weighted_boxes_fusion(
        [[box[1:] for box in by_rad[rad]] for rad in rads],
        [[1.0] * len(by_rad[rad]) for rad in rads],
        [[box[0] for box in by_rad[rad]] for rad in rads],
        iou_thr=iou_thr,
        skip_box_thr=0.0,
    )
    return [(int(label), *box) for label, box in zip(fused_labels, fused_boxes.tolist())]


def to_yolo_line(class_id, x1, y1, x2, y2):
    return f"{class_id} {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}"


def convert_image(task):
    """Конвертує один знімок; запускається в окремому процесі."""
    dicom_path, image_out, label_out, boxes, iou_thr = task
    try:
        ds = pydicom.dcmread(dicom_path)
        image = dicom_to_uint8(ds)
    except Exception as e:  # пошкоджений файл не має зупиняти решту конвертації
        return {"error": f"{dicom_path.name}: {type(e).__name__}: {e}"}
    img_h, img_w = image.shape

    fused = fuse_boxes(boxes, img_w, img_h, iou_thr)
    Image.fromarray(image).save(image_out)
    label_out.write_text("".join(to_yolo_line(*box) + "\n" for box in fused), encoding="utf-8", newline="\n")
    return {
        "raw": len(boxes),
        "classes": Counter(box[0] for box in fused),
        "rads": len({box[0] for box in boxes}),
        "inverted": ds.PhotometricInterpretation == "MONOCHROME1",
    }


def convert(raw_dir: Path, out_dir: Path, val_fraction: float, seed: int, iou_thr: float, workers: int, limit: int):
    parts = split_annotations(raw_dir / "annotations", val_fraction, seed)
    reset_split_dirs(out_dir, SPLITS)

    for split in SPLITS:
        source, boxes_by_image = parts[split]
        image_ids = sorted(boxes_by_image)[:limit] if limit else sorted(boxes_by_image)

        tasks, missing = [], []
        for image_id in image_ids:
            dicom_path = raw_dir / f"{source}_images" / f"{image_id}.dicom"
            if not dicom_path.exists():
                missing.append(image_id)
                continue
            tasks.append((dicom_path, out_dir / "images" / split / f"{image_id}.png",
                          out_dir / "labels" / split / f"{image_id}.txt", boxes_by_image[image_id], iou_thr))

        raw_boxes, images, negatives, inverted = 0, 0, 0, 0
        class_counts, rad_counts, errors = Counter(), Counter(), []
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for done, result in enumerate(pool.map(convert_image, tasks, chunksize=8), start=1):
                if done % 500 == 0 or done == len(tasks):
                    print(f"  {split}: {done}/{len(tasks)}", flush=True)
                if "error" in result:
                    errors.append(result["error"])
                    continue
                images += 1
                raw_boxes += result["raw"]
                class_counts += result["classes"]
                negatives += not result["classes"]
                inverted += result["inverted"]
                if result["rads"]:
                    rad_counts[result["rads"]] += 1

        fused_boxes = sum(class_counts.values())
        print(f"{split}: {images} знімків ({negatives} без bbox, {inverted} MONOCHROME1), "
              f"bbox: {raw_boxes} у CSV -> {fused_boxes} після обрізання та WBF")
        print("  радіологів на знімок (з bbox): "
              + ", ".join(f"{n} -> {count} знімків" for n, count in sorted(rad_counts.items())))
        for idx, name in enumerate(CLASS_NAMES):
            print(f"  {idx} {name}: {class_counts[idx]}")
        if missing:
            print(f"  УВАГА: немає DICOM для {len(missing)} знімків, напр.: {missing[:5]}")
        if errors:
            print(f"  УВАГА: не вдалося прочитати {len(errors)} DICOM, напр.: {errors[:5]}")
        if not images:
            raise SystemExit(f"Помилка: {split} порожній — перевірте {raw_dir / f'{source}_images'}.")

    yaml_path = write_data_yaml(out_dir, CLASS_NAMES, SPLITS)
    print(f"Готово. Конфіг збережено: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=DATASET_DIR / "raw",
                        help="Папка з train_images/, test_images/, annotations/")
    parser.add_argument("--out", type=Path, default=DATASET_DIR / "yolo",
                        help="Куди зберегти YOLO-датасет (images/, labels/, classes.txt, data.yaml)")
    parser.add_argument("--val-fraction", type=float, default=0.1,
                        help="Частка досліджень (study_id) з train.csv, що йде у val")
    parser.add_argument("--seed", type=int, default=0, help="Seed для розподілу досліджень на train/val")
    parser.add_argument("--iou-thr", type=float, default=0.5,
                        help="Поріг IoU, вище якого bbox-и одного класу вважаються дублікатом і зливаються (WBF)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                        help="Кількість процесів для декодування DICOM")
    parser.add_argument("--limit", type=int, default=0,
                        help="Конвертувати лише перші N знімків кожної частини (для sanity check)")
    args = parser.parse_args()

    convert(args.raw_dir, args.out, args.val_fraction, args.seed, args.iou_thr, args.workers, args.limit)
