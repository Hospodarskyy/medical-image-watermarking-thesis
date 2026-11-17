"""
Аналіз детектора RoI: recall по типах уражень і розміру bbox, FROC, ROC на рівні знімка
та покриття уражень RoI-ами.

Детектор тренується з одним класом "RoI", тому його mAP не каже, які саме ураження він
пропускає і скільки зайвих RoI дає. Скрипт проганяє ваги по знімках частини датасету
(типово val) і рахує:

1. Recall по типах уражень і площі bbox. Для кожного bbox розмітки перевіряється, чи
   перекрито його хоч одним передбаченням (IoU >= поріг). Клас передбачення ігнорується,
   справжній клас береться з файлу міток. Зіставлення не один-до-одного: одне передбачення
   може "покрити" кілька bbox — для водяного маркування важливо саме, чи потрапило
   ураження в якусь RoI.
       --conf — пороги впевненості (0.25 — типовий для Ultralytics predict)
       --iou  — пороги IoU (0.5 — як у mAP50; 0.3 і 0.1 — для розмитих уражень)
2. FROC (free-response ROC) — стандартна крива для детекції уражень у медичних
   зображеннях: частка знайдених уражень (sensitivity) залежно від середньої кількості
   хибних RoI на знімок при зміні порогу conf. Хибна RoI — передбачення, що не перекриває
   жодного ураження з IoU >= поріг. Зведене число — CPM: середня sensitivity при
   1/8, 1/4, 1/2, 1, 2, 4, 8 хибних RoI на знімок (як у LUNA16).
3. ROC на рівні знімка — "чи є на знімку ураження", оцінка знімка = найвища впевненість
   RoI на ньому. Можлива лише там, де є знімки без уражень (VinDr-SpineXR); у val
   ChestX-ray14 і FracAtlas таких знімків немає, тоді ROC пропускається.
4. Покриття уражень — головна метрика для водяного маркування. IoU карає за завелику
   рамку, хоча для маркування важливо інше: чи потрапило ураження всередину RoI (тоді воно
   отримає крихкий маркер). Ураження вважається захищеним, якщо щонайменше --cover (типово
   90%) його площі лежить усередині об'єднання RoI: площа(ураження ∩ ⋃RoI) / площа(ураження).
   Ціна — частка площі знімка під RoI: вона забирає місце в RoNI для стійкого маркера і
   бітів відновлення. Рахується для сітки порогів conf і розширення рамок (кожна рамка
   збільшується на N% своєї ширини й висоти з кожного боку). Площі рахуються точно
   (стиснення координат), без растеризації.

З --tta передбачення робляться з аугментацією на інференсі (Ultralytics augment=True:
віддзеркалення й кілька масштабів, результати об'єднуються). Без перенавчання це підвищує
частку знайдених уражень ціною ~3× повільнішого передбачення. Метрики тоді пишуться з
префіксом roi_<частина>_tta/, файли — з суфіксом -tta, щоб не змішувати з базовими.

Вихід: таблиці в консолі; roi-recall-<частина>.csv, froc-<частина>.png, roc-<частина>.png,
roi-coverage-<частина>.csv, roi-coverage-<частина>.png
поруч з вагами (runs/<датасет>/<запуск>/). Якщо встановлено MLflow і знайдено запуск
тренування з такою самою назвою (для test — <назва>-test), метрики roi_<частина>/* і
файли записуються в нього (артефакти roi-analysis-<частина>/). train_yolo.py викликає
цей аналіз автоматично після тренування і в --eval-val / --eval-test.

Використання:
    python scripts/analyze_roi_recall.py chestxray14 runs/chestxray14/<запуск>/weights/best.pt
    python scripts/analyze_roi_recall.py chestxray14 <ваги> --conf 0.25 0.1 0.01 --iou 0.5 0.3 0.1
    python scripts/analyze_roi_recall.py vindr-spinexr <ваги> --tta
"""

import argparse
import csv
from pathlib import Path

import numpy as np

from yolo_utils import REPO_ROOT, mlflow_tracking_uri

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Потрібен Ultralytics: pip install ultralytics")

try:
    import mlflow
except ImportError:
    mlflow = None

import matplotlib

matplotlib.use("Agg")  # без вікон: графіки лише зберігаються у файли
import matplotlib.pyplot as plt  # noqa: E402

DATASETS = ("chestxray14", "fracatlas", "vindr-spinexr")
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
# межі груп за площею bbox, частка площі знімка
AREA_BINS = [(0.0, 0.01, "< 1%"), (0.01, 0.05, "1–5%"), (0.05, 0.15, "5–15%"), (0.15, 1.01, "> 15%")]
# найнижча впевненість, з якою беруться передбачення: криві мають покривати весь діапазон порогів
MIN_CONF = 0.001
FROC_FP_RATES = (0.125, 0.25, 0.5, 1, 2, 4, 8)
FROC_IOUS = (0.5, 0.3)
# покриття: розширення рамок (частка ширини/висоти з кожного боку) і пороги conf
COVERAGE_PADS = (0.0, 0.1, 0.25, 0.5)
COVERAGE_TABLE_CONFS = (0.25, 0.1, 0.05)  # точки, що йдуть у таблицю й метрики MLflow
COVERAGE_CURVE_CONFS = (0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01)


def load_ground_truth(labels_dir: Path):
    """Повертає image_stem -> масив [[class_id, x1, y1, x2, y2], ...] у нормалізованих координатах."""
    ground_truth = {}
    for label_path in sorted(labels_dir.glob("*.txt")):
        rows = [line.split() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        boxes = np.array([[float(v) for v in row] for row in rows], dtype=np.float64).reshape(-1, 5)
        cls, xc, yc, w, h = boxes.T
        ground_truth[label_path.stem] = np.stack([cls, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2], axis=1)
    return ground_truth


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU кожного bbox з a (N x 4) з кожним bbox з b (M x 4), формат x1, y1, x2, y2."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter)


def output_dir(dataset: str, weights: Path) -> Path:
    """runs/<датасет>/<запуск>/ для ваг з runs/, інакше runs/<датасет>/<назва ваг>/."""
    weights = weights.resolve()
    if weights.parent.name == "weights":
        return weights.parent.parent
    return REPO_ROOT / "runs" / dataset / weights.stem


def predict(weights: Path, images: list, imgsz, batch: int, augment: bool = False):
    """Повертає (image_stem -> (bbox-и N x 4 нормалізовані, впевненості N), imgsz).

    augment=True — TTA: кожен знімок проганяється у кількох масштабах і з віддзеркаленням,
    тому порції вдвічі менші, щоб уміститись у пам'ять GPU.
    """
    model = YOLO(weights)
    if imgsz is None:
        imgsz = (getattr(model, "ckpt", None) or {}).get("train_args", {}).get("imgsz", 640)
    # розігрів: перший виклик повільний (ініціалізація CUDA/NMS) і може перевищити ліміт часу
    # NMS в Ultralytics, через що передбачення частини знімків батчу обрізаються
    model.predict(source=[str(images[0])], imgsz=imgsz, conf=MIN_CONF, verbose=False)
    predictions = {}
    # Список шляхів Ultralytics завантажує в пам'ять цілим, як один батч, і параметр batch
    # ігнорує; для 854 знімків VinDr-SpineXR у повній роздільності це ~10 ГБ RAM. Тому порціями.
    chunk_size = max(1, batch // 2) if augment else batch
    for start in range(0, len(images), chunk_size):
        chunk = [str(p) for p in images[start:start + chunk_size]]
        # agnostic_nms: класи ігноруються і при придушенні дублікатів, як у детекторі RoI
        for result in model.predict(source=chunk, imgsz=imgsz, conf=MIN_CONF, agnostic_nms=True, augment=augment,
                                    verbose=False):
            predictions[Path(result.path).stem] = (result.boxes.xyxyn.cpu().numpy().astype(np.float64),
                                                   result.boxes.conf.cpu().numpy().astype(np.float64))
    return predictions, imgsz


def recall_tables(images, class_names, confs, ious):
    """Recall bbox розмітки по типах і площі. Повертає (рядки для консолі, рядки CSV, метрики)."""
    records = []  # по одному на bbox розмітки: клас, площа, найкращий IoU при кожному порозі conf
    empty, counts = dict.fromkeys(confs, 0), dict.fromkeys(confs, 0)
    lesion_images = [img for img in images if len(img["gt"])]
    for img in lesion_images:
        gt, conf, ious_gt_pred = img["gt"], img["conf"], img["ious"]
        areas = (gt[:, 3] - gt[:, 1]) * (gt[:, 4] - gt[:, 2])
        best = {}
        for c in confs:
            keep = conf >= c
            counts[c] += int(keep.sum())
            empty[c] += not keep.any()
            best[c] = ious_gt_pred[:, keep].max(axis=1) if keep.any() else np.zeros(len(gt))
        for i in range(len(gt)):
            records.append({"class_id": int(gt[i, 0]), "area": float(areas[i]), **{c: best[c][i] for c in confs}})

    columns = [(c, t) for c in confs for t in ious]
    header = [f"R conf≥{c} IoU≥{t}" for c, t in columns]

    def recall_row(group):
        return [float(np.mean([r[c] >= t for r in group])) if group else float("nan") for c, t in columns]

    n = len(lesion_images)
    lines = [f"знімків з ураженнями: {n}, bbox розмітки: {len(records)}",
             "Знімки з ураженнями без ЖОДНОЇ RoI: "
             + ", ".join(f"conf≥{c}: {empty[c]} ({100 * empty[c] / n:.0f}%)" for c in confs),
             # при низькому порозі recall росте частково тому, що RoI просто стає багато
             "Середня кількість RoI на знімок з ураженнями: " + ", ".join(f"conf≥{c}: {counts[c] / n:.1f}" for c in confs)]
    csv_rows = []
    groups = {
        "За типом ураження": [(f"{i} {name}", [r for r in records if r["class_id"] == i]) for i, name in enumerate(class_names)],
        "За площею bbox (частка знімка)": [(label, [r for r in records if lo <= r["area"] < hi]) for lo, hi, label in AREA_BINS],
    }
    for title, rows in groups.items():
        lines += ["", title, f"  {'':24} {'bbox':>5}  " + "  ".join(f"{h:>21}" for h in header)]
        for label, group in rows:
            values = recall_row(group)
            lines.append(f"  {label:24} {len(group):>5}  " + "  ".join(f"{v:>21.2f}" for v in values))
            csv_rows.append([title, label, len(group), *[round(v, 4) for v in values]])
    total = recall_row(records)
    lines.append(f"  {'Усі':24} {len(records):>5}  " + "  ".join(f"{v:>21.2f}" for v in total))
    csv_rows.append(["Усі", "Усі", len(records), *[round(v, 4) for v in total]])

    metrics = {f"recall_conf{c}_iou{t}": v for (c, t), v in zip(columns, total)}
    metrics.update({f"empty_share_conf{c}": empty[c] / n for c in confs})
    metrics.update({f"rois_per_image_conf{c}": counts[c] / n for c in confs})
    return lines, ["group_type", "group", "n_boxes", *header], csv_rows, metrics


def froc_curve(images, iou_thr: float):
    """Повертає (хибні RoI на знімок, sensitivity) для всіх порогів conf, від найвищого до найнижчого."""
    gt_best_conf, fp_conf = [], []
    for img in images:
        hit = img["ious"] >= iou_thr  # bbox розмітки x передбачення
        # ураження знайдене при порозі t, якщо його перекриває передбачення з conf >= t
        gt_best_conf += [img["conf"][row].max() if row.any() else -1.0 for row in hit]
        fp_conf += img["conf"][~hit.any(axis=0)].tolist()
    gt_best_conf, fp_conf = np.sort(gt_best_conf), np.sort(fp_conf)
    thresholds = np.unique(np.concatenate([gt_best_conf[gt_best_conf >= 0], fp_conf]))[::-1]
    sensitivity = (len(gt_best_conf) - np.searchsorted(gt_best_conf, thresholds, side="left")) / max(len(gt_best_conf), 1)
    fp_per_image = (len(fp_conf) - np.searchsorted(fp_conf, thresholds, side="left")) / len(images)
    return fp_per_image, sensitivity


def sensitivity_at(fp_per_image, sensitivity, rate: float) -> float:
    """Найвища sensitivity, досяжна при не більше ніж rate хибних RoI на знімок."""
    allowed = fp_per_image <= rate
    return float(sensitivity[allowed].max()) if allowed.any() else 0.0


def union_area(boxes: np.ndarray) -> float:
    """Точна площа об'єднання прямокутників x1, y1, x2, y2 (стиснення координат)."""
    if not len(boxes):
        return 0.0
    xs, ys = np.unique(boxes[:, [0, 2]]), np.unique(boxes[:, [1, 3]])
    covered = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    for x1, y1, x2, y2 in boxes:
        covered[np.searchsorted(ys, y1):np.searchsorted(ys, y2), np.searchsorted(xs, x1):np.searchsorted(xs, x2)] = True
    return float((np.outer(np.diff(ys), np.diff(xs)) * covered).sum())


def expand_boxes(boxes: np.ndarray, pad: float) -> np.ndarray:
    """Розширює кожну рамку на pad її ширини/висоти з кожного боку й обрізає по межах знімка."""
    w, h = boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]
    grown = np.stack([boxes[:, 0] - pad * w, boxes[:, 1] - pad * h, boxes[:, 2] + pad * w, boxes[:, 3] + pad * h], axis=1)
    return np.clip(grown, 0.0, 1.0)


def lesion_coverage(lesion: np.ndarray, rois: np.ndarray) -> float:
    """Частка площі ураження (x1, y1, x2, y2), що лежить усередині об'єднання RoI."""
    area = (lesion[2] - lesion[0]) * (lesion[3] - lesion[1])
    if area <= 0 or not len(rois):
        return 0.0
    clipped = np.column_stack([np.maximum(rois[:, 0], lesion[0]), np.maximum(rois[:, 1], lesion[1]),
                               np.minimum(rois[:, 2], lesion[2]), np.minimum(rois[:, 3], lesion[3])])
    clipped = clipped[(clipped[:, 2] > clipped[:, 0]) & (clipped[:, 3] > clipped[:, 1])]
    return union_area(clipped) / area


def coverage_point(images, conf: float, pad: float, cover: float) -> dict:
    """Покриття уражень і площа RoI при одному порозі conf і розширенні pad."""
    protected, full, area_lesion, area_normal = [], [], [], []
    for img in images:
        rois = expand_boxes(img["boxes"][img["conf"] >= conf], pad)
        area = union_area(rois)
        if len(img["gt"]):
            ok = [lesion_coverage(g[1:], rois) >= cover for g in img["gt"]]
            protected += ok
            full.append(all(ok))
            area_lesion.append(area)
        else:
            area_normal.append(area)
    nan = float("nan")
    return {"protected": float(np.mean(protected)) if protected else nan,
            "full_images": float(np.mean(full)) if full else nan,
            "roi_area_lesion": float(np.mean(area_lesion)) if area_lesion else nan,
            "roi_area_normal": float(np.mean(area_normal)) if area_normal else nan}


def image_roc(images):
    """ROC "чи є на знімку ураження" за найвищою впевненістю RoI. None, якщо немає знімків одного з типів."""
    from sklearn.metrics import roc_auc_score, roc_curve

    labels = np.array([len(img["gt"]) > 0 for img in images])
    if labels.all() or not labels.any():
        return None
    scores = np.array([img["conf"].max() if len(img["conf"]) else 0.0 for img in images])
    fpr, tpr, _ = roc_curve(labels, scores)
    return fpr, tpr, float(roc_auc_score(labels, scores)), int(labels.sum()), int((~labels).sum())


def analyze(dataset: str, weights: Path, split: str = "val", confs=(0.25, 0.1, 0.01), ious=(0.5, 0.3, 0.1),
            imgsz=None, batch: int = 16, cover: float = 0.9, tta: bool = False):
    """Рахує все й зберігає файли. Повертає {"metrics", "files", "tag"} для логування в MLflow.

    tag — назва частини, з суфіксом -tta для TTA; входить у назви файлів і папку артефактів.
    """
    tag = f"{split}-tta" if tta else split
    yolo_dir = REPO_ROOT / "data" / dataset / "yolo"
    class_names = (yolo_dir / "classes.txt").read_text(encoding="utf-8").split("\n")[:-1]
    image_paths = sorted(p for p in (yolo_dir / "images" / split).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    ground_truth = load_ground_truth(yolo_dir / "labels" / split)
    predictions, imgsz = predict(weights, image_paths, imgsz, batch, augment=tta)

    images = []
    for path in image_paths:
        gt = ground_truth.get(path.stem, np.zeros((0, 5)))
        boxes, conf = predictions[path.stem]
        images.append({"gt": gt, "boxes": boxes, "conf": conf, "ious": iou_matrix(gt[:, 1:], boxes)})

    out_dir = output_dir(dataset, weights)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nВаги: {weights}  |  частина: {split}{'  |  TTA' if tta else ''}  |  imgsz: {imgsz}  |  знімків: {len(images)}")

    lines, header, csv_rows, metrics = recall_tables(images, class_names, confs, ious)
    print("\n".join(lines))
    csv_path = out_dir / f"roi-recall-{tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_rows)
    files = [csv_path]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    print("\nFROC (sensitivity при N хибних RoI на знімок):")
    for iou_thr in FROC_IOUS:
        fp, sens = froc_curve(images, iou_thr)
        points = {rate: sensitivity_at(fp, sens, rate) for rate in FROC_FP_RATES}
        cpm = float(np.mean(list(points.values())))
        metrics.update({f"froc_iou{iou_thr}/sens_at_{rate}fp": v for rate, v in points.items()})
        metrics[f"froc_iou{iou_thr}/cpm"] = cpm
        print(f"  IoU≥{iou_thr}: " + ", ".join(f"{rate}: {v:.2f}" for rate, v in points.items()) + f"  |  CPM {cpm:.3f}")
        ax.plot(fp, sens, label=f"IoU ≥ {iou_thr} (CPM {cpm:.3f})")
    ax.set(xscale="log", xlim=(FROC_FP_RATES[0] / 2, FROC_FP_RATES[-1] * 2), ylim=(0, 1),
           xlabel="Хибних RoI на знімок", ylabel="Sensitivity (частка знайдених уражень)",
           title=f"FROC — {dataset}, {tag}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="lower right")
    froc_path = out_dir / f"froc-{tag}.png"
    fig.tight_layout()
    fig.savefig(froc_path, dpi=150)
    plt.close(fig)
    files.append(froc_path)

    roc = image_roc(images)
    if roc is None:
        print("\nROC на рівні знімка: пропущено — у цій частині немає знімків без уражень (або лише такі).")
    else:
        fpr, tpr, auc, n_pos, n_neg = roc
        metrics["image_roc_auc"] = auc
        print(f"\nROC на рівні знімка ({n_pos} з ураженнями, {n_neg} без): AUC {auc:.3f}")
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(fpr, tpr, label=f"AUC {auc:.3f}")
        ax.plot([0, 1], [0, 1], "--", color="grey", linewidth=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False positive rate", ylabel="True positive rate",
               title=f"ROC на рівні знімка — {dataset}, {tag}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")
        roc_path = out_dir / f"roc-{tag}.png"
        fig.tight_layout()
        fig.savefig(roc_path, dpi=150)
        plt.close(fig)
        files.append(roc_path)

    files += coverage_analysis(images, dataset, tag, cover, out_dir, metrics)

    print("\nФайли: " + ", ".join(str(p) for p in files))
    prefix = f"roi_{split}_tta" if tta else f"roi_{split}"
    return {"metrics": {f"{prefix}/{key}": value for key, value in metrics.items()}, "files": files, "tag": tag}


def coverage_analysis(images, dataset: str, split: str, cover: float, out_dir: Path, metrics: dict):
    """Таблиця, CSV і графік "захищено уражень ↔ площа RoI"; дописує метрики в metrics. Повертає файли."""
    has_normal = any(not len(img["gt"]) for img in images)
    points = {(conf, pad): coverage_point(images, conf, pad, cover)
              for pad in COVERAGE_PADS for conf in sorted(set(COVERAGE_CURVE_CONFS) | set(COVERAGE_TABLE_CONFS))}

    print(f"\nПокриття уражень (захищене = ≥{cover:.0%} площі ураження всередині RoI):")
    print(f"  {'conf':>5} {'розширення':>10} | {'уражень захищено':>16} {'хворих знімків повністю':>23} | "
          f"{'площа RoI: хворі':>16} {'нормальні':>10}")
    for conf in COVERAGE_TABLE_CONFS:
        for pad in COVERAGE_PADS:
            pt = points[(conf, pad)]
            normal = f"{pt['roi_area_normal']:>10.1%}" if has_normal else f"{'—':>10}"
            print(f"  {conf:>5} {f'+{pad:.0%}':>10} | {pt['protected']:>16.0%} {pt['full_images']:>23.0%} | "
                  f"{pt['roi_area_lesion']:>16.1%} {normal}")
            key = f"conf{conf}_pad{pad}"
            metrics[f"coverage/protected_{key}"] = pt["protected"]
            metrics[f"coverage/full_images_{key}"] = pt["full_images"]
            metrics[f"coverage/roi_area_lesion_{key}"] = pt["roi_area_lesion"]
            if has_normal:
                metrics[f"coverage/roi_area_normal_{key}"] = pt["roi_area_normal"]

    csv_path = out_dir / f"roi-coverage-{split}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["conf", "pad", f"lesions_protected_cover{cover}", "lesion_images_fully_protected",
                         "roi_area_lesion_images", "roi_area_normal_images"])
        for (conf, pad), pt in sorted(points.items(), key=lambda kv: (kv[0][1], -kv[0][0])):
            writer.writerow([conf, pad, *[round(pt[k], 4) for k in ("protected", "full_images", "roi_area_lesion",
                                                                     "roi_area_normal")]])

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for pad in COVERAGE_PADS:
        curve = [points[(conf, pad)] for conf in COVERAGE_CURVE_CONFS]
        ax.plot([100 * pt["roi_area_lesion"] for pt in curve], [pt["protected"] for pt in curve], marker="o",
                markersize=3, label=f"розширення +{pad:.0%}")
    # підписи порогів conf на одній з кривих, щоб можна було прочитати робочу точку
    for conf in COVERAGE_TABLE_CONFS:
        pt = points[(conf, 0.25)]
        ax.annotate(f"conf {conf}", (100 * pt["roi_area_lesion"], pt["protected"]), textcoords="offset points",
                    xytext=(4, -10), fontsize=7)
    ax.set(ylim=(0, 1), xlabel="Площа RoI, % площі знімка (знімки з ураженнями)",
           ylabel=f"Частка захищених уражень (≥{cover:.0%} площі в RoI)",
           title=f"Покриття уражень — {dataset}, {split}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    png_path = out_dir / f"roi-coverage-{split}.png"
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return [csv_path, png_path]


def log_to_mlflow(result: dict):
    """Пише результат analyze() в активний запуск MLflow (артефакти — у roi-analysis-<tag>/)."""
    mlflow.log_metrics(result["metrics"])
    for path in result["files"]:
        mlflow.log_artifact(str(path), artifact_path=f"roi-analysis-{result['tag']}")


def find_mlflow_run(dataset: str, run_name: str):
    """run_id запуску з такою назвою в експерименті датасету (останнього, якщо їх кілька) або None."""
    mlflow.set_tracking_uri(mlflow_tracking_uri())
    experiment = mlflow.get_experiment_by_name(f"roi-detector/{dataset}")
    if experiment is None:
        return None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="list",
                              filter_string=f"attributes.run_name = '{run_name}'",
                              order_by=["attributes.start_time DESC"], max_results=1)
    return runs[0].info.run_id if runs else None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=DATASETS)
    parser.add_argument("weights", type=Path, help="Ваги детектора, напр. runs/<датасет>/<запуск>/weights/best.pt")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"),
                        help="Частина датасету (test — лише для фінально обраної моделі)")
    parser.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.1, 0.01], help="Пороги впевненості для таблиць recall")
    parser.add_argument("--iou", type=float, nargs="+", default=[0.5, 0.3, 0.1], help="Пороги IoU для таблиць recall")
    parser.add_argument("--imgsz", type=int, default=None, help="Типово — той, з яким модель тренувалась")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--cover", type=float, default=0.9,
                        help="Ураження вважається захищеним, якщо така частка його площі всередині RoI")
    parser.add_argument("--tta", action="store_true",
                        help="Передбачення з аугментацією на інференсі (віддзеркалення + масштаби)")
    args = parser.parse_args()

    result = analyze(args.dataset, args.weights, args.split, args.conf, args.iou, args.imgsz, args.batch, args.cover,
                     args.tta)
    if mlflow is not None:
        run_name = output_dir(args.dataset, args.weights).name + ("" if args.split == "val" else f"-{args.split}")
        run_id = find_mlflow_run(args.dataset, run_name)
        if run_id is None:
            print(f"У MLflow немає запуску '{run_name}' — результат лише у файлах.")
        else:
            with mlflow.start_run(run_id=run_id):
                log_to_mlflow(result)
            prefix = next(iter(result["metrics"])).split("/")[0]
            print(f"Записано в MLflow: запуск '{run_name}', метрики {prefix}/*, артефакти roi-analysis-{result['tag']}/")
