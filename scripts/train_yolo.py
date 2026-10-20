"""
Тренування YOLO-детектора RoI на одному з трьох датасетів (fine-tune ваг, pretrained на COCO).

Датасет має бути вже зібраний у data/<датасет>/yolo/ відповідним скриптом підготовки.
Під час тренування використовуються лише train і val; test — незалежний набір для
фінальної оцінки, він рахується окремим запуском з --eval-test.

Типово всі класи патологій зводяться до одного класу "RoI" (single_cls): для водяного
маркування потрібне лише розташування області інтересу, а не тип патології.
З --multi-class модель вчиться розрізняти класи (а --cls-pw додає ваги класів проти
дисбалансу), але її mAP під час тренування — середнє по класах і з одноклассовою моделлю
не порівнюваний. Тому після кожного тренування best.pt додатково оцінюється на val як
детектор RoI — клас ігнорується, рахується лише локалізація (метрики roi_val/*).
Так само, як детектор RoI, оцінюють --eval-val і --eval-test. Саме roi_val/* — метрика
для порівняння будь-яких моделей між собою. Разом з цим запускається analyze_roi_recall.py:
recall по типах уражень, FROC і ROC на рівні знімка (де є знімки без уражень) — метрики
roi_val/* і графіки в артефактах roi-analysis-val/.

Аугментація задається пресетом --aug:
    default      — типові налаштування Ultralytics (розраховані на фото COCO): mosaic 1.0,
                   fliplr 0.5, scale 0.5, translate 0.1, hsv_v 0.4
    xray         — реалістичні для рентгену варіації: без mosaic і віддзеркалення, scale 0.2,
                   translate 0.05, поворот ±5°, яскравість ±20%
    xray-mosaic  — те саме, але mosaic 0.5: компроміс, якщо без mosaic швидко перенавчається
    nomosaic     — default без mosaic: решта стандартної аугментації (fliplr, scale 0.5,
                   translate, hsv_v) лишається як регуляризація
Для FracAtlas віддзеркалення анатомічно допустиме (ліва/права кінцівка), для грудної клітки
й хребта — ні.

Результати (ваги, графіки, метрики) пишуться в runs/<датасет>/<назва>-<дата>-<час>/,
напр. runs/chestxray14/yolov8n-20261001-2315/. Назва — це --name або, якщо його немає,
назва моделі; дата й час старту додаються завжди, тому запуски не перезаписують один
одного, а папка на диску й запуск у MLflow називаються однаково.

Якщо встановлено MLflow (pip install mlflow), кожен запуск додатково логується в
експеримент "roi-detector/<датасет>": гіперпараметри, метрики по епохах, ваги та графіки —
вбудованою інтеграцією Ultralytics, а цей скрипт додає теги (датасет, git-коміт, етап)
метрики roi_val/* і окремі запуски з метриками для --eval-val / --eval-test. База — mlflow.db у корені
репозиторію, артефакти — mlartifacts/ (інше місце: змінна MLFLOW_TRACKING_URI).
Перегляд: mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://127.0.0.1:5000

Використання:
    # sanity check: 10% train-знімків, 5 епох
    python scripts/train_yolo.py chestxray14 --fraction 0.1 --epochs 5 --name sanity
    # повне тренування
    python scripts/train_yolo.py chestxray14 --epochs 100
    # окремі класи з вагами проти дисбалансу (порівнювати з іншими за roi_val/*)
    python scripts/train_yolo.py chestxray14 --epochs 50 --multi-class --cls-pw 0.5
    # оцінка вже натренованих ваг на val як детектора RoI (напр. для старих запусків)
    python scripts/train_yolo.py chestxray14 --eval-val runs/chestxray14/yolov8n-20261001-2315/weights/best.pt
    # фінальна оцінка на test — лише для остаточно обраної моделі
    python scripts/train_yolo.py chestxray14 --eval-test runs/chestxray14/yolov8n-20261001-2315/weights/best.pt
"""

import argparse
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from analyze_roi_recall import analyze, log_to_mlflow
from yolo_utils import REPO_ROOT, mlflow_tracking_uri

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Потрібен Ultralytics: pip install ultralytics")

try:
    import mlflow
except ImportError:
    mlflow = None

DATASETS = ("chestxray14", "fracatlas", "vindr-spinexr")

XRAY_AUG = {"mosaic": 0.0, "fliplr": 0.0, "scale": 0.2, "translate": 0.05, "degrees": 5.0,
            "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.2}
AUG_PRESETS = {
    "default": {},  # типові значення Ultralytics
    "xray": XRAY_AUG,
    "xray-mosaic": {**XRAY_AUG, "mosaic": 0.5},
    "nomosaic": {"mosaic": 0.0},
}


def data_yaml_path(dataset: str) -> Path:
    path = REPO_ROOT / "data" / dataset / "yolo" / "data.yaml"
    if not path.exists():
        raise SystemExit(f"Немає {path} — спершу зберіть YOLO-датасет скриптом підготовки (див. README.md).")
    return path


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


@contextmanager
def mlflow_run(dataset: str, run_name: str, stage: str, tags: dict = None):
    """Відкриває запуск MLflow в експерименті датасету; без встановленого MLflow нічого не робить.

    Вбудований callback Ultralytics під час тренування підхоплює вже активний запуск і пише
    в нього параметри, метрики та артефакти. Експеримент і запуск він шукає через змінні
    середовища, тому вони виставляються тут.
    """
    if mlflow is None:
        print("MLflow не встановлено — запуск логується лише в runs/ (pip install mlflow).")
        yield None
        return

    experiment = f"roi-detector/{dataset}"
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri()
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
    os.environ["MLFLOW_RUN"] = run_name

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if mlflow.get_experiment_by_name(experiment) is None:
        # явне розташування артефактів: типове залежало б від папки, з якої запущено скрипт
        mlflow.create_experiment(experiment, artifact_location=(REPO_ROOT / "mlartifacts" / dataset).as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"dataset": dataset, "stage": stage, "git_commit": git_commit(), **(tags or {})})
        yield run


def evaluate_roi(model, dataset: str, split: str, imgsz: int, args, save_dir: Path):
    """Оцінює модель як детектор RoI: клас передбачення й розмітки ігнорується.

    single_cls=True у валідації Ultralytics зводить і розмітку, і передбачення до класу 0
    (і робить NMS незалежним від класу), тому для багатокласової моделі рахується лише,
    чи знайдено область, а не чи правильно названо ураження.
    Повертає (результат валідації, {метрика: значення}).
    """
    metrics = model.val(
        data=str(data_yaml_path(dataset)),
        split=split,
        imgsz=imgsz,
        batch=args.batch,
        workers=args.workers,
        single_cls=True,
        project=str(save_dir.parent),
        name=save_dir.name,
        exist_ok=True,
    )
    results = {"mAP50": metrics.box.map50, "mAP50-95": metrics.box.map,
               "precision": metrics.box.mp, "recall": metrics.box.mr}
    return metrics, {key: float(value) for key, value in results.items()}


def print_results(split: str, results: dict):
    print(f"RoI {split}: " + ", ".join(f"{key} = {value:.4f}" for key, value in results.items()))


def train(args):
    run_name = f"{args.name or Path(args.model).stem}-{datetime.now():%Y%m%d-%H%M}"
    imgsz = args.imgsz or 640
    model = YOLO(args.model)
    with mlflow_run(args.dataset, run_name, stage="train", tags={"aug": args.aug}) as run:
        model.train(
            **AUG_PRESETS[args.aug],
            data=str(data_yaml_path(args.dataset)),
            epochs=args.epochs,
            imgsz=imgsz,
            batch=args.batch,
            workers=args.workers,
            fraction=args.fraction,
            single_cls=not args.multi_class,
            cls_pw=args.cls_pw,
            seed=args.seed,
            project=str(REPO_ROOT / "runs" / args.dataset),
            name=run_name,
        )
        run_dir = Path(model.trainer.save_dir)
        _, results = evaluate_roi(YOLO(model.trainer.best), args.dataset, "val", imgsz, args, run_dir / "roi-val")
        analysis = analyze(args.dataset, Path(model.trainer.best), "val", imgsz=imgsz, batch=args.batch)
        if run:
            mlflow.log_metrics({f"roi_val/{key}": value for key, value in results.items()})
            log_to_mlflow(analysis, "val")
    print_results("val", results)


def evaluate(args, split: str, weights: Path):
    """Окремий запуск оцінки вже натренованих ваг на val або test."""
    weights = weights.resolve()
    if weights.parent.name == "weights":  # runs/<датасет>/<запуск>/weights/best.pt
        run_dir = weights.parent.parent
    else:  # ваги поза runs/, напр. завантажені готові
        run_dir = REPO_ROOT / "runs" / args.dataset / weights.stem
    run_name = (args.name or run_dir.name) + f"-{split}"
    model = YOLO(weights)
    train_args = (getattr(model, "ckpt", None) or {}).get("train_args", {})
    imgsz = args.imgsz or train_args.get("imgsz", 640)  # типово — той самий розмір, що й під час тренування
    with mlflow_run(args.dataset, run_name, stage=split) as run:
        metrics, results = evaluate_roi(model, args.dataset, split, imgsz, args, run_dir / f"roi-{split}")
        analysis = analyze(args.dataset, weights, split, imgsz=imgsz, batch=args.batch)
        if run:
            log_to_mlflow(analysis, split)
            mlflow.log_params({"weights": str(weights), "split": split, "imgsz": imgsz})
            mlflow.log_params({f"train.{key}": train_args[key] for key in
                               ("model", "epochs", "imgsz", "single_cls", "cls_pw", "seed", "fraction")
                               if key in train_args})
            mlflow.log_metrics({f"roi_{split}/{key}": value for key, value in results.items()})
            mlflow.log_artifacts(str(metrics.save_dir))
    print_results(split, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=DATASETS, help="Датасет з data/<датасет>/yolo/")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Початкові ваги Ultralytics (yolov8n.pt, yolov8s.pt, yolo11n.pt, ...)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Розмір довшої сторони знімка на вході моделі (тренування — типово 640, "
                             "оцінка — типово той, з яким модель тренувалась)")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="Кількість процесів завантаження даних")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Частка train-знімків для тренування (менше 1.0 — для sanity check)")
    parser.add_argument("--multi-class", action="store_true",
                        help="Лишити окремі класи патологій замість одного класу RoI")
    parser.add_argument("--cls-pw", type=float, default=0.0,
                        help="Ваги класів проти дисбалансу: 0 — вимкнено, 1 — обернено пропорційні частоті "
                             "класу (Ultralytics cls_pw). Лише разом з --multi-class")
    parser.add_argument("--aug", choices=AUG_PRESETS, default="default",
                        help="Пресет аугментації: default (Ultralytics), xray, xray-mosaic, nomosaic — див. опис вище")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Назва запуску в runs/<датасет>/ і в MLflow (типово — назва моделі); "
                             "дата й час старту додаються до неї завжди")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--eval-val", type=Path, default=None, metavar="WEIGHTS",
                            help="Не тренувати, а оцінити вказані ваги на val як детектор RoI")
    evaluation.add_argument("--eval-test", type=Path, default=None, metavar="WEIGHTS",
                            help="Не тренувати, а оцінити вказані ваги на test як детектор RoI")
    args = parser.parse_args()

    if not 0.0 <= args.cls_pw <= 1.0:
        parser.error("--cls-pw має бути в межах [0, 1]")
    if args.cls_pw and not args.multi_class:
        parser.error("--cls-pw діє лише з --multi-class: з одним класом RoI зважувати нічого")

    if args.eval_val:
        evaluate(args, "val", args.eval_val)
    elif args.eval_test:
        evaluate(args, "test", args.eval_test)
    else:
        train(args)
