"""
Тренування YOLO-детектора RoI на одному з трьох датасетів (fine-tune ваг, pretrained на COCO).

Датасет має бути вже зібраний у data/<датасет>/yolo/ відповідним скриптом підготовки.
Під час тренування використовуються лише train і val; test — незалежний набір для
фінальної оцінки, він рахується окремим запуском з --eval-test.

Типово всі класи патологій зводяться до одного класу "RoI" (single_cls): для водяного
маркування потрібне лише розташування області інтересу, а не тип патології.

Результати (ваги, графіки, метрики) пишуться в runs/<датасет>/<назва>-<дата>-<час>/,
напр. runs/chestxray14/yolov8n-20261001-2315/. Назва — це --name або, якщо його немає,
назва моделі; дата й час старту додаються завжди, тому запуски не перезаписують один
одного, а папка на диску й запуск у MLflow називаються однаково.

Якщо встановлено MLflow (pip install mlflow), кожен запуск додатково логується в
експеримент "roi-detector/<датасет>": гіперпараметри, метрики по епохах, ваги та графіки —
вбудованою інтеграцією Ultralytics, а цей скрипт додає теги (датасет, git-коміт, етап)
і окремий запуск із метриками на test для --eval-test. База — mlflow.db у корені
репозиторію, артефакти — mlartifacts/ (інше місце: змінна MLFLOW_TRACKING_URI).
Перегляд: mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://127.0.0.1:5000

Використання:
    # sanity check: 10% train-знімків, 5 епох
    python scripts/train_yolo.py chestxray14 --fraction 0.1 --epochs 5 --name sanity
    # повне тренування
    python scripts/train_yolo.py chestxray14 --epochs 100
    # фінальна оцінка на test
    python scripts/train_yolo.py chestxray14 --eval-test runs/chestxray14/yolov8n-20261001-2315/weights/best.pt
"""

import argparse
import os
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from yolo_utils import REPO_ROOT

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("Потрібен Ultralytics: pip install ultralytics")

try:
    import mlflow
except ImportError:
    mlflow = None

DATASETS = ("chestxray14", "fracatlas", "vindr-spinexr")


def data_yaml_path(dataset: str) -> Path:
    path = REPO_ROOT / "data" / dataset / "yolo" / "data.yaml"
    if not path.exists():
        raise SystemExit(f"Немає {path} — спершу зберіть YOLO-датасет скриптом підготовки (див. README.md).")
    return path


def git_commit() -> str:
    result = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


@contextmanager
def mlflow_run(dataset: str, run_name: str, stage: str):
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
    os.environ.setdefault("MLFLOW_TRACKING_URI", f"sqlite:///{(REPO_ROOT / 'mlflow.db').as_posix()}")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = experiment
    os.environ["MLFLOW_RUN"] = run_name

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    if mlflow.get_experiment_by_name(experiment) is None:
        # явне розташування артефактів: типове залежало б від папки, з якої запущено скрипт
        mlflow.create_experiment(experiment, artifact_location=(REPO_ROOT / "mlartifacts" / dataset).as_uri())
    mlflow.set_experiment(experiment)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"dataset": dataset, "stage": stage, "git_commit": git_commit()})
        yield run


def train(args):
    run_name = f"{args.name or Path(args.model).stem}-{datetime.now():%Y%m%d-%H%M}"
    model = YOLO(args.model)
    with mlflow_run(args.dataset, run_name, stage="train"):
        model.train(
            data=str(data_yaml_path(args.dataset)),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            fraction=args.fraction,
            single_cls=not args.multi_class,
            seed=args.seed,
            project=str(REPO_ROOT / "runs" / args.dataset),
            name=run_name,
        )


def eval_test(args):
    run_name = (args.name or args.eval_test.resolve().parent.parent.name) + "-test"
    model = YOLO(args.eval_test)
    with mlflow_run(args.dataset, run_name, stage="test") as run:
        metrics = model.val(
            data=str(data_yaml_path(args.dataset)),
            split="test",
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            single_cls=not args.multi_class,
            project=str(REPO_ROOT / "runs" / args.dataset),
            name=run_name,
        )
        results = {"mAP50": metrics.box.map50, "mAP50-95": metrics.box.map,
                   "precision": metrics.box.mp, "recall": metrics.box.mr}
        if run:
            mlflow.log_params({"weights": str(args.eval_test), "split": "test", "imgsz": args.imgsz,
                               "single_cls": not args.multi_class})
            mlflow.log_metrics({f"test/{key}": float(value) for key, value in results.items()})
            mlflow.log_artifacts(str(metrics.save_dir))
    print("test: " + ", ".join(f"{key} = {value:.4f}" for key, value in results.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", choices=DATASETS, help="Датасет з data/<датасет>/yolo/")
    parser.add_argument("--model", default="yolov8n.pt",
                        help="Початкові ваги Ultralytics (yolov8n.pt, yolov8s.pt, yolo11n.pt, ...)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640, help="Розмір довшої сторони знімка на вході моделі")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4, help="Кількість процесів завантаження даних")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="Частка train-знімків для тренування (менше 1.0 — для sanity check)")
    parser.add_argument("--multi-class", action="store_true",
                        help="Лишити окремі класи патологій замість одного класу RoI")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Назва запуску в runs/<датасет>/ і в MLflow (типово — назва моделі); "
                             "дата й час старту додаються до неї завжди")
    parser.add_argument("--eval-test", type=Path, default=None, metavar="WEIGHTS",
                        help="Не тренувати, а оцінити вказані ваги на test")
    args = parser.parse_args()

    if args.eval_test:
        eval_test(args)
    else:
        train(args)
