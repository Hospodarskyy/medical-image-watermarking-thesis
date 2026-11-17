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
roi_val/* і графіки в артефактах roi-analysis-val/. З --tta аналіз додатково повторюється з
аугментацією на інференсі (метрики roi_val_tta/*, артефакти roi-analysis-val-tta/).

Рідкісні ураження можна показувати частіше: --oversample-rare T вмикає repeat factor sampling
(Gupta et al., LVIS, CVPR 2019). Для класу c з часткою f(c) train-знімків, де він є, множник
r(c) = max(1, sqrt(T / f(c))); знімок повторюється стільки разів, скільки найбільший множник
серед його класів (дробова частина — стохастичне округлення з --seed). Знімки без уражень і
з частими класами не повторюються. Список train пишеться в data/<датасет>/yolo/train-rfs<T>.txt,
конфіг — у data-rfs<T>.yaml поруч; val і test не змінюються.

Оптимізатор: типово optimizer=auto, і Ultralytics сам обирає AdamW з lr0 = 0.002·5/(4+nc), але
якщо кроків тренування більше 10 000 (ceil(train-знімків / 64) · епохи) — MuSGD з lr0 0.01.
Тобто збільшення епох чи датасету може непомітно змінити оптимізатор. Щоб порівнювати
запуски чесно, задавайте його явно: --optimizer AdamW --lr0 0.002 (як auto для одного класу).

Аугментація задається пресетом --aug:
    default      — типові налаштування Ultralytics (розраховані на фото COCO): mosaic 1.0,
                   fliplr 0.5, scale 0.5, translate 0.1, hsv_v 0.4
    xray         — реалістичні для рентгену варіації: без mosaic і віддзеркалення, scale 0.2,
                   translate 0.05, поворот ±5°, яскравість ±20%
    xray-mosaic  — те саме, але mosaic 0.5: компроміс, якщо без mosaic швидко перенавчається
    nomosaic     — default без mosaic: решта стандартної аугментації (fliplr, scale 0.5,
                   translate, hsv_v) лишається як регуляризація
    lowscale     — default зі scale 0.25 (0.75–1.25× замість 0.5–1.5×): окремо ефект масштабу,
                   який зменшує й так дрібні ураження
    spine        — для рентгену хребта: mosaic 0.5, scale 0.25 (0.75–1.25×, щоб не зменшувати
                   й так дрібні ураження вдвічі), поворот ±5° (нахил хребта), яскравість ±20%;
                   віддзеркалення лишається (на прямих знімках хребет майже симетричний)
Для FracAtlas віддзеркалення анатомічно допустиме (ліва/права кінцівка), для грудної клітки
й хребта — ні.

Результати (ваги, графіки, метрики) пишуться в runs/<датасет>/<назва>-<дата>-<час>/,
напр. runs/chestxray14/yolov8n-20261001-2315/. Назва — це --name або, якщо його немає,
назва моделі; дата й час старту додаються завжди, тому запуски не перезаписують один
одного, а папка на диску й запуск у MLflow називаються однаково.

Якщо встановлено MLflow (pip install mlflow), кожен запуск додатково логується в
експеримент "roi-detector/<датасет>": гіперпараметри, метрики по епохах, ваги та графіки —
вбудованою інтеграцією Ultralytics, а цей скрипт додає теги (датасет, git-коміт, етап),
метрики roi_val/* і окремі запуски з метриками для --eval-val / --eval-test. База — mlflow.db у корені
репозиторію, артефакти — mlartifacts/ (інше місце: змінна MLFLOW_TRACKING_URI).
Перегляд: mlflow ui --backend-store-uri sqlite:///mlflow.db  ->  http://127.0.0.1:5000

Перерване тренування (Ctrl+C, вимкнення ПК) продовжується з --resume: Ultralytics після кожної
епохи зберігає weights/last.pt зі станом оптимізатора, номером епохи й EMA, тож тренування
йде далі з наступної епохи з тими самими налаштуваннями — у тій самій папці runs/ і в тому
самому запуску MLflow, а після останньої епохи, як завжди, рахується оцінка RoI на val.
Епоха, на якій тренування перервано, втрачається й проходиться заново. --tta треба вказати
знову, якщо він був; --workers можна змінити, решту налаштувань береться з last.pt.

Використання:
    # sanity check: 10% train-знімків, 5 епох
    python scripts/train_yolo.py chestxray14 --fraction 0.1 --epochs 5 --name sanity
    # повне тренування
    python scripts/train_yolo.py chestxray14 --epochs 100
    # окремі класи з вагами проти дисбалансу (порівнювати з іншими за roi_val/*)
    python scripts/train_yolo.py chestxray14 --epochs 50 --multi-class --cls-pw 0.5
    # оцінка вже натренованих ваг на val як детектора RoI (напр. для старих запусків)
    python scripts/train_yolo.py chestxray14 --eval-val runs/chestxray14/yolov8n-20261001-2315/weights/best.pt
    # більша модель, більший вхід, довше, частіше рідкісні ураження, оптимізатор зафіксовано
    python scripts/train_yolo.py vindr-spinexr --model yolov8s.pt --imgsz 1536 --batch 8 --epochs 80 \
        --oversample-rare 0.1 --optimizer AdamW --lr0 0.002 --tta
    # продовжити перерване тренування
    python scripts/train_yolo.py vindr-spinexr --resume runs/vindr-spinexr/v8s-e80-img1536-rfs-20261117-1211 --tta
    # фінальна оцінка на test — лише для остаточно обраної моделі
    python scripts/train_yolo.py chestxray14 --eval-test runs/chestxray14/yolov8n-20261001-2315/weights/best.pt
"""

import argparse
import math
import os
import random
import subprocess
from collections import Counter
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from analyze_roi_recall import IMAGE_SUFFIXES, analyze, find_mlflow_run, log_to_mlflow
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
    "lowscale": {"scale": 0.25},
    "spine": {"mosaic": 0.5, "scale": 0.25, "degrees": 5.0, "hsv_h": 0.0, "hsv_s": 0.0, "hsv_v": 0.2},
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
def mlflow_run(dataset: str, run_name: str, stage: str, tags: dict = None, run_id: str = None):
    """Відкриває запуск MLflow в експерименті датасету; без встановленого MLflow нічого не робить.

    З run_id продовжує вже наявний запуск (--resume) і лишає його початкові теги.

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
    with mlflow.start_run(**({"run_id": run_id} if run_id else {"run_name": run_name})) as run:
        if not run_id:
            mlflow.set_tags({"dataset": dataset, "stage": stage, "git_commit": git_commit()})
        mlflow.set_tags(tags or {})
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


def oversampled_data_yaml(dataset: str, threshold: float, seed: int):
    """Repeat factor sampling для train (див. опис модуля). Повертає (шлях до yaml, параметри для MLflow)."""
    yolo_dir = REPO_ROOT / "data" / dataset / "yolo"
    class_names = (yolo_dir / "classes.txt").read_text(encoding="utf-8").split("\n")[:-1]
    images = sorted(p for p in (yolo_dir / "images" / "train").iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    classes_per_image = []
    for image in images:
        label = yolo_dir / "labels" / "train" / f"{image.stem}.txt"
        lines = label.read_text(encoding="utf-8").splitlines() if label.exists() else []
        classes_per_image.append({int(line.split()[0]) for line in lines if line.strip()})

    image_count = Counter(c for classes in classes_per_image for c in classes)
    class_repeat = {c: max(1.0, math.sqrt(threshold / (n / len(images)))) for c, n in image_count.items()}
    rng = random.Random(seed)
    entries = []
    for image, classes in zip(images, classes_per_image):
        repeat = max((class_repeat[c] for c in classes), default=1.0)
        copies = int(repeat) + (rng.random() < repeat - int(repeat))  # стохастичне округлення
        entries += [str(image)] * copies

    list_path = yolo_dir / f"train-rfs{threshold}.txt"
    list_path.write_text("\n".join(entries) + "\n", encoding="utf-8", newline="\n")
    base_yaml = data_yaml_path(dataset).read_text(encoding="utf-8")
    assert "train: images/train" in base_yaml, "неочікуваний формат data.yaml"
    yaml_path = yolo_dir / f"data-rfs{threshold}.yaml"
    yaml_path.write_text(base_yaml.replace("train: images/train", f"train: {list_path.name}"),
                         encoding="utf-8", newline="\n")

    print(f"Repeat factor sampling (T={threshold}): train {len(images)} -> {len(entries)} знімків; множники: "
          + ", ".join(f"{class_names[c]} ×{r:.2f}" for c, r in sorted(class_repeat.items())))
    params = {"oversample_rare": threshold, "train_images_original": len(images), "train_images_oversampled": len(entries)}
    params.update({f"repeat_factor.{class_names[c]}": round(r, 3) for c, r in class_repeat.items()})
    return yaml_path, params


def train(args):
    run_name = f"{args.name or Path(args.model).stem}-{datetime.now():%Y%m%d-%H%M}"
    imgsz = args.imgsz or 640
    data_yaml, data_params = data_yaml_path(args.dataset), {}
    if args.oversample_rare:
        data_yaml, data_params = oversampled_data_yaml(args.dataset, args.oversample_rare, args.seed)
    # lr0 передається лише коли заданий: з optimizer=auto Ultralytics його однаково ігнорує
    optimizer_args = {"optimizer": args.optimizer, **({"lr0": args.lr0} if args.lr0 is not None else {})}
    model = YOLO(args.model)
    with mlflow_run(args.dataset, run_name, stage="train", tags={"aug": args.aug}) as run:
        if run and data_params:
            mlflow.log_params(data_params)
        model.train(
            **AUG_PRESETS[args.aug],
            **optimizer_args,
            data=str(data_yaml),
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
        evaluate_trained(model, args, imgsz, run)


def evaluate_trained(model, args, imgsz: int, run):
    """Після тренування: best.pt на val як детектор RoI і аналіз пропусків — у той самий запуск MLflow."""
    run_dir = Path(model.trainer.save_dir)
    _, results = evaluate_roi(YOLO(model.trainer.best), args.dataset, "val", imgsz, args, run_dir / "roi-val")
    analyses = [analyze(args.dataset, Path(model.trainer.best), "val", imgsz=imgsz, batch=args.batch, tta=tta)
                for tta in ([False, True] if args.tta else [False])]
    if run:
        mlflow.log_metrics({f"roi_val/{key}": value for key, value in results.items()})
        for analysis in analyses:
            log_to_mlflow(analysis)
    print_results("val", results)


def resume(args):
    """Продовжує перерване тренування з last.pt (див. опис модуля)."""
    last = args.resume.resolve()
    if last.is_dir():  # передано папку запуску runs/<датасет>/<запуск>/
        last = last / "weights" / "last.pt"
    if not last.exists():
        raise SystemExit(f"Немає {last}")
    model = YOLO(last)
    ckpt = getattr(model, "ckpt", None) or {}
    if ckpt.get("optimizer") is None:  # інакше Ultralytics мовчки почав би нове тренування
        raise SystemExit(f"{last} — ваги вже завершеного тренування, продовжувати нічого")
    train_args = ckpt.get("train_args", {})
    run_name = last.parent.parent.name
    args.batch = train_args.get("batch", args.batch)  # для оцінки після тренування — як у тренуванні
    run_id = find_mlflow_run(args.dataset, run_name) if mlflow else None
    if mlflow and run_id is None:
        print(f"У MLflow немає запуску {run_name} — буде створено новий з цією назвою.")
    print(f"Продовження {run_name} з епохи {ckpt.get('epoch', -1) + 2} з {train_args.get('epochs')}")
    with mlflow_run(args.dataset, run_name, stage="train", tags={"resumed": "true"}, run_id=run_id) as run:
        if run:
            # Ultralytics на старті знову логує всі аргументи тренування, а model і resume тепер інші:
            # MLflow не дає змінювати параметри, і callback Ultralytics вимикає логування епох.
            # Запуск уже відкрито тут, тож логування вмикається назад перед першою епохою.
            model.add_callback("on_train_start", lambda trainer: setattr(trainer, "_mlflow_active", True))
        model.train(resume=True, workers=args.workers)
        evaluate_trained(model, args, train_args.get("imgsz", 640), run)


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
        analyses = [analyze(args.dataset, weights, split, imgsz=imgsz, batch=args.batch, tta=tta)
                    for tta in ([False, True] if args.tta else [False])]
        if run:
            for analysis in analyses:
                log_to_mlflow(analysis)
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
                        help="Пресет аугментації: default (Ultralytics), xray, xray-mosaic, nomosaic, lowscale, spine — див. опис вище")
    parser.add_argument("--oversample-rare", type=float, default=0.0, metavar="T",
                        help="Repeat factor sampling: частіше показувати знімки з рідкісними ураженнями "
                             "(типове значення для спроби — 0.1; 0 — вимкнено)")
    parser.add_argument("--optimizer", default="auto", choices=("auto", "AdamW", "SGD", "MuSGD"),
                        help="auto сам перемикається з AdamW на MuSGD, якщо кроків > 10 000 — див. опис вище")
    parser.add_argument("--lr0", type=float, default=None, help="Початковий learning rate (з --optimizer auto ігнорується)")
    parser.add_argument("--tta", action="store_true",
                        help="Додатково оцінити з аугментацією на інференсі (метрики roi_val_tta/*)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name", default=None, help="Назва запуску в runs/<датасет>/ і в MLflow (типово — назва моделі); "
                             "дата й час старту додаються до неї завжди")
    evaluation = parser.add_mutually_exclusive_group()
    evaluation.add_argument("--resume", type=Path, default=None, metavar="RUN",
                            help="Продовжити перерване тренування: папка запуску runs/<датасет>/<запуск>/ "
                                 "або її weights/last.pt (налаштування беруться звідти, --tta — вказати знову)")
    evaluation.add_argument("--eval-val", type=Path, default=None, metavar="WEIGHTS",
                            help="Не тренувати, а оцінити вказані ваги на val як детектор RoI")
    evaluation.add_argument("--eval-test", type=Path, default=None, metavar="WEIGHTS",
                            help="Не тренувати, а оцінити вказані ваги на test як детектор RoI")
    args = parser.parse_args()

    if not 0.0 <= args.cls_pw <= 1.0:
        parser.error("--cls-pw має бути в межах [0, 1]")
    if args.cls_pw and not args.multi_class:
        parser.error("--cls-pw діє лише з --multi-class: з одним класом RoI зважувати нічого")
    if not 0.0 <= args.oversample_rare < 1.0:
        parser.error("--oversample-rare має бути в межах [0, 1)")
    if args.lr0 is not None and args.optimizer == "auto":
        parser.error("--lr0 з --optimizer auto ігнорується Ultralytics — задайте й --optimizer")

    if args.resume:
        resume(args)
    elif args.eval_val:
        evaluate(args, "val", args.eval_val)
    elif args.eval_test:
        evaluate(args, "test", args.eval_test)
    else:
        train(args)
