# Магістерська робота — робочий репозиторій

Контекст усього проєкту для Claude Code — у файлі `CLAUDE.md` (читається автоматично при запуску `claude` у цій папці).

## Структура

```
MASTERS_DIPLOMA/
├── CLAUDE.md                  # повний контекст роботи — тема, план, рішення
├── README.md                  # цей файл
├── scripts/
│   ├── prepare_chestxray14.sh              # розпакування архівів + запуск конвертера
│   ├── convert_chestxray_bbox_to_yolo.py   # анотації NIH -> YOLO-датасет (спліт по пацієнтах)
│   ├── prepare_fracatlas_yolo.py           # розкладка FracAtlas за офіційним сплітом
│   └── yolo_utils.py                       # спільні функції (структура папок, data.yaml)
└── data/                      # датасети (не в git — див. .gitignore)
    ├── chestxray14/
    │   ├── raw/               # як завантажено: CSV, archives/*.tar.gz, images/*.png
    │   └── yolo/              # генерується скриптом: images/, labels/, data.yaml
    ├── fracatlas/
    │   ├── raw/               # images/, Annotations/, Utilities/, dataset.csv
    │   └── yolo/
    └── vindr-spinexr/
        └── raw/               # annotations/, далі train_images/, test_images/
```

`raw/` не редагується вручну. `yolo/` можна видалити і відтворити скриптом у будь-який момент.
У `yolo/images/` потрапляють лише анотовані знімки: знімок без файлу міток Ultralytics вважає фоном.

## Швидкий старт (ChestX-ray14)

1. Завантажити `Data_Entry_2017_v2020.csv` та `BBox_List_2017.csv` у `data/chestxray14/raw/`,
   архіви зображень — у `data/chestxray14/raw/archives/`
   (https://nihcc.app.box.com/v/ChestXray-NIHCC). 880 анотованих знімків розкидані
   по всіх 12 архівах (у першому їх лише 37).

2. Розпакувати архіви та зібрати YOLO-датасет однією командою (Git Bash або WSL):
   ```bash
   bash scripts/prepare_chestxray14.sh
   ```
   Скрипт розпаковує лише нові архіви (розпакований позначається файлом `<архів>.extracted`)
   і запускає конвертер. Інший інтерпретатор: `PYTHON=.venv/bin/python bash scripts/prepare_chestxray14.sh`.

3. Те саме вручну:
   ```bash
   tar -xzf data/chestxray14/raw/archives/images_001.tar.gz -C data/chestxray14/raw
   python scripts/convert_chestxray_bbox_to_yolo.py
   ```
   Після розпакування нових архівів конвертер запускається ще раз — розподіл пацієнтів на train/val не зміниться.

4. Встановити Ultralytics:
   ```bash
   pip install ultralytics
   ```

5. Sanity-check тренування на малій підмножині (5-10 епох) перед повним прогоном:
   ```bash
   yolo detect train data=data/chestxray14/yolo/data.yaml model=yolov8n.pt epochs=5 single_cls=True
   ```
   `single_cls=True` — усі 8 класів патологій зводяться до одного класу "RoI".

## FracAtlas

Розпакувати датасет у `data/fracatlas/raw/` (так, щоб `images/`, `Annotations/`, `Utilities/`
лежали безпосередньо в `raw/`) і запустити:

```bash
python scripts/prepare_fracatlas_yolo.py
```

## Примітка щодо VinDr-SpineXR

Датасет вимагає credentialed-доступу через PhysioNet (DUA вже підписано).
Дотримуватись умов угоди при роботі з даними — див. CLAUDE.md, розділ
"Обмеження доступу до VinDr-SpineXR".
