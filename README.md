# Магістерська робота — робочий репозиторій

Контекст усього проєкту для Claude Code — у файлі `CLAUDE.md` (читається автоматично при запуску `claude` у цій папці).

## Структура

```
MASTERS_DIPLOMA/
├── CLAUDE.md                  # повний контекст роботи — тема, план, рішення
├── README.md                  # цей файл
├── scripts/
│   ├── prepare_chestxray14.sh              # розпакування архівів + запуск конвертера
│   ├── convert_chestxray_bbox_to_yolo.py   # анотації NIH -> YOLO-датасет (train/val/test по пацієнтах)
│   ├── prepare_fracatlas_yolo.py           # розкладка FracAtlas за офіційним сплітом
│   ├── convert_vindr_spinexr_to_yolo.py    # DICOM + CSV VinDr-SpineXR -> YOLO-датасет (WBF, офіційний test)
│   ├── train_yolo.py                       # тренування YOLO-детектора RoI та оцінка на test
│   └── yolo_utils.py                       # спільні функції (структура папок, data.yaml)
└── data/                      # датасети (не в git — див. .gitignore)
    ├── chestxray14/
    │   ├── raw/               # як завантажено: CSV, archives/*.tar.gz, images/*.png
    │   └── yolo/              # генерується скриптом: images/, labels/, data.yaml, classes.txt
    ├── fracatlas/
    │   ├── raw/               # images/, Annotations/, Utilities/, dataset.csv
    │   └── yolo/
    └── vindr-spinexr/
        ├── raw/               # annotations/{train,test}.csv, train_images/, test_images/ (*.dicom)
        └── yolo/              # генерується скриптом: images/, labels/, classes.txt, data.yaml
```

`raw/` не редагується вручну. `yolo/` можна видалити і відтворити скриптом у будь-який момент
(скрипт щоразу перезбирає `yolo/` з нуля, тому в `raw/` мають бути всі потрібні знімки).

Усі три датасети мають однакову схему `images/{train,val,test}` + `labels/{train,val,test}`,
розподіл скрізь на рівні пацієнта/дослідження:

| Датасет | train / val | test |
|---|---|---|
| ChestX-ray14 | власний спліт по пацієнтах, 70% / 15% (`--val-fraction`) | власний спліт по пацієнтах, 15% (`--test-fraction`) |
| FracAtlas | офіційні `train.csv` / `valid.csv` | офіційний `test.csv` |
| VinDr-SpineXR | офіційний `train.csv`, поділений по `study_id` 90% / 10% (`--val-fraction`) | офіційний `test.csv` цілком |

`val` використовується під час тренування (early stopping, підбір гіперпараметрів). `test` —
незалежний набір лише для фінальної оцінки (зокрема baseline vs. watermarked у Фазі 4):
`yolo detect val data=... split=test`.

У `yolo/images/` ChestX-ray14 та FracAtlas потрапляють лише анотовані знімки: знімок без файлу
міток Ultralytics вважає фоном. У VinDr-SpineXR знімки "No finding" додаються навмисно —
як негативні приклади з порожнім файлом міток.

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
   Після розпакування нових архівів конвертер запускається ще раз — розподіл пацієнтів на train/val/test не зміниться.

4. Встановити Ultralytics:
   ```bash
   pip install ultralytics
   ```

5. Далі — тренування, див. розділ "Тренування YOLO-детектора RoI".

## Тренування YOLO-детектора RoI

Один скрипт для всіх трьох датасетів (`chestxray14`, `fracatlas`, `vindr-spinexr`):

```bash
# sanity check на малій підмножині перед повним прогоном
python scripts/train_yolo.py chestxray14 --fraction 0.1 --epochs 5 --name sanity
# повне тренування
python scripts/train_yolo.py chestxray14 --epochs 100
# фінальна оцінка на незалежному test
python scripts/train_yolo.py chestxray14 --eval-test runs/chestxray14/yolov8n/weights/best.pt
```

Типово всі класи патологій зводяться до одного класу "RoI" (`single_cls`); окремі класи
лишає `--multi-class`. Модель обирається через `--model` (типово `yolov8n.pt`).
Результати пишуться в `runs/<датасет>/<name>/` (не в git).

### Логування експериментів (MLflow)

```bash
pip install mlflow
mlflow ui --backend-store-uri sqlite:///mlflow.db    # http://127.0.0.1:5000
```

Якщо MLflow встановлено, `train_yolo.py` логує кожен запуск в експеримент
`roi-detector/<датасет>`: гіперпараметри, метрики по епохах (loss, mAP, precision, recall, lr),
ваги й графіки, а також теги `dataset`, `stage` (`train`/`test`) і `git_commit`.
`--eval-test` створює окремий запуск `<name>-test` з метриками `test/*`.
База — `mlflow.db`, артефакти — `mlartifacts/` у корені репозиторію (обидва не в git);
інше місце задається змінною `MLFLOW_TRACKING_URI`. Без MLflow скрипт працює як раніше.

## FracAtlas

Розпакувати датасет у `data/fracatlas/raw/` (так, щоб `images/`, `Annotations/`, `Utilities/`
лежали безпосередньо в `raw/`) і запустити:

```bash
python scripts/prepare_fracatlas_yolo.py
```

## VinDr-SpineXR

Розпакувати датасет у `data/vindr-spinexr/raw/` (так, щоб `train_images/`, `test_images/`,
`annotations/` лежали безпосередньо в `raw/`) і запустити:

```bash
pip install pydicom ensemble-boxes Pillow
python scripts/convert_vindr_spinexr_to_yolo.py
```

Повна конвертація 10 466 DICOM триває близько 35 хв на 6 процесах (`--workers`).
Швидка перевірка на частині даних: `--limit 100 --out data/vindr-spinexr/yolo-sanity`.

Що робить конвертер:
- **Спліт:** офіційний `test.csv` цілком → `test`; `train.csv` ділиться на `train`/`val`
  по `study_id` (`--val-fraction 0.1 --seed 0`). Офіційний розподіл уже patient-level
  (`study_id` у `train.csv` і `test.csv` не перетинаються).
- **Зображення:** DICOM → 8-бітний PNG у повній роздільності: VOI LUT windowing
  (`apply_voi_lut`), інверсія для `MONOCHROME1`, розтягування до [0, 255]. У частини файлів
  `WindowCenter`/`WindowWidth` записані з зайвою комою (`"32688,"`) — конвертер це виправляє.
- **Розмір зображення** береться з DICOM (у CSV його немає).
- **Bbox:** кути обрізаються по межах зображення, дублікати анотацій (bbox-и одного класу
  з IoU > 0.5) зливаються через Weighted Box Fusion (`--iou-thr`), потім координати
  перераховуються в center/width/height.
- **"No finding"** → порожній `.txt` (негативний приклад).
- 7 класів: Disc space narrowing, Foraminal stenosis, Osteophytes, Other lesions,
  Spondylolysthesis, Surgical implant, Vertebral collapse.

Кожен знімок розмічений рівно одним радіологом (`rad_id`), тому WBF тут не зливає думки
кількох радіологів, а усуває дублікати анотацій одного радіолога. Окремі ураження одного
класу на одному знімку (multi-instance, напр. Osteophytes на різних рівнях хребта) лишаються
окремими bbox: у `train.csv` з 3403 груп "знімок + клас" із кількома bbox 97.71% мають
max IoU < 0.1, і лише 1.62% — max IoU > 0.5 (bbox введено двічі), їх WBF і зливає.

Датасет вимагає credentialed-доступу через PhysioNet (DUA вже підписано).
Дотримуватись умов угоди при роботі з даними — див. CLAUDE.md, розділ
"Обмеження доступу до VinDr-SpineXR".
