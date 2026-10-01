# Магістерська робота — робочий репозиторій

Контекст усього проєкту для Claude Code — у файлі `CLAUDE.md` (читається автоматично при запуску `claude` у цій папці).

## Структура

```
MASTERS_DIPLOMA/
├── CLAUDE.md                  # повний контекст роботи — тема, план, рішення
├── README.md                  # цей файл
├── scripts/
│   └── convert_chestxray_bbox_to_yolo.py   # конвертер анотацій NIH -> YOLO формат
├── data/                      # сюди завантажувати датасети (не в git!)
│   ├── chestxray14/
│   ├── fracatlas/
│   └── vindr-spinexr/
└── configs/                   # YOLO data.yaml конфіги (створити по аналогії для кожного датасету)
```

## Швидкий старт (ChestX-ray14)

1. Завантажити `Data_Entry_2017_v2020.csv`, `BBox_List_2017.csv` та 1-2 архіви зображень
   з https://nihcc.app.box.com/v/ChestXray-NIHCC у `data/chestxray14/`

2. Розпакувати зображення в `data/chestxray14/images/`

3. Конвертувати bbox-анотації у формат YOLO:
   ```bash
   python scripts/convert_chestxray_bbox_to_yolo.py \
       --bbox-csv data/chestxray14/BBox_List_2017.csv \
       --images-dir data/chestxray14/images \
       --labels-out data/chestxray14/labels \
       --classes-out data/chestxray14/classes.txt
   ```

4. Встановити Ultralytics:
   ```bash
   pip install ultralytics
   ```

5. Sanity-check тренування на малій підмножині (5-10 епох, невелика кількість зображень)
   перед повним прогоном — перевірити коректність конвертації та відсутність помилок пайплайну.

## Примітка щодо VinDr-SpineXR

Датасет вимагає credentialed-доступу через PhysioNet (DUA вже підписано).
Дотримуватись умов угоди при роботі з даними — див. CLAUDE.md, розділ
"Обмеження доступу до VinDr-SpineXR".
